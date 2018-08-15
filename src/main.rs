use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::Ordering::AcqRel;
use std::sync::atomic::Ordering::Release;

extern crate core;
use core::marker::PhantomData;

use std::sync::Arc;
use std::option::Option;

// Lock-free atomic Option<Arc<T>> based on Folly::AtomicSharedPtr
//
// struct AtomicOptionArc
//
//      new
//      swap
//      store
//      load
//      compare_exchange
//      compare_and_swap
//      compare_exchange_weak
//      drop
//      default 

// Recreate implementation detail of Arc so we can get at its refcount.
// Obviously wicked

struct ArcInner<T : ?Sized> {
    strong : AtomicUsize,
    weak : AtomicUsize,
    value : T,
}

// AtomicOptionArc packs a pointer and a count into a single usize
// Lower 48: pointer
// Upper 16: count
// Requires a 64 bit architecture with 16 high bits unused (x86_64, AArch64)
// Todo: for other architectures, use alignment low bits

struct AtomicOptionArc<T> {
    ptr: AtomicUsize,         // *const ArcInner<T> : 48, usize : 16
    phantom : PhantomData<T>, // Use the type parameter
}

// Useful constants to help us pack and unpack the pointer and count to/from
// the atomic integer

const SHIFT : usize = 48;
const MASK  : usize = (1 << SHIFT) - 1;
const DELTA : usize = (1 << 16);
const LIMIT : usize = DELTA - 1;

fn into_usize<T>(val: Option<Arc<T>>) -> usize {
    match val {
        None => 0,
        Some(b) => Arc::into_raw(b) as usize,
    }
}

fn into_option_arc<T>(n: usize) -> Option<Arc<T>> {
    match n {
        0 => None,
        _ => Some(unsafe { Arc::from_raw((n & MASK) as *const T) } ),
    }
}

fn inc_global(n: usize, delta: usize) -> usize {
    assert!(delta > 0);
    match n {
        0 => 0,
        _ => {
            let p = ((n & MASK) - 16) as *const ArcInner<()>;
            let m = unsafe { (*p).strong.fetch_add(delta, Relaxed) };
            assert!(m > 0);
            //println!("{} -> {}", m, m + delta);
            m
        },
    }
}

fn dec_global(n: usize, delta: usize) -> usize {
    assert!(delta > 0);
    match n {
        0 => 0,
        _ => { 
            let p = ((n & MASK) - 16) as *const ArcInner<()>;
            let m = unsafe { (*p).strong.fetch_sub(delta, Release) };
            assert!(m > delta);
            //println!("{} -> {}", m, m - delta);
            m
        }
    }
}

impl<T> AtomicOptionArc<T> {

    pub fn new(data: Option<Arc<T>>) -> AtomicOptionArc<T> {
        let n = into_usize(data);
        inc_global(n, LIMIT);
        AtomicOptionArc {
            ptr: AtomicUsize::new(n), 
            phantom:PhantomData
        }
    }

    fn swap(&self, val: Option<Arc<T>>) -> Option<Arc<T>> {
        let a = into_usize(val);          // Convert to packed integer
        inc_global(a, LIMIT);             // Add (2 << 16) - 1 owners
        let b = self.ptr.swap(a, AcqRel); // Swap atomically
        let n = b >> SHIFT;               // Extract count of ownerships relinquished
        dec_global(b, LIMIT - n);         // Subtract (2 << 16) - 1 - n owners
        into_option_arc(b)                // Convert from packed integer
    }

    fn store(&self, val: Option<Arc<T>>) {
        self.swap(val);
        // Old value is dropped here
        // Todo: we are touching .strong twice, in dec_global and then in
        // Arc::drop, here and in many other places
    }

    fn load(&self) -> Option<Arc<T>> {      

        // To load we must increment the local count.  The atomic then holds
        // one less unit of ownership, which is used by the returned Arc
        //
        // It is almost sufficient to perform
        //
        //     self.ptr.fetch_add(1 << SHIFT, Acquire)
        //
        // but we must handle the occasional overflow of the 16-bit counter,
        // which occurs when we perform 64k loads without any exchanges.  This
        // infrequent path locks other loads (including the load in
        // compare_exchange), so we are "mostly" lock-free
        
        let mut desired : usize;
        let mut expected : usize = self.ptr.load(Relaxed);
        loop {
            if expected == 0 {
                // Null pointer; we are done
                return None;
            }
            if expected >> SHIFT == LIMIT {
                // Count is saturated; anticipate another thread resetting it
                expected &= MASK;
                std::thread::yield_now();
            }
            // Increment the count
            desired = expected + (1 << SHIFT);
            match self.ptr.compare_exchange_weak(expected, desired, Acquire, Relaxed) {
                Ok(_) => break, // Successfully incremented
                Err(x) => { expected = x } // Start over
            }
        }

        if desired >> SHIFT == LIMIT {
            // We saturated the local counter and other threads cannot proceed
            // Add the local count to the global count
            inc_global(desired, LIMIT);
            // Try to set the local count to zero
            expected = desired;
            desired = expected & MASK;
            match self.ptr.compare_exchange(expected, desired, AcqRel, Relaxed) {
                Ok(_) => {
                    // Successfully transferred local count to global count
                },
                Err(_) => {
                    // A strong failure means the value really changed, so
                    // either it is now a different object, or a different and
                    // thus non-saturated count.  No longer our problem.

                    // Undo the refcount manipulation
                    dec_global(desired, LIMIT);
                }
            }
        }

        // If we were more aggressive about attempting to zero the counter
        // (for example, when more than half-full) we would become lock-free
        // but at the cost of increased contention.  Profile.

        into_option_arc::<T>(desired)
    }

    fn compare_exchange(&self, current: Option<Arc<T>>, new: Option<Arc<T>>) 
        -> Result<Option<Arc<T>>, Option<Arc<T>>>
    {
        let new2 = into_usize(new);
        inc_global(new2, LIMIT);
        let cur2 = into_usize(current);
        let mut expected : usize = self.ptr.load(Relaxed);
        loop {
            if (expected & MASK) == cur2 {

                // Expected value.  Try to swap in new value.

                match self.ptr.compare_exchange_weak(expected, new2, AcqRel, Relaxed) {
                    Ok(_) => {
                        let n = expected >> SHIFT;
                        dec_global(cur2, DELTA - n);
                        break Ok(into_option_arc(cur2));
                    },
                    Err(x) => {
                        expected = x; // Try again
                    }
                }

            } else {

                // Unexpected value.  We need to take ownership of it.  See load

                if expected == 0 {
                    // Null pointer.  We are done.
                    dec_global(new2, LIMIT);
                    into_option_arc::<T>(new2);
                    into_option_arc::<T>(cur2);
                    return Err(None);
                }

                // Build increment
                let new3 = expected + (1 << SHIFT);                
                // Todo: handle counter saturation here, as in load
                assert!(new3 >> SHIFT != LIMIT); 
                
                // Install increment
                match self.ptr.compare_exchange_weak(expected, new3, Acquire, Relaxed) {
                    Ok(_) => {
                        dec_global(new2, LIMIT);
                        into_option_arc::<T>(new2);
                        into_option_arc::<T>(cur2);
                        break Err(into_option_arc(new3));
                    },
                    Err(x) => {
                        expected = x; // Try again
                    }
                }
            }
        }
    }

    fn compare_and_swap(&self, current: Option<Arc<T>>, new: Option<Arc<T>>) 
    -> Option<Arc<T>> {
        match self.compare_exchange(current, new) {
            Ok(x) => x,
            Err(x) => x,
        }
    }

    fn compare_exchange_weak(&self, current: Option<Arc<T>>, new: Option<Arc<T>>)
    -> Result<Option<Arc<T>>, Option<Arc<T>>> {
        self.compare_exchange(current, new)
    }

}

impl<T> Default for AtomicOptionArc<T> {
    fn default() -> AtomicOptionArc<T> {
        AtomicOptionArc { 
            ptr: AtomicUsize::new(0), 
            phantom: PhantomData,
        }
    } 
}

impl<T> Drop for AtomicOptionArc<T> {
    fn drop(&mut self) {
        self.store(None);
    }
}

// Singly-linked-list Node for a simple unbounded concurrent Stack

struct Node<T : std::fmt::Debug> {
    next: Option<Arc<Node<T>>>,
    value: T,
}

impl<T : Clone + std::fmt::Debug> Node<T> {
    fn new(val: T) -> Node<T> {
        Node { next: None, value: val }
    }
}

impl<T : std::fmt::Debug> Drop for Node<T> {
    fn drop(&mut self) {
        //println!("Dropping {:?}", self.value);
        // Automatic drop is correct, but it recurses along a linked list and
        // thus tends to blow the stack.  We have to convert the iteration
        // into recursion.  If we are the sole owner of the next node, we take
        // its next field and overwrite our own.  That node is then deleted
        // with no dependents.  We keep looping until we hit a node that
        // somebody else owns too.

        loop {
            let tmp : Option<Arc<Node<T>>>;
            match self.next {
                None => return,
                Some(ref mut a) => {
                    match Arc::get_mut(a) {
                        None => return,
                        Some(ref mut nn) => {
                            tmp = nn.next.take();
                        },
                    }
                }
            }
            self.next = tmp;
        }
    }
}

// Simple unbounded concurrent Stack using AtomicOptionArc
//
// Does not suffer from the ABA problem because the (atomic) Arcs guarantee
// memory is not recycled while we hold any pointers to it

struct Stack<T : std::fmt::Debug> {
    head: AtomicOptionArc<Node<T>>,
}

impl<T : Clone + std::fmt::Debug> Stack<T> {

    fn push(&self, val: T) {
        let mut current = self.head.load();
        loop {
            let new  = Some(Arc::new(Node { next: current.clone(), value: val.clone() }));
            match self.head.compare_exchange(current, new) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
            // Todo: On failure, the newly allocated node is discarded, which
            // is wasteful.  Work out how to reuse it; probably requires
            // unsafe code.
        }
    }

    fn pop(&self) -> Option<T> {
        let mut current = self.head.load();
        loop {
            let tmp : Option<Arc<Node<T>>>;
            match current {
                Some(ref node) => {
                    let new = node.next.clone();
                    let payload = node.value.clone();
                    match self.head.compare_exchange(current.clone(), new) {
                        Ok(_) => return Some(payload),
                        Err(actual) => tmp = actual, 
                    }
                    // Todo: Can we avoid these clones?
                },
                None => return None
            }
            current = tmp;
        }
    }

}

impl<T : std::fmt::Debug> Default for Stack<T> {
    fn default() -> Stack<T> {
        Stack { head: AtomicOptionArc::default() }
    }
}


fn main() {

    // Exercise the Stack.  This doesn't prove anything but does catch some
    // basic bugs.  Note that the debugging output introduces extra
    // synchronization between the threads and tends to serialize everything.

    let arcstack : Arc<Stack<usize>> = Arc::new(Stack::default());

    let mut v : Vec<std::thread::JoinHandle<()>> = Vec::default();

    for k in 0..16 {
        let s = arcstack.clone();
        v.push(std::thread::spawn(move || {
            //let mut u : Vec<Option<usize>> = Vec::default();
            for i in 0..4 {
                let j = i * 2 + k * 8;
                s.push(j);
                println!("Pushed {}", j);
                s.push(j + 1);
                println!("Pushed {}", j + 1);
                let p = s.pop();
                println!("Popped {:?}", p)
                //u.push(p);
            }
            for _ in 0..5 {
                let p = s.pop();
                println!("Popped {:?}", p)
                //u.push(p);
            }
            //println!("Thread {:?}", k);
            //for x in u {
            //    println!("Popped {:?}", x);
            //}
        }));
    }

    for h in v {
        match h.join() {
            Ok(_) => println!("Joined"),
            Err(_) => println!("Join error"),
        }
    }

}
