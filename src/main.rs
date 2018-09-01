extern crate barc;
use barc::{WeightedArc, AtomicOptionWeightedArc};

use std::fmt::Debug;
use std::clone::Clone;

// Singly-linked-list Node for a simple unbounded concurrent Stack

struct Node<T : Debug> {
    next: Option<WeightedArc<Node<T>>>,
    value: T,
}

impl<T : Clone + Debug> Node<T> {
    fn new(val: T) -> Node<T> {
        println!("Newing {:?}", val);
        Node { next: None, value: val }
    }

    fn into_inner(self) -> T {
        //let Self { next, value: v } = this;
        //v
        unsafe {
            let _ = std::ptr::read(&self.next);
            let v = std::ptr::read(&self.value);
            std::mem::forget(self);
            v
        }
    }
}

impl<T : Debug> Drop for Node<T> {
    fn drop(&mut self) {
        println!("Dropping {:?}", self.value);
        // Automatic drop is correct, but it recurses along a linked list and
        // thus tends to blow the stack.  We have to convert the iteration
        // into recursion.  If we are the sole owner of the next node, we take
        // its next field and overwrite our own.  That node is then deleted
        // with no dependents.  We keep looping until we hit a node that
        // somebody else owns too.

        loop {
            let tmp : Option<WeightedArc<Node<T>>>;
            match self.next {
                None => return,
                Some(ref mut a) => {
                    match WeightedArc::get_mut(a) {
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
    head: AtomicOptionWeightedArc<Node<T>>,
}

impl<T : Clone + std::fmt::Debug> Stack<T> {

    fn push(&self, val: T) {
        let mut current = self.head.load();
        let mut new = WeightedArc::new( Node { next: current.clone(), value: val, } );
        loop {
            match self.head.compare_exchange(current, Some(new.clone_mut())) {
                Ok(_) => {
                    return
                },
                Err(actual) => {
                    current = actual;
                    unsafe { WeightedArc::get_mut_unchecked(&mut new).next = current.clone() }
                },
            }
        }
    }

    fn pop(&self) -> Option<T> {
        let mut current = self.head.load();
        loop {
            let tmp : Option<WeightedArc<Node<T>>>;
            match current {
                Some(ref node) => {
                    let new = node.next.clone();
                    let payload = node.value.clone();
                    match self.head.compare_exchange(current.clone(), new) {
                        Ok(_old) => return Some(payload),
                        Err(actual) => tmp = actual,
                    }
                },
                None => return None
            }
            current = tmp;
        }
    }

}

impl<T : std::fmt::Debug> Default for Stack<T> {
    fn default() -> Stack<T> {
        Stack { head: AtomicOptionWeightedArc::default() }
    }
}


fn main() {

    // Exercise the Stack.  This doesn't prove anything but does catch some
    // basic bugs.  Note that the debugging output introduces extra
    // synchronization between the threads and tends to serialize everything.

    let arcstack : WeightedArc<Stack<usize>> = WeightedArc::new(Stack::default());

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
            for _i in 0..5 {
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






/*


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

// AtomicOptionArc packs a pointer and a count into a single usize
// Lower 48: pointer
// Upper 16: count
// Requires a 64 bit architecture with 16 high bits unused (x86_64, AArch64)
// Todo: for other architectures, use alignment low bits

pub struct AtomicOptionArc<T> {
    ptr: AtomicUsize,         // *const ArcInner<T> : 48, usize : 16
    phantom : PhantomData<T>, // Use the type parameter
}

fn into_usize<T>(val: Option<Arc<T>>) -> usize {
    match val {
        None => 0,
        Some(b) => Arc::into_raw(b) as usize,
    }
}

fn into_option_arc<T>(n: usize) -> Option<Arc<T>> {
    // If lower 48 bits are zero, upper 16 bits must be zero
    debug_assert!(((n & MASK) != 0) || (n == 0));
    match n {
        0 => None,
        _ => Some(unsafe { Arc::from_raw((n & MASK) as *const T) } ),
    }
}

fn inc_global(n: usize, delta: usize) -> usize {
    debug_assert!(((n & MASK) != 0) || (n == 0));
    debug_assert!(delta > 0);
    match n {
        0 => 0,
        _ => {
            let p = ((n & MASK) - 16) as *const ArcInner<()>;
            let m = unsafe { (*p).strong.fetch_add(delta, Relaxed) };
            debug_assert!(m > 0);
            //println!("{} -> {}", m, m + delta);
            m
        },
    }
}

fn dec_global(n: usize, delta: usize) -> usize {
    debug_assert!(((n & MASK) != 0) || (n == 0));
    debug_assert!(delta > 0);
    match n {
        0 => 0,
        _ => {
            let p = ((n & MASK) - 16) as *const ArcInner<()>;
            let m = unsafe { (*p).strong.fetch_sub(delta, Release) };
            debug_assert!(m > delta);
            //println!("{} -> {}", m, m - delta);
            m
        }
    }
}

fn increment_or_yield(expected : &mut usize) -> usize {
    debug_assert!(((*expected & MASK) != 0) || (*expected == 0));
    if *expected >> SHIFT == LIMIT {
        // Count is saturated; anticipate another thread resetting it
        *expected = *expected & MASK;
        std::thread::yield_now();
    }
    // Increment the count
    *expected + (1 << SHIFT)
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

    pub fn into_inner(self) -> Option<Arc<T>> {
        self.swap(None)
    }

    pub fn swap(&self, val: Option<Arc<T>>) -> Option<Arc<T>> {
        let a = into_usize(val);          // Convert to packed integer
        inc_global(a, LIMIT);             // Add (2 << 16) - 1 owners
        let b = self.ptr.swap(a, AcqRel); // Swap atomically
        let n = b >> SHIFT;               // Extract count of ownerships relinquished
        dec_global(b, LIMIT - n);         // Subtract (2 << 16) - 1 - n owners
        into_option_arc(b)                // Convert from packed integer
    }

    pub fn store(&self, val: Option<Arc<T>>) {
        self.swap(val);
    }

    fn maybe_desaturate(&self, expected : usize) {
        debug_assert!((expected & MASK) != 0);
        if expected >> SHIFT == LIMIT {
            // We saturated the local counter and other threads cannot proceed
            // Add the local count to the global count
            inc_global(expected, LIMIT);
            // Try to set the local count to zero
            let desired = expected & MASK;
            match self.ptr.compare_exchange(expected, desired, AcqRel, Relaxed) {
                Ok(_) => {
                    // Successfully transferred local count to global count
                },
                Err(_) => {
                    // A strong failure means the value really changed, so
                    // either it is now a different object, or a different and
                    // thus non-saturated count.  No longer our problem.

                    // Undo the refcount manipulation
                    dec_global(expected, LIMIT);
                }
            }
        }
    }

    pub fn load(&self) -> Option<Arc<T>> {

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
            if expected & MASK == 0 {
                // Null pointer; we are done
                debug_assert!(expected == 0); // We should not have incremented a null ptr
                return None;
            }
            desired = increment_or_yield(&mut expected);
            match self.ptr.compare_exchange_weak(expected, desired, Acquire, Relaxed) {
                Ok(_) => break, // Successfully incremented
                Err(x) => { expected = x } // Start over
            }
        }

        // If we were more aggressive about attempting to zero the counter
        // (for example, when more than half-full) we would become lock-free
        // but at the cost of increased contention.  Profile.
        self.maybe_desaturate(desired);

        into_option_arc::<T>(desired)
    }

    pub fn compare_exchange(
        &self,
        current: Option<Arc<T>>,
        new: Option<Arc<T>>
    ) -> Result<Option<Arc<T>>, Option<Arc<T>>> {

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

                if expected & MASK == 0 {
                    // Null pointer.  We are done.
                    assert!(expected == 0);
                    dec_global(new2, LIMIT);
                    into_option_arc::<T>(new2);
                    into_option_arc::<T>(cur2);
                    break Err(None);
                }

                // Build increment
                let new3 = increment_or_yield(&mut expected);

                // Install increment
                match self.ptr.compare_exchange_weak(expected, new3, Acquire, Relaxed) {
                    Ok(_) => {
                        self.maybe_desaturate(new3);
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

    pub fn compare_and_swap(
        &self,
        current: Option<Arc<T>>,
        new: Option<Arc<T>>
    ) -> Option<Arc<T>> {
        match self.compare_exchange(current, new) {
            Ok(x) => x,
            Err(x) => x,
        }
    }

    pub fn compare_exchange_weak(&self,
        current: Option<Arc<T>>,
        new: Option<Arc<T>>
    ) -> Result<Option<Arc<T>>, Option<Arc<T>>> {
        self.compare_exchange(current, new)
    }

}


impl<T : std::fmt::Debug> std::fmt::Debug for AtomicOptionArc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.load(), f)
    }
}


impl<T> Default for AtomicOptionArc<T> {
    fn default() -> Self {
        Self {
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


impl<T> From<Option<Arc<T>>> for AtomicOptionArc<T> {
    fn from(a : Option<Arc<T>>) -> Self {
        Self::new(a)
    }
}


// Implement AtomicArc as a wrapper around AtomicOptionArc

pub struct AtomicArc<T> {
    value : AtomicOptionArc<T>,
}


impl<T> AtomicArc<T> {

    pub fn new(val : Arc<T>) -> AtomicArc<T> {
        AtomicArc { value: AtomicOptionArc::new(Some(val)), }
    }

    pub fn into_inner(self) -> Arc<T> {
        AtomicOptionArc::into_inner(self.value).unwrap()
    }

    pub fn load(&self) -> Arc<T> {
        self.value.load().unwrap()
    }

    pub fn store(&self, val : Arc<T>) {
        self.value.store(Some(val))
    }

    pub fn swap(&self, val : Arc<T>) -> Arc<T> {
        self.value.swap(Some(val)).unwrap()
    }

    pub fn compare_and_swap(&self, current: Arc<T>, new: Arc<T>) -> Arc<T> {
        self.value.compare_and_swap(Some(current), Some(new)).unwrap()
    }

    pub fn compare_exchange(&self, current: Arc<T>, new: Arc<T>) -> Result<Arc<T>, Arc<T>> {
        match self.value.compare_exchange(Some(current), Some(new)) {
            Ok(a) => Ok(a.unwrap()),
            Err(b) => Ok(b.unwrap()),
        }
    }

    pub fn compare_exchange_weak(&self, current: Arc<T>, new: Arc<T>) -> Result<Arc<T>, Arc<T>> {
        match self.value.compare_exchange_weak(Some(current), Some(new)) {
            Ok(a) => Ok(a.unwrap()),
            Err(b) => Ok(b.unwrap()),
        }
    }

}

impl<T : std::fmt::Debug> std::fmt::Debug for AtomicArc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.load(), f)
    }
}

impl<T: Default> Default for AtomicArc<T> {
    fn default() -> Self {
        Self::new(Arc::<T>::default())
    }
}


impl<T> From<Arc<T>> for AtomicArc<T> {
    fn from(a : Arc<T>) -> Self {
        Self::new(a)
    }
}





*/


/*
struct AtomicOptionCountedNonNullPtr<T> {
    ptr: AtomicUsize,
    phantom: PhantomData<Option<CountedNonNullPtr<T>>>,
}

impl<T> AtomicOptionCountedNonNullPtr<T> {

    fn to_usize(x: Option<CountedNonNullPtr<T>>) -> usize {
        match x {
            None => 0,
            Some(y) => y.ptr.get(),
        }
    }

    fn from_usize(x: usize) -> Option<CountedNonNullPtr<T>> {
        match x {
            0 => None,
            y => Some(CountedNonNullPtr {
                ptr: unsafe { NonZeroUsize::new_unchecked(y) },
                phantom: PhantomData,
                }
            ),
        }
    }

    fn new(x: Option<CountedNonNullPtr<T>>) -> Self {
        Self {
            ptr: AtomicUsize::new(Self::to_usize(x)),
            phantom: PhantomData,
        }
    }

    fn load(&self, order: Ordering) -> Option<CountedNonNullPtr<T>> {
        Self::from_usize(self.ptr.load(order))
    }

    fn store(&self, val: Option<CountedNonNullPtr<T>>, order: Ordering) {
        self.ptr.store(Self::to_usize(val), order)
    }

    fn swap(&self, val: Option<CountedNonNullPtr<T>>, order: Ordering) -> Option<CountedNonNullPtr<T>> {
        Self::from_usize(self.ptr.swap(Self::to_usize(val), order))
    }

    fn compare_and_swap(
        &self,
        current: Option<CountedNonNullPtr<T>>,
        new: Option<CountedNonNullPtr<T>>,
        order: Ordering
    ) -> Option<CountedNonNullPtr<T>> {
        Self::from_usize(
            self.ptr.compare_and_swap(
                Self::to_usize(current),
                Self::to_usize(new),
                order
            )
        )
    }

    fn compare_exchange(
        &self,
        current: Option<CountedNonNullPtr<T>>,
        new: Option<CountedNonNullPtr<T>>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<CountedNonNullPtr<T>>, Option<CountedNonNullPtr<T>>> {
        match self.ptr.compare_exchange(
            Self::to_usize(current),
            Self::to_usize(new),
            success,
            failure,
        ) {
            Ok(x) => Ok(Self::from_usize(x)),
            Err(x) => Err(Self::from_usize(x)),
        }
    }

    fn compare_exchange_weak(
        &self,
        current: Option<CountedNonNullPtr<T>>,
        new: Option<CountedNonNullPtr<T>>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Option<CountedNonNullPtr<T>>, Option<CountedNonNullPtr<T>>> {
        match self.ptr.compare_exchange_weak(
            Self::to_usize(current),
            Self::to_usize(new),
            success,
            failure
        ) {
            Ok(x) => Ok(Self::from_usize(x)),
            Err(x) => Err(Self::from_usize(x)),
        }
    }

}

struct AtomicCountedNonNullPtr<T> {
    ptr: AtomicUsize,
    phantom: PhantomData<CountedNonNullPtr<T>>
}

impl<T> AtomicCountedNonNullPtr<T> {

    fn to_usize(x: CountedNonNullPtr<T>) -> usize {
        x.ptr.get()
    }

    fn from_usize(x: usize) -> CountedNonNullPtr<T> {
        CountedNonNullPtr {
            ptr: unsafe { NonZeroUsize::new_unchecked(x) },
            phantom: PhantomData,
        }
    }

    fn new(p: CountedNonNullPtr<T>) -> Self {
        Self {
            ptr: AtomicUsize::new(Self::to_usize(p)),
            phantom: PhantomData,
        }
    }

    fn load(&self, order: Ordering) -> CountedNonNullPtr<T> {
        Self::from_usize(self.ptr.load(order))
    }

    fn store(&self, p: CountedNonNullPtr<T>, order: Ordering) {
        self.ptr.store(Self::to_usize(p), order)
    }

    fn swap(&self, p: CountedNonNullPtr<T>, order: Ordering) -> CountedNonNullPtr<T> {
        Self::from_usize(self.ptr.swap(Self::to_usize(p), order))
    }

    fn compare_and_swap(
        &self,
        current: CountedNonNullPtr<T>,
        new: CountedNonNullPtr<T>,
        order: Ordering,
    ) -> CountedNonNullPtr<T> {
        Self::from_usize(
            self.ptr.compare_and_swap(
                Self::to_usize(current),
                Self::to_usize(new),
                order,
            )
        )
    }

    fn compare_exchange(
        &self,
        current: CountedNonNullPtr<T>,
        new: CountedNonNullPtr<T>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<CountedNonNullPtr<T>, CountedNonNullPtr<T>> {
        match self.ptr.compare_exchange(
            Self::to_usize(current),
            Self::to_usize(new),
            success,
            failure,
        ) {
            Ok(x) => Ok(Self::from_usize(x)),
            Err(x) => Err(Self::from_usize(x)),
        }
    }

    fn compare_exchange_weak(
        &self,
        current: CountedNonNullPtr<T>,
        new: CountedNonNullPtr<T>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<CountedNonNullPtr<T>, CountedNonNullPtr<T>> {
        match self.ptr.compare_exchange_weak(
            Self::to_usize(current),
            Self::to_usize(new),
            success,
            failure,
        ) {
            Ok(x) => Ok(Self::from_usize(x)),
            Err(x) => Err(Self::from_usize(x)),
        }
    }

}
*/
