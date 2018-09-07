extern crate barc;
use barc::*;

use std::fmt::Debug;
use std::clone::Clone;

use std::sync::atomic::AtomicIsize;
use std::sync::atomic::Ordering::Relaxed;


/// Singly-linked-list Node for a simple unbounded concurrent Stack
///
/// To pop a !Clone value, the stack must be able to take that value from a node that is still
/// shared, so we store it behind an AtomicOptionWeightedArc.  This might be a valid use case for
/// an AtomicOptionBox, since we only unconditionally take it.

struct Node<T : Debug> {
    next: Option<WeightedArc<Node<T>>>,
    value: AtomicOptionWeightedArc<T>,
}

impl<T : Debug> Node<T> {
    fn new(nxt: Option<WeightedArc<Node<T>>>, val: T) -> Node<T> {
        println!("Newing {:?}", val);
        Node {
            next: nxt,
            value: AtomicOptionWeightedArc::new(
                Some(WeightedArc::new(val))
            )
        }
    }

    fn into_inner(self) -> T {
        //let Self { next, value: v } = this;
        //v
        unsafe {
            let _ = std::ptr::read(&self.next);
            let v = std::ptr::read(&self.value);
            std::mem::forget(self);
            WeightedArc::try_unwrap(v.into_inner().unwrap()).expect("Empty node")
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
            let tmp;
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
        let mut new = WeightedArc::new(Node::new(current.clone(), val));
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
            let tmp;
            match current {
                Some(ref node) => {
                    match self.head.compare_exchange(current.clone(), node.next.clone()) {
                        Ok(old) => {
                            match old {
                                Some(x) => {
                                    match x.value.swap(None) {
                                        Some(y) => {
                                            return Some(WeightedArc::try_unwrap(y).expect("shared"))
                                        },
                                        None => { return None },
                                    }
                                },
                                None => { return None },
                            }
                        },
                        Err(actual) => { tmp = actual },
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

    // Ham-fisted benchmarks for performance under contention.  Mutex is always terrible, so we
    // have that win.  WeightedArc's swaps and stores are about as good as raw AtomicPtr, which
    // of course they almost always are.
    // Loads are bad because they are basically compare-exchanges.

    // Question: how different is fetch_add[sub] to load-compare_exchange_weak?
    // load() vs clone()

    {
        // Stress-test swaps

        let a = 100;
        let b = WeightedArc::new(100);
        let c = AtomicWeightedArc::new(b);
        let d = WeightedArc::new(c);

        let n = 10000000;

        let mut v = Vec::default();

        let now = std::time::Instant::now();

        let bar = WeightedArc::new(std::sync::Barrier::new(9));

        for i in 0..8 {
            let e = d.clone();
            let g = bar.clone();
            v.push(std::thread::spawn(move || {
                let f = &*e;
                g.wait();
                let mut h = WeightedArc::new(i);
                for j in 0..n {
                    h = f.swap(h);
                }
            }));
        }

        bar.wait();


        for h in v {
            h.join().expect("oops");
        }

        let then = now.elapsed();
        let t = (then.as_secs() as f64) * 1e9 + (then.subsec_nanos() as f64);

        println!("AtomicOpionWeightedArc ns per swap {}", t / ((n * 8) as f64));

    }

    {
        // Stress-test swaps

        let a = 100;
        let b = std::sync::Arc::new(100);
        let c = std::sync::Mutex::new(b);
        let d = std::sync::Arc::new(c);

        let n = 100000;

        let mut v = Vec::default();

        let now = std::time::Instant::now();

        let bar = WeightedArc::new(std::sync::Barrier::new(9));

        for i in 0..8 {
            let e = d.clone();
            let g = bar.clone();
            v.push(std::thread::spawn(move || {
                let f = &*e;
                g.wait();
                let mut h = std::sync::Arc::new(i);
                for j in 0..n {
                    // h = f.lock().unwrap().swap(h);
                    let tmp;
                    {
                        let mut p = f.lock().unwrap();
                        tmp = p.clone();
                        *p = h;
                    }
                    h = tmp;
                }
            }));
        }

        bar.wait();

        for h in v {
            h.join().expect("oops");
        }

        let then = now.elapsed();
        let t = (then.as_secs() as f64) * 1e9 + (then.subsec_nanos() as f64);

        println!("Mutex ns per swap {}", t / ((n * 8) as f64));

    }

    {

        // Stress-test loads
        //
        // Get multiple threads a &AtomicWeightedArc<AtomicIsize>, and then load the Arc and
        // increment the payload many times to test highly contended loads

        // This is the worst case for distributed rc since load is as much work as
        // compare_exchange

        let a = AtomicIsize::new(0);
        let b = WeightedArc::new(a);
        let c = AtomicWeightedArc::new(b);
        let d = WeightedArc::new(c);

        let n = 10000000;

        let mut v = Vec::default();

        let now = std::time::Instant::now();

        let bar = WeightedArc::new(std::sync::Barrier::new(9));

        for _ in 0..8 {
            let e = d.clone();
            let g = bar.clone();
            v.push(std::thread::spawn(move || {
                let f = &*e;
                g.wait();
                while f.load().fetch_add(1, Relaxed) < n {}
            }));
        }

        bar.wait();

        for h in v {
            h.join().expect("oops");
        }

        let then = now.elapsed();
        let t = (then.as_secs() as f64) * 1e9 + (then.subsec_nanos() as f64);

        println!("strong_bound {}", WeightedArc::strong_bound(&d.swap(WeightedArc::new(AtomicIsize::new(0)))));
        println!("AtomicOpionWeightedArc ns per thing {}", t / (n as f64));

    }

    {

        // Stress-test loads
        //
        // Get multiple threads a &AtomicWeightedArc<AtomicIsize>, and then load the Arc and
        // increment they payload many times to test highly contended loads

        let a = AtomicIsize::new(0);
        let b = std::sync::Arc::new(a);
        let c = std::sync::Mutex::new(b);
        let d = std::sync::Arc::new(c);

        let n = 100000;

        let mut v = Vec::default();

        let now = std::time::Instant::now();

        let bar = WeightedArc::new(std::sync::Barrier::new(9));

        for _ in 0..8 {
            let e = d.clone();
            let g = bar.clone();
            v.push(std::thread::spawn(move || {
                let f = &*e;
                g.wait();
                // while f.lock().unwrap().fetch_add(1, Relaxed) < n {}
                loop {
                    let h = f.lock().unwrap().clone();
                    if h.fetch_add(1, Relaxed) < n { continue; }
                    break;
                }
            }));
        }

        bar.wait();

        for h in v {
            h.join().expect("oops");
        }

        let then = now.elapsed();
        let t = (then.as_secs() as f64) * 1e9 + (then.subsec_nanos() as f64);

        //println!("strong_bound {}", WeightedArc::strong_bound(&d.swap(WeightedArc::new(AtomicIsize::new(0)))));
        println!("Mutex ns per thing {}", t / (n as f64));

    }

    {

        // Stress-test loads
        //
        // Get multiple threads a &AtomicWeightedArc<AtomicIsize>, and then load the Arc and
        // increment they payload many times to test highly contended loads

        let a = AtomicIsize::new(0);
        let b = std::sync::Arc::new(a);
        let c = std::sync::RwLock::new(b);
        let d = std::sync::Arc::new(c);

        let n = 1000000;

        let mut v = Vec::default();

        let now = std::time::Instant::now();

        let bar = WeightedArc::new(std::sync::Barrier::new(9));

        for _ in 0..8 {
            let e = d.clone();
            let g = bar.clone();
            v.push(std::thread::spawn(move || {
                let f = &*e;
                g.wait();
                while f.read().unwrap().fetch_add(1, Relaxed) < n {}
                //loop {
                    //let h = f.read().unwrap().clone();
                    //if h.fetch_add(1, Relaxed) < n { continue; }
                    //break;
                //}
            }));
        }

        bar.wait();

        for h in v {
            h.join().expect("oops");
        }

        let then = now.elapsed();
        let t = (then.as_secs() as f64) * 1e9 + (then.subsec_nanos() as f64);

        //println!("strong_bound {}", WeightedArc::strong_bound(&d.swap(WeightedArc::new(AtomicIsize::new(0)))));
        println!("RwLock ns per thing {}", t / (n as f64));

    }



    return;

    {

        // Exercise the Stack.  This doesn't prove anything but does catch some
        // basic bugs.  Note that the debugging output introduces extra
        // synchronization between the threads and tends to serialize everything.

        let arcstack = WeightedArc::new(Stack::default());

        let mut v = Vec::default();

        let bar = WeightedArc::new(std::sync::Barrier::new(8));

        for k in 0..8 {
            let s = arcstack.clone();
            let q = bar.clone();
            v.push(std::thread::spawn(move || {
                q.wait();
                for i in 0..4 {
                    let j = i * 2 + k * 8;
                    s.push(j);
                    println!("Pushed {}", j);
                    s.push(j + 1);
                    println!("Pushed {}", j + 1);
                    let p = s.pop();
                    println!("Popped {:?}", p)
                }
                for _i in 0..5 {
                    let p = s.pop();
                    println!("Popped {:?}", p)
                }
            }));
        }

        for h in v {
            h.join().expect("oops");
        }

    }

}
