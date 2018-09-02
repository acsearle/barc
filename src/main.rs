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
                    // Avoid these clones
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
