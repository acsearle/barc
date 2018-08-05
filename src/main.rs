use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Release;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::Ordering::AcqRel;

extern crate core;
use core::marker::PhantomData;
use core::mem::forget;



struct BarcInner<T> {
    strong : AtomicUsize,
    weak : AtomicUsize,
    value : T
}

struct Barc<T> {
    ptr : *mut BarcInner<T>
}

impl<T> Barc<T> {

    fn new(data : T) -> Barc<T> {
        println!("new Barc");
        let x = Box::new(BarcInner {
            strong: AtomicUsize::new(1), 
            weak: AtomicUsize::new(1), 
            value:data
        });
        Barc { ptr: Box::into_raw(x) }
    }

    fn into_raw(this: Self) -> *mut BarcInner<T> {
        let ptr = this.ptr;
        forget(this);
        ptr
    }

    fn from_raw(ptr: *mut BarcInner<T>) -> Barc<T> {
        Barc { ptr: ptr }
    }

    fn strong_inc(&self, n : usize) {
        unsafe {
            (*self.ptr).strong.fetch_add(n, Relaxed);
        }
    }

    fn strong_dec(&self, n : usize) {
        unsafe {
            if (*self.ptr).strong.fetch_sub(n, Release) == n {
                std::sync::atomic::fence(Acquire);
                println!("dropping Barc<T>");
                Box::from_raw(self.ptr);
            }
        }
    }

}

impl<T> Drop for Barc<T> {
    fn drop(&mut self) {
        self.strong_dec(1);
    }
}

struct AtomicBarc<T> {
    ptr: AtomicUsize,
    phantom : PhantomData<Barc<T>>
}

const DELTA : usize = (1 << 16);
const MASK : usize = (1 << 48) - 1;
const SHIFT : usize = 48;

impl<T> AtomicBarc<T> {

    fn unpack(val : usize) -> (*mut BarcInner<T>, usize) {
        ((val & MASK) as *mut BarcInner<T>, val >> SHIFT)
    }

    fn pack(ptr: *mut BarcInner<T>, cnt: usize) -> usize {
        (ptr as usize) | (cnt << SHIFT)
    }

    fn new(data: Barc<T>) -> AtomicBarc<T> {
        data.strong_inc(DELTA - 1);
        AtomicBarc { 
            ptr: AtomicUsize::new(AtomicBarc::pack(Barc::into_raw(data), 0)), 
            phantom:PhantomData
        }
    }

    fn load(&self) -> Barc<T> {
        let n = 1 << 48;
        let p = self.ptr.fetch_add(n, Acquire);
        println!("{}", p);
        let p2 = p & MASK;
        let q = p2 as *mut BarcInner<T>;
        //println!("{}", q);
        Barc::from_raw(q)
    }

    fn swap(&self, val: Barc<T>) -> Barc<T> {
        val.strong_inc(DELTA - 1);
        let p = Barc::into_raw(val) as usize;
        let q = self.ptr.swap(p, AcqRel);
        let n = q >> 48;
        let q2 = q & MASK;
        let p2 = q2 as *mut BarcInner<T>;
        println!("{}\n{}\n{}", p, q, q2);
        let a = Barc::from_raw(p2);
        a.strong_dec(DELTA - n - 1);
        a
    }

    fn store(&self, val: Barc<T>) {
        self.swap(val);
    }

    fn compare_exchange(&self, current: Barc<T>, new: Barc<T>) -> Result<Barc<T>, Barc<T>> {
        let new2 = new.ptr as usize;
        new.strong_inc(DELTA - 1);
        let mut expected : usize = self.ptr.load(Relaxed);
        loop {
            if expected & MASK == new2 {
                match self.ptr.compare_exchange(expected, new2, AcqRel, Relaxed) {
                    Ok(_) => {
                        let n = expected >> SHIFT;
                        current.strong_dec(DELTA - n - 1);
                        forget(new);
                        break Ok(current)
                    },
                    Err(x) => {
                        expected = x; // Try again
                    }
                }
            } else {
                let new3 = expected + (1 << SHIFT);
                match self.ptr.compare_exchange(expected, new3, Acquire, Relaxed) {
                    Ok(_) => {
                        new.strong_dec(DELTA - 1);
                        break Err(Barc::from_raw((expected & MASK) as *mut BarcInner<T>));
                    },
                    Err(x) => {
                        expected = x; // Try again
                    }
                }
            }
        }
    }

}

impl<T> Drop for AtomicBarc<T> {
    fn drop(&mut self) {
        println!("drop AtomicBarc");
        let a : usize = *self.ptr.get_mut();
        let b = a & MASK;
        let c = b as *mut BarcInner<T>;
        let d = Barc::from_raw(c);
        let n = b >> SHIFT;
        d.strong_dec(DELTA - n - 1);
    }
}

fn main() {
    let a = Barc::new(7);
    let b = AtomicBarc::new(a);
    let c = b.load();
    let d = b.swap(Barc::new(8));
    println!("Hello, world!");
}
