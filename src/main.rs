#![feature(cfg_target_has_atomic)]
#![cfg(all(target_pointer_width = "64", target_has_atomic = "64"))]
#![feature(allocator_api)]
#![feature(box_into_raw_non_null)]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::Ordering::AcqRel;
use std::sync::atomic::Ordering::Release;
use std::sync::atomic::Ordering;

use std::marker::PhantomData;

use std::sync::Arc;
use std::option::Option;

use std::mem;
use std::num::NonZeroUsize;

use std::alloc::Alloc;
use std::ptr::NonNull;

use std::ops::Deref;
use std::ops::DerefMut;
use std::clone::Clone;

// Useful constants to help us pack and unpack the pointer and count to/from
// the atomic integer

const SHIFT : usize = 48;
const MASK  : usize = (1 << SHIFT) - 1;
const LIMIT : usize = (!MASK) >> SHIFT;
const DELTA : usize = LIMIT + 1;
const N : usize = 1 << 16;

struct CountedPtr<T> {
    ptr: usize,
    phantom: PhantomData<*mut T>,
}

impl<T> CountedPtr<T> {

    fn new(count: usize, pointer: *mut T) -> Self {
        Self {
            ptr: ((count - 1) << SHIFT) | (pointer as usize),
            phantom: PhantomData,
        }
    }

    fn get(&self) -> (usize, *mut T) {
        ((self.ptr >> SHIFT) + 1, (self.ptr & MASK) as *mut T)
    }

    fn set_count(&mut self, count: usize) {
        let (_, p) = self.get();
        *self = Self::new(count, p);
    }

}

impl<T> Clone for CountedPtr<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            phantom: PhantomData,
        }
    }
}

impl<T> Copy for CountedPtr<T> {}

impl<T> Deref for CountedPtr<T> {
    type Target = T;
    fn deref(&self) -> &T {
        let (_, p) = self.get();
        unsafe { &*p }
    }
}

impl<T> DerefMut for CountedPtr<T> {
    fn deref_mut(&mut self) ->&mut T {
        let (_, p) = self.get();
        unsafe { &mut *p }
    }
}






struct CountedNonNullPtr<T> {
    ptr: NonZeroUsize,
    phantom: PhantomData<NonNull<T>>,
}

impl<T> CountedNonNullPtr<T> {

    fn new(count: usize, pointer: NonNull<T>) -> Self {
        let x = ((count - 1) << SHIFT) | (pointer.as_ptr() as usize);
        Self {
            ptr: unsafe { NonZeroUsize::new_unchecked(x) },
            phantom: PhantomData,
        }
    }

    fn get(&self) -> (usize, NonNull<T>) {
        let n = self.ptr.get();
        let p = (n & MASK) as *mut T;
        ((n >> SHIFT) + 1, unsafe { NonNull::new_unchecked(p) })
    }

    fn set_count(&mut self, count: usize) {
        let (_, p) = self.get();
        *self = Self::new(count, p);
    }

}

impl<T> Clone for CountedNonNullPtr<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            phantom: PhantomData,
        }
    }
}

impl<T> Copy for CountedNonNullPtr<T> {}

impl<T> Deref for CountedNonNullPtr<T> {
    type Target = T;
    fn deref(&self) -> &T {
        let (_, p) = self.get();
        unsafe { &*p.as_ptr() }
    }
}

impl<T> DerefMut for CountedNonNullPtr<T> {
    fn deref_mut(&mut self) ->&mut T {
        let (_, p) = self.get();
        unsafe { &mut *p.as_ptr() }
    }
}

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





struct AtomicCountedPtr<T> {
    ptr: AtomicUsize,
    phantom: PhantomData<CountedPtr<T>>
}

impl<T> AtomicCountedPtr<T> {

    fn to_usize(x: CountedPtr<T>) -> usize {
        x.ptr
    }

    fn from_usize(x: usize) -> CountedPtr<T> {
        CountedPtr {
            ptr: x,
            phantom: PhantomData,
        }
    }

    fn new(p: CountedPtr<T>) -> Self {
        Self {
            ptr: AtomicUsize::new(Self::to_usize(p)),
            phantom: PhantomData,
        }
    }

    fn load(&self, order: Ordering) -> CountedPtr<T> {
        Self::from_usize(self.ptr.load(order))
    }

    fn store(&self, p: CountedPtr<T>, order: Ordering) {
        self.ptr.store(Self::to_usize(p), order)
    }

    fn swap(&self, p: CountedPtr<T>, order: Ordering) -> CountedPtr<T> {
        Self::from_usize(self.ptr.swap(Self::to_usize(p), order))
    }

    fn compare_and_swap(
        &self,
        current: CountedPtr<T>,
        new: CountedPtr<T>,
        order: Ordering,
    ) -> CountedPtr<T> {
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
        current: CountedPtr<T>,
        new: CountedPtr<T>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<CountedPtr<T>, CountedPtr<T>> {
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
        current: CountedPtr<T>,
        new: CountedPtr<T>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<CountedPtr<T>, CountedPtr<T>> {
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

struct ArcInner<T : ?Sized> {
    strong : AtomicUsize,
    weak : AtomicUsize,
    data : T,
}

// WeightedArc packs a weight into the spare bits of its pointer, the weight ranging from 1 to N
// (represented as 0 to N - 1).  An Arc and a WeightedArc with weight one have the same
// bit representation.  The weight is a measure of how much ownership object has, and it can be
// reallocated between objects without touching the global .strong count, enabling lock-free
// implementation of AtomicWeightedArc and some optimizing extensions to Arc's interface such as
// split and merge, which sometimes enable us to clone and drop without touching the the global
// reference count

pub struct WeightedArc<T> {
    ptr: CountedNonNullPtr<ArcInner<T>>
}

pub struct WeightedWeak<T> {
    ptr : CountedNonNullPtr<ArcInner<T>>
}

impl<T> WeightedArc<T> {

    pub fn new(data: T) -> Self {
        Self {
            ptr: CountedNonNullPtr::new(
                N,
                Box::into_raw_non_null(
                    Box::new(
                        ArcInner {
                            strong : AtomicUsize::new(N),
                            weak : AtomicUsize::new(N),
                            data : data,
                        }
                    )
                )
            )
        }
    }

    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        let (n, p) = this.ptr.get();
        match this.ptr.strong.compare_exchange(n, 0, Release, Relaxed) {
            Ok(_) => {
                std::sync::atomic::fence(Acquire);
                let data = unsafe { std::ptr::read(&p.as_ref().data) };
                let _weak : WeightedWeak<T> = WeightedWeak {
                    ptr : CountedNonNullPtr::new(N, p),
                };
                std::mem::forget(this);
                Ok(data)
            }
            Err(_) => Err(this),
        }
    }

    pub fn into_raw(this: Self) -> *const T {
        let (n, _) = this.ptr.get();
        if n > 1 {
            // Release all but one ownership
            let m = this.ptr.strong.fetch_sub(n - 1, Relaxed);
            debug_assert!(m > 0);
        }
        let ptr : *const T = &*this;
        mem::forget(this);
        ptr
    }

    pub unsafe fn from_raw(ptr: *const T) -> Self {
        let fake_ptr = 0 as *const ArcInner<T>;
        let offset = &(*fake_ptr).data as *const T as usize;
        let p = NonNull::new_unchecked(((ptr as usize) - offset) as *mut ArcInner<T>);
        Self {
            ptr : CountedNonNullPtr::new(1, p),
        }
    }

    pub fn downgrade(this: &Self) -> WeightedWeak<T> {
        let (_, p) = this.ptr.get();
        let mut cur = this.ptr.weak.load(Relaxed);
        loop {
            if cur == std::usize::MAX {
                // The weak count is locked by is_unique.  Spin.
                cur = this.ptr.weak.load(Relaxed);
                continue
            }
            match this.ptr.weak.compare_exchange_weak(cur, cur + N, Acquire, Relaxed) {
                Ok(_) => break WeightedWeak {
                    ptr: CountedNonNullPtr::new(N, p),
                },
                Err(old) => cur = old,
            }
        }
    }

    // This is not a count, but an upper bound on the number of WeightedWeaks.  Returns zero if
    // and only if there were no WeightedWeaks at some time, but races against the downgrading
    // of any WeightedArcs.
    pub fn weak_bound(this: &Self) -> usize {
        // I don't understand why this load is SeqCst in std::sync::Arc.  I believe that SeqCst
        // synchronizes only with itself, not Acqure/Release.
        let cnt = this.ptr.weak.load(Relaxed);
        if cnt == std::usize::MAX {
            // .weak is locked, so must have been N
            0
        } else {
            // We are calling this on a WeightedArc, so at least one WeightedArc is extant, so
            // the offset of N is active on .weak
            cnt - N
        }
    }

    // This is not a count, but an upper bound on the number of WeightedArcs.  Since it is invoked
    // on a WeightedArc, a lower bound is 1.  Returns if and only if the WeightedArc was unique
    // at some time, but races against the upgrading of any WeightedWeaks
    pub fn stong_bound(this: &Self) -> usize {
        let (n, _) = this.ptr.get();
        // I don't understand why this load is SeqCst in std::sync::Arc.  I beleive that SeqCst
        // synchronizes only with itself not Acquire/Release
        let m = this.ptr.strong.load(Relaxed);
        m - n
    }

    unsafe fn drop_slow(&mut self) {
        // We have just set .strong to zero
        let (_, mut p) = self.ptr.get();
        std::ptr::drop_in_place(&mut p.as_mut().data);
        if self.ptr.weak.fetch_sub(N, Release) == N {
            // Upgrade memory order to synchronze with dropped WeightedWeaks on other threads
            std::sync::atomic::fence(Acquire);
            std::alloc::Global.dealloc(
                std::ptr::NonNull::new_unchecked(p.as_ptr() as *mut u8),
                std::alloc::Layout::for_value(p.as_ref())
            );
        }
    }

    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        let (_, p) = this.ptr.get();
        let (_, q) = other.ptr.get();
        p == q
    }

    fn is_unique(&mut self) -> bool {
        if self.ptr.weak.compare_exchange(N, std::usize::MAX, Acquire, Relaxed).is_ok() {
            let (n, _) = self.ptr.get();
            let u = self.ptr.strong.load(Relaxed) == n;
            self.ptr.weak.store(N, Release);
            u
        } else {
            false
        }
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.is_unique() {
            Some(&mut self.ptr.data)
        } else {
            None
        }
    }

    // Clone self, sharing its weight and sometimes avoiding touching the reference count
    pub fn split(&mut self) -> Self {
        let (n, p) = self.ptr.get();
        if n == 1 {
            // We have no spare weight, we have to hit the global count so max it
            self.ptr.strong.fetch_add(N * 2 - 1, Relaxed);
            self.ptr.set_count(N);
            WeightedArc {
                ptr: self.ptr,
            }
        } else {
            // We have enough weight to share
            let m = n >> 1;
            let a = (n - m) << SHIFT;
            let b = m << SHIFT;
            self.ptr.set_count(a);
            WeightedArc {
                ptr: CountedNonNullPtr::new(b, p),
            }
        }
    }

    // Consume other, transferring its weight onto self, and sometimes avoiding touching the
    // global reference count
    pub fn merge(&mut self, other : Self) {
        assert!(WeightedArc::ptr_eq(self, &other));
        let (n, p) = self.ptr.get();
        let (n2, p2) = other.ptr.get();
        assert!(p.as_ptr() == p2.as_ptr());
        let n3 = n + n2;
        if n3 > N {
            // We have to release the excess
            // This can be relaxed because we know we are not releasing the last owner
            self.ptr.strong.fetch_sub(n3 - N, Relaxed);
            self.ptr.set_count(N);
        } else {
            // We can consolidate all the ownership into ourself
            self.ptr.set_count(n3);
        }
        std::mem::forget(other);
    }

    pub fn fortify(&mut self) {
        let (n, _) = self.ptr.get();
        if n < N {
            self.ptr.strong.fetch_add(N - n, Relaxed);
            self.ptr.set_count(N);
        }
    }

    fn condition(&mut self) {
        let (n, _) = self.ptr.get();
        if n == 1 {
            self.ptr.strong.fetch_add(N - 1, Relaxed);
            self.ptr.set_count(N);
        }
    }

}

impl<T : Clone> WeightedArc<T> {

    pub fn make_mut(this: &mut Self) -> &mut T {
        // This function is very subtle
        let (n, p) = this.ptr.get();
        if this.ptr.strong.compare_exchange(n, 0, Acquire, Relaxed).is_err() {
            // Another strong pointer exists, so clone .data into a new ArcInner
            *this = WeightedArc::new((**this).clone());
        } else {
            // We are the only strong pointer,
            // and have set .strong to zero,
            // but not dropped_in_place .data

            // Weak cannot be locked since it is only locked when in a method on this object
            // which we have exclusive access to and have just shown is alone

            if this.ptr.weak.load(Relaxed) != N {
                // There are weak pointers to the control block.
                // We need to move the value and release N from weak.
                let _weak : WeightedWeak<T> = WeightedWeak {
                    ptr : CountedNonNullPtr::new(N, p),
                };

                unsafe {
                    // Move the data into a new WeightedArc, leaving the old one with destroyed
                    // data, just as if it was dropped
                    let mut swap = WeightedArc::new(std::ptr::read(&p.as_ref().data));
                    std::mem::swap(this, &mut swap);
                    mem::forget(swap);
                }

            } else {
                // We are the only strong pointer
                // and have set .strong to zero
                // and are a unique reference to this arc, so downgrade is not being called on us
                // and after that, the weak count was zero
                // thus we are the only reference
                this.ptr.strong.store(N, Release);
            }

        }
        // Return whatever we point to now
        &mut this.ptr.data
    }


}


// Because WeightedArc is Sync we can't touch the local count when cloning.  Use split when we
// have a &mut self
impl<T> Clone for WeightedArc<T> {
    fn clone(&self) -> Self {
        self.ptr.strong.fetch_add(N, Relaxed);
        let (_, p) = self.ptr.get();
        Self {
            ptr: CountedNonNullPtr::new(N, p),
        }
    }
}


impl<T> Deref for WeightedArc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.ptr.data
    }
}

impl<T> Drop for WeightedArc<T> {
    fn drop(&mut self) {
        let (n, _) = self.ptr.get();
        if self.ptr.strong.fetch_sub(n, Release) != n {
            return
        }
        std::sync::atomic::fence(Acquire);
        unsafe { self.drop_slow() }
    }
}



impl<T> WeightedWeak<T> {

    pub fn new() -> WeightedWeak<T> {
        // A standalone WeightedWeak is created in a half-destroyed state, can never be upgraded
        // and isn't very useful!
        Self {
            ptr : CountedNonNullPtr::new(
                N,
                unsafe {
                    Box::into_raw_non_null(
                        Box::<ArcInner<T>>::new(
                            ArcInner {
                                strong : AtomicUsize::new(0),
                                weak : AtomicUsize::new(N),
                                data : std::mem::uninitialized(),
                            }
                        )
                    )
                }
            ),
        }
    }

    pub fn upgrade(&self) -> Option<WeightedArc<T>> {
        let mut s = self.ptr.strong.load(Relaxed);
        loop {
            if s == 0 {
                break None
            }
            match self.ptr.strong.compare_exchange_weak(s, s + N, Relaxed, Relaxed) {
                Ok(_) => {
                    let (_, p) = self.ptr.get();
                    break Some(
                        WeightedArc {
                            ptr: CountedNonNullPtr::new(N, p),
                        }
                    )
                },
                Err(old) => s = old,
            }
        }
    }

}

impl<T> Clone for WeightedWeak<T> {
    fn clone(&self) -> Self {
        self.ptr.weak.fetch_add(N, Relaxed);
        let (_, p) = self.ptr.get();
        Self {
            ptr: CountedNonNullPtr::new(N, p),
        }
    }
}

impl<T : std::fmt::Debug> std::fmt::Debug for WeightedArc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}

impl<T : std::cmp::PartialEq> std::cmp::PartialEq for WeightedArc<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T : std::cmp::Eq> std::cmp::Eq for WeightedArc<T> {
}



struct AtomicOptionWeightedArc<T> {
    ptr: AtomicCountedPtr<ArcInner<T>>,
}

impl<T> AtomicOptionWeightedArc<T> {

    fn to_ptr(p: Option<WeightedArc<T>>) -> CountedPtr<ArcInner<T>> {
        match p {
            None => CountedPtr { ptr: 0, phantom: PhantomData },
            Some(mut q) => {
                // Make sure we have more than 1 ownership, so we can be installed
                q.condition();
                let x : usize = q.ptr.ptr.get();
                std::mem::forget(q);
                CountedPtr { ptr: x, phantom: PhantomData, }
            },
        }
    }

    fn from_ptr(p: CountedPtr<ArcInner<T>>) -> Option<WeightedArc<T>> {
        match p.ptr {
            0 => None,
            x => Some(
                WeightedArc {
                    ptr: CountedNonNullPtr {
                        ptr: unsafe { NonZeroUsize::new_unchecked(x) },
                        phantom: PhantomData,
                    }
                }
            )
        }
    }

    fn new(p: Option<WeightedArc<T>>) -> Self {
        Self { ptr: AtomicCountedPtr::new(Self::to_ptr(p)) }
    }

    fn swap(&self, new: Option<WeightedArc<T>>) -> Option<WeightedArc<T>> {
        Self::from_ptr(self.ptr.swap(Self::to_ptr(new), AcqRel))
    }

    fn store(&self, new: Option<WeightedArc<T>>) {
        self.swap(new);
    }

    fn load(&self) -> Option<WeightedArc<T>> {
        let mut expected = self.ptr.load(Relaxed);
        let mut desired : CountedPtr<ArcInner<T>>;
        loop {
            let (n, p) = expected.get();
            if p.is_null() {
                return None
            }
            if n == 1 {
                // Spin until the count is increased
                std::thread::yield_now();
                expected = self.ptr.load(Relaxed);
                continue
            }
            desired = CountedPtr::new(n - 1, p);
            match self.ptr.compare_exchange_weak(
                expected,
                desired,
                Acquire,
                Relaxed
            ) {
                Ok(_) => break,
                Err(e) => expected = e,
            }
        }
        expected = desired;
        let (n, _) = expected.get();
        if n == 1 {
            // We have weight 1 in our load, and weight 1 in the atomic, locking it against
            // any further loads.  We need to get more weight for the atomic, so we also get
            // more for the return value
            expected.strong.fetch_add(2 * (N - 1), Relaxed);
            desired.set_count(N);
            match self.ptr.compare_exchange(expected, desired, Release, Relaxed) {
                Ok(_) => {},
                Err(_) => {
                    // We failed because the expected value was not there, so we aren't blocked
                    // anyway.  Give back the excess weight.
                    expected.strong.fetch_sub(N - 1, Relaxed);
                }
            }
            // No matter what, we have strengthened the
            expected.set_count(N);
        } else {
            expected.set_count(1);
        }
        Self::from_ptr(expected)
    }

}


mod tests {
    use super::*;

    #[test]
    fn test_new() {
        {
            assert!(std::mem::size_of::<Option<WeightedArc<usize>>>() == 8);
        }
        {
            let a = WeightedArc::new(0);
            assert!(*a == 0);
        }
        {
            let a = WeightedArc::new(4);
            assert!(WeightedArc::ptr_eq(&a, &a));
            let b = WeightedArc::new(5);
            assert!(!WeightedArc::ptr_eq(&a, &b));
        }
        {
            let a = WeightedArc::new(3);
            let b = a.clone();
            assert_eq!(a, b);
            assert!(WeightedArc::ptr_eq(&a, &b));
        }
        {
            {
                let a = WeightedArc::new(1);
                assert_eq!(WeightedArc::try_unwrap(a), Ok(1));
            }
            {
                let a = WeightedArc::new(2);
                let b = a.clone();
                println!("{:?}", a);
                assert_eq!(WeightedArc::try_unwrap(a), Err(b));
            }
        }

    }
}













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
