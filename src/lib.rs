//! # Lock-free atomic Arc for 64-bit architectures
//!
//! This crate provides a lock-free atomic smart pointer, a useful building block for lock-free
//! concurrent data structures.  WeightedArc is a drop-in replacement for Arc, and
//! AtomicWeightedArc provides the interface of the std::sync::atomic types for it.  For example,
//! we can compare_exchange WeightedArcs.
//!
//! WeightedArc is implemented with what is variously called external, distributed or weighted
//! reference counting.  Each WeightedArc packs a count into the spare bits of its pointer, marking
//! how many units of ownership it possesses, with the global strong count in the control block
//! being the total of all extant WeightedArc counts.  We can clone a WeightedArc by transferring
//! some of its count to the clone, without touching the reference count.  This allows us to
//! perform an atomic load by subtracting a constant from a one pointer-sized atomic value.
//!
//! Rarely (after 64k consecutive loads) the counter is depleted, and we must spin until the
//! thread that depleted it requests
//! more ownership from the global strong count and updates the atomic.  The load and
//! compare_exchange family methods are not atomic in this case.
//!
//! The crate also provides AtomicOptionWeightedArc, which seems more useful in practice, and
//! WeightedWeak, AtomicWeightedWeak and AtomicOptionWeightedWeak, all implemented identically.
//!
//! The crate relies heavily on std::sync::Arc and folly::AtomicSharedPtr
//!
//! Drawbacks of WeightedArc relative to std::sync::Arc are:
//! * Less tested
//! * Small cost to mask the pointer on each access?
//! * The strong_count and weak_count become upper bounds (but note there was no way to use these
//!   functions safely anyway)
//!
//! Drawbacks relative to Mutex<Arc> are:
//! * Uncontended performance?

#![feature(cfg_target_has_atomic)]
#![cfg(all(target_pointer_width = "64", target_has_atomic = "64"))]
#![feature(allocator_api)]
#![feature(box_into_raw_non_null)]
#![feature(extern_prelude)]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::Ordering::AcqRel;
use std::sync::atomic::Ordering::Release;
use std::sync::atomic::Ordering;

use std::marker::PhantomData;

use std::option::Option;

use std::mem;
use std::num::NonZeroUsize;

use std::alloc::Alloc;
use std::ptr::NonNull;

use std::ops::Deref;
use std::ops::DerefMut;
use std::clone::Clone;

use std::fmt::Debug;
use std::cmp::Eq;
use std::cmp::PartialEq;

// # Pointer packing
//
// To perform atomic operations, we often need to atomically update both a pointer and a flag or
// a pointer and an integer.  In the case of `Arc`, the essential problem is that we must
// atomically load a pointer, and change the strong reference count associated with it.
//
// On x86_64 and AArch64, the address space is only 48 bits.  The 17 most significant bits of
// a pointer must be all zeros or all ones.  Ones are used to indicate protected kernel memory.
// We will also be pointing at structures that are at least 8-byte aligned, so the 3 least
// significant bits are also unused.
//
// For now we will use the most significant 16 bits to represent an integer stored with the
// pointer.
//
// Rust does not yet have bit fields.  We define some constants to help us manually extract the
// pointer and the count:
const SHIFT : usize = 48;
const MASK  : usize = (1 << SHIFT) - 1;
const N : usize = 1 << 16;

/// Pointer and small counter packed into a 64 bit value
///
/// This trivial non-owning struct solves the problems of packing and unpacking the count and
/// pointer so that WeightedArc and related classes can concentrate on ownership.
///
/// The count can take on the values of `1` to `N` (not `0` to `N-1`), but is stored as `0` to `N-1`.
/// This is confusing, but the confusion is at least encapsulated in this struct.
/// The payoff is that the representation of a pointer and a count of 1 is bitwise identical to
/// just the pointer,
/// and bitwise identical to the corresponding `std::sync::Arc`.
struct CountedPtr<T> {
    ptr: usize,
    phantom: PhantomData<*mut T>,
}

impl<T> CountedPtr<T> {

    // 1 <= count <= N
    fn new(count: usize, pointer: *mut T) -> Self {
        debug_assert!(0 < count);
        debug_assert!(count <= N);
        debug_assert!(pointer as usize & !MASK == 0);
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

    fn ptr_eq(left: Self, right: Self) -> bool {
        let (_n, p) = left.get();
        let (_m, q) = right.get();
        p == q
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




/// Non-null pointer and small counter packed into a 64 bit value
///
/// This trivial non-owning struct solves the problems of packing and unpacking the count and
/// pointer so that `WeightedArc` and related classes can concentrate on ownership.
/// The use of a `NonZeroUsize` field lets `Option` recognize that it can use `0usize` to
/// represent `None`, ultimately allowing `Option<WeightedArc<T>>` to have the same representation as
/// the corresponding `CountedPtr<ArcInner<T>>`.
struct CountedNonNullPtr<T> {
    ptr: NonZeroUsize,
    phantom: PhantomData<NonNull<T>>,
}

impl<T> CountedNonNullPtr<T> {

    fn new(count: usize, pointer: NonNull<T>) -> Self {
        debug_assert!(0 < count);
        debug_assert!(count <= N);
        debug_assert!(pointer.as_ptr() as usize & !MASK == 0);
        let x = ((count - 1) << SHIFT) | (pointer.as_ptr() as usize);
        Self {
            ptr: unsafe { NonZeroUsize::new_unchecked(x) },
            phantom: PhantomData,
        }
    }

    fn get(&self) -> (usize, NonNull<T>) {
        let x = self.ptr.get();
        let p = (x & MASK) as *mut T;
        debug_assert!(!p.is_null());
        ((x >> SHIFT) + 1, unsafe { NonNull::new_unchecked(p) })
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

    fn into_inner(self) -> CountedPtr<T> {
        Self::from_usize(self.ptr.into_inner())
    }

    fn get_mut(&mut self) -> &CountedPtr<T> {
        // Relies on layout
        unsafe { &mut *(self.ptr.get_mut() as *mut usize as *mut CountedPtr<T>) }
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

/// Drop-in replacement for [`std::sync::Arc`] compatible with lock-free atomics
///
/// (represented as 0 to N - 1).  An Arc and a WeightedArc with weight one have the same
/// bit representation.  The weight is a measure of how much ownership object has, and it can be
/// reallocated between objects without touching the global .strong count, enabling lock-free
/// implementation of AtomicWeightedArc and some optimizing extensions to Arc's interface such as
/// split and merge, which sometimes enable us to clone and drop without touching the the global
/// reference count
pub struct WeightedArc<T> {
    ptr: CountedNonNullPtr<ArcInner<T>>
}

/// Drop-in replacement for [`std::sync::Weak`] compatible with lock-free atomics
///
/// WeightedWeak packs a weight into the spare bits of its pointer, the weight ranging from 1 to N
/// (represented as 0 to N - 1).  A Weak and a WeightedWeak with weight one have the same
/// bit representation.  The weight is a measure of how much weak ownership the object has
pub struct WeightedWeak<T> {
    ptr : CountedNonNullPtr<ArcInner<T>>
}

impl<T> WeightedArc<T> {

    /// Create a new WeightedArc managing the lifetime of a value on the heap
    pub fn new(data: T) -> Self {
        Self {
            ptr: CountedNonNullPtr::new(
                N,
                Box::into_raw_non_null(
                    Box::new(
                        ArcInner {
                            strong : AtomicUsize::new(N),
                            weak : AtomicUsize::new(1),
                            data : data,
                        }
                    )
                )
            )
        }
    }

    /// Attempt to move the value from the heap.  This is only allowed if no other WeightedArc is
    /// sharing it.  WeightedWeaks do not prevent unwrapping, though upgrading them will race with
    /// it.
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        let (n, p) = this.ptr.get();
        match this.ptr.strong.compare_exchange(n, 0, Release, Relaxed) {
            Ok(_) => {
                std::sync::atomic::fence(Acquire);
                let data = unsafe { std::ptr::read(&p.as_ref().data) };
                let _weak : WeightedWeak<T> = WeightedWeak {
                    ptr : CountedNonNullPtr::new(1, p),
                };
                std::mem::forget(this);
                Ok(data)
            }
            Err(_) => Err(this),
        }
    }

    /// Get a raw pointer to the object on the heap.  If this pointer is not eventually returned
    /// to management via from_raw, the object will be leaked.  Useful for foreign functions?
    /// If we have more than one weight, we return the excess to the global count first.
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
            match this.ptr.weak.compare_exchange_weak(cur, cur + 1, Acquire, Relaxed) {
                Ok(_) => break WeightedWeak {
                    ptr: CountedNonNullPtr::new(1, p),
                },
                Err(old) => cur = old,
            }
        }
    }

    /// This is not a count, but an upper bound on the number of WeightedWeaks.  Returns zero if
    /// and only if there were no WeightedWeaks at some time, but races against the downgrading
    /// of any WeightedArcs.
    pub fn weak_bound(this: &Self) -> usize {
        // I don't understand why this load is SeqCst in std::sync::Arc.  I believe that SeqCst
        // synchronizes only with itself, not Acqure/Release?
        let cnt = this.ptr.weak.load(Relaxed);
        if cnt == std::usize::MAX {
            // .weak is locked, so must have been 1
            0
        } else {
            // We are calling this on a WeightedArc, so at least one WeightedArc is extant, so
            // the offset of 1 is active on .weak
            cnt - 1
        }
    }

    /// This is not a count, but an upper bound on the number of WeightedArcs.  Since it is invoked
    /// on a WeightedArc, a lower bound is 1.  Returns if and only if the WeightedArc was unique
    /// at some time, but races against the upgrading of any WeightedWeaks
    pub fn strong_bound(this: &Self) -> usize {
        let (n, _) = this.ptr.get();
        // I don't understand why this load is SeqCst in std::sync::Arc.  I beleive that SeqCst
        // synchronizes only with itself not Acquire/Release
        let m = this.ptr.strong.load(Relaxed);
        m - n + 1
    }

    unsafe fn drop_slow(&mut self) {
        // We have just set .strong to zero
        let (_, mut p) = self.ptr.get();
        std::ptr::drop_in_place(&mut p.as_mut().data);
        if self.ptr.weak.fetch_sub(1, Release) == 1 {
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
        if self.ptr.weak.compare_exchange(1, std::usize::MAX, Acquire, Relaxed).is_ok() {
            let (n, _) = self.ptr.get();
            let u = self.ptr.strong.load(Relaxed) == n;
            self.ptr.weak.store(1, Release);
            u
        } else {
            false
        }
    }

    /// Get `&mut T` if `self` is the only `WeightedArc` managing the `T`
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            Some(&mut this.ptr.data)
        } else {
            None
        }
    }

    /// Useful when we can statically prove that the `WeightedArc` is unique.
    ///
    /// # Safety
    ///
    /// None
    ///
    /// # Panics
    ///
    /// Only in debug mode, panics if the `WeightedArc` is not unique.
    ///
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        debug_assert!(this.is_unique());
        &mut this.ptr.data
    }

    /// Clone, but allowed to mutate self in a way that does not change its equivalence class
    /// so that the postcondition still holds.  For WeightedArc, we can usually avoid touching
    /// the global strong count by stealing some weight from self.
    pub fn clone_mut(&mut self) -> Self {
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
            let a = n - m;
            let b = m;
            self.ptr.set_count(a);
            WeightedArc {
                ptr: CountedNonNullPtr::new(b, p),
            }
        }
    }

    pub fn split(this: Self) -> (Self, Self) {
        let (mut n, p) = this.ptr.get();
        std::mem::forget(this);
        if n == 1 {
            unsafe { p.as_ref().strong.fetch_add(N + N - 1, Relaxed) };
            n = N + N;
        }
        let m = n >> 1;
        (
            WeightedArc{
                ptr: CountedNonNullPtr::new(n - m, p),
            },
            WeightedArc{
                ptr: CountedNonNullPtr::new(n, p),
            }
        )
    }

    pub fn merge(left: Self, right: Self) -> Self {
        assert!(WeightedArc::ptr_eq(&left, &right));
        let (n, p) = left.ptr.get();
        let (m, _) = right.ptr.get();
        std::mem::forget(left);
        std::mem::forget(right);
        let mut s = n + m;
        if s > N {
            unsafe { p.as_ref().strong.fetch_sub(s - N, Relaxed) };
            s = N;
        }
        WeightedArc {
            ptr: CountedNonNullPtr::new(s, p)
        }
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

            if this.ptr.weak.load(Relaxed) != 1 {
                // There are weak pointers to the control block.
                // We need to move the value and release 1 from weak.
                let _weak : WeightedWeak<T> = WeightedWeak {
                    ptr : CountedNonNullPtr::new(1, p),
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
                this.ptr.set_count(N);
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

unsafe impl<T: Send + Sync> Send for WeightedArc<T> {}
unsafe impl<T: Send + Sync> Sync for WeightedArc<T> {}



impl<T> WeightedWeak<T> {

    /// Creates a control block without an object.  It can never be upgraded.  Only useful for
    /// providing an initial non-null state?  Consider Option<WeightedWeak<T>>.
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

    /// Attempt to make a strong reference to the object.  This will only succeed if there is
    /// somewhere else a WeightedArc keeping the object alive.  Races with dropping those other
    // WeightedArcs.
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
        // The weak count cannot be locked because it is only locked if there are no WeightedWeak
        // objects
        self.ptr.weak.fetch_add(N, Relaxed);
        let (_, p) = self.ptr.get();
        Self {
            ptr: CountedNonNullPtr::new(N, p),
        }
    }
}

impl<T : Debug> Debug for WeightedArc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}

impl<T : PartialEq> PartialEq for WeightedArc<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T : Eq> Eq for WeightedArc<T> {
}

/// Lock-free concurrent `Option<WeightedArc<T>>`
///
/// `AtomicOptionWeightedArc` provides an (almost) lock-free thread-safe nullable smart pointer,
/// providing an alternative to `Mutex<Option<Arc<T>>>`,
/// with an interface similar to [`std::sync::atomic::AtomicPtr`].
///
/// `store` and `swap` are always lock-free.
/// `load` and `compare_exchange` are lock-free,
/// except after 64k consecutive loads (or failed exchanges).
/// Their hot paths consist of an
/// [`AtomicUsize::load`] followed by an [`AtomicUsize::compare_exchange_weak`] loop.
///
/// `compare_and_swap` and `compare_exchange_weak` are both implemented in terms of
/// `compare_exchange`, as there is little to be gained by exploiting spurious failure.
///
/// .
///
pub struct AtomicOptionWeightedArc<T> {
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

    pub fn new(p: Option<WeightedArc<T>>) -> Self {
        Self { ptr: AtomicCountedPtr::new(Self::to_ptr(p)) }
    }

    /// Provide non-atomic access when not shared.
    /// Not to be confused with [`WeightedArc::get_mut`]
    pub fn get_mut(&mut self) -> &mut Option<WeightedArc<T>> {
        // Rely on layout compatibility
        unsafe { &mut *(self.ptr.get_mut() as *mut CountedPtr<T> as *mut Option<WeightedArc<T>>) }
    }

    /// Consume, returning the inner value.
    /// Not to be confused with [`WeightedArc::try_unwrap`]
    pub fn into_inner(mut self) -> Option<WeightedArc<T>> {
        let a = Self::from_ptr(*self.ptr.get_mut());
        std::mem::forget(self);
        a
    }

    /// Loading is the key operation for the structure.
    ///
    /// Typically a load and looping compare_exchange_weak, locking only when the counter is
    /// exhausted after 64k consecutive loads
    pub fn load(&self) -> Option<WeightedArc<T>> {

        // load is almost exactly fetch_sub(1 << SHIFT, Acquire)
        //
        // The complexity arises entirely from handling the rare (but not impossible) case where
        // the Count is depleted.  Then other threads spin waiting for the last successful loader
        // to replenish the CountedPtr.

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
            expected.strong.fetch_add((N - 1) + (N - 1), Relaxed);
            desired.set_count(N);
            match self.ptr.compare_exchange(expected, desired, Release, Relaxed) {
                Ok(_) => {},
                Err(_) => {
                    // We failed because the expected value was not there, so we aren't blocked
                    // anyway.  Give back the excess weight.
                    expected.strong.fetch_sub(N - 1, Relaxed);
                    // The loaded value is presumably stale now, but we don't care (we wouldn't
                    // know if we hadn't looked!)
                }
            }
            // No matter what, we have strengthened the reesult
            expected.set_count(N);
        } else {
            expected.set_count(1);
        }
        Self::from_ptr(expected)
    }

    /// Always lock-free
    fn store(&self, new: Option<WeightedArc<T>>) {

        // store is swap, dropping the old value

        self.swap(new);
    }

    /// Always lock-free
    fn swap(&self, new: Option<WeightedArc<T>>) -> Option<WeightedArc<T>> {

        // swap is simply an atomic swap.  The conversion of new to a CountedPtr will increase its
        // weight if it is low.

        Self::from_ptr(self.ptr.swap(Self::to_ptr(new), AcqRel))
    }

    /// See compare_exchange
    pub fn compare_and_swap(
        &self,
        current: Option<WeightedArc<T>>,
        new: Option<WeightedArc<T>>,
    ) -> Option<WeightedArc<T>> {
        match self.compare_exchange(current, new) {
            Ok(old) => old,
            Err(old) => old,
        }
    }

    /// Typically a load and looping compare_exchange_weak, spinning only when the counter is
    /// exhausted
    pub fn compare_exchange(&self, current: Option<WeightedArc<T>>, new: Option<WeightedArc<T>>)
    -> Result<Option<WeightedArc<T>>, Option<WeightedArc<T>>> {

        // compare_exchange must load its value on failure, so its loop combines the code for
        // swap and load

        let current_cp = Self::to_ptr(current);
        let new_cp = Self::to_ptr(new);
        let mut expected_cp = self.ptr.load(Relaxed);
        loop {
            if CountedPtr::ptr_eq(expected_cp, current_cp) {
                // attempt to swap
                match self.ptr.compare_exchange_weak(
                    expected_cp,
                    new_cp,
                    AcqRel,
                    Relaxed,
                ) {
                    Ok(old) => {
                        // Success linearization point
                        // Todo: merge these two objects rather than drop one
                        Self::from_ptr(current_cp);
                        return Ok(Self::from_ptr(old))
                    },
                    Err(old) => {
                        expected_cp = old;
                        continue;
                    }
                }
            } else {
                let (n, p) = expected_cp.get();
                if p.is_null() {
                    Self::from_ptr(current_cp);
                    Self::from_ptr(new_cp);
                    return Err(Self::from_ptr(expected_cp))
                }
                if n == 1 {
                    // Rare branch
                    // The atomic has no extra weight to give us.
                    // Spin until another thread replenishes the count
                    std::thread::yield_now();
                    expected_cp = self.ptr.load(Relaxed);
                    continue;
                }
                let desired_cp = CountedPtr::new(n - 1, p);
                match self.ptr.compare_exchange_weak(
                    expected_cp,
                    desired_cp,
                    Acquire,
                    Relaxed,
                ) {
                    Ok(_) => {
                        // Failure linearization point
                        let mut m = 1;
                        if n == 2 {
                            // Rare branch
                            // No more loads are possible until we put more weight into the atomic
                            // Since we are touching the refcount anyway, also get more weight
                            // for the return value
                            unsafe { (*p).strong.fetch_add((N - 1) + (N - 1), Relaxed); }
                            m = N;
                            // Deliberately not compare_exchange_weak: real failure is fine, but
                            // we can't accept spurious failure
                            match self.ptr.compare_exchange(
                                desired_cp,
                                CountedPtr::new(N, p), // Maximum weight
                                AcqRel, // Ordered with the refcount manipulations above and below
                                Relaxed,
                            ) {
                                Ok(_) => {
                                    // We replenished the count
                                },
                                Err(_) => {
                                    // True failure, so some other thread has really changed the
                                    // value.  Give back some of the extra weight, using the rest
                                    // to top up the return value
                                    unsafe { (*p).strong.fetch_sub(N - 1, Relaxed) };
                                    // The loaded value is presumably stale now, but we don't care
                                    // (we wouldn't know if we hadn't looked!)
                                }
                            }
                        }
                        Self::from_ptr(current_cp);
                        Self::from_ptr(new_cp);
                        return Err(Self::from_ptr(CountedPtr::new(m, p)))
                    },
                    Err(old) => {
                        expected_cp = old;
                        continue;
                    }
                }
            }
        }
    }

    pub fn compare_exchange_weak(
        &self,
        current: Option<WeightedArc<T>>,
        new: Option<WeightedArc<T>>,
    ) -> Result<Option<WeightedArc<T>>, Option<WeightedArc<T>>> {
        // Beacuse we must load on failure, there are limited opportunities to fail usefully
        self.compare_exchange(current, new)
    }

}

impl<T> Default for AtomicOptionWeightedArc<T> {
    fn default() -> Self {
        Self::new(None)
    }
}

impl<T> Drop for AtomicOptionWeightedArc<T> {
    fn drop(&mut self) {
        // Jump through some hoops to avoid doing an atomic load, since we are sole owner
        Self::from_ptr(*self.ptr.get_mut());
    }
}

unsafe impl<T: Send + Sync> Send for AtomicOptionWeightedArc<T> {}
unsafe impl<T: Send + Sync> Sync for AtomicOptionWeightedArc<T> {}





/// Lock-free concurrent [`WeightedArc`]
///
/// [`std::sync::Arc`] has atomic reference counts that manage the lifetime of the stored value,
/// but any particular `Arc` cannot be safely mutated by multiple threads.
/// [`std::sync::atomic::AtomicPtr`] can be concurrently accessed and modified by multiple threads,
/// but does not manage the lifetime of the pointed-to value.
/// [`AtomicWeightedArc`] provides both atomic reference counting and atomic mutability, without
/// locking, making it a useful primitive for constructing lock-free concurrent data structures.
/// Like `Arc`, it does not make the managed value thread-safe.  `AtomicWeightedArc<T>` can
/// safely replace `Mutex<Arc<T>>`, potentially increasing performance under contention.
/// It cannot replace `Arc<Mutex<T>>`.
///
/// Compare interface to `AtomicPtr`.
/// Currently just a wrapper around [`AtomicOptionWeightedArc`]
pub struct AtomicWeightedArc<T> {
    value: AtomicOptionWeightedArc<T>,
}

impl<T> AtomicWeightedArc<T> {

    pub fn new(val: WeightedArc<T>) -> Self {
        Self { value: AtomicOptionWeightedArc::new(Some(val)) }
    }

    pub fn into_inner(self) -> WeightedArc<T> {
        self.value.into_inner().unwrap()
    }

    pub fn get_mut(&mut self) -> &mut WeightedArc<T> {
        // Rely on layout compatibility
        unsafe { &mut *(self.value.get_mut() as *mut Option<WeightedArc<T>> as *mut WeightedArc<T>) }
    }

    /// See [`AtomicOptionWeightedArc`] for details
    pub fn load(&self) -> WeightedArc<T> {
        self.value.load().unwrap()
    }

    pub fn store(&self, new: WeightedArc<T>) {
        self.value.store(Some(new))
    }

    pub fn swap(&self, new: WeightedArc<T>) -> WeightedArc<T> {
        self.value.swap(Some(new)).unwrap()
    }

    pub fn compare_and_swap(
        &self,
        current: WeightedArc<T>,
        new: WeightedArc<T>,
    ) -> WeightedArc<T> {
        self.value.compare_and_swap(Some(current), Some(new)).unwrap()
    }

    /// See [`AtomicOptionWeightedArc`] for details
    pub fn compare_exchange(
        &self,
        current: WeightedArc<T>,
        new: WeightedArc<T>,
    ) -> Result<WeightedArc<T>, WeightedArc<T>> {
        match self.value.compare_exchange(Some(current), Some(new)) {
            Ok(old) => Ok(old.unwrap()),
            Err(old) => Err(old.unwrap()),
        }
    }

    /// The current implementation calls ['compare_exchange`] as there is no particular benefit
    /// to allowing spurious failure
    pub fn compare_exchange_weak(
        &self,
        current: WeightedArc<T>,
        new: WeightedArc<T>,
    ) -> Result<WeightedArc<T>, WeightedArc<T>> {
        match self.value.compare_exchange_weak(Some(current), Some(new)) {
            Ok(old) => Ok(old.unwrap()),
            Err(old) => Err(old.unwrap()),
        }
    }

}

impl<T: Default> Default for AtomicWeightedArc<T> {
    fn default() -> Self {
        Self::new(WeightedArc::default())
    }
}

unsafe impl<T: Send + Sync> Send for AtomicWeightedArc<T> {}
unsafe impl<T: Send + Sync> Sync for AtomicWeightedArc<T> {}

/// Lock-free concurrent `Option<WeightedWeak<T>>`
struct AtomicOptionWeightedWeak<T> {
    ptr: AtomicCountedPtr<ArcInner<T>>,
}

/// Lock-free concurrent `WeightedWeak<T>`
struct AtomicWeightedWeak<T> {
    value: AtomicOptionWeightedWeak<T>,
}

// Todo: AtomicOptionWeightedWeak, AtomicWeightedWeak

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicIsize;


    #[derive(Debug, PartialEq, Eq)]
    struct Canary {
        value: usize,
    }

    static CANARY_COUNT: AtomicIsize = AtomicIsize::new(0);

    impl Canary {
        fn new(x: usize) -> Self {
            let n = CANARY_COUNT.fetch_add(1, AcqRel);
            assert!(n >= 0);
            println!("Creating with value {:?}", x);
            Canary { value: x }
        }
        fn check() {
            assert_eq!(CANARY_COUNT.load(Acquire), 0);
        }
    }

    impl Clone for Canary {
        fn clone(&self) -> Self {
            Canary::new(self.value)
        }
    }

    impl Drop for Canary {
        fn drop(&mut self) {
            println!("Dropping with value {:?}", self.value);
            let n = CANARY_COUNT.fetch_sub(1, AcqRel);
            assert!(n > 0);
        }
    }

    //#[test]
    fn test_new() {
        CANARY_COUNT.store(0, Release);
        Canary::check();
        {
            assert!(std::mem::size_of::<Option<WeightedArc<Canary>>>() == 8);
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(0));
            assert_eq!(*a, Canary::new(0));
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(4));
            assert!(WeightedArc::ptr_eq(&a, &a));
            let b = WeightedArc::new(Canary::new(5));
            assert!(!WeightedArc::ptr_eq(&a, &b));
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(3));
            let b = a.clone();
            assert_eq!(a, b);
            assert!(WeightedArc::ptr_eq(&a, &b));
        }
        Canary::check();
        {
            {
                let a = WeightedArc::new(Canary::new(1));
                assert_eq!(WeightedArc::try_unwrap(a), Ok(Canary::new(1)));
            }
            Canary::check();
            {
                let a = WeightedArc::new(Canary::new(2));
                let b = a.clone();
                println!("{:?}", a);
                assert_eq!(WeightedArc::try_unwrap(a), Err(b));
            }
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(1));
            let b = a.clone();
            assert!(WeightedArc::ptr_eq(&a, &b));
            let c = WeightedArc::into_raw(a);
            let d = unsafe { WeightedArc::from_raw(c) };
            assert!(WeightedArc::ptr_eq(&d, &b));
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(1));
            let b = WeightedArc::downgrade(&a);
            assert_eq!(WeightedWeak::upgrade(&b), Some(a));
            // a is dropped here
            assert_eq!(WeightedWeak::upgrade(&b), None);
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(1));
            assert_eq!(WeightedArc::strong_bound(&a), 1);
            assert_eq!(WeightedArc::weak_bound(&a), 0);
            let b = a.clone();
            assert!(WeightedArc::strong_bound(&a) > 1);
            let c = WeightedArc::downgrade(&a);
            assert!(WeightedArc::weak_bound(&a) > 0);
        }
        Canary::check();
        {
            let mut a = WeightedArc::new(Canary::new(1));
            assert_eq!(*a, Canary::new(1));
            *WeightedArc::make_mut(&mut a) = Canary::new(2);
            assert_eq!(*a, Canary::new(2));
            let b = a.clone();
            *WeightedArc::make_mut(&mut a) = Canary::new(3);
            assert_eq!(*a, Canary::new(3));
            assert_eq!(*b, Canary::new(2));
        }
        Canary::check();
        {
            let mut a = WeightedArc::new(Canary::new(1));
            assert_eq!(*a, Canary::new(1));
            *WeightedArc::get_mut(&mut a).unwrap() = Canary::new(2);
            assert_eq!(*a, Canary::new(2));
            let mut b = a.clone();
            assert_eq!(WeightedArc::get_mut(&mut a), None);
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(1));
            let b = WeightedArc::new(Canary::new(1));
            assert_eq!(a, b);
            assert!(!WeightedArc::ptr_eq(&a, &b));
        }
        Canary::check();
    }

    #[test]
    fn test_atomic() {
        CANARY_COUNT.store(0, Release);
        Canary::check();
        {
            let a = AtomicOptionWeightedArc::new(Some(WeightedArc::new(Canary::new(99))));
        }
        Canary::check();
        {
            let a = AtomicOptionWeightedArc::new(Some(WeightedArc::new(Canary::new(1))));
            assert_eq!(a.load().unwrap().value, 1);
            //let b : AtomicOptionWeightedArc<usize> = AtomicOptionWeightedArc::new(None);
            //assert_eq!(b.load(), None);
        }
        Canary::check();
        {
            let a = AtomicOptionWeightedArc::new(Some(WeightedArc::new(Canary::new(1))));
            assert_eq!(a.into_inner(), Some(WeightedArc::new(Canary::new(1))));
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(1));
            let b = AtomicOptionWeightedArc::new(Some(a.clone()));
            let c = b.load();
            assert_eq!(&a, &c.unwrap());
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(1));
            let b = WeightedArc::new(Canary::new(2));
            let c = AtomicOptionWeightedArc::new(Some(a));
            c.store(Some(b));
            assert_eq!(*c.load().unwrap(), Canary::new(2));
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(1));
            let b = WeightedArc::new(Canary::new(2));
            let c = AtomicOptionWeightedArc::new(Some(a));
            let d = c.swap(Some(b));
            assert_eq!(*d.unwrap(), Canary::new(1));
            let e = c.load();
            assert_eq!(*e.unwrap(), Canary::new(2));
        }
        Canary::check();
        {
            let a = WeightedArc::new(Canary::new(1));
            let b = WeightedArc::new(Canary::new(2));
            let c = AtomicOptionWeightedArc::new(Some(a.clone()));
            let d = c.compare_and_swap(Some(a.clone()), Some(b.clone()));
            assert!(WeightedArc::ptr_eq(&d.unwrap(), &a));
            let e = c.compare_and_swap(Some(a.clone()), Some(b.clone()));
            assert!(WeightedArc::ptr_eq(&e.unwrap(), &b));
        }
        Canary::check();
    }
}
