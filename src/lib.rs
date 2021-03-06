//! # A lock-free atomic `Arc` for some 64-bit architectures
//!
//! This crate provides `WeightedArc`, a mostly compatible replacement for `std::sync::Arc`,
//! and `AtomicWeightedArc`, which can be concurrently mutated by multiple threads, just like
//! `std::sync::atomic::AtomicPtr`.  The implementation of `AtomicWeightedArc` is lock-free
//! and typically much faster than the similar functionality provided by
//! `std::sync::Mutex<std::sync::Arc>`.  Characterization against other lock-free schemes such
//! as `crossbeam` is far from complete, but `AtomicWeightedArc` probably has relatively
//! expensive loads (and failed compare_and_swap), and relatively cheap other operations.  It
//! is probably uncompetitive for occasionally-updated data but may be useful for frequently
//! updated data such as the head of linked structures.  Notably, rather than modeling
//! `Cell<T>`, we model `AtomicT`, and we also support the full range of atomic operations
//! on `Weak`s, which have uses such as concurrent weak caches.
//!
//! # Example
//!
//! ```
//! use barc::*;
//! let a = AtomicWeightedArc::new(WeightedArc::new(1));
//! let current = a.load();
//! let new = WeightedArc::new(*current + 1);
//! a.compare_exchange(current, new);
//! assert_eq!(*a.load(), 2);
//! ```
//!
//! # How it works
//!
//! On common 64-bit architectures the full address space is not used, leaving some free bits in
//! the pointer.  We pack into these bits a 16 bit integer recording how much of the total reference
//! count is ours.  To atomically load a WeightedArc, we perform an atomic subtraction, taking
//! one unit of ownership and getting the old count and current pointer, which is enough to
//! construct a new valid `WeightedArc`.  If we perform tens of thousands of loads without
//! storing a new value, the counter will be almost depleted.  Threads will then get more weight
//! from the reference count and race to add it to the counter.  Only if all these threads are
//! pre-empted and the available weight drops to zero will new loads have to spin until one of
//! them manages to repair the counter.  Thus we claim the algorithm is lock-free up to some
//! number of threads simultanesously loading the same object, 256 by default.
//!
//! The crate also provides `AtomicOptionWeightedArc`, which seems more useful in practice, and
//! `WeightedWeak`, `AtomicWeightedWeak` and `AtomicOptionWeightedWeak`, all implemented similarly.
//!
//! ## Why we can't use `std::sync::Arc`
//!
//! To provide lock-free atomics with this technique we need to add and subtract arbitrary values
//! from the reference count of `std::sync::Arc` and there is no safe way to
//! perform these operations.  One unsafe option would be to subvert lifetimes to manipulate the
//! counts, as by calling clone and forget n times.  Another unsafe option would be to attempt to
//! deduce the address of the reference counts and access them directly.
//!
//! We instead choose the safer option of creating our own version of `Arc`.  (However, `Arc` is
//! notoriously difficult to get right, so our new implementation is likely to be less correct.)
//! Our implementation provides some minor optimizations like the ability to clone from &mut self
//! without touching the atomic reference count.  It also has some deficiencies ranging from the
//! signifcant limitation of only supporting `Sized` types, to the trivial limitation of not
//! supporting the impossible-to-use-correctly strong_count.
//!
//! The techniques used in this crate are drawn from std::sync::Arc and the C++ folly::AtomicSharedPtr.
//! In particular the implementation of `WeightedArc` closely follows `std::sync::Arc`, just
//! generalizing it to non-unit weights, and uses the same tricks for locking the weak count etc.
//!
//! (How to appropriately credit?)
//!
//! # Safety
//!
//! The implementation inherently relies on architecture-specific details of how pointers are
//! stored, and the library is not available on incompatible architectures.  We support
//! x86_64 and AArch64, covering the majority of desktop and mobile devices (?).  In
//! debug mode we also validate incoming pointers before packing them, which will
//! detect if we are trying to run on an incompatible system.  On other
//! architectures it should be possible to use alignment bits, at a significant performance
//! penalty, ultimately degenerating into a one-bit spinlock.
//!
//! The control block and counts used by `WeightedArc` are identical to the current implementation
//! of `std::sync::Arc` to the extent that it should be possible to freely convert between the
//! types with `Arc::from_raw(WeightedArc::into_raw(wa))` and vice versa.  This is of course
//! wildly unsafe.

#![feature(cfg_target_has_atomic)]
#![cfg(all(target_pointer_width = "64", target_has_atomic = "64"))]
#![cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#![feature(allocator_api)]
#![feature(box_into_raw_non_null)]
#![feature(extern_prelude)]

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::atomic::Ordering::{Relaxed, Acquire, AcqRel, Release};

use std::mem;

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::num::NonZeroUsize;
use std::option::Option;

use std::alloc::Alloc;

use std::ops::{Add, Sub, Deref};
use std::clone::Clone;
use std::fmt::Debug;
use std::cmp::{PartialEq, Eq};
use std::hash::{Hash, Hasher};

// # Pointer packing
//
// To perform atomic operations, we often need to atomically update both a pointer and a flag or
// a pointer and an integer.  In the case of `Arc`, the essential problem is that we must
// atomically load a pointer, and change the strong reference count associated with it.
//
// On x86_64 and AArch64, the address space is only 48 bits.  On x86_64 the 17 most significant
// bits of a pointer must be all zeros or all ones.  On some systems, ones are used to indicate
// protected kernel memory.
// We will also be pointing at structures that are at least 8-byte aligned, so the 3 least
// significant bits are also unused.  For now we will use the most significant 16 bits to
// represent an integer stored with the pointer.

// Rust does not yet have bit fields.  We define some constants to help us manually extract the
// pointer and the count:
const SHIFT : usize = 48;
const MASK  : usize = (1 << SHIFT) - 1;
const N : usize = 1 << 16;

// If fewer than this number of threads simultaneously access the types below, the acesses are
// lock-free.
const APPROX_MAX_LOCKFREE_THREADS : usize = 1 << 8;

/// Pointer and small counter packed into a 64 bit value
///
/// This trivial non-owning struct solves the problems of packing and unpacking the count and
/// pointer so that WeightedArc and related classes can concentrate on ownership.
///
/// The count can take on the values of `1` to `N` (not `0` to `N-1`), but is stored as `0` to `N-1`.
/// The payoff is that the representation of a pointer and a count of 1 is bitwise identical to
/// just the pointer, and thus bitwise identical to the corresponding `std::sync::Arc`.
// #[derive(Copy, Clone)] doesn't work
#[derive(Debug, Eq, PartialEq)]
struct CountedPtr<T> {
    ptr: usize,
    phantom: PhantomData<*mut T>,
}

impl<T> CountedPtr<T> {

    fn new(count: usize, pointer: *mut T) -> Self {
        debug_assert!(0 < count);
        debug_assert!(count <= N);
        // The most significant 16 bits of the pointer must be unset
        debug_assert!(pointer as usize & !MASK == 0);
        Self {
            ptr: ((count - 1) << SHIFT) | (pointer as usize),
            phantom: PhantomData,
        }
    }

    fn get(&self) -> (usize, *mut T) {
        (self.get_count(), self.get_ptr())
    }

    fn get_count(&self) -> usize {
        (self.ptr >> SHIFT) + 1
    }

    pub fn get_ptr(&self) -> *mut T {
        (self.ptr & MASK) as *mut T
    }

    fn set(&mut self, count: usize, pointer: *mut T) {
        *self = Self::new(count, pointer);
    }

    fn set_count(&mut self, count: usize) {
        let (_, p) = self.get();
        *self = Self::new(count, p);
    }

    fn set_ptr(&mut self, pointer: *mut T) {
        let (n, _) = self.get();
        *self = Self::new(n, pointer);
    }

    fn ptr_eq(left: Self, right: Self) -> bool {
        let (_n, p) = left.get();
        let (_m, q) = right.get();
        p == q
    }

    fn is_null(&self) -> bool {
        (self.ptr & MASK) != 0
    }

    unsafe fn as_ref(&self) -> &T {
        &*self.get_ptr()
    }

    unsafe fn as_mut(&mut self) -> &mut T {
        &mut *self.get_ptr()
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

/// Directly manipulate the count, leaving the pointer unchanged
impl<T> Sub<usize> for CountedPtr<T> {
    type Output = Self;

    fn sub(self, rhs: usize) -> Self::Output {
        let (n, p) = self.get();
        debug_assert!(n > rhs);
        Self::new(n - rhs, p)
    }
}

impl<T> Add<usize> for CountedPtr<T> {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        let (n, p) = self.get();
        debug_assert!(n + rhs <= N);
        Self::new(n + rhs, p)
    }
}

unsafe impl<T> Send for CountedPtr<T> {}
unsafe impl<T> Sync for CountedPtr<T> {}




/// Non-null pointer and small counter packed into a 64 bit value
///
/// This trivial non-owning struct solves the problems of packing and unpacking the count and
/// pointer so that `WeightedArc` and related classes can concentrate on ownership.
/// The use of a `NonZeroUsize` field lets `Option` recognize that it can use `0usize` to
/// represent `None`, ultimately allowing `Option<WeightedArc<T>>` to have the same representation as
/// the corresponding `CountedPtr<ArcInner<T>>`.
///
#[derive(Debug, Eq, PartialEq)]
struct CountedNonNull<T> {
    ptr: NonZeroUsize,
    phantom: PhantomData<NonNull<T>>,
}

impl<T> CountedNonNull<T> {

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

    fn get_count(&self) -> usize {
        (self.ptr.get() >> SHIFT) + 1
    }

    fn set_count(&mut self, count: usize) {
        let (_, p) = self.get();
        *self = Self::new(count, p);
    }

    fn get_ptr(&self) -> *mut T {
        let (_, p) = self.get();
        p.as_ptr()
    }

    fn set_ptr(&mut self, pointer: NonNull<T>) {
        let (count, _) = self.get();
        *self = Self::new(count, pointer);
    }

    pub unsafe fn as_ref(&self) -> &T {
        let (_, p) = self.get();
        &*p.as_ptr()
    }

    pub unsafe fn as_mut(&mut self) -> &mut T {
        let (_, p) = self.get();
        &mut *p.as_ptr()
    }

}

// Todo: why is this necessary?
impl<T> Clone for CountedNonNull<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            phantom: PhantomData,
        }
    }
}

impl<T> Copy for CountedNonNull<T> {}

unsafe impl<T> Send for CountedNonNull<T> {}
unsafe impl<T> Sync for CountedNonNull<T> {}


/// AtomicCountedPtr
///
/// Wrapper providing conversions between CountedPtr<T> and usize
///
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

    fn get_mut(&mut self) -> &mut CountedPtr<T> {
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

unsafe impl<T> Send for AtomicCountedPtr<T> {}
unsafe impl<T> Sync for AtomicCountedPtr<T> {}



/// Control block compatible with private struct std::sync::ArcInner
///
///
struct ArcInner<T : ?Sized> {
    strong : AtomicUsize,
    weak : AtomicUsize,
    data : T,
}

/// Mostly compaible replacement for [`std::sync::Arc`] that can be used with lock-free atomics
///
/// Except where noted, the documentation and guarantees of `Arc` are applicable.
///
pub struct WeightedArc<T> {
    ptr: CountedNonNull<ArcInner<T>>
}

/// Mostly compaible replacement for [`std::sync::Weak`] that can be used with lock-free atomics
///
/// Except where noted, the documentation and guarantees of `Weak` are applicable.
///
pub struct WeightedWeak<T> {
    ptr : CountedNonNull<ArcInner<T>>
}

impl<T> WeightedArc<T> {

    fn inner(&self) -> &ArcInner<T> {
        unsafe { self.ptr.as_ref() }
    }

    /// See [`std::sync::Arc::new`]
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::*;
    ///
    /// let a = WeightedArc::new(1);
    /// ```
    ///
    pub fn new(data: T) -> Self {
        // The initial weak count is 1 (rather than N) to maintain binary compatibility with
        // std:sync::Arc.
        Self {
            ptr: CountedNonNull::new(
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

    /// See [`std::sync::Arc::new`]
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let a = WeightedArc::new(7);
    /// {
    ///     let b = a.clone();
    ///     let c = WeightedArc::try_unwrap(b);
    ///     assert!(c.is_err());
    /// }
    /// let d = WeightedArc::try_unwrap(a);
    /// assert_eq!(d.unwrap(), 7);
    /// ```
    ///
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        let (n, p) = this.ptr.get();
        match this.inner().strong.compare_exchange(n, 0, Release, Relaxed) {
            Ok(_) => {
                std::sync::atomic::fence(Acquire);
                let data = unsafe { std::ptr::read(&p.as_ref().data) };
                let _weak : WeightedWeak<T> = WeightedWeak {
                    ptr : CountedNonNull::new(1, p),
                };
                std::mem::forget(this);
                Ok(data)
            }
            Err(_) => Err(this),
        }
    }

    /// See [`std::sync::Arc::into_raw`]
    ///
    /// # Safety
    ///
    /// It is safe to call this function and dereference the pointer.  If the pointer is not
    /// returned to management via [`WeightedArc::from_raw`] the object will be leaked.
    ///
    /// The returned pointer is into a private `barc::ArcInner<T>` structure that is layout and
    /// usage compatible with the private `std::sync::ArcInner<T>`, so passing it to
    /// [`std::sync::Arc::from_raw`] should work but is not recommended.
    ///
    /// # Example
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let a = WeightedArc::new(7);
    /// let p = WeightedArc::into_raw(a.clone());
    /// let b = unsafe { WeightedArc::from_raw(p) };
    /// assert!(WeightedArc::ptr_eq(&a, &b));
    /// ```
    ///
    pub fn into_raw(this: Self) -> *const T {
        let (n, _) = this.ptr.get();
        if n > 1 {
            // We can't store weight information in the pointer we return, so give back non-unit
            // weight to produce a normal pointer (that is pseudo-compatible with std::sync::Arc)
            let m = this.inner().strong.fetch_sub(n - 1, Relaxed);
            debug_assert!(m >= n);
        }
        let ptr : *const T = &*this;
        mem::forget(this);
        ptr
    }

    /// See [`std::sync::Arc::from_raw`].
    ///
    /// # Safety
    ///
    /// Unsafe.  The caller must ensure that the pointer originated from `WeightedArc::into_raw`.
    ///
    /// The pointer is expected to point into a private struct `barc::ArcInner` which is layout
    /// and usage compatible with the private struct `std::sync::ArcInner`, so passing a pointer
    /// from [`std::sync::Arc::into_raw`] should work but is not recommended.
    ///
    /// # Examples
    ///
    /// See [`WeightedArc::into_raw`].
    ///
    pub unsafe fn from_raw(ptr: *const T) -> Self {
        // todo: is this a good way to compute the offset?
        let fake_ptr = 0 as *const ArcInner<T>;
        let offset = &(*fake_ptr).data as *const T as usize;
        let p = NonNull::new_unchecked(((ptr as usize) - offset) as *mut ArcInner<T>);
        Self {
            ptr : CountedNonNull::new(1, p),
        }
    }

    /// See [`std::sync::Arc::downgrade`]
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, WeightedWeak};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = WeightedArc::downgrade(&a);
    /// let c = WeightedWeak::upgrade(&b);
    /// assert!(WeightedArc::ptr_eq(&a, &c.unwrap()));
    /// drop(a);
    /// let d = WeightedWeak::upgrade(&b);
    /// assert_eq!(d, None);
    /// ```
    pub fn downgrade(this: &Self) -> WeightedWeak<T> {
        let (_, p) = this.ptr.get();
        let mut cur = this.inner().weak.load(Relaxed);
        loop {
            if cur == std::usize::MAX {
                // The weak count is locked by is_unique.  Spin.
                cur = this.inner().weak.load(Relaxed);
                continue
            }
            match this.inner().weak.compare_exchange_weak(cur, cur + N, Acquire, Relaxed) {
                Ok(_) => break WeightedWeak {
                    ptr: CountedNonNull::new(N, p),
                },
                Err(old) => cur = old,
            }
        }
    }

    /// Return the total strong ownership of all `WeightedArc`s for the managed object.  Consider
    /// using strong_bound instead.  Users of the public interface don't have enough information
    /// to do anything useful with the returned value.
    ///
    /// # Compatibility
    ///
    /// The returned value will >= that returned by `Arc::strong_count` for equivalent
    /// code, because each object can have more than one ownership 'unit'.  This is one of two
    /// observable incompatibilities with Arc.  It will often be roughly 2^16 x.
    ///
    /// # Safety
    ///
    /// Though safe to call, the result is difficult to use safely, typically requiring some
    /// additional synchronization or guarantees.
    /// Prefer [`WeightedArc::try_unwrap`], [`WeightedArc::get_mut`] or [`WeightedArc::make_mut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let a = WeightedArc::new(1);
    /// let n = WeightedArc::strong_count(&a);
    /// assert!(n > 0);
    /// let b = a.clone();
    /// let m = WeightedArc::strong_count(&a);
    /// assert!(m > n);
    /// ```
    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong.load(Relaxed)
    }

    /// Return the total weak ownership of all `WeightedWeak`s for the managed object
    ///
    /// # Compatibility
    ///
    /// The returned value will differ from that returned by `Arc::weak_count` for equivalent
    /// code.  This is one of two observable incompatibilities with Arc.
    ///
    /// # Safety
    ///
    /// Though safe to call, the result is difficult to use safely, typically requiring some
    /// additional synchronization or guarantees.
    /// Prefer [`WeightedArc::try_unwrap`], [`WeightedArc::get_mut`] or [`WeightedArc::make_mut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let a = WeightedArc::new(1);
    /// let n = WeightedArc::weak_count(&a);
    /// assert_eq!(n, 0);
    /// let b = WeightedArc::downgrade(&a);
    /// let m = WeightedArc::weak_count(&a);
    /// assert!(m > 0);
    /// ```
    pub fn weak_count(this: &Self) -> usize {
        let n = this.inner().weak.load(Relaxed);
        if n == std::usize::MAX {
            0
        } else {
            n - 1
        }
    }

    /// Provide an upper bound on the number of `WeightedWeak`s associated with the object.
    ///
    /// # Safety
    ///
    /// It is safe to call this function, but difficult to use the result without further
    /// synchronization or guarantees.
    /// Prefer [`WeightedArc::try_unwrap`], [`WeightedArc::get_mut`] or [`WeightedArc::make_mut`].
    ///
    /// # Example
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let a = WeightedArc::new(1);
    /// let n = WeightedArc::weak_bound(&a);
    /// assert_eq!(n, 0);
    /// let b = WeightedArc::downgrade(&a);
    /// let m = WeightedArc::weak_bound(&a);
    /// assert!(m >= 1);
    /// ```
    pub fn weak_bound(this: &Self) -> usize {
        // I don't understand why this load is SeqCst in std::sync::Arc.  I believe that SeqCst
        // synchronizes only with itself, not Acqure/Release?
        let n = this.inner().weak.load(Relaxed);
        if n == std::usize::MAX {
            // .weak is locked, so must have been 1
            0
        } else {
            // We are calling this on a WeightedArc, so at least one WeightedArc is extant, so
            // the offset of 1 is active on .weak
            n - 1
        }
    }

    /// Provide an upper bound on the number of `WeightedArc`s associated with the object.
    ///
    /// # Safety
    ///
    /// It is safe to call this function, but difficult to use the result without further
    /// synchronization or guarantees.
    /// Prefer [`WeightedArc::try_unwrap`], [`WeightedArc::get_mut`] or [`WeightedArc::make_mut`].
    ///
    /// # Example
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let a = WeightedArc::new(1);
    /// let n = WeightedArc::strong_bound(&a);
    /// assert_eq!(n, 1);
    /// let b = a.clone();
    /// let m = WeightedArc::strong_bound(&a);
    /// assert!(m >= 2);
    /// ```
    pub fn strong_bound(this: &Self) -> usize {
        let (n, _) = this.ptr.get();
        // I don't understand why this load is SeqCst in std::sync::Arc.  I beleive that SeqCst
        // synchronizes only with itself not Acquire/Release
        let m = this.inner().strong.load(Relaxed);
        m - n + 1
    }

    unsafe fn drop_slow(&mut self) {
        // We have just set .strong to zero
        let (_, mut p) = self.ptr.get();
        std::ptr::drop_in_place(&mut p.as_mut().data);
        if self.inner().weak.fetch_sub(1, Release) == 1 {
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
        if self.inner().weak.compare_exchange(1, std::usize::MAX, Acquire, Relaxed).is_ok() {
            let (n, _) = self.ptr.get();
            let u = self.inner().strong.load(Relaxed) == n;
            self.inner().weak.store(1, Release);
            u
        } else {
            false
        }
    }

    /// Get `&mut T` only if `self` is the only `WeightedArc` managing the `T`
    ///
    /// # Examples
    ///
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            unsafe { Some(&mut this.ptr.as_mut().data) }
        } else {
            None
        }
    }

    /// Useful when we can statically prove that the `WeightedArc` is unique.  Subverts one of the
    /// basic services provided by the `Arc` interface and will likley be removed in the future.
    ///
    /// # Safety
    ///
    /// Misuse will cause a data race.
    ///
    /// # Panics
    ///
    /// In debug mode, panics if the `WeightedArc` is not truly unique.
    ///
    /// # Example
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let mut a = WeightedArc::new(1);
    /// unsafe { *WeightedArc::get_mut_unchecked(&mut a) += 1 };
    /// assert_eq!(*a, 2);
    /// ```
    ///
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        debug_assert!(this.is_unique());
        &mut this.ptr.as_mut().data
    }

    /// Clone a `&mut WeightedArc` without touching the reference count, by taking some of
    /// `self`'s local weight.
    ///
    /// This changes the representation of self, but not its equivalence class under `Eq` or
    /// `WeightedArc::ptr_eq`, so the invariant is upheld.
    /// As `WeightedArc` is `Sync` we cannot use this implementation for `clone`.
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let mut a = WeightedArc::new(2);
    /// let n = WeightedArc::strong_count(&a);
    /// let b = a.clone_mut();
    /// assert!(WeightedArc::ptr_eq(&a, &b));
    /// let m = WeightedArc::strong_count(&a);
    /// assert_eq!(n, m);
    /// ```
    ///
    pub fn clone_mut(&mut self) -> Self {
        let (n, p) = self.ptr.get();
        if n == 1 {
            // We have no spare weight, we have to hit the global count, so take enough to
            // maximize both pointers
            let old = self.inner().strong.fetch_add(N * 2 - 1, Relaxed);
            debug_assert!(old >= 1);
            self.ptr.set_count(N);
            WeightedArc {
                ptr: self.ptr,
            }
        } else {
            // We have enough weight to share
            let m = n >> 1;
            let a = n - m;
            let b = m;
            debug_assert!(a > 0);
            debug_assert!(b > 0);
            debug_assert!((a + b) == n);
            self.ptr.set_count(a);
            WeightedArc {
                ptr: CountedNonNull::new(b, p),
            }
        }
    }

    /// Split one `WeightedArc` into two, without touching the reference count if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let (a, b) = WeightedArc::split(WeightedArc::new(2));
    /// assert!(WeightedArc::ptr_eq(&a, &b));
    /// ```
    ///
    pub fn split(this: Self) -> (Self, Self) {
        let (mut n, p) = this.ptr.get();
        std::mem::forget(this);
        if n == 1 {
            let old = unsafe { p.as_ref().strong.fetch_add(N + N - 1, Relaxed) };
            debug_assert!(old >= 1);
            n = N + N;
        }
        let m = n >> 1;
        (
            WeightedArc{
                ptr: CountedNonNull::new(n - m, p),
            },
            WeightedArc{
                ptr: CountedNonNull::new(m, p),
            }
        )
    }

    /// Merge two `WeightedArc`s into one, without touching the reference count if possible.
    ///
    /// # Panics
    ///
    /// Panics if the arguments are not `WeghtedArc::ptr_eq`
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let a = WeightedArc::new(2);
    /// let (b, c) = WeightedArc::split(a);
    /// let d = WeightedArc::merge(b, c);
    /// ```
    pub fn merge(left: Self, right: Self) -> Self {
        assert!(WeightedArc::ptr_eq(&left, &right));
        let (n, p) = left.ptr.get();
        let (m, _) = right.ptr.get();
        std::mem::forget(left);
        std::mem::forget(right);
        let mut s = n + m;
        if s > N {
            let old = unsafe { p.as_ref().strong.fetch_sub(s - N, Relaxed) };
            debug_assert!(old >= s);
            s = N;
        }
        WeightedArc {
            ptr: CountedNonNull::new(s, p)
        }
    }

    /// Adds more weight to self if it is running low
    fn fortify(&mut self) {
        let (n, _) = self.ptr.get();
        if n < APPROX_MAX_LOCKFREE_THREADS {
            let old = self.inner().strong.fetch_add(N - n, Relaxed);
            debug_assert!(old >= n);
            self.ptr.set_count(N);
        }
    }

}

impl<T : Clone> WeightedArc<T> {

    /// Provide a `&mut T` by cloning the managed object if necessary
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let mut a = WeightedArc::new(1);
    /// *WeightedArc::make_mut(&mut a) += 1;
    /// assert_eq!(*a, 2);
    /// let b = a.clone();
    /// assert!(WeightedArc::ptr_eq(&a, &b));
    /// *WeightedArc::make_mut(&mut a) += 1;
    /// assert_eq!(*b, 2);
    /// assert_eq!(*a, 3);
    /// assert!(!WeightedArc::ptr_eq(&a, &b));
    /// ```
    ///
    pub fn make_mut(this: &mut Self) -> &mut T {
        // This function is very subtle!
        let (n, p) = this.ptr.get();
        if this.inner().strong.compare_exchange(n, 0, Acquire, Relaxed).is_err() {
            // Another strong pointer exists, so clone .data into a new ArcInner
            *this = WeightedArc::new((**this).clone());
        } else {
            // We are the only strong pointer,
            // and have set .strong to zero,
            // but not dropped_in_place .data

            // Weak cannot be locked since it is only locked when in a method on this object
            // which we have exclusive access to and have just shown is alone

            if this.inner().weak.load(Relaxed) != 1 {
                // There are weak pointers to the control block.
                // We need to move the value and release 1 from weak.
                let _weak : WeightedWeak<T> = WeightedWeak {
                    ptr : CountedNonNull::new(1, p),
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
                this.inner().strong.store(N, Release);
                this.ptr.set_count(N);
            }

        }
        // Return whatever we point to now
        unsafe { &mut this.ptr.as_mut().data }
    }


}


// Because WeightedArc is Sync we can't touch the local count when cloning.  Use split when we
// have a &mut self
impl<T> Clone for WeightedArc<T> {

    /// See `std::sync::Arc` `clone`
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let a = WeightedArc::new(1);
    /// let b = a.clone();
    /// assert!(WeightedArc::ptr_eq(&a, &b));
    /// assert_eq!(*a, *b);
    /// ```
    ///
    fn clone(&self) -> Self {
        let (n, p) = self.ptr.get();
        let old = self.inner().strong.fetch_add(N, Relaxed);
        debug_assert!(old >= n);
        Self {
            ptr: CountedNonNull::new(N, p),
        }
    }
}

impl<T: Default> Default for WeightedArc<T> {

    /// See [`std::sync::Arc`] `default`
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedArc;
    ///
    /// let a = WeightedArc::<usize>::default();
    /// assert_eq!(*a, usize::default());
    /// ```
    ///
    fn default() -> Self {
        WeightedArc::new(T::default())
    }
}

impl<T> Deref for WeightedArc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T> Drop for WeightedArc<T> {
    fn drop(&mut self) {
        let (n, _) = self.ptr.get();
        let old = self.inner().strong.fetch_sub(n, Release);
        debug_assert!(old >= n);
        if old != n {
            return
        }
        std::sync::atomic::fence(Acquire);
        unsafe { self.drop_slow() }
    }
}

impl<T : PartialEq<T>> PartialEq<WeightedArc<T>> for WeightedArc<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T : Eq> Eq for WeightedArc<T> {
}

impl<T : Hash> Hash for WeightedArc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

unsafe impl<T: Send + Sync> Send for WeightedArc<T> {}
unsafe impl<T: Send + Sync> Sync for WeightedArc<T> {}



impl<T> WeightedWeak<T> {

    fn inner(&self) -> &ArcInner<T> {
        unsafe { self.ptr.as_ref() }
    }

    /// Create an alreay expired `WeightedWeak`
    ///
    /// See [`std::sync::Weak::new`]
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::WeightedWeak;
    ///
    /// let a = WeightedWeak::<usize>::new();
    /// assert_eq!(WeightedWeak::upgrade(&a), None);
    ///
    /// ```
    ///
    pub fn new() -> WeightedWeak<T> {
        // A standalone WeightedWeak is created in a half-destroyed state, can never be upgraded
        // and isn't very useful!
        Self {
            ptr : CountedNonNull::new(
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

    /// See [`std::sync::Weak::upgrade`]
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, WeightedWeak};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = WeightedArc::downgrade(&a);
    /// assert_eq!(WeightedWeak::upgrade(&b), Some(a));
    /// // a is dropped
    /// assert_eq!(WeightedWeak::upgrade(&b), None);
    /// ```
    ///
    pub fn upgrade(&self) -> Option<WeightedArc<T>> {
        let mut s = self.inner().strong.load(Relaxed);
        loop {
            if s == 0 {
                break None
            }
            match self.inner().strong.compare_exchange_weak(s, s + N, Relaxed, Relaxed) {
                Ok(_) => {
                    let (_, p) = self.ptr.get();
                    break Some(
                        WeightedArc {
                            ptr: CountedNonNull::new(N, p),
                        }
                    )
                },
                Err(old) => s = old,
            }
        }
    }

    /// Clone from `&mut WeightedWeak` without touching the weak reference count.
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, WeightedWeak};
    ///
    /// let a = WeightedArc::new(7);
    /// let mut b = WeightedArc::downgrade(&a);
    /// let n = WeightedArc::weak_count(&a);
    /// let c = b.clone_mut();
    /// let m = WeightedArc::weak_count(&a);
    /// assert_eq!(n, m);
    /// assert_eq!(WeightedWeak::upgrade(&c), Some(a));
    /// ```
    ///
    pub fn clone_mut(&mut self) -> Self {
        let (n, p) = self.ptr.get();
        if n == 1 {
            let old = self.inner().weak.fetch_add(N + N - n, Relaxed);
            debug_assert!(old >= 1);
            self.ptr.set_count(N);
            WeightedWeak { ptr: CountedNonNull::new(N, p) }
        } else {
            let m = n >> 1;
            self.ptr.set_count(n - m);
            WeightedWeak { ptr: CountedNonNull::new(m, p) }
        }
    }

    pub fn split(mut this: Self) -> (Self, Self) {
        let a = this.clone_mut();
        (this, a)
    }

    // pub fn merge(a: Self, b: Self) -> Self

    fn fortify(&mut self) {
        if self.ptr.get_count() == 1 {
            let old = self.inner().weak.fetch_add(N - 1, Relaxed);
            debug_assert!(old >= 1);
            self.ptr.set_count(N);
        }
    }

}

impl<T> Clone for WeightedWeak<T> {

    /// See `std::sync::Weak` `clone`
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, WeightedWeak};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = WeightedArc::downgrade(&a);
    /// let c = b.clone();
    /// assert_eq!(WeightedWeak::upgrade(&c), Some(a));
    /// ```
    ///
    fn clone(&self) -> Self {
        // The weak count cannot be locked because it is only locked if there are no WeightedWeak
        // objects
        let (n, p) = self.ptr.get();
        let old = self.inner().weak.fetch_add(N, Relaxed);
        debug_assert!(old >= n);
        Self {
            ptr: CountedNonNull::new(N, p),
        }
    }
}

impl<T : Debug> Debug for WeightedArc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}

unsafe impl<T: Send + Sync> Send for WeightedWeak<T> {}
unsafe impl<T: Send + Sync> Sync for WeightedWeak<T> {}

/// Lock-free concurrent `Option<WeightedArc<T>>`
///
/// `AtomicOptionWeightedArc` provides an (almost) lock-free thread-safe nullable smart pointer,
/// providing an alternative to `Mutex<Option<Arc<T>>>`,
/// with an interface similar to [`std::sync::atomic::AtomicPtr`].
///
/// `store` and `swap` are always lock-free.
///
/// `load` and `compare_exchange` are lock-free unless there have been ~64k consecutive loads
/// (or failed compare_exchanges) AND ~256 threads concurrently calling these methods on the
/// object.  This should be infrequent.
///
/// Their hot paths consist of an
/// [`AtomicUsize::load`] followed by an [`AtomicUsize::compare_exchange_weak`] loop.  Since
/// read() involves decrementing the local count, it degrades under contention.
///
/// `compare_and_swap` and `compare_exchange_weak` are both implemented in terms of
/// `compare_exchange`, as there is little to be gained by exploiting spurious failure.
///
/// We do not support user-specified Orderings yet.  (We could use the stronger of user's and
/// what we currently use)
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
                q.fortify();
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
                    ptr: CountedNonNull {
                        ptr: unsafe { NonZeroUsize::new_unchecked(x) },
                        phantom: PhantomData,
                    }
                }
            )
        }
    }

    /// Constructs a new `AtomicOptionWeightedArc`
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::*;
    ///
    /// let a = WeightedArc::new(3);
    /// let b = AtomicOptionWeightedArc::new(Some(a.clone()));
    /// assert_eq!(b.load(), Some(a));
    /// let c = AtomicOptionWeightedArc::<usize>::new(None);
    /// assert_eq!(c.load(), None);
    /// ```
    ///
    pub fn new(p: Option<WeightedArc<T>>) -> Self {
        Self { ptr: AtomicCountedPtr::new(Self::to_ptr(p)) }
    }

    /// Non-atomic access to an unshared `AtomicOptionWeightedArc`
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::*;
    ///
    /// let a = WeightedArc::new(3);
    /// let mut b = AtomicOptionWeightedArc::new(Some(a.clone()));
    /// assert_eq!(*b.get_mut(), Some(a));
    /// b.store(None);
    /// assert_eq!(*b.get_mut(), None);
    /// ```
    ///
    pub fn get_mut(&mut self) -> &mut Option<WeightedArc<T>> {
        // Rely on layout compatibility
        unsafe { &mut *(self.ptr.get_mut() as *mut CountedPtr<ArcInner<T>> as *mut Option<WeightedArc<T>>) }
    }

    /// Consume the `AtomicOptionWeightedArc` returning the stored value
    /// (an `Option<WeightedArc<T>>`).
    /// Not to be confused with [`WeightedArc::try_unwrap`].
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::*;
    ///
    /// let a = WeightedArc::new(3);
    /// let b = AtomicOptionWeightedArc::new(Some(a.clone()));
    /// assert_eq!(b.into_inner(), Some(a));
    /// let c = AtomicOptionWeightedArc::<usize>::new(None);
    /// assert_eq!(c.into_inner(), None);
    /// ```
    ///
    pub fn into_inner(mut self) -> Option<WeightedArc<T>> {
        let a = unsafe { std::ptr::read(self.get_mut() as *mut Option<WeightedArc<T>>) };
        std::mem::forget(self);
        a
    }

    /// Atomically load the stored `Option<WeightedArc<T>>`
    ///
    /// # Implementation
    ///
    /// Typically lock-free
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::*;
    ///
    /// let a = WeightedArc::new(3);
    /// let b = AtomicOptionWeightedArc::new(Some(a.clone()));
    /// assert_eq!(b.load(), Some(a));
    /// b.store(None);
    /// assert_eq!(b.load(), None);
    /// ```
    ///
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

                // This should be practically impossible: requires draining the weight and then
                // APPROX_MAX_LOCKFREE_THREADS
                // threads to be inside .load or failing .compare_exchange.
                // While developing we want to know if this happens!
                debug_assert!(false, "Not lock-free!");
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

        // Check the weight and add more if it is running out
        self.maybe_replenish(&mut expected);
        Self::from_ptr(expected)
    }

    /// Takes the raw result of a load, checks its weight, maybe gets new weight from the
    /// global reference count and replenishes counter
    ///
    ///
    fn maybe_replenish(&self, expected: &mut CountedPtr<ArcInner<T>>) {
        let (n, p) = expected.get();
        debug_assert!(!p.is_null());
        if n <= APPROX_MAX_LOCKFREE_THREADS { // rare path
            // Try to replenish the counter before it depletes and threads start spinning
            let mut current = *expected;

            // Total weight available to us is n + 1
            // Total weight that can be stored in the atomic and the result is N + N
            // Get extra weight to maximize both
            let old = unsafe { (*p).strong.fetch_add((N - 1) + (N - n), Relaxed) };
            debug_assert!(old >= (n + 1));
            expected.set_count(N);
            let k = N - n;

            loop {
                let new = current + k;
                match self.ptr.compare_exchange_weak(current, new, Release, Relaxed) {
                    Ok(_) => {
                        break
                    },
                    Err(actual) => {
                        let (m, q) = actual.get();
                        if (q != p) || (m > n) {
                            // We can't install; either the count increased (so problem solved)
                            // or the pointer changed (so problem solved).  Give back some of the
                            // count
                            let old = unsafe { (*p).strong.fetch_sub(k, Relaxed) };
                            debug_assert!(old >= (N + N));
                            break
                        }
                        current = actual;
                    },
                }
            }
        } else {
            // The counter is fine
            expected.set_count(1);
        }
    }

    /// Atomically store a value.  The previous value is dropped.
    ///
    /// # Safety
    ///
    /// Always lock-free.
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::*;
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicOptionWeightedArc::new(Some(a.clone()));
    /// b.store(None);
    /// assert_eq!(b.load(), None);
    /// b.store(Some(a.clone()));
    /// assert_eq!(b.load(), Some(a));
    /// let c = WeightedArc::new(8);
    /// b.store(Some(c.clone()));
    /// assert_eq!(b.load(), Some(c));
    /// ```
    ///
    pub fn store(&self, new: Option<WeightedArc<T>>) {

        // store is swap, dropping the old value

        self.swap(new);
    }

    /// Atomically swap a value.  The previous value is returned.
    ///
    /// # Safety
    ///
    /// Always lock-free.
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::*;
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicOptionWeightedArc::new(Some(a.clone()));
    /// let mut c = b.swap(None);
    /// assert_eq!(c, Some(a.clone()));
    /// c = b.swap(Some(a.clone()));
    /// assert_eq!(c, None);
    /// ```
    ///
    pub fn swap(&self, new: Option<WeightedArc<T>>) -> Option<WeightedArc<T>> {

        // swap is simply an atomic swap.  The conversion of new to a CountedPtr will increase its
        // weight if it is low.

        Self::from_ptr(self.ptr.swap(Self::to_ptr(new), AcqRel))
    }

    /// See [`AtomicOptionWeightedArc::compare_and_swap`].
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

    /// Atomic compare exchange.  Typically lock-free.
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::*;
    /// let a = WeightedArc::new(7);
    /// let b = AtomicOptionWeightedArc::new(Some(a.clone()));
    /// let mut c = b.compare_exchange(Some(a.clone()), None);
    /// assert_eq!(c, Ok(Some(a.clone())));
    /// c = b.compare_exchange(Some(a.clone()), None);
    /// assert_eq!(c, Err(None));
    /// c = b.compare_exchange(None, Some(a.clone()));
    /// assert_eq!(c, Ok(None));
    /// c = b.compare_exchange(None, Some(a.clone()));
    /// assert_eq!(c, Err(Some(a)));
    /// ```
    ///
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
                        // Todo: merge these two objects rather than drop one.  Since current was
                        // likely split off old, should be common that they can be merged.
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

                    // While developing we want to know if this happens!
                    debug_assert!(false, "Not lock-free!");
                    std::thread::yield_now();
                    expected_cp = self.ptr.load(Relaxed);
                    continue;
                }
                let mut desired_cp = expected_cp - 1;
                match self.ptr.compare_exchange_weak(
                    expected_cp,
                    desired_cp,
                    Acquire,
                    Relaxed,
                ) {
                    Ok(_) => {
                        self.maybe_replenish(&mut desired_cp);
                        Self::from_ptr(current_cp);
                        Self::from_ptr(new_cp);
                        return Err(Self::from_ptr(desired_cp));
                    },
                    Err(old) => {
                        expected_cp = old;
                        continue;
                    }
                }
            }
        }
    }

    /// See [`AtomicOptionWeightedArc::compare_exchange`]
    ///
    /// The current implementation just calls `compare_exchange`.  Can we reimplement without
    /// too much duplication?
    pub fn compare_exchange_weak(
        &self,
        current: Option<WeightedArc<T>>,
        new: Option<WeightedArc<T>>,
    ) -> Result<Option<WeightedArc<T>>, Option<WeightedArc<T>>> {
        self.compare_exchange(current, new)
    }

}

impl<T: std::fmt::Debug> std::fmt::Debug for AtomicOptionWeightedArc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.load(), f)
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
/// Currently just a wrapper around [`AtomicOptionWeightedArc`], at slight(?) runtime cost
pub struct AtomicWeightedArc<T> {
    value: AtomicOptionWeightedArc<T>,
}

impl<T> AtomicWeightedArc<T> {

    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, AtomicWeightedArc};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicWeightedArc::new(a);
    /// ```
    ///
    pub fn new(val: WeightedArc<T>) -> Self {
        Self { value: AtomicOptionWeightedArc::new(Some(val)) }
    }

    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, AtomicWeightedArc};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicWeightedArc::new(a.clone());
    /// let c = b.into_inner();
    /// assert!(WeightedArc::ptr_eq(&a, &c));
    /// ```
    ///
    pub fn into_inner(self) -> WeightedArc<T> {
        self.value.into_inner().unwrap()
    }

    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, AtomicWeightedArc};
    ///
    /// let a = WeightedArc::new(7);
    /// let mut b = AtomicWeightedArc::new(a.clone());
    /// let c = WeightedArc::new(8);
    /// *b.get_mut() = c.clone();
    /// let d = b.into_inner();
    /// assert!(WeightedArc::ptr_eq(&c, &d));
    /// ```
    ///
    pub fn get_mut(&mut self) -> &mut WeightedArc<T> {
        // Rely on layout compatibility
        unsafe { &mut *(self.value.get_mut() as *mut Option<WeightedArc<T>> as *mut WeightedArc<T>) }
    }

    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, AtomicWeightedArc};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicWeightedArc::new(a.clone());
    /// let c = b.load();
    /// assert!(WeightedArc::ptr_eq(&a, &c));
    /// ```
    ///
    pub fn load(&self) -> WeightedArc<T> {
        self.value.load().unwrap()
    }

    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, AtomicWeightedArc};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicWeightedArc::new(a.clone());
    /// let c = WeightedArc::new(8);
    /// b.store(c.clone());
    /// let d = b.load();
    /// assert!(WeightedArc::ptr_eq(&c, &d));
    /// ```
    ///
    pub fn store(&self, new: WeightedArc<T>) {
        self.value.store(Some(new))
    }

    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, AtomicWeightedArc};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicWeightedArc::new(a.clone());
    /// let c = WeightedArc::new(8);
    /// let d = b.swap(c.clone());
    /// assert!(WeightedArc::ptr_eq(&a, &d));
    /// ```
    ///
    pub fn swap(&self, new: WeightedArc<T>) -> WeightedArc<T> {
        self.value.swap(Some(new)).unwrap()
    }

    /// See [`AtomicOptionWeightedArc::compare_exchange`] for details
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, AtomicWeightedArc};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicWeightedArc::new(a.clone());
    /// let c = WeightedArc::new(8);
    /// let mut d = b.compare_and_swap(a.clone(), c.clone());
    /// assert!(WeightedArc::ptr_eq(&a, &d)); // Swap succeeded
    /// d = b.compare_and_swap(a.clone(), c.clone());
    /// assert!(WeightedArc::ptr_eq(&c, &d)); // Swap failed
    /// ```
    ///
    pub fn compare_and_swap(
        &self,
        current: WeightedArc<T>,
        new: WeightedArc<T>,
    ) -> WeightedArc<T> {
        self.value.compare_and_swap(Some(current), Some(new)).unwrap()
    }

    /// See [`AtomicOptionWeightedArc::compare_exchange`] for details
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, AtomicWeightedArc};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicWeightedArc::new(a.clone());
    /// let c = WeightedArc::new(8);
    /// let mut d = b.compare_exchange(a.clone(), c.clone());
    /// assert_eq!(d, Ok(a.clone()));
    /// d = b.compare_exchange(a.clone(), c.clone());
    /// assert_eq!(d, Err(c.clone()));
    /// ```
    ///
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

    /// See [`AtomicOptionWeightedArc::compare_exchange_weak`] for details
    ///
    /// # Examples
    ///
    /// ```
    /// use barc::{WeightedArc, AtomicWeightedArc};
    ///
    /// let a = WeightedArc::new(7);
    /// let b = AtomicWeightedArc::new(a.clone());
    /// let c = WeightedArc::new(8);
    /// loop {
    ///     match b.compare_exchange_weak(a.clone(), c.clone()) {
    ///         Ok(old) => {
    ///             assert!(WeightedArc::ptr_eq(&a, &old));
    ///             break
    ///         },
    ///         Err(actual) => {
    ///             // Spurious failure must return the actual state
    ///             assert!(WeightedArc::ptr_eq(&a, &actual));
    ///             continue
    ///         }
    ///     }
    /// }
    /// let d = b.compare_exchange(a.clone(), c.clone());
    /// assert_eq!(d, Err(c.clone()));
    /// ```
    ///
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
///
/// Despite the differences between `WeightedWeak` and `WeightedArc`, the implementation of
/// `AtomicOptionWeightedWeak` is only concerned with manipulating reference counts and is thus
/// almost identical to `AtomicOptionWeightedArc`
pub struct AtomicOptionWeightedWeak<T> {
    ptr: AtomicCountedPtr<ArcInner<T>>,
}

impl<T> AtomicOptionWeightedWeak<T> {

    // todo: is there a way to share the duplicate implementation of AOWW and AOWA, that differ
    // only in manipulating .strong or .weak, without recourse to an enormous macro?

    // todo: doc tests

    fn to_ptr(mut oww: Option<WeightedWeak<T>>) -> CountedPtr<ArcInner<T>> {
        match oww.as_mut() {
            Some(r) => r.fortify(),
            None => ()
        }
        unsafe { std::mem::transmute::<Option<WeightedWeak<T>>, CountedPtr<ArcInner<T>>>(oww) }
    }

    fn from_ptr(p: CountedPtr<ArcInner<T>>) -> Option<WeightedWeak<T>> {
        unsafe { std::mem::transmute::<CountedPtr<ArcInner<T>>, Option<WeightedWeak<T>>>(p) }
    }

    pub fn new(oww: Option<WeightedWeak<T>>) -> Self {
        Self { ptr: AtomicCountedPtr::new(Self::to_ptr(oww)) }
    }

    pub fn get_mut(&mut self) -> &mut Option<WeightedWeak<T>> {
        unsafe { &mut *(self.ptr.get_mut() as *mut CountedPtr<ArcInner<T>> as *mut Option<WeightedWeak<T>>) }
    }

    pub fn into_inner(mut self) -> Option<WeightedWeak<T>> {
        let oww = Self::from_ptr(*self.ptr.get_mut());
        std::mem::forget(self);
        oww
    }

    pub fn load(&self) -> Option<WeightedWeak<T>> {
        let mut expected = self.ptr.load(Relaxed);
        loop {
            let (n, p) = expected.get();
            if p.is_null() {
                return None
            }
            if n == 1 {
                std::thread::yield_now();
                expected = self.ptr.load(Relaxed);
                continue;
            }
            match self.ptr.compare_exchange_weak(
                expected,
                expected - 1,
                Acquire,
                Relaxed
            ) {
                Ok(_) => {
                    break
                },
                Err(actual) => {
                    expected = actual;
                    continue;
                },
            }
        }
        expected = expected - 1;
        self.maybe_replenish(&mut expected);
        Self::from_ptr(expected)
    }

    fn maybe_replenish(&self, expected: &mut CountedPtr<ArcInner<T>>) {
        // todo: duplicated from WeightedArc::, find a better way
        let (n, p) = expected.get();
        debug_assert!(!p.is_null());
        if n <= APPROX_MAX_LOCKFREE_THREADS { // rare path
            // Try to replenish the counter before it depletes and threads start spinning
            let mut current = *expected;

            // Total weight available to us is n + 1
            // Total weight that can be stored in the atomic and the result is N + N
            // Get extra weight to maximize both
            let old = unsafe { (*p).weak.fetch_add((N - 1) + (N - n), Relaxed) };
            debug_assert!(old >= (n + 1));
            expected.set_count(N);
            let k = N - n;

            loop {
                let new = current + k;
                match self.ptr.compare_exchange_weak(current, new, Release, Relaxed) {
                    Ok(_) => {
                        break
                    },
                    Err(actual) => {
                        let (m, q) = actual.get();
                        if (q != p) || (m > n) {
                            // We can't install; either the count increased (so problem solved)
                            // or the pointer changed (so problem solved).  Give back some of the
                            // count
                            let old = unsafe { (*p).weak.fetch_sub(k, Relaxed) };
                            debug_assert!(old >= (N + N));
                            break
                        }
                        current = actual;
                    },
                }
            }
        } else {
            // The counter is fine
            expected.set_count(1);
        }
    }


    pub fn swap(&self, oww: Option<WeightedWeak<T>>) -> Option<WeightedWeak<T>> {
        Self::from_ptr(self.ptr.swap(Self::to_ptr(oww), AcqRel))
    }

    pub fn store(&self, oww: Option<WeightedWeak<T>>) {
        let _dropped = self.swap(oww);
    }

    pub fn compare_exchange(&self, current: Option<WeightedWeak<T>>, new: Option<WeightedWeak<T>>)
    -> Result<Option<WeightedWeak<T>>, Option<WeightedWeak<T>>> {
        let current_cp = Self::to_ptr(current);
        let new_cp = Self::to_ptr(new);
        let mut expected = self.ptr.load(Relaxed);
        loop {
            if CountedPtr::ptr_eq(expected, current_cp) {
                match self.ptr.compare_exchange_weak(
                    expected,
                    new_cp,
                    AcqRel,
                    Relaxed
                ) {
                    Ok(_) => {
                        let _drop_current = Self::from_ptr(current_cp);
                        return Ok(Self::from_ptr(expected))
                    },
                    Err(actual) => {
                        expected = actual;
                        continue
                    }
                }
            } else {
                if expected.is_null() {
                    let _drop_current = Self::from_ptr(current_cp);
                    let _drop_new = Self::from_ptr(new_cp);
                    return Err(None);
                }
                if expected.get_count() == 1 {
                    // While developing, we want to know if we hit this case
                    debug_assert!(false, "Lock-freedom violated!");
                    std::thread::yield_now();
                    expected = self.ptr.load(Relaxed);
                    continue
                }
                match self.ptr.compare_exchange_weak(
                    expected,
                    expected - 1,
                    Acquire,
                    Relaxed
                ) {
                    Ok(_) => {
                        expected = expected - 1;
                        self.maybe_replenish(&mut expected);
                        let _drop_current = Self::from_ptr(current_cp);
                        let _drop_new = Self::from_ptr(new_cp);
                        return Err(Self::from_ptr(expected))
                    }
                    Err(actual) => {
                        expected = actual;
                        continue;
                    },
                }
            }
        }
    }

    pub fn compare_and_swap(&self, current: Option<WeightedWeak<T>>, new: Option<WeightedWeak<T>>)
    -> Option<WeightedWeak<T>> {
        match self.compare_exchange(current, new) {
            Ok(old) => old,
            Err(old) => old,
        }
    }

    pub fn compare_exchange_weak(&self, current: Option<WeightedWeak<T>>, new: Option<WeightedWeak<T>>)
    -> Result<Option<WeightedWeak<T>>, Option<WeightedWeak<T>>> {
        self.compare_exchange(current, new)
    }

}

impl<T> Drop for AtomicOptionWeightedWeak<T> {
    fn drop(&mut self) {
        // Avoid doing an atomic operation since we are unique
        Self::from_ptr(*self.ptr.get_mut());
    }
}

/// Lock-free concurrent `WeightedWeak<T>`
///
/// A concurrently accessible `Weak`.  Useful building block for concurrent data structures,
/// for example a weak value dictionary.
///
/// Despite the unusual properties of `Weak` (and `WeightedWeak`),
/// the implementation of `AtomicWeightedWeak` is only concerned with manipulating the weak'
/// reference count and is practically identical to `AtomicWeightedArc`.
///
/// This struct is currently implemented in terms of `AtomicOptionWeightedWeak` incurring a small
/// runtime cost to wrap and `unwrap` `WeightedWeak`s
///
/// Compare `Mutex<Weak<T>>`.  Compare `AtomicPtr`.
pub struct AtomicWeightedWeak<T> {
    value: AtomicOptionWeightedWeak<T>,
}

impl<T> AtomicWeightedWeak<T> {

    pub fn new(ww: WeightedWeak<T>) -> Self {
        Self { value: AtomicOptionWeightedWeak::new(Some(ww)) }
    }

    pub fn into_inner(self) -> WeightedWeak<T> {
        self.value.into_inner().unwrap()
    }

    pub fn get_mut(&mut self) -> &mut WeightedWeak<T> {
        self.value.get_mut().as_mut().unwrap()
    }

    pub fn load(&self) -> WeightedWeak<T> {
        self.value.load().unwrap()
    }

    pub fn store(&self, ww: WeightedWeak<T>) {
        self.value.store(Some(ww))
    }

    pub fn swap(&self, ww: WeightedWeak<T>) -> WeightedWeak<T> {
        self.value.swap(Some(ww)).unwrap()
    }

    pub fn compare_and_swap(&self, current: WeightedWeak<T>, new: WeightedWeak<T>)
    -> WeightedWeak<T> {
        self.value.compare_and_swap(Some(current), Some(new)).unwrap()
    }

    pub fn compare_exchange(&self, current: WeightedWeak<T>, new: WeightedWeak<T>)
    -> Result<WeightedWeak<T>, WeightedWeak<T>> {
        match self.value.compare_exchange(Some(current), Some(new)) {
            Ok(old) => Ok(old.unwrap()),
            Err(old) => Err(old.unwrap()),
        }
    }

    pub fn compare_exchange_weak(&self, current: WeightedWeak<T>, new: WeightedWeak<T>)
    -> Result<WeightedWeak<T>, WeightedWeak<T>> {
        match self.value.compare_exchange_weak(Some(current), Some(new)) {
            Ok(old) => Ok(old.unwrap()),
            Err(old) => Err(old.unwrap()),
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicIsize;
    use std::sync::Arc; // todo: become "self-hosting"

    /// The service provided by Arc is to drop its contents at the appropriate time,
    /// so a key failure mode is not running drop at the end of scope.
    /// We can't test this with an assert_eq.
    ///
    /// We create a Canary type for testing.  For the test scope, one Nest hatches
    /// Canaries.  Each new or cloned Canary increments a counter shared with the Nest.  Each
    /// dropped Canary decrements the counter.  When the Nest is dropped it asserts that no
    /// Canaries are alive.  The analogy is becoming strained.
    ///

    #[derive(Debug)]
    struct Nest {
        counter: Arc<AtomicIsize>,
    }

    impl Nest {

        fn new() -> Self {
            Self { counter: Arc::new(AtomicIsize::new(0)) }
        }

        fn hatch(&self, x: isize) -> Canary {
            assert!(self.counter.fetch_add(1, Relaxed) >= 0);
            Canary { counter: self.counter.clone(), value: x }
        }

        fn check(&self) {
            assert_eq!(self.counter.load(Relaxed), 0);
        }

    }

    impl Drop for Nest {
        fn drop(&mut self) {
            assert_eq!(self.counter.load(Relaxed), 0);
        }
    }

    #[derive(Debug)]
    struct Canary {
        counter: Arc<AtomicIsize>,
        value: isize,
    }

    impl Canary {
    }

    impl Clone for Canary {
        fn clone(&self) -> Self {
            self.counter.fetch_add(1, Relaxed);
            Canary { counter: self.counter.clone(), value: self.value }
        }
    }

    impl Drop for Canary {
        fn drop(&mut self) {
            assert!(self.counter.fetch_sub(1, Relaxed) > 0);
        }
    }

    impl PartialEq for Canary {
        fn eq(&self, other: &Self) -> bool {
            assert!(Arc::ptr_eq(&self.counter, &other.counter));
            return &self.value == &other.value;
        }
    }

    impl Eq for Canary {

    }

    #[test]
    fn test_new() {
        {
            let nest = Nest::new();
            assert!(std::mem::size_of::<Option<WeightedArc<Canary>>>() == 8);
            assert!(std::mem::size_of::<Option<WeightedWeak<Canary>>>() == 8);
            assert!(APPROX_MAX_LOCKFREE_THREADS <= N);
            assert!(1 <= APPROX_MAX_LOCKFREE_THREADS);
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(0));
            assert_eq!(*a, nest.hatch(0));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(4));
            assert!(WeightedArc::ptr_eq(&a, &a));
            let b = WeightedArc::new(nest.hatch(5));
            assert!(!WeightedArc::ptr_eq(&a, &b));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(3));
            let b = a.clone();
            assert_eq!(a, b);
            assert!(WeightedArc::ptr_eq(&a, &b));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            assert_eq!(WeightedArc::try_unwrap(a), Ok(nest.hatch(1)));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(2));
            let b = a.clone();
            assert_eq!(WeightedArc::try_unwrap(a), Err(b));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            let b = a.clone();
            assert!(WeightedArc::ptr_eq(&a, &b));
            let c = WeightedArc::into_raw(a);
            let d = unsafe { WeightedArc::from_raw(c) };
            assert!(WeightedArc::ptr_eq(&d, &b));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            let b = WeightedArc::downgrade(&a);
            assert_eq!(WeightedWeak::upgrade(&b), Some(a));
            // a is dropped here
            assert_eq!(WeightedWeak::upgrade(&b), None);
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            assert_eq!(WeightedArc::strong_bound(&a), 1);
            assert_eq!(WeightedArc::weak_bound(&a), 0);
            let b = a.clone();
            assert!(WeightedArc::strong_bound(&a) > 1);
            let c = WeightedArc::downgrade(&a);
            assert!(WeightedArc::weak_bound(&a) > 0);
        }
        {
            let nest = Nest::new();
            let mut a = WeightedArc::new(nest.hatch(1));
            assert_eq!(*a, nest.hatch(1));
            *WeightedArc::make_mut(&mut a) = nest.hatch(2);
            assert_eq!(*a, nest.hatch(2));
            let b = a.clone();
            *WeightedArc::make_mut(&mut a) = nest.hatch(3);
            assert_eq!(*a, nest.hatch(3));
            assert_eq!(*b, nest.hatch(2));
        }
        {
            let nest = Nest::new();
            let mut a = WeightedArc::new(nest.hatch(1));
            assert_eq!(*a, nest.hatch(1));
            *WeightedArc::get_mut(&mut a).unwrap() = nest.hatch(2);
            assert_eq!(*a, nest.hatch(2));
            let mut b = a.clone();
            assert_eq!(WeightedArc::get_mut(&mut a), None);
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            let b = WeightedArc::new(nest.hatch(1));
            assert_eq!(a, b);
            assert!(!WeightedArc::ptr_eq(&a, &b));
        }
    }

    #[test]
    fn test_atomic() {
        {
            let nest = Nest::new();
            let a = AtomicOptionWeightedArc::new(Some(WeightedArc::new(nest.hatch(99))));
        }
        {
            let nest = Nest::new();
            let a = AtomicOptionWeightedArc::new(Some(WeightedArc::new(nest.hatch(1))));
            assert_eq!(a.load().unwrap().value, 1);
            //let b : AtomicOptionWeightedArc<usize> = AtomicOptionWeightedArc::new(None);
            //assert_eq!(b.load(), None);
        }
        {
            let nest = Nest::new();
            let a = AtomicOptionWeightedArc::new(Some(WeightedArc::new(nest.hatch(1))));
            assert_eq!(a.into_inner(), Some(WeightedArc::new(nest.hatch(1))));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            let b = AtomicOptionWeightedArc::new(Some(a.clone()));
            let c = b.load();
            assert_eq!(&a, &c.unwrap());
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            let b = WeightedArc::new(nest.hatch(2));
            let c = AtomicOptionWeightedArc::new(Some(a));
            c.store(Some(b));
            assert_eq!(*c.load().unwrap(), nest.hatch(2));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            let b = WeightedArc::new(nest.hatch(2));
            let c = AtomicOptionWeightedArc::new(Some(a));
            let d = c.swap(Some(b));
            assert_eq!(*d.unwrap(), nest.hatch(1));
            let e = c.load();
            assert_eq!(*e.unwrap(), nest.hatch(2));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            let b = WeightedArc::new(nest.hatch(2));
            let c = AtomicOptionWeightedArc::new(Some(a.clone()));
            let d = c.compare_and_swap(Some(a.clone()), Some(b.clone()));
            assert!(WeightedArc::ptr_eq(&d.unwrap(), &a));
            let e = c.compare_and_swap(Some(a.clone()), Some(b.clone()));
            assert!(WeightedArc::ptr_eq(&e.unwrap(), &b));
        }
        {
            let nest = Nest::new();
            let a = WeightedArc::new(nest.hatch(1));
            let b = WeightedArc::new(nest.hatch(2));
            let c = AtomicOptionWeightedArc::new(Some(a.clone()));
            let d = c.compare_exchange(Some(a.clone()), Some(b.clone()));
            assert_eq!(d, Ok(Some(a.clone())));
            let e = c.compare_exchange(Some(a.clone()), Some(b.clone()));
            assert_eq!(e, Err(Some(b.clone())));
            let f = c.compare_exchange(None, Some(a.clone()));
            assert_eq!(f, Err(Some(b.clone())));
            let g = c.compare_exchange(Some(b.clone()), None);
            assert_eq!(g, Ok(Some(b.clone())));
            let h = c.compare_exchange(Some(a.clone()), None);
            assert_eq!(h, Err(None));
            let i = c.compare_exchange(None, Some(a.clone()));
            assert_eq!(i, Ok(None));
        }
    }

    // todo: stress test mixed operations
}
