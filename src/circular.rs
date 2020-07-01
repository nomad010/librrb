//! A circular buffer.
//!
//! A fixed-size container of elements that supports fast operations on either side of the buffer.

use crate::{Side, RRB_WIDTH};
use std::fmt::{self, Debug, Formatter};
use std::iter::FusedIterator;
use std::mem;
use std::ops::Range;
use thiserror::Error;

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CircularBufferError {
    #[error("Tried to pop an element from an empty buffer")]
    EmptyBuffer,
    #[error("tried to insert an element into a full buffer")]
    FullBuffer,
    // #[error("tried to access an element past the end of the buffer")]
    // IndexOutOfBounds,
}

type Result<T> = std::result::Result<T, CircularBufferError>;

pub(crate) struct BorrowBufferMut<A: Debug> {
    pub(crate) range: Range<usize>,
    data: *mut mem::MaybeUninit<A>,
    front: usize,
    len: usize,
}

impl<A: Debug> Debug for BorrowBufferMut<A> {
    fn fmt(&self, f: &mut Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.write_str("[")?;
        if !self.is_empty() {
            for i in 0..self.len() {
                f.write_str(&format!("{:?}, ", self.get(i).unwrap()))?;
            }
        }
        f.write_str("]")?;
        Ok(())
    }
}

impl<'a, A: Debug> BorrowBufferMut<A> {
    pub fn from_same_source(&self, other: &Self) -> bool {
        self.data == other.data
    }

    pub fn combine(&mut self, other: Self) {
        assert!(self.from_same_source(&other));
        assert!(other.range.start == self.range.end);
        self.range.end = other.range.end;
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn index_for(&self, idx: usize) -> usize {
        (self.front + idx) % RRB_WIDTH
    }

    pub fn get(&self, idx: usize) -> Option<&A> {
        let idx = idx + self.range.start;
        if self.range.contains(&idx) {
            let index = self.index_for(idx);
            unsafe {
                let data = std::slice::from_raw_parts(self.data, RRB_WIDTH);
                Some(&*data[index].as_ptr())
            }
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut A> {
        let idx = idx + self.range.start;
        if self.range.contains(&idx) {
            let index = self.index_for(idx);
            unsafe {
                let data = std::slice::from_raw_parts_mut(self.data, RRB_WIDTH);
                Some(&mut *data[index].as_mut_ptr())
            }
        } else {
            None
        }
    }

    pub fn front_mut(&mut self) -> Option<&mut A> {
        self.get_mut(0)
    }

    pub fn split_at(&mut self, index: usize) -> Self {
        let first_end = self.range.start + index;
        assert!(first_end <= self.range.end);
        let other = BorrowBufferMut {
            data: self.data,
            range: first_end..self.range.end,
            front: self.front,
            len: self.len,
        };
        self.range.end = first_end;
        other
    }
}

/// A fixed-sized circular buffer. The buffer can hold up to `RRB_WIDTH` items and supports fast
/// operations on either end, however operations in the middle of the buffer are typically O(N).
pub(crate) struct CircularBuffer<A: Debug> {
    front: usize,
    len: usize,
    data: [mem::MaybeUninit<A>; RRB_WIDTH],
}

impl<A: Debug> CircularBuffer<A> {
    /// Creates an empty `CircularBuffer`.
    pub fn new() -> Self {
        let data: [mem::MaybeUninit<A>; RRB_WIDTH] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };
        CircularBuffer {
            front: 0,
            len: 0,
            data,
        }
    }

    /// Creates a `CircularBuffer` with a single element.
    pub fn with_item(item: A) -> Self {
        let mut result = CircularBuffer::new();
        result.push_back(item);
        result
    }

    /// Returns the length of the buffer. This function differentiates between empty and full
    /// buffers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer is completely empty and `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the amount of free space left in the buffer.
    pub fn free_space(&self) -> usize {
        RRB_WIDTH - self.len()
    }

    /// Returns `true` if the buffer is completely full and `false` otherwise.
    pub fn is_full(&self) -> bool {
        self.len() == RRB_WIDTH
    }

    /// Determines the true position of the index in the buffer.
    fn index_for(&self, idx: usize) -> usize {
        (self.front + idx) % RRB_WIDTH
    }

    /// Attempts to add an element to the end of the buffer.
    ///
    /// Returns Err() in the event the buffer is full.
    pub fn try_push_back(&mut self, item: A) -> Result<()> {
        if self.is_full() {
            Err(CircularBufferError::FullBuffer)
        } else {
            let back = (self.front + self.len) % RRB_WIDTH;
            self.data[back] = mem::MaybeUninit::new(item);
            self.len += 1;
            Ok(())
        }
    }

    /// Attempts to add an element to the end of the buffer.
    ///
    /// Panics in the event the buffer is full.
    pub fn push_back(&mut self, item: A) {
        self.try_push_back(item).unwrap()
    }

    /// Attempts to remove an element from the end of the buffer.
    ///
    /// Returns Err(()) if the buffer is empty.
    pub fn try_pop_back(&mut self) -> Result<A> {
        if self.is_empty() {
            Err(CircularBufferError::EmptyBuffer)
        } else {
            self.len -= 1;
            let back = (self.front + self.len) % RRB_WIDTH;
            let result = unsafe {
                mem::replace(&mut self.data[back], mem::MaybeUninit::uninit()).assume_init()
            };
            Ok(result)
        }
    }

    /// Attempts to remove an element from the end of the buffer.
    ///
    /// Panics if the buffer is empty.
    pub fn pop_back(&mut self) -> A {
        self.try_pop_back().expect("Circular buffer is empty")
    }

    /// Attempts to add an element to the front of the buffer.
    ///
    /// Returns Err(()) in the event the buffer is already full.
    pub fn try_push_front(&mut self, item: A) -> Result<()> {
        if self.is_full() {
            Err(CircularBufferError::FullBuffer)
        } else {
            self.front = (self.front + RRB_WIDTH - 1) % RRB_WIDTH;
            self.len += 1;
            self.data[self.front] = mem::MaybeUninit::new(item);
            Ok(())
        }
    }

    /// Attempts to add an element to the front of the buffer.
    ///
    /// Panics in the event the buffer is full.
    pub fn push_front(&mut self, item: A) {
        self.try_push_front(item).unwrap()
    }

    /// Attempts to remove an element from the front of the buffer.
    ///
    /// Returns Err(()) if the buffer is empty.
    pub fn try_pop_front(&mut self) -> Result<A> {
        if self.is_empty() {
            Err(CircularBufferError::EmptyBuffer)
        } else {
            let result = unsafe {
                mem::replace(&mut self.data[self.front], mem::MaybeUninit::uninit()).assume_init()
            };
            self.len -= 1;
            self.front = (self.front + 1) % RRB_WIDTH;
            Ok(result)
        }
    }

    /// Attempts to remove an element from the front of the buffer.
    ///
    /// Panics if the buffer is empty.
    pub fn pop_front(&mut self) -> A {
        self.try_pop_front().expect("Circular buffer is empty")
    }

    /// Attempts to push an element to a side of the buffer.
    ///
    /// Panics if the buffer is full.
    pub fn push(&mut self, side: Side, item: A) {
        match side {
            Side::Back => self.push_back(item),
            Side::Front => self.push_front(item),
        }
    }

    /// Attempts to remove an element from a side of the buffer.
    ///
    /// Returns Err(()) if the buffer is empty.
    pub fn try_pop(&mut self, side: Side) -> Result<A> {
        match side {
            Side::Back => self.try_pop_back(),
            Side::Front => self.try_pop_front(),
        }
    }

    /// Attempts to remove an element from a side of the buffer.
    ///
    /// Panics if the buffer is empty.
    pub fn pop(&mut self, side: Side) -> A {
        match side {
            Side::Back => self.pop_back(),
            Side::Front => self.pop_front(),
        }
    }

    /// Gets a reference to an element in the buffer.
    ///
    /// Returns None if the index does not exist.
    pub fn get(&self, idx: usize) -> Option<&A> {
        if idx < self.len() {
            let index = self.index_for(idx);
            unsafe { Some(&*self.data[index].as_ptr()) }
        } else {
            None
        }
    }

    /// Gets a mutable reference to an element in the buffer.
    ///
    /// Returns None if the index does not exist.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut A> {
        if idx < self.len() {
            let index = self.index_for(idx);
            unsafe { Some(&mut *self.data[index].as_mut_ptr()) }
        } else {
            None
        }
    }

    /// Gets a reference to the front of the buffer.
    ///
    /// Returns None if the buffer is empty.
    pub fn front(&self) -> Option<&A> {
        if !self.is_empty() {
            self.get(0)
        } else {
            None
        }
    }

    /// Gets a reference to the end of the buffer.
    ///
    /// Returns None if the buffer is empty.
    pub fn back(&self) -> Option<&A> {
        if !self.is_empty() {
            self.get(self.len() - 1)
        } else {
            None
        }
    }

    /// Gets a mutable reference to the front of the buffer.
    ///
    /// Returns None if the buffer is empty.
    pub fn front_mut(&mut self) -> Option<&mut A> {
        if !self.is_empty() {
            self.get_mut(0)
        } else {
            None
        }
    }

    /// Gets a mutable reference to the end of the buffer.
    ///
    /// Returns None if the buffer is empty.
    pub fn back_mut(&mut self) -> Option<&mut A> {
        if !self.is_empty() {
            self.get_mut(self.len() - 1)
        } else {
            None
        }
    }

    /// Creates an iterator over the elements in the buffer.
    pub fn iter(&self) -> Iter<A> {
        Iter {
            consumed: 0,
            front: self.front,
            back: (self.front + self.len) % RRB_WIDTH,
            buffer: self,
        }
    }

    /// Creates a mutable iterator over the elements in the buffer.
    pub fn iter_mut(&mut self) -> IterMut<A> {
        IterMut {
            consumed: 0,
            front: self.front,
            back: (self.front + self.len) % RRB_WIDTH,
            buffer: self,
        }
    }

    /// Returns a pair of mutable references to the data at the given positions.
    ///
    /// # Panics
    ///
    /// This panics if both indices are equal.
    /// This panics if either indices are out of bounds.
    pub fn pair_mut(&mut self, first: usize, second: usize) -> (&mut A, &mut A) {
        assert_ne!(first, second);
        assert!(first < self.len());
        assert!(second < self.len());

        let first_idx = self.index_for(first);
        let second_idx = self.index_for(second);
        if first_idx > second_idx {
            let result = self.pair_mut(second, first);
            return (result.1, result.0);
        }

        let (first_slice, second_slice) = self.data.split_at_mut(second_idx);
        let first_ptr = first_slice[first_idx].as_mut_ptr();
        let second_ptr = second_slice[0].as_mut_ptr();

        unsafe { (&mut *first_ptr, &mut *second_ptr) }
    }

    /// Removes elements from `self` and inserts these elements into `destination`. At most `len`
    /// elements will be removed. The actual number of elements decanted is returned. Elements are
    /// popped from `share_side` but pushed into the destination via `share_side.negate()`.
    pub fn decant_into(&mut self, destination: &mut Self, share_side: Side, len: usize) -> usize {
        if destination.is_empty() && len >= RRB_WIDTH {
            mem::swap(self, destination);
            destination.len()
        } else {
            let result = len.min(self.len()).min(destination.free_space());
            for _ in 0..result {
                let item = self.pop(share_side);
                destination.push(share_side.negate(), item);
            }
            result
        }
    }

    pub(crate) fn mutable_view(&mut self) -> BorrowBufferMut<A> {
        assert_ne!(self.len(), 0);
        BorrowBufferMut {
            range: 0..self.len(),
            data: self.data.as_mut_ptr(),
            front: self.front,
            len: self.len,
        }
    }
}

impl<A: Debug> Default for CircularBuffer<A> {
    fn default() -> Self {
        CircularBuffer::new()
    }
}

impl<A: Debug> Debug for CircularBuffer<A> {
    fn fmt(&self, fmt: &mut Formatter) -> std::result::Result<(), fmt::Error> {
        fmt.debug_list().entries(self).finish()
    }
}

impl<A: Clone + Debug> Clone for CircularBuffer<A> {
    fn clone(&self) -> Self {
        let mut data: [mem::MaybeUninit<A>; RRB_WIDTH] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };
        let cloner = |x: &mem::MaybeUninit<A>| unsafe {
            let data_ptr = x.as_ptr();
            let new_data = (&*data_ptr).clone();
            mem::MaybeUninit::new(new_data)
        };
        let back = (self.front + self.len) % RRB_WIDTH;

        if self.is_full() {
            for (i, x) in self.data.iter().enumerate() {
                data[i] = cloner(x);
            }
        } else if back >= self.front {
            let len = self.len();
            for (i, x) in self.data.iter().enumerate().skip(self.front).take(len) {
                data[i] = cloner(x);
            }
        } else {
            for (i, x) in self.data.iter().enumerate().take(back) {
                data[i] = cloner(x);
            }
            for (i, x) in self.data.iter().enumerate().skip(self.front) {
                data[i] = cloner(x);
            }
        }

        CircularBuffer {
            len: self.len,
            front: self.front,
            data,
        }
    }
}

#[cfg(not(feature = "may_dangle"))]
impl<A: Debug> Drop for CircularBuffer<A> {
    fn drop(&mut self) {
        let back = (self.front + self.len) % RRB_WIDTH;
        if self.is_full() {
            for x in self.data.iter_mut() {
                unsafe {
                    mem::replace(x, mem::MaybeUninit::uninit()).assume_init();
                }
            }
        } else if back >= self.front {
            for x in &mut self.data[self.front..back] {
                unsafe {
                    mem::replace(x, mem::MaybeUninit::uninit()).assume_init();
                }
            }
        } else {
            for x in &mut self.data[..back] {
                unsafe {
                    mem::replace(x, mem::MaybeUninit::uninit()).assume_init();
                }
            }
            for x in &mut self.data[self.front..] {
                unsafe {
                    mem::replace(x, mem::MaybeUninit::uninit()).assume_init();
                }
            }
        }
    }
}

#[cfg(feature = "may_dangle")]
unsafe impl<#[may_dangle] A: Debug> Drop for CircularBuffer<A> {
    fn drop(&mut self) {
        let back = (self.front + self.len) % RRB_WIDTH;
        if self.is_full() {
            for x in self.data.iter_mut() {
                unsafe {
                    mem::replace(x, mem::MaybeUninit::uninit()).assume_init();
                }
            }
        } else if back >= self.front {
            for x in &mut self.data[self.front..back] {
                unsafe {
                    mem::replace(x, mem::MaybeUninit::uninit()).assume_init();
                }
            }
        } else {
            for x in &mut self.data[..back] {
                unsafe {
                    mem::replace(x, mem::MaybeUninit::uninit()).assume_init();
                }
            }
            for x in &mut self.data[self.front..] {
                unsafe {
                    mem::replace(x, mem::MaybeUninit::uninit()).assume_init();
                }
            }
        }
    }
}

unsafe impl<A: Clone + Debug + Send> Send for CircularBuffer<A> {}
unsafe impl<A: Clone + Debug + Sync> Sync for CircularBuffer<A> {}

/// An iterator over a buffer that is obtained by the `CircularBuffer::iter()` method.
pub struct Iter<'a, A: 'a + Debug> {
    consumed: usize,
    front: usize,
    back: usize,
    buffer: &'a CircularBuffer<A>,
}

impl<'a, A: 'a + Debug> Iterator for Iter<'a, A> {
    type Item = &'a A;

    fn next(&mut self) -> Option<&'a A> {
        if self.consumed == self.buffer.len() {
            None
        } else {
            let result = unsafe { &*self.buffer.data[self.front].as_ptr() };
            self.front = (self.front + 1) % RRB_WIDTH;
            self.consumed += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.buffer.len() - self.consumed;
        (len, Some(len))
    }
}

impl<'a, A: 'a + Debug> DoubleEndedIterator for Iter<'a, A> {
    fn next_back(&mut self) -> Option<&'a A> {
        if self.consumed == self.buffer.len() {
            None
        } else {
            self.consumed += 1;
            self.back = (self.back + RRB_WIDTH - 1) % RRB_WIDTH;
            let result = unsafe { &*self.buffer.data[self.back].as_ptr() };
            Some(result)
        }
    }
}

impl<'a, A: 'a + Debug> ExactSizeIterator for Iter<'a, A> {}

impl<'a, A: 'a + Debug> FusedIterator for Iter<'a, A> {}

/// A mutable iterator over a buffer that is obtained by the `CircularBuffer::iter_mut()` method.
pub struct IterMut<'a, A: 'a + Debug> {
    consumed: usize,
    front: usize,
    back: usize,
    buffer: &'a mut CircularBuffer<A>,
}

impl<'a, A: 'a + Debug> Iterator for IterMut<'a, A> {
    type Item = &'a mut A;

    fn next(&mut self) -> Option<&'a mut A> {
        if self.consumed == self.buffer.len() {
            None
        } else {
            self.consumed += 1;
            let result = unsafe { &mut *self.buffer.data[self.front].as_mut_ptr() };
            self.front = (self.front + 1) % RRB_WIDTH;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.buffer.len() - self.consumed;
        (len, Some(len))
    }
}

impl<'a, A: 'a + Debug> DoubleEndedIterator for IterMut<'a, A> {
    fn next_back(&mut self) -> Option<&'a mut A> {
        if self.consumed == self.buffer.len() {
            None
        } else {
            self.consumed += 1;
            self.back = (self.back + RRB_WIDTH - 1) % RRB_WIDTH;
            let result = unsafe { &mut *self.buffer.data[self.back].as_mut_ptr() };
            Some(result)
        }
    }
}

impl<'a, A: 'a + Debug> ExactSizeIterator for IterMut<'a, A> {}

impl<'a, A: 'a + Debug> FusedIterator for IterMut<'a, A> {}

/// A consuming iterator over a vector that is obtained by the `Vector::into_iter()` method.
pub struct IntoIter<A: Debug> {
    consumed: usize,
    front: usize,
    back: usize,
    buffer: CircularBuffer<A>,
}

impl<A: Debug> Iterator for IntoIter<A> {
    type Item = A;

    fn next(&mut self) -> Option<A> {
        if self.consumed == self.buffer.len() {
            None
        } else {
            let result = unsafe {
                mem::replace(
                    &mut self.buffer.data[self.front],
                    mem::MaybeUninit::uninit(),
                )
                .assume_init()
            };
            self.front = (self.front + 1) % RRB_WIDTH;
            self.consumed += 1;
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.buffer.len() - self.consumed;
        (len, Some(len))
    }
}

impl<A: Debug> DoubleEndedIterator for IntoIter<A> {
    fn next_back(&mut self) -> Option<A> {
        if self.consumed == self.buffer.len() {
            None
        } else {
            self.consumed += 1;
            self.back = (self.back + RRB_WIDTH - 1) % RRB_WIDTH;
            let result = unsafe {
                mem::replace(
                    &mut self.buffer.data[self.front],
                    mem::MaybeUninit::uninit(),
                )
                .assume_init()
            };
            Some(result)
        }
    }
}

impl<A: Debug> ExactSizeIterator for IntoIter<A> {}

impl<A: Debug> FusedIterator for IntoIter<A> {}

impl<'a, A: 'a + Debug> IntoIterator for &'a CircularBuffer<A> {
    type IntoIter = Iter<'a, A>;
    type Item = &'a A;

    fn into_iter(self) -> Iter<'a, A> {
        self.iter()
    }
}

impl<'a, A: 'a + Debug> IntoIterator for &'a mut CircularBuffer<A> {
    type IntoIter = IterMut<'a, A>;
    type Item = &'a mut A;

    fn into_iter(self) -> IterMut<'a, A> {
        self.iter_mut()
    }
}

impl<A: Debug> IntoIterator for CircularBuffer<A> {
    type IntoIter = IntoIter<A>;
    type Item = A;

    fn into_iter(self) -> IntoIter<A> {
        let back = (self.front + self.len) % RRB_WIDTH;
        IntoIter {
            back,
            front: self.front,
            consumed: 0,
            buffer: self,
        }
    }
}

#[allow(clippy::cognitive_complexity)]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn empty() {
        let mut empty: CircularBuffer<usize> = CircularBuffer::new();
        let empty_vector: Vec<usize> = Vec::new();
        let empty_ref_vector: Vec<&usize> = Vec::new();
        let empty_ref_mut_vector: Vec<&mut usize> = Vec::new();

        // Len
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert!(!empty.is_full() || RRB_WIDTH == 0);

        // Back
        assert_eq!(empty.back(), None);
        assert_eq!(empty.try_pop_back(), Err(CircularBufferError::EmptyBuffer));
        assert_eq!(empty.back_mut(), None);

        // Front
        assert_eq!(empty.front(), None);
        assert_eq!(empty.front_mut(), None);
        assert_eq!(empty.try_pop_front(), Err(CircularBufferError::EmptyBuffer));

        // Iter
        assert_eq!(empty.iter().collect::<Vec<_>>(), empty_ref_vector);
        assert_eq!(empty.iter_mut().collect::<Vec<_>>(), empty_ref_mut_vector);
        assert_eq!(empty.into_iter().collect::<Vec<_>>(), empty_vector);
    }

    #[test]
    pub fn single() {
        let mut item = 9;
        let mut single = CircularBuffer::new();
        single.push_back(item);
        let mut vector: Vec<usize> = vec![item];

        // Len
        assert!(!single.is_empty());
        assert_eq!(single.len(), 1);
        assert!(!single.is_full() || RRB_WIDTH == 1);

        // Back
        assert_eq!(single.back(), Some(&item));
        assert_eq!(single.back_mut(), Some(&mut item));
        let mut back = single.clone();
        assert_eq!(back.try_pop_back(), Ok(item));
        assert_eq!(back.try_pop_back(), Err(CircularBufferError::EmptyBuffer));
        assert_eq!(back.back(), None);
        assert_eq!(back.back_mut(), None);

        // Front
        assert_eq!(single.front(), Some(&item));
        assert_eq!(single.front_mut(), Some(&mut item));
        let mut front = single.clone();
        assert_eq!(front.try_pop_front(), Ok(item));
        assert_eq!(front.try_pop_front(), Err(CircularBufferError::EmptyBuffer));

        // Iter
        assert_eq!(
            single.iter().collect::<Vec<_>>(),
            vector.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            single.iter_mut().collect::<Vec<_>>(),
            vector.iter_mut().collect::<Vec<_>>()
        );
        assert_eq!(single.into_iter().collect::<Vec<_>>(), vector);
    }

    #[test]
    pub fn sort() {
        let items = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        let mut buffer = CircularBuffer::new();
        for item in &items {
            buffer.push_back(*item);
        }

        // Len
        assert!(!buffer.is_empty());
        assert_eq!(buffer.len(), 10);
        assert!(!buffer.is_full() || RRB_WIDTH == 10);

        // Back
        assert_eq!(buffer.back(), Some(&0));
        assert_eq!(buffer.back_mut(), Some(&mut 0));
        let mut back = buffer.clone();
        assert_eq!(back.try_pop_back(), Ok(0));
        assert_eq!(back.try_pop_back(), Ok(1));
        assert_eq!(back.try_pop_back(), Ok(2));
        assert_eq!(back.try_pop_back(), Ok(3));
        assert_eq!(back.try_pop_back(), Ok(4));
        assert_eq!(back.try_pop_back(), Ok(5));
        assert_eq!(back.try_pop_back(), Ok(6));
        assert_eq!(back.try_pop_back(), Ok(7));
        assert_eq!(back.try_pop_back(), Ok(8));
        assert_eq!(back.try_pop_back(), Ok(9));
        assert_eq!(back.try_pop_back(), Err(CircularBufferError::EmptyBuffer));
        assert_eq!(back.back(), None);
        assert_eq!(back.back_mut(), None);

        // Front
        assert_eq!(buffer.front(), Some(&9));
        assert_eq!(buffer.front_mut(), Some(&mut 9));
        let mut front = buffer.clone();
        assert_eq!(front.try_pop_front(), Ok(9));
        assert_eq!(front.try_pop_front(), Ok(8));
        assert_eq!(front.try_pop_front(), Ok(7));
        assert_eq!(front.try_pop_front(), Ok(6));
        assert_eq!(front.try_pop_front(), Ok(5));
        assert_eq!(front.try_pop_front(), Ok(4));
        assert_eq!(front.try_pop_front(), Ok(3));
        assert_eq!(front.try_pop_front(), Ok(2));
        assert_eq!(front.try_pop_front(), Ok(1));
        assert_eq!(front.try_pop_front(), Ok(0));
        assert_eq!(front.try_pop_front(), Err(CircularBufferError::EmptyBuffer));
    }

    #[test]
    fn drop() {
        let mut s = "derp".to_owned();
        let mut buffer = CircularBuffer::with_item(&mut s);
        buffer.front_mut().unwrap().push('l');
        assert_eq!(buffer.front().unwrap(), &"derpl");
    }

    #[test]
    fn drop_generics() {
        let mut s = "derp".to_owned();
        {
            let mut buffer = CircularBuffer::with_item(&mut s);
            buffer.front_mut().unwrap().push('l');
        }
        assert_eq!(s, "derpl");
    }

    #[cfg(feature = "may_dangle")]
    #[test]
    fn drop_generics_dangle() {
        let mut s = "derp".to_owned();
        let mut buffer = CircularBuffer::with_item(&mut s);
        buffer.front_mut().unwrap().push('l');
        assert_eq!(s, "derpl");
    }

    #[test]
    fn zst_buffer() {
        let mut buffer: CircularBuffer<()> = CircularBuffer::new();
        assert_eq!(mem::size_of_val(&buffer), 16); // We need to allocate 2 usizes for front and len
        buffer.push_back(());
        assert_eq!(buffer.try_pop_back(), Ok(()));
        assert_eq!(buffer.try_pop_back(), Err(CircularBufferError::EmptyBuffer));
    }
}
