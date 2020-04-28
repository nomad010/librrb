//! A circular buffer.
//!
//! A fixed-size container of elements that supports fast operations on either side of the buffer.

use crate::{Side, RRB_WIDTH};
use std::cmp;
use std::fmt::{self, Debug, Formatter};
use std::iter::FusedIterator;
use std::mem;
use std::ops::Range;

// #[derive(Debug)]
pub(crate) struct BorrowBufferMut<A: Debug> {
    pub(crate) range: Range<usize>,
    data: *mut mem::MaybeUninit<A>,
    front: usize,
    // len: usize,
}

impl<A: Clone + Debug> Debug for BorrowBufferMut<A> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
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
    pub fn empty(&mut self) -> Self {
        BorrowBufferMut {
            range: self.range.start..self.range.start,
            data: self.data,
            front: 0,
            // len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn free_space(&self) -> usize {
        RRB_WIDTH - self.len()
    }

    pub fn is_full(&self) -> bool {
        self.len() == RRB_WIDTH
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

    pub fn front(&self) -> Option<&A> {
        self.get(0)
    }

    pub fn back(&self) -> Option<&A> {
        self.get(self.len() - 1)
    }

    pub fn front_mut(&mut self) -> Option<&mut A> {
        self.get_mut(0)
    }

    pub fn back_mut(&mut self) -> Option<&mut A> {
        self.get_mut(self.len() - 1)
    }

    pub fn end(&self, side: Side) -> Option<&A> {
        match side {
            Side::Back => self.back(),
            Side::Front => self.front(),
        }
    }

    pub fn end_mut(&mut self, side: Side) -> Option<&mut A> {
        match side {
            Side::Back => self.back_mut(),
            Side::Front => self.front_mut(),
        }
    }

    pub unsafe fn take_item(&mut self, idx: usize) -> A {
        let idx = idx + self.range.start;
        assert!(self.range.contains(&idx));
        let index = self.index_for(idx);
        let data = std::slice::from_raw_parts_mut(self.data, RRB_WIDTH);
        mem::replace(&mut data[index], mem::MaybeUninit::uninit()).assume_init()
    }

    pub unsafe fn return_item(&mut self, idx: usize, item: A) {
        let idx = idx + self.range.start;
        assert!(self.range.contains(&idx));
        let index = self.index_for(idx);
        let data = std::slice::from_raw_parts_mut(self.data, RRB_WIDTH);
        data[index] = mem::MaybeUninit::new(item);
    }

    // pub fn pair_mut(&mut self, first: usize, second: usize) -> (&mut A, &mut A) {
    // assert_ne!(first, second);
    // assert!(first < self.len());
    // assert!(second < self.len());

    // let first_idx = self.index_for(first);
    // let second_idx = self.index_for(second);
    // if first_idx > second_idx {
    //     let result = self.pair_mut(second, first);
    //     return (result.1, result.0);
    // }

    // let (first_slice, second_slice) = self.data.split_at_mut(second_idx);
    // let first_ptr = first_slice[first_idx].as_mut_ptr();
    // let second_ptr = second_slice[0].as_mut_ptr();

    // unsafe { (&mut *first_ptr, &mut *second_ptr) }
    // }

    pub fn split_at(&mut self, index: usize) -> Self {
        let first_end = self.range.start + index;
        assert!(first_end <= self.range.end);
        let other = BorrowBufferMut {
            data: self.data,
            range: first_end..self.range.end,
            front: self.front,
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

    /// Creates a new `CircularBuffer` with the elements from the given `Vec`.
    ///
    /// Panics if the collection contains more than `RRB_WIDTH` items.
    pub fn with_items(items: Vec<A>) -> Self {
        let mut result = CircularBuffer::new();
        assert!(items.len() <= RRB_WIDTH);
        for item in items.into_iter() {
            result.push_back(item);
        }
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
    /// Returns Err(()) in the event the buffer is full.
    pub fn try_push_back(&mut self, item: A) -> Result<(), ()> {
        if self.is_full() {
            Err(())
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
        self.try_push_back(item).expect("Circular buffer is full")
    }

    /// Attempts to remove an element from the end of the buffer.
    ///
    /// Returns Err(()) if the buffer is empty.
    pub fn try_pop_back(&mut self) -> Result<A, ()> {
        if self.is_empty() {
            Err(())
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
    pub fn try_push_front(&mut self, item: A) -> Result<(), ()> {
        if self.is_full() {
            Err(())
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
        self.try_push_front(item).expect("Circular buffer is full")
    }

    /// Attempts to remove an element from the front of the buffer.
    ///
    /// Returns Err(()) if the buffer is empty.
    pub fn try_pop_front(&mut self) -> Result<A, ()> {
        if self.is_empty() {
            Err(())
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

    /// Attempts to remove an element from the buffer.
    ///
    /// Returns Err(()) if the requested element does not exist.
    pub fn try_remove(&mut self, idx: usize) -> Result<A, ()> {
        if self.len() > idx {
            let position = (self.front + idx) % RRB_WIDTH;
            let back = (self.front + self.len) % RRB_WIDTH;

            if back > self.front {
                // We rotate the subarray from idx to self.back left 1
                let slice = &mut self.data[position..back];
                slice.rotate_left(1);
                self.try_pop_back()
            } else {
                // The buffer is split up into self.front..SIZE and 0..self.back
                // If the idx lies in the second part we can follow the same proccess as above
                // If idx lies in the first part we need to shift both parts around
                if position > self.front {
                    // Part 1
                    // Slide the first element in the second part to the last position and pop it
                    let second_slice = &mut self.data[0..back];
                    second_slice.rotate_right(1);
                    let moved_element = mem::MaybeUninit::new(self.pop_back());

                    // We rotate the item we'd like to remove the very end and replace it with the
                    // above
                    let first_slice = &mut self.data[position..RRB_WIDTH];
                    first_slice.rotate_left(1);
                    unsafe {
                        Ok(
                            mem::replace(&mut self.data[RRB_WIDTH - 1], moved_element)
                                .assume_init(),
                        )
                    }
                } else {
                    // Part 2
                    let slice = &mut self.data[position..back];
                    slice.rotate_left(1);
                    self.try_pop_back()
                }
            }
        } else {
            Err(())
        }
    }

    /// Attempts to remove an element from the buffer.
    ///
    /// Panics if the requested element does not exist.
    pub fn remove(&mut self, idx: usize) -> A {
        self.try_remove(idx)
            .expect("Tried to remove element that does not exist")
    }

    /// Attempts to push an element to a side of the buffer.
    ///
    /// Returns Err(()) if the buffer is full.
    pub fn try_push(&mut self, side: Side, item: A) -> Result<(), ()> {
        match side {
            Side::Back => self.try_push_back(item),
            Side::Front => self.try_push_front(item),
        }
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
    pub fn try_pop(&mut self, side: Side) -> Result<A, ()> {
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

    /// Gets a reference the element on a given side of the buffer.
    ///
    /// Returns None if the buffer is empty.
    pub fn end(&self, side: Side) -> Option<&A> {
        match side {
            Side::Back => self.back(),
            Side::Front => self.front(),
        }
    }

    /// Gets a mutable  reference the element on a given side of the buffer.
    ///
    /// Returns None if the buffer is empty.
    pub fn end_mut(&mut self, side: Side) -> Option<&mut A> {
        match side {
            Side::Back => self.back_mut(),
            Side::Front => self.front_mut(),
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

    /// Shuffles the circular buffer in memory so that the buffer is contiguous instead of
    /// potentially being split up into two areas. This allows the buffer to be returned as a slice.
    pub fn align_contents(&mut self) {
        self.data.rotate_left(self.front);
        self.front = 0;
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

    /// Returns the circular buffer as a single slice if possible.
    pub fn slice_get_ref(&self) -> Option<&[A]> {
        let back = (self.front + self.len) % RRB_WIDTH;
        if back >= self.front {
            let slice = &self.data[self.front..back];
            unsafe { Some(&*(slice as *const [mem::MaybeUninit<A>] as *const [A])) }
        } else {
            None
        }
    }

    /// Returns the circular buffer as a single mutable slice if possible.
    pub fn slice_get_mut(&mut self) -> Option<&mut [A]> {
        let back = (self.front + self.len) % RRB_WIDTH;
        if back >= self.front {
            let slice = &mut self.data[self.front..back];
            unsafe { Some(&mut *(slice as *mut [mem::MaybeUninit<A>] as *mut [A])) }
        } else {
            None
        }
    }

    /// Temporarily removes an item at the requested index if it exists. The item must be put back
    /// into the buffer before ANY other operations on the cirular buffer are performed. The item
    /// can be returned to the buffer with the `return_item` function. It is not safe to clone, drop
    /// or do anything except return the item to the buffer while the item is taken.
    ///
    /// # Panics
    ///
    /// This panics if the index points to an out of bound item.
    ///
    /// # Safety
    ///
    /// This is a very unsafe function. If the buffer is dropped while an item is taken out,
    /// undefined behaviour will occur. If the same item is taken multiple items, instant undefined
    /// behaviour will occur. Reading taken items via calls such as `get`, `get_mut` or `pop` will
    /// invoke undefined behaviour. Putting an item back in the wrong place, will invoke undefined
    /// behaviour when the buffer is dropped.
    pub unsafe fn take_item(&mut self, idx: usize) -> A {
        assert!(idx < self.len());
        let position = (self.front + idx) % RRB_WIDTH;
        mem::replace(&mut self.data[position], mem::MaybeUninit::uninit()).assume_init()
    }

    /// Returns an item to the given index. Calling this on an existing item that was not taken out,
    /// will leak the memory of the item that is replaced.
    ///
    /// # Panics
    ///
    /// This panics if the index points to an out of bound item.
    ///
    /// # Safety
    ///
    /// Putting an item back in the wrong place will invoke undefined behaviour when the buffer is
    /// dropped.
    pub unsafe fn return_item(&mut self, idx: usize, item: A) {
        assert!(idx < self.len());
        let position = (self.front + idx) % RRB_WIDTH;
        self.data[position] = mem::MaybeUninit::new(item);
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

    /// Sorts the buffer by the order induced by the given comparator.
    pub fn sort_by<F: FnMut(&A, &A) -> cmp::Ordering>(&mut self, range: Range<usize>, f: &mut F) {
        self.align_contents();
        let data = self.slice_get_mut().unwrap();
        data[range].sort_by(f);
    }

    /// Tests whether the buffer is sorted by the order induced by the given comparator.
    pub fn is_sorted_by<F: FnMut(&A, &A) -> Option<cmp::Ordering>>(
        &self,
        range: Range<usize>,
        f: &mut F,
    ) -> bool {
        let mut iter = self.iter().skip(range.start).take(range.end - range.start);
        if let Some(mut item) = iter.next() {
            for next in iter {
                if let Some(cmp) = f(item, next) {
                    if cmp != cmp::Ordering::Greater {
                        item = next;
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            true
        } else {
            true
        }
    }

    /// Performs a binary search through the buffer that has already been sorted by the order
    /// induced by the given comparator. If the function returns `Ok(position)` then it is
    /// guaranteed that the `f(buffer[position]) == Ordering::Equal`. If the function returns
    /// `Err(position)` then `f(buffer[position]) == Ordering::Greater`. This would indicate the
    /// position where searched for element should be inserted in order to maintain sorted order.
    /// Tgis is hoisted directly from the algorithm used in the rust std library.
    pub fn binary_search_by<F: FnMut(&A) -> cmp::Ordering>(
        &self,
        range: Range<usize>,
        f: &mut F,
    ) -> Result<usize, usize> {
        let range = range.start.min(self.len())..range.end.min(self.len());
        if range.start == range.end {
            panic!()
        }
        let mut size = range.end - range.start;
        if size == 0 {
            return Err(0);
        }
        let mut base = range.start;
        while size > 1 {
            let half = size / 2;
            let mid = base + half;
            let cmp = f(self.get(mid).unwrap());
            base = if cmp == cmp::Ordering::Greater {
                base
            } else {
                mid
            };
            size -= half;
        }
        let cmp = f(self.get(base).unwrap());
        if cmp == cmp::Ordering::Equal {
            Ok(base)
        } else {
            Err(base + (cmp == cmp::Ordering::Less) as usize)
        }
    }

    pub(crate) fn mutable_view(&mut self) -> BorrowBufferMut<A> {
        BorrowBufferMut {
            range: 0..self.len(),
            data: self.data.as_mut_ptr(),
            front: self.front,
        }
    }
}

impl<A: PartialOrd + Debug> CircularBuffer<A> {
    /// Tests whether the buffer is sorted by the order induced by the natural `PartialOrd`
    /// comparator.
    pub fn is_sorted(&self, range: Range<usize>) -> bool {
        let mut f = |a: &A, b: &A| a.partial_cmp(b);
        self.is_sorted_by(range, &mut f)
    }
}

impl<A: Ord + Debug> CircularBuffer<A> {
    /// Sorts the buffer by the order induced by the natural `Ord` comparator.
    pub fn sort(&mut self, range: Range<usize>) {
        let mut f = |a: &A, b: &A| a.cmp(b);
        self.sort_by(range, &mut f);
    }

    /// Performs a binary search through the buffer that has already been sorted by the order
    /// induced by the natural `Ord` comparator. For more information see the `binary_search_by`
    /// function.
    pub fn binary_search(&self, range: Range<usize>, item: &A) -> Result<usize, usize> {
        let mut f = |p: &A| p.cmp(item);
        self.binary_search_by(range, &mut f)
    }
}

impl<A: Debug> Default for CircularBuffer<A> {
    fn default() -> Self {
        CircularBuffer::new()
    }
}

impl<A: Debug> Debug for CircularBuffer<A> {
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), fmt::Error> {
        fmt.write_str("[")?;
        let mut first = true;
        for item in self.iter() {
            if first {
                first = false;
            } else {
                fmt.write_str(", ")?;
            }
            item.fmt(fmt)?;
        }
        fmt.write_str("]")?;
        Ok(())
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
        assert_eq!(empty.try_pop_back(), Err(()));
        assert_eq!(empty.back_mut(), None);
        assert_eq!(empty.end(Side::Back), None);
        assert_eq!(empty.end_mut(Side::Back), None);

        // Front
        assert_eq!(empty.front(), None);
        assert_eq!(empty.front_mut(), None);
        assert_eq!(empty.try_pop_front(), Err(()));
        assert_eq!(empty.end(Side::Front), None);
        assert_eq!(empty.end_mut(Side::Front), None);

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
        assert_eq!(single.end(Side::Back), Some(&item));
        assert_eq!(single.end_mut(Side::Back), Some(&mut item));
        let mut back = single.clone();
        assert_eq!(back.try_pop_back(), Ok(item));
        assert_eq!(back.try_pop_back(), Err(()));
        assert_eq!(back.back(), None);
        assert_eq!(back.back_mut(), None);
        assert_eq!(back.end(Side::Back), None);
        assert_eq!(back.end_mut(Side::Back), None);

        // Front
        assert_eq!(single.front(), Some(&item));
        assert_eq!(single.front_mut(), Some(&mut item));
        assert_eq!(single.end(Side::Front), Some(&item));
        assert_eq!(single.end_mut(Side::Front), Some(&mut item));
        let mut front = single.clone();
        assert_eq!(front.try_pop_front(), Ok(item));
        assert_eq!(front.try_pop_front(), Err(()));
        assert_eq!(front.end(Side::Front), None);
        assert_eq!(front.end_mut(Side::Front), None);

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
        assert_eq!(buffer.end(Side::Back), Some(&0));
        assert_eq!(buffer.end_mut(Side::Back), Some(&mut 0));
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
        assert_eq!(back.try_pop_back(), Err(()));
        assert_eq!(back.back(), None);
        assert_eq!(back.back_mut(), None);
        assert_eq!(back.end(Side::Back), None);
        assert_eq!(back.end_mut(Side::Back), None);

        // Front
        assert_eq!(buffer.front(), Some(&9));
        assert_eq!(buffer.front_mut(), Some(&mut 9));
        assert_eq!(buffer.end(Side::Front), Some(&9));
        assert_eq!(buffer.end_mut(Side::Front), Some(&mut 9));
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
        assert_eq!(front.try_pop_front(), Err(()));
        assert_eq!(front.end(Side::Front), None);
        assert_eq!(front.end_mut(Side::Front), None);

        // Sort
        buffer.sort(0..buffer.len());
        assert_eq!(buffer.try_pop_front(), Ok(0));
        assert_eq!(buffer.try_pop_front(), Ok(1));
        assert_eq!(buffer.try_pop_front(), Ok(2));
        assert_eq!(buffer.try_pop_front(), Ok(3));
        assert_eq!(buffer.try_pop_front(), Ok(4));
        assert_eq!(buffer.try_pop_front(), Ok(5));
        assert_eq!(buffer.try_pop_front(), Ok(6));
        assert_eq!(buffer.try_pop_front(), Ok(7));
        assert_eq!(buffer.try_pop_front(), Ok(8));
        assert_eq!(buffer.try_pop_front(), Ok(9));
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
}
