//! An annotation table
//!
//! The size table is a structure for mapping from a position to a child index in an RRB tree.
use crate::circular::CircularBuffer;
use crate::{Side, RRB_WIDTH};
use std::ops::{Add, Sub};

/// A naive implementation of an size table for nodes in the RRB tree.
#[derive(Clone, Debug)]
pub(crate) struct AnnotationTable<T>
where
    T: Clone + std::fmt::Debug + Ord + Add<Output = T> + Sub<Output = T>,
{
    pub(crate) buffer: CircularBuffer<T>,
}

impl<T> AnnotationTable<T>
where
    T: Clone + std::fmt::Debug + Ord + Add<Output = T> + Sub<Output = T>,
{
    /// Construct a new SizeTable for a node at the given level. The SizeTable is initially empty.
    pub fn new() -> Self {
        AnnotationTable {
            buffer: CircularBuffer::new(),
        }
    }

    /// Returns the total size of this node.
    pub fn cumulative_size(&self) -> T {
        self.get_cumulative_child_size(self.len() - 1)
    }

    /// Returns the sum of the total sizes of all children up to the given index.
    pub fn get_cumulative_child_size(&self, idx: usize) -> T {
        let mut iter = self.buffer.iter().take(idx + 1).cloned();
        let mut sum = iter.next().unwrap().clone();
        for item in iter {
            sum = sum + item;
        }
        sum
    }

    /// Returns the size of the child at the given index.
    pub fn get_child_size(&self, idx: usize) -> Option<&T> {
        self.buffer.get(idx)
    }

    /// Returns the position of the child that corresponds to the given index along with a new
    /// index to query in that child.
    pub fn position_info_for(&self, value: &T) -> Option<(usize, T)> {
        let mut iter = self.buffer.iter().cloned();
        let mut sum = iter.next().unwrap();
        for (i, sz) in iter.enumerate() {
            if value < &sum {
                return Some((i, sum - value.clone()));
            }
            sum = sum + sz;
        }
        if value < &sum {
            Some((self.len() - 1, sum - value.clone()))
        } else {
            None
        }
    }

    /// Adds a number of elements to the child at the given index.
    pub fn increment_child_size(&mut self, idx: usize, increment: T) {
        unsafe {
            let value = self.buffer.take_item(idx);
            self.buffer.return_item(idx, value + increment)
        }
    }

    /// Removes a number of elements to the child at the given index.
    pub fn decrement_child_size(&mut self, idx: usize, decrement: T) {
        unsafe {
            let value = self.buffer.take_item(idx);
            self.buffer.return_item(idx, value - decrement)
        }
    }

    /// Adds a number of elements to the child at a given side of the node.
    pub fn increment_side_size(&mut self, side: Side, increment: T) {
        if self.is_empty() {
            self.push_child(side, increment);
        } else {
            let idx = match side {
                Side::Back => self.len() - 1,
                Side::Front => 0,
            };
            self.increment_child_size(idx, increment)
        }
    }

    /// Removes a number of elements to the child at a given side of the node.
    pub fn decrement_side_size(&mut self, side: Side, decrement: T) {
        let idx = match side {
            Side::Back => self.len() - 1,
            Side::Front => 0,
        };
        self.decrement_child_size(idx, decrement)
    }

    /// Removes the child at the given index from the node and returns its size.
    pub fn remove_child(&mut self, idx: usize) -> T {
        self.buffer.remove(idx)
    }

    /// Adds a new child to a side of the node of a given size.
    pub fn push_child(&mut self, side: Side, size: T) {
        self.buffer.push(side, size);
    }

    /// Removes a child from a side of the node.
    pub fn pop_child(&mut self, side: Side) -> Option<T> {
        self.buffer.try_pop(side).ok()
    }

    /// Returns the number of children of the node, not the size of the node!
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if this node is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[allow(clippy::cognitive_complexity)]
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn empty() {
        let empty: AnnotationTable<usize> = AnnotationTable::new();

        // Len
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }
    #[allow(clippy::cognitive_complexity)]
    #[test]
    pub fn linear() {
        let length = RRB_WIDTH;
        assert!(length <= RRB_WIDTH);
        let mut linear = AnnotationTable::new();
        for i in 0..length {
            linear.push_child(Side::Back, i);
        }

        // Len
        assert!(!linear.is_empty());
        assert_eq!(linear.len(), length);

        // Iterate over the table from ensuring that the cumulative values and individual values are correct
        for i in 0..length {
            assert_eq!(linear.get_cumulative_child_size(i), i * (i + 1) / 2);
            assert_eq!(linear.get_child_size(i), Some(&i));
        }

        // Test if the cumulative sum of the every child is correct
        assert_eq!(linear.cumulative_size(), length * (length - 1) / 2);

        // Do some simple mutation to see whether these correctly modify the data structure
        let modification = 10;
        linear.decrement_side_size(Side::Back, modification);

        assert_eq!(
            linear.cumulative_size(),
            length * (length - 1) / 2 - modification
        );
        assert_eq!(
            linear.get_cumulative_child_size(length - 1),
            length * (length - 1) / 2 - modification
        );
        assert_eq!(
            linear.get_child_size(length - 1),
            Some(&(length - 1 - modification))
        );

        // Re-increment the value
        linear.increment_side_size(Side::Back, modification);

        assert_eq!(linear.cumulative_size(), length * (length - 1) / 2);
        assert_eq!(
            linear.get_cumulative_child_size(length - 1),
            length * (length - 1) / 2
        );
        assert_eq!(linear.get_child_size(length - 1).unwrap(), &(length - 1));

        // Increment again
        linear.increment_side_size(Side::Back, modification);

        assert_eq!(
            linear.cumulative_size(),
            length * (length - 1) / 2 + modification
        );
        assert_eq!(
            linear.get_cumulative_child_size(length - 1),
            length * (length - 1) / 2 + modification
        );
        assert_eq!(
            linear.get_child_size(length - 1).unwrap(),
            &(length - 1 + modification)
        );

        // Increment the front side now
        linear.increment_side_size(Side::Front, modification);

        assert_eq!(
            linear.cumulative_size(),
            length * (length - 1) / 2 + 2 * modification
        );
        assert_eq!(
            linear.get_cumulative_child_size(length - 1),
            length * (length - 1) / 2 + 2 * modification
        );
        assert_eq!(
            linear.get_child_size(length - 1).unwrap(),
            &(length - 1 + modification)
        );
        assert_eq!(*linear.get_child_size(0).unwrap(), modification);

        // Decrement the front side now
        linear.decrement_side_size(Side::Front, modification);

        assert_eq!(
            linear.cumulative_size(),
            length * (length - 1) / 2 + modification
        );
        assert_eq!(
            linear.get_cumulative_child_size(length - 1),
            length * (length - 1) / 2 + modification
        );
        assert_eq!(
            linear.get_child_size(length - 1).unwrap(),
            &(length - 1 + modification)
        );
        assert_eq!(linear.get_child_size(0).unwrap(), &0);

        // Decrement the back to get to the original state and now we will just increment
        // every second element and decrement the others and ensure this gets reflected
        linear.decrement_side_size(Side::Back, modification);
        for i in 0..length / 2 {
            linear.increment_child_size(2 * i, 2 * i);
            linear.decrement_child_size(2 * i + 1, i);
        }
        // Sizes now look like like:
        // Idx: 0, 1. 2, 3, 4, 5, 6, 7, 8, 9
        // New: 0, 1, 4, 2, 8, 3, 12, 4, 16, 5
        // These are 2 intermingled arrays were the one is a simple counter and the other is that multipled by 4
        // This is except for the first 0 and the last element which is length / 2 in this case

        for i in 0..length / 2 {
            assert_eq!(*linear.get_child_size(2 * i).unwrap(), 4 * i);
            assert_eq!(*linear.get_child_size(2 * i + 1).unwrap(), i + 1);
        }

        assert_eq!(
            linear.get_cumulative_child_size(length - 1),
            5 * ((length / 2) * ((length / 2) - 1)) / 2 + length / 2
        );
        assert_eq!(
            linear.cumulative_size(),
            5 * ((length / 2) * ((length / 2) - 1)) / 2 + length / 2
        );
    }

    #[test]
    pub fn constant() {
        let level = 1;
        let length = RRB_WIDTH;
        assert!(length <= RRB_WIDTH);
        let mut constant = AnnotationTable::new();
        let k = 10;
        for _ in 0..length {
            constant.push_child(Side::Back, k);
        }

        // Len
        assert!(!constant.is_empty());
        assert_eq!(constant.len(), length);

        // Iterate over the table from ensuring that the cumulative values and individual values are correct
        for i in 0..length {
            assert_eq!(constant.get_cumulative_child_size(i), (i + 1) * k);
            assert_eq!(*constant.get_child_size(i).unwrap(), k);
        }

        // Test if the cumulative sum of the every child is correct
        assert_eq!(constant.cumulative_size(), length * k);

        // Do some simple mutation to see whether these correctly modify the data structure
        let modification = 2;
        constant.decrement_side_size(Side::Back, modification);

        assert_eq!(constant.cumulative_size(), length * k - modification);
        assert_eq!(
            constant.get_cumulative_child_size(length - 1),
            length * k - modification
        );
        assert_eq!(
            *constant.get_child_size(length - 1).unwrap(),
            k - modification
        );

        // Re-increment the value
        constant.increment_side_size(Side::Back, modification);

        assert_eq!(constant.cumulative_size(), length * k);
        assert_eq!(constant.get_cumulative_child_size(length - 1), length * k);
        assert_eq!(*constant.get_child_size(length - 1).unwrap(), k);

        // Increment again
        constant.increment_side_size(Side::Back, modification);

        assert_eq!(constant.cumulative_size(), length * k + modification);
        assert_eq!(
            constant.get_cumulative_child_size(length - 1),
            length * k + modification
        );
        assert_eq!(
            *constant.get_child_size(length - 1).unwrap(),
            k + modification
        );

        // Increment the front side now
        constant.increment_side_size(Side::Front, modification);

        assert_eq!(constant.cumulative_size(), length * k + 2 * modification);
        assert_eq!(
            constant.get_cumulative_child_size(length - 1),
            length * k + 2 * modification
        );
        assert_eq!(
            *constant.get_child_size(length - 1).unwrap(),
            k + modification
        );
        assert_eq!(*constant.get_child_size(0).unwrap(), k + modification);

        // Decrement the front side now
        constant.decrement_side_size(Side::Front, modification);

        assert_eq!(constant.cumulative_size(), length * k + modification);
        assert_eq!(
            constant.get_cumulative_child_size(length - 1),
            length * k + modification
        );
        assert_eq!(
            *constant.get_child_size(length - 1).unwrap(),
            k + modification
        );
        assert_eq!(*constant.get_child_size(0).unwrap(), k);

        // Decrement the back to get to the original state and now we will just increment
        // every second element and decrement the others and ensure this gets reflected
        constant.decrement_side_size(Side::Back, modification);
        for i in 0..length / 2 {
            constant.increment_child_size(2 * i, k);
            constant.decrement_child_size(2 * i + 1, k / 2);
        }
        // Sizes now look like like:
        // Idx: 0, 1. 2, 3, 4, 5, 6, 7, 8, 9
        // New: 2K , K/2, 2K, K/2, 2K, K/2, 2K, K/2, 2K, K/2
        // These are 2 intermingled constant arrays

        for i in 0..length / 2 {
            assert_eq!(*constant.get_child_size(2 * i).unwrap(), 2 * k);
            assert_eq!(*constant.get_child_size(2 * i + 1).unwrap(), k / 2);
        }

        assert_eq!(
            constant.get_cumulative_child_size(length - 1),
            (k / 2 + 2 * k) * length / 2
        );
        assert_eq!(constant.cumulative_size(), (k / 2 + 2 * k) * length / 2);
    }
}
