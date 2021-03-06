//! A size table implementation.
//!
//! The size table is a structure for mapping from a position to a child index in an RRB tree.
use crate::circular::CircularBuffer;
use crate::Side;

use std::ops::Range;

/// A naive implementation of an size table for nodes in the RRB tree. This implementation keeps
/// track of the size of each of child as well as the level of the node in the tree.
#[derive(Clone, Debug)]
pub struct SizeTable {
    level: usize,
    buffer: CircularBuffer<usize>,
}

impl SizeTable {
    /// Construct a new SizeTable for a node at the given level. The SizeTable is initially empty.
    pub fn new(level: usize) -> Self {
        SizeTable {
            level,
            buffer: CircularBuffer::new(),
        }
    }

    /// Returns the level that the table should be associated with.
    pub fn level(&self) -> usize {
        self.level
    }

    /// Returns the total size of this node.
    pub fn cumulative_size(&self) -> usize {
        self.buffer.back().copied().unwrap_or_default()
    }

    /// Returns the sum of the total sizes of all children up to the given index.
    pub fn get_cumulative_child_size(&self, idx: usize) -> Option<&usize> {
        self.buffer.get(idx)
    }

    /// Returns the size of the child at the given index.
    pub fn get_child_size(&self, idx: usize) -> Option<usize> {
        if idx == 0 {
            self.buffer.get(0).cloned()
        } else if idx < self.buffer.len() {
            Some(self.buffer.get(idx).unwrap() - self.buffer.get(idx - 1).unwrap())
        } else {
            None
        }
    }

    /// Returns the size of the child at the given index.
    pub fn get_child_range(&self, idx: usize) -> Option<Range<usize>> {
        if idx < self.buffer.len() {
            if idx == 0 {
                Some(0..*self.buffer.get(0).unwrap())
            } else {
                Some(*self.buffer.get(idx - 1).unwrap()..*self.buffer.get(idx).unwrap())
            }
        } else {
            None
        }
    }

    /// Returns the position of the child that corresponds to the given index along with a new
    /// index to query in that child.
    pub fn position_info_for(&self, idx: usize) -> Option<(usize, usize)> {
        let mut last = 0;
        for (i, sz) in self.buffer.iter().enumerate() {
            if idx < *sz {
                return Some((i, idx - last));
            }
            last = *sz;
        }
        None
    }

    /// Adds a number of elements to the child at the given index.
    #[cfg(test)]
    pub fn increment_child_size(&mut self, idx: usize, increment: usize) {
        for item in self.buffer.iter_mut().skip(idx) {
            *item += increment;
        }
    }

    /// Removes a number of elements to the child at the given index.
    #[cfg(test)]
    pub fn decrement_child_size(&mut self, idx: usize, decrement: usize) {
        for item in self.buffer.iter_mut().skip(idx) {
            *item -= decrement;
        }
    }

    /// Adds a number of elements to the child at a given side of the node.
    #[cfg(test)]
    pub fn increment_side_size(&mut self, side: Side, increment: usize) {
        if self.len() == 0 {
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
    #[cfg(test)]
    pub fn decrement_side_size(&mut self, side: Side, decrement: usize) {
        let idx = match side {
            Side::Back => self.len() - 1,
            Side::Front => 0,
        };
        self.decrement_child_size(idx, decrement)
    }

    /// Adds a new child to a side of the node of a given size.
    pub fn push_child(&mut self, side: Side, size: usize) {
        match side {
            Side::Back => {
                if let Some(last_size) = self.buffer.back().copied() {
                    self.buffer.push(side, last_size + size);
                } else {
                    self.buffer.push(side, size);
                }
            }
            Side::Front => {
                for item in self.buffer.iter_mut() {
                    *item += size;
                }
                self.buffer.push_front(size);
            }
        }
    }

    /// Removes a child from a side of the node.
    pub fn pop_child(&mut self, side: Side) -> usize {
        match side {
            Side::Back => {
                let final_size = self.buffer.pop(side);
                if let Some(last_size) = self.buffer.back() {
                    final_size - last_size
                } else {
                    final_size
                }
            }
            Side::Front => {
                let size = self.buffer.pop_front();
                for item in self.buffer.iter_mut() {
                    *item -= size;
                }
                size
            }
        }
    }

    /// Returns the number of children of the node, not the size of the node!
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

#[allow(clippy::cognitive_complexity)]
#[cfg(test)]
mod test {
    use super::*;
    use crate::RRB_WIDTH;

    #[test]
    fn empty() {
        let level = 1;
        let empty = SizeTable::new(level);

        // Len
        assert_eq!(empty.len(), 0);

        // Level
        assert_eq!(empty.level(), level);
    }
    #[allow(clippy::cognitive_complexity)]
    #[test]
    fn linear() {
        let level = 1;
        let length = RRB_WIDTH;
        assert!(length <= RRB_WIDTH);
        let mut linear = SizeTable::new(level);
        for i in 0..length {
            linear.push_child(Side::Back, i);
        }

        // Len
        assert!(linear.len() != 0);
        assert_eq!(linear.len(), length);

        // Iterate over the table from ensuring that the cumulative values and individual values are correct
        for i in 0..length {
            assert_eq!(
                *linear.get_cumulative_child_size(i).unwrap(),
                i * (i + 1) / 2
            );
            assert_eq!(linear.get_child_size(i).unwrap(), i);
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
            *linear.get_cumulative_child_size(length - 1).unwrap(),
            length * (length - 1) / 2 - modification
        );
        assert_eq!(
            linear.get_child_size(length - 1).unwrap(),
            length - 1 - modification
        );

        // Re-increment the value
        linear.increment_side_size(Side::Back, modification);

        assert_eq!(linear.cumulative_size(), length * (length - 1) / 2);
        assert_eq!(
            *linear.get_cumulative_child_size(length - 1).unwrap(),
            length * (length - 1) / 2
        );
        assert_eq!(linear.get_child_size(length - 1).unwrap(), length - 1);

        // Increment again
        linear.increment_side_size(Side::Back, modification);

        assert_eq!(
            linear.cumulative_size(),
            length * (length - 1) / 2 + modification
        );
        assert_eq!(
            *linear.get_cumulative_child_size(length - 1).unwrap(),
            length * (length - 1) / 2 + modification
        );
        assert_eq!(
            linear.get_child_size(length - 1).unwrap(),
            length - 1 + modification
        );

        // Increment the front side now
        linear.increment_side_size(Side::Front, modification);

        assert_eq!(
            linear.cumulative_size(),
            length * (length - 1) / 2 + 2 * modification
        );
        assert_eq!(
            *linear.get_cumulative_child_size(length - 1).unwrap(),
            length * (length - 1) / 2 + 2 * modification
        );
        assert_eq!(
            linear.get_child_size(length - 1).unwrap(),
            length - 1 + modification
        );
        assert_eq!(linear.get_child_size(0).unwrap(), modification);

        // Decrement the front side now
        linear.decrement_side_size(Side::Front, modification);

        assert_eq!(
            linear.cumulative_size(),
            length * (length - 1) / 2 + modification
        );
        assert_eq!(
            *linear.get_cumulative_child_size(length - 1).unwrap(),
            length * (length - 1) / 2 + modification
        );
        assert_eq!(
            linear.get_child_size(length - 1).unwrap(),
            length - 1 + modification
        );
        assert_eq!(linear.get_child_size(0).unwrap(), 0);

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
            assert_eq!(linear.get_child_size(2 * i).unwrap(), 4 * i);
            assert_eq!(linear.get_child_size(2 * i + 1).unwrap(), i + 1);
        }

        assert_eq!(
            *linear.get_cumulative_child_size(length - 1).unwrap(),
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
        let mut constant = SizeTable::new(level);
        let k = 10;
        for _ in 0..length {
            constant.push_child(Side::Back, k);
        }

        // Len
        assert!(constant.len() != 0);
        assert_eq!(constant.len(), length);

        // Iterate over the table from ensuring that the cumulative values and individual values are correct
        for i in 0..length {
            assert_eq!(*constant.get_cumulative_child_size(i).unwrap(), (i + 1) * k);
            assert_eq!(constant.get_child_size(i).unwrap(), k);
        }

        // Test if the cumulative sum of the every child is correct
        assert_eq!(constant.cumulative_size(), length * k);

        // Do some simple mutation to see whether these correctly modify the data structure
        let modification = 2;
        constant.decrement_side_size(Side::Back, modification);

        assert_eq!(constant.cumulative_size(), length * k - modification);
        assert_eq!(
            *constant.get_cumulative_child_size(length - 1).unwrap(),
            length * k - modification
        );
        assert_eq!(
            constant.get_child_size(length - 1).unwrap(),
            k - modification
        );

        // Re-increment the value
        constant.increment_side_size(Side::Back, modification);

        assert_eq!(constant.cumulative_size(), length * k);
        assert_eq!(
            *constant.get_cumulative_child_size(length - 1).unwrap(),
            length * k
        );
        assert_eq!(constant.get_child_size(length - 1).unwrap(), k);

        // Increment again
        constant.increment_side_size(Side::Back, modification);

        assert_eq!(constant.cumulative_size(), length * k + modification);
        assert_eq!(
            *constant.get_cumulative_child_size(length - 1).unwrap(),
            length * k + modification
        );
        assert_eq!(
            constant.get_child_size(length - 1).unwrap(),
            k + modification
        );

        // Increment the front side now
        constant.increment_side_size(Side::Front, modification);

        assert_eq!(constant.cumulative_size(), length * k + 2 * modification);
        assert_eq!(
            *constant.get_cumulative_child_size(length - 1).unwrap(),
            length * k + 2 * modification
        );
        assert_eq!(
            constant.get_child_size(length - 1).unwrap(),
            k + modification
        );
        assert_eq!(constant.get_child_size(0).unwrap(), k + modification);

        // Decrement the front side now
        constant.decrement_side_size(Side::Front, modification);

        assert_eq!(constant.cumulative_size(), length * k + modification);
        assert_eq!(
            *constant.get_cumulative_child_size(length - 1).unwrap(),
            length * k + modification
        );
        assert_eq!(
            constant.get_child_size(length - 1).unwrap(),
            k + modification
        );
        assert_eq!(constant.get_child_size(0).unwrap(), k);

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
            assert_eq!(constant.get_child_size(2 * i).unwrap(), 2 * k);
            assert_eq!(constant.get_child_size(2 * i + 1).unwrap(), k / 2);
        }

        assert_eq!(
            *constant.get_cumulative_child_size(length - 1).unwrap(),
            (k / 2 + 2 * k) * length / 2
        );
        assert_eq!(constant.cumulative_size(), (k / 2 + 2 * k) * length / 2);
    }
}
