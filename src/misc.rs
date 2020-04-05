//! Collection of utility structs used for the RRB tree and its algorithms.

// Sorting algorithm works as follows:
// 1) Choose a pivot and swap it to the end (potentially `take` it)
// 2) Move from left to right to find element(a) that is >= pivot
// 3) Move from right to left to find element(b) that is <= pivot
// 4) Swap both elements
// 5) If a == pivot then we swap with left equal range end
// 6) If b == pivot then we swap with right equal range start

// Four positions must be tracked 2, 3, 5, 6.
// 2: Left iter(proceeds forward)
// 3: Right iter(proceeds backwards)
// 5: Left equal iter(proceeds forwards)
// 6: Right equal iter(proceed backwards)

// Invariants
// 5 is not after 2, but can be equal
// 6 is not before or equal to 5

// When 2 and 6 cross, the algorithm moves equal areas and recurses
use crate::nodes::{ChildList, Internal, Leaf, NodeRc};

use std::fmt::Debug;
use std::mem;
use std::ops::Range;
use std::rc::Rc;

pub(crate) enum LeafGetResult<'a, A: Clone + Debug> {
    SharedLeaf(&'a mut Leaf<A>),
    NonSharedLeaf(&'a mut Leaf<A>, &'a mut Leaf<A>),
}

pub(crate) struct SortFocusMut<A: Clone + Debug> {
    left_iter_index: usize,
    left_equal_iter_index: usize,
    right_iter_index: usize,
    right_equal_iter_index: usize,

    root: NodeRc<A>,

    // This a path from the root (inclusive) to the last node shared between the leading left and
    // right iterators.
    shared_path: Vec<(NodeRc<A>, usize, Range<usize>)>,

    // This is a path that picks off where the `shared_path` leaves off. It is the rest of the paths
    // to left/right leading iterators.
    left_iter_path: Vec<(NodeRc<A>, usize, Range<usize>)>,
    right_iter_path: Vec<(NodeRc<A>, usize, Range<usize>)>,

    // These paths form the non-shared path to each of the lagging iterators
    left_equal_iter_path: Vec<(NodeRc<A>, usize, Range<usize>)>,
    right_equal_iter_path: Vec<(NodeRc<A>, usize, Range<usize>)>,

    // This is the position in the relevant leaf that each iterator is in.
    left_iter_leaf_position: usize,
    right_iter_leaf_position: usize,
    left_equal_leaf_position: usize,
    right_equal_leaf_position: usize,
    /*
    How the focus works:

    It is important to note that we are only allowed to store nodes in exactly one place in the
    focus. Secondly, it is important that the tree is in PACKED form.

    The `shared_path` vector contains the part of the path that is shared between both leading
    iterators. It is important to note that this path might not be shared between the lagging
    iterators. This is explained in more detail later.

    The non-shared paths of the left and right leading iterators are stored in `left_iter_path` and
    `right_iter_path`, respectively.

    The path to the lagging iterators is represented in `left_equal_iter_path` and
    `right_equal_iter_path`. Only nodes which are different to `left_iter_path` and
    `right_iter_path` are represented in the vector. In some situation the entire non-shared iter
    paths are replaced and bits of `shared_path` are overridden.
    */
}

impl<A: Clone + Debug> SortFocusMut<A> {
    fn new_from_root(mut root: NodeRc<A>, range: Range<usize>) -> Self {
        assert_ne!(range.start, range.end);
        let mut focus_start = range.start;
        let mut focus_end = range.end - 1;
        let mut shared_path = Vec::new();
        let mut left_iter_path = Vec::new();
        let mut right_iter_path = Vec::new();
        if !root.is_leaf() {
            let mut parent = &mut root;

            // This sets up the shared path
            while let NodeRc::Internal(node) = parent {
                let start_position = node.position_info_for(focus_start).unwrap();
                let end_position = node.position_info_for(focus_end).unwrap();

                if start_position.0 != end_position.0 {
                    break;
                }

                let new_node = unsafe {
                    match Rc::make_mut(node).children {
                        ChildList::Internals(ref mut children) => {
                            NodeRc::Internal(children.take_item(start_position.0))
                        }
                        ChildList::Leaves(ref mut children) => {
                            NodeRc::Leaf(children.take_item(start_position.0))
                        }
                    }
                };

                shared_path.push((new_node, start_position.0, focus_start..focus_end));
                parent = shared_path.last_mut().map(|x| &mut x.0).unwrap();
                focus_start = start_position.1;
                focus_end = end_position.1;
            }

            // We now set up the left iter path
            while let NodeRc::Internal(node) = parent {
                let start_position = node.position_info_for(focus_start).unwrap();

                let new_node = unsafe {
                    match Rc::make_mut(node).children {
                        ChildList::Internals(ref mut children) => {
                            NodeRc::Internal(children.take_item(start_position.0))
                        }
                        ChildList::Leaves(ref mut children) => {
                            NodeRc::Leaf(children.take_item(start_position.0))
                        }
                    }
                };

                left_iter_path.push((new_node, start_position.0, focus_start..focus_end));
                parent = left_iter_path.last_mut().map(|x| &mut x.0).unwrap();
                focus_start = start_position.1;
            }

            // We now set up the right iter path node
            // NOTE: We need to reset parent to the end of the shared_path
            parent = shared_path
                .last_mut()
                .map(|x| &mut x.0)
                .unwrap_or(&mut root);

            while let NodeRc::Internal(node) = parent {
                let end_position = node.position_info_for(focus_end).unwrap();

                let new_node = unsafe {
                    match Rc::make_mut(node).children {
                        ChildList::Internals(ref mut children) => {
                            NodeRc::Internal(children.take_item(end_position.0))
                        }
                        ChildList::Leaves(ref mut children) => {
                            NodeRc::Leaf(children.take_item(end_position.0))
                        }
                    }
                };

                right_iter_path.push((new_node, end_position.0, focus_start..focus_end));
                parent = right_iter_path.last_mut().map(|x| &mut x.0).unwrap();
                focus_end = end_position.1;
            }
        }
        SortFocusMut {
            root,
            shared_path,
            left_iter_leaf_position: focus_start,
            left_equal_leaf_position: focus_start,
            right_iter_leaf_position: focus_end,
            right_equal_leaf_position: focus_end,
            left_equal_iter_index: range.start,
            left_equal_iter_path: Vec::new(),
            left_iter_index: range.start,
            left_iter_path,
            right_iter_index: range.end,
            right_iter_path,
            right_equal_iter_index: range.end,
            right_equal_iter_path: Vec::new(),
        }
    }

    fn take_root(mut self) -> NodeRc<A> {
        // TODO Restore the root here
        self.root
    }

    fn read_left_iter(&mut self) -> Option<&A> {
        if self.left_iter_index == self.right_iter_index {
            None
        } else if let Some(left_iter_last) = self.left_iter_path.last() {
            left_iter_last
                .0
                .leaf_ref()
                .get(self.left_iter_leaf_position)
        } else if let Some(shared_path_last) = self.shared_path.last() {
            shared_path_last
                .0
                .leaf_ref()
                .get(self.left_iter_leaf_position)
        } else {
            // Root is a leaf node so we just get a reference to the required item
            self.root.leaf_ref().get(self.left_iter_leaf_position)
        }
    }

    fn read_right_iter(&self) -> Option<&A> {
        if self.left_iter_index == self.right_iter_index {
            None
        } else if let Some(right_iter_last) = self.right_iter_path.last() {
            right_iter_last
                .0
                .leaf_ref()
                .get(self.right_iter_leaf_position)
        } else if let Some(shared_path_last) = self.shared_path.last() {
            shared_path_last
                .0
                .leaf_ref()
                .get(self.right_iter_leaf_position)
        } else {
            // Root is a leaf node so we just get a reference to the required item
            self.root.leaf_ref().get(self.right_iter_leaf_position)
        }
    }

    fn swap_left_iters(&mut self) -> bool {
        if self.left_equal_iter_index == self.left_iter_index {
            return false;
        }

        let left_iter_leaf = if let Some(left_iter_last) = self.left_iter_path.last_mut() {
            left_iter_last.0.leaf_mut()
        } else if let Some(shared_path_last) = self.shared_path.last_mut() {
            shared_path_last.0.leaf_mut()
        } else {
            // Root is a leaf node so we just get a reference to the required item
            self.root.leaf_mut()
        };
        let left_iter_leaf = Rc::make_mut(left_iter_leaf);

        if let Some(left_equal_iter_leaf) = self.left_equal_iter_path.last_mut() {
            let left_equal_iter_leaf = Rc::make_mut(left_equal_iter_leaf.0.leaf_mut());

            let left_iter_value = left_iter_leaf
                .get_mut(self.left_iter_leaf_position)
                .unwrap();

            let equal_iter_value = left_equal_iter_leaf
                .get_mut(self.left_equal_leaf_position)
                .unwrap();

            mem::swap(left_iter_value, equal_iter_value);
        } else {
            let (equal_iter_value, left_iter_value) = left_iter_leaf
                .pair_mut(self.left_equal_leaf_position, self.left_iter_leaf_position);
            mem::swap(left_iter_value, equal_iter_value);
        }

        true
    }

    fn swap_right_iters(&mut self) -> bool {
        if self.right_equal_iter_index == self.right_iter_index {
            return false;
        }

        let right_iter_leaf = if let Some(right_iter_last) = self.right_iter_path.last_mut() {
            right_iter_last.0.leaf_mut()
        } else if let Some(shared_path_last) = self.shared_path.last_mut() {
            shared_path_last.0.leaf_mut()
        } else {
            // Root is a leaf node so we just get a reference to the required item
            self.root.leaf_mut()
        };
        let right_iter_leaf = Rc::make_mut(right_iter_leaf);

        if let Some(right_equal_iter_leaf) = self.right_equal_iter_path.last_mut() {
            let right_equal_iter_leaf = Rc::make_mut(right_equal_iter_leaf.0.leaf_mut());

            let right_iter_value = right_iter_leaf
                .get_mut(self.right_iter_leaf_position)
                .unwrap();

            let equal_iter_value = right_equal_iter_leaf
                .get_mut(self.right_equal_leaf_position)
                .unwrap();

            mem::swap(right_iter_value, equal_iter_value);
        } else {
            let (equal_iter_value, right_iter_value) = right_iter_leaf.pair_mut(
                self.right_equal_leaf_position,
                self.right_iter_leaf_position,
            );
            mem::swap(right_iter_value, equal_iter_value);
        }

        true
    }

    fn swap_leading_iters(&mut self) -> bool {
        if self.left_iter_index == self.right_iter_index {
            return false;
        }

        if let Some(left_iter_leaf) = self.left_iter_path.last_mut() {
            let left_iter_leaf = Rc::make_mut(left_iter_leaf.0.leaf_mut());
            let right_iter_leaf =
                Rc::make_mut(self.right_iter_path.last_mut().unwrap().0.leaf_mut());

            let left_iter_value = left_iter_leaf
                .get_mut(self.left_iter_leaf_position)
                .unwrap();

            let right_iter_value = right_iter_leaf
                .get_mut(self.right_iter_leaf_position)
                .unwrap();

            mem::swap(left_iter_value, right_iter_value);
        } else {
            let leaf = self
                .shared_path
                .last_mut()
                .map(|x| &mut x.0)
                .unwrap_or(&mut self.root)
                .leaf_mut();
            let (left_iter_value, right_iter_value) = Rc::make_mut(leaf).pair_mut(
                self.right_equal_leaf_position,
                self.right_iter_leaf_position,
            );
            mem::swap(left_iter_value, right_iter_value);
        }
        true
    }

    fn advance_left_iterator(&mut self) {
        if self.left_iter_index == self.right_iter_index {
            return;
        }
        // We need to handle two cases
    }
}
