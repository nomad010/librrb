//! A focus for a vector.
//!
//! A focus tracks the last leaf and positions which was read. The path down this tree is saved in
//! the focus and is used to accelerate lookups in nearby locations.
use crate::nodes::{ChildList, Internal, Leaf, NodeRc};
use crate::vector::Vector;
use crate::Side;
use std::fmt::Debug;
use std::ops::Range;
use std::rc::Rc;

/// A focus for a particular node in the spine.
///
/// This tracks the path down to a particular leaf in the tree.
#[derive(Clone, Debug)]
struct PartialFocus<A: Clone + Debug> {
    path: Vec<(Rc<Internal<A>>, Range<usize>)>,
    leaf: Rc<Leaf<A>>,
    leaf_range: Range<usize>,
}

impl<'a, A: Clone + Debug> PartialFocus<A> {
    /// A helper method to compute the remainder of a path down a tree to a particular index.
    fn tree_path(
        nodes: &mut Vec<(Rc<Internal<A>>, Range<usize>)>,
        mut idx: usize,
    ) -> (Range<usize>, Rc<Leaf<A>>) {
        let mut skipped_items = nodes.last().unwrap().1.clone();
        while let Some((previous_root, _)) = nodes.last() {
            if let Some((array_idx, new_idx)) = previous_root.sizes.position_info_for(idx) {
                let this_skipped_items = idx - new_idx;
                match &previous_root.children {
                    ChildList::Internals(children) => {
                        let new_root = children.get(array_idx).unwrap().clone();
                        skipped_items.start += this_skipped_items;
                        skipped_items.end = skipped_items.start + new_root.len();
                        nodes.push((new_root.clone(), skipped_items.clone()));
                        idx = new_idx;
                    }
                    ChildList::Leaves(leaves) => {
                        let leaf = leaves.get(array_idx).unwrap().clone();
                        skipped_items.start += this_skipped_items;
                        skipped_items.end = skipped_items.start + leaf.len();
                        return (skipped_items, leaf);
                    }
                }
            } else {
                panic!("Attempt to move a focus to an out of bounds location.")
            }
        }
        unreachable!()
    }

    /// Constructs the focus from a tree node. This will focus on the first element in the node.
    fn from_tree(tree: &'a NodeRc<A>) -> Self {
        match tree {
            NodeRc::Internal(internal) => {
                let mut path = vec![(internal.clone(), 0..tree.len())];
                let (leaf_range, leaf) = PartialFocus::tree_path(&mut path, 0);
                PartialFocus {
                    path,
                    leaf,
                    leaf_range,
                }
            }
            NodeRc::Leaf(leaf) => PartialFocus {
                path: Vec::new(),
                leaf: leaf.clone(),
                leaf_range: 0..leaf.len(),
            },
        }
    }

    /// Moves the focus to a new index in the tree.
    pub fn move_focus(&mut self, idx: usize) {
        if !self.leaf_range.contains(&idx) {
            while !self.path.last().unwrap().1.contains(&idx) {
                self.path.pop();
            }
            let new_idx = idx - self.path.last().unwrap().1.start;
            let (leaf_range, leaf) = PartialFocus::tree_path(&mut self.path, new_idx);
            self.leaf_range = leaf_range;
            self.leaf = leaf;
        }
    }

    /// Gets an element from the tree. If the element does not exist this will return `None`. This
    /// will move the focus along if necessary.
    fn get(&mut self, idx: usize) -> Option<&A> {
        if self.path.is_empty() {
            self.leaf.get(idx)
        } else if idx >= self.path[0].0.len() {
            None
        } else {
            self.move_focus(idx);
            self.leaf.get(idx - self.leaf_range.start)
        }
    }

    /// Returns the length of the focus. This is equivalent to the length of the root of the focus.
    fn len(&self) -> usize {
        if self.path.is_empty() {
            self.leaf.len()
        } else {
            self.path[0].0.len()
        }
    }

    /// Returns the leaf node and its range associated with the particular index. This will move the
    /// focus if necessary.
    fn leaf_at(&mut self, idx: usize) -> Option<(Range<usize>, &Leaf<A>)> {
        if idx < self.len() {
            self.move_focus(idx);
            Some((self.leaf_range.clone(), &self.leaf))
        } else {
            None
        }
    }
}

/// A focus for the entire the tree. Like a `PartialFocus`, but this also takes the position in the
/// spine into account.
#[derive(Clone, Debug)]
pub struct Focus<'a, A: Clone + Debug> {
    tree: &'a Vector<A>,
    spine_position: Option<(Side, usize)>,
    spine_node_focus: PartialFocus<A>,
    focus_range: Range<usize>,
    range: Range<usize>,
}

impl<'a, A: Clone + Debug> Focus<'a, A> {
    /// Constructs a new focus for a Vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// let v = vector![1, 2, 3];
    /// let mut focus = Focus::new(&v);
    /// assert_eq!(focus.get(0), Some(&1));
    /// ```
    pub fn new(tree: &'a Vector<A>) -> Self {
        Focus::narrowed_tree(tree, 0..tree.len())
    }

    /// Constructs a new focus for a Vector. The focus is narrowed by the given range, only
    /// elements within this range are accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// let v = vector![1, 2, 3];
    /// let mut focus = Focus::narrowed_tree(&v, 1..3);
    /// assert_eq!(focus.get(0), Some(&2));
    /// ```
    pub fn narrowed_tree(tree: &'a Vector<A>, mut range: Range<usize>) -> Self {
        if range.start >= tree.len() {
            range.start = tree.len();
        }
        if range.end >= tree.len() {
            range.end = tree.len();
        }
        let tree_iter = tree.spine_iter();
        let mut range_end = 0;
        let mut focus_range = 0..0;
        let mut spine_position = None;
        let mut focus_node = &tree.root;
        for (node_position, node) in tree_iter {
            let range_start = range_end;
            range_end = range_start + node.len();

            if range.start < range_end && range.end > range_start {
                focus_range = range_start..range_end;
                spine_position = node_position;
                focus_node = node;
            }
        }
        Focus {
            tree,
            spine_position,
            spine_node_focus: PartialFocus::from_tree(focus_node),
            focus_range,
            range,
        }
    }

    /// Refocuses to a different index within the tree. This will move to a different spine node if
    /// necessary.
    fn refocus(&mut self, idx: usize) {
        debug_assert!(idx < self.focus_range.start || idx >= self.focus_range.end);
        if idx < self.focus_range.start {
            let skip_amount = match self.spine_position {
                Some((Side::Front, position)) => {
                    self.tree.right_spine.len() + 1 + self.tree.left_spine.len() - position
                }
                None => self.tree.right_spine.len() + 1,
                Some((Side::Back, position)) => position + 1,
            };
            let mut range_end = self.focus_range.start;
            for (position, node) in self.tree.spine_iter().rev().skip(skip_amount) {
                let range_start = range_end - node.len();
                let range = range_start..range_end;
                if range.contains(&idx) {
                    self.spine_position = position;
                    self.focus_range = range;
                    self.spine_node_focus = PartialFocus::from_tree(node);
                    self.spine_node_focus.move_focus(idx - range_start);
                    break;
                }
                range_end = range_start;
            }
        } else {
            let skip_amount = match self.spine_position {
                Some((Side::Front, position)) => position + 1,
                None => self.tree.left_spine.len() + 1,
                Some((Side::Back, position)) => {
                    self.tree.left_spine.len() + 1 + self.tree.right_spine.len() - position
                }
            };
            let mut range_start = self.focus_range.end;
            for (position, node) in self.tree.spine_iter().skip(skip_amount) {
                let range_end = range_start + node.len();
                let range = range_start..range_end;
                if range.contains(&idx) {
                    self.spine_position = position;
                    self.focus_range = range;
                    self.spine_node_focus = PartialFocus::from_tree(node);
                    self.spine_node_focus.move_focus(idx - range_start);
                    break;
                }
                range_start = range_end;
            }
        }
    }

    /// Returns a reference to the element at the given position relative to the start of the range
    /// that the focus is narrowed for. Returns `None` if the position is out of bounds of the
    /// range.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// let v = vector![1, 2, 3];
    /// let mut focus = Focus::new(&v);
    /// assert_eq!(focus.get(0), Some(&1));
    /// assert_eq!(focus.get(1), Some(&2));
    /// assert_eq!(focus.get(2), Some(&3));
    /// assert_eq!(focus.get(3), None);
    /// ```
    pub fn get(&mut self, idx: usize) -> Option<&A> {
        let new_idx = idx + self.range.start;
        if self.range.contains(&new_idx) {
            if !self.focus_range.contains(&new_idx) {
                self.refocus(new_idx);
            }
            self.spine_node_focus.get(new_idx - self.focus_range.start)
        } else {
            None
        }
    }

    /// Returns the length of the range that is accessible through the focus.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// let v = vector![1, 2, 3];
    /// assert_eq!(Focus::new(&v).len(), 3);
    /// assert_eq!(Focus::narrowed_tree(&v, 3..3).len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.range.end - self.range.start
    }

    /// Tests whether no elements are accessible through the focus.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// let v = vector![1, 2, 3];
    /// assert!(!Focus::new(&v).is_empty());
    /// assert!(Focus::narrowed_tree(&v, 3..3).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the leaf node and its range associated with the particular index. This will move the
    /// focus if necessary.
    fn leaf_at(&mut self, idx: usize) -> Option<(Range<usize>, &Leaf<A>)> {
        let new_idx = idx + self.range.start;
        if self.range.contains(&new_idx) {
            if !self.focus_range.contains(&new_idx) {
                self.refocus(new_idx);
            }
            self.spine_node_focus
                .leaf_at(new_idx - self.focus_range.start)
        } else {
            None
        }
    }

    /// Narrows the focus, limiting it to a particular range.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// let v = vector![1, 2, 3];
    /// let mut focus = Focus::new(&v);
    /// assert_eq!(focus.get(0), Some(&1));
    /// focus.narrow(1..3);
    /// assert_eq!(focus.get(0), Some(&2));
    /// focus.narrow(1..2);
    /// assert_eq!(focus.get(0), Some(&3));
    /// focus.narrow(1..1);
    /// assert_eq!(focus.get(0), None);
    /// ```
    pub fn narrow(&mut self, range: Range<usize>) {
        if range.end > self.range.end {
            panic!("Range must be inside the parent range");
        }
        let new_start = self.range.start + range.start;
        let new_end = new_start + range.end - range.start;
        let new_focus = Focus::narrowed_tree(&self.tree, new_start..new_end);
        *self = new_focus;
    }

    /// Splits the focus into two narrowed foci. The first focus is narrowed to the start of the
    /// current focus to the given index exclusive, while the second focus is narrowed from the
    /// given index to the end of the current range.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// let v = vector![1, 2, 3];
    /// let focus = Focus::new(&v);
    /// let (mut first_focus, mut second_focus) = focus.split_at(1);
    /// assert_eq!(first_focus.get(0), Some(&1));
    /// assert_eq!(first_focus.get(1), None);
    /// assert_eq!(second_focus.get(0), Some(&2));
    /// assert_eq!(second_focus.get(1), Some(&3));
    /// ```
    pub fn split_at(self, idx: usize) -> (Self, Self) {
        if idx >= self.len() {
            panic!("Split is out of range")
        }
        let first_range = self.range.start..idx;
        let second_range = idx..self.range.end;
        let first = Focus::narrowed_tree(self.tree, first_range);
        let second = Focus::narrowed_tree(self.tree, second_range);
        (first, second)
    }
}
