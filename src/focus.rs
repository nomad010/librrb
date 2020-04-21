//! A focus for a vector.
//!
//! A focus tracks the last leaf and positions which was read. The path down this tree is saved in
//! the focus and is used to accelerate lookups in nearby locations.
use crate::nodes::{BorrowedChildList, BorrowedNode, ChildList, Internal, Leaf, NodeRc};
use crate::vector::Vector;
use crate::Side;
use std::fmt::Debug;
use std::mem;
use std::ops::{Bound, Range, RangeBounds};
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

/// A focus of the elements of a vector. The focus allows mutation of the elements in the vector.
#[derive(Debug)]
pub struct FocusMut<'a, A: Clone + Debug> {
    origin: Rc<&'a mut Vector<A>>,
    pub(crate) nodes: Vec<BorrowedNode<A>>,
    len: usize,
    // Focus part
    // This indicates the index of the root in the node list and the range of that is covered by it
    root: Option<(usize, Range<usize>)>,
    // The listing of internal nodes below the borrowed root node along with their associated ranges
    path: Vec<(*mut Internal<A>, Range<usize>)>,
    // The leaf of the focus part, might not exist if the borrowed root is a leaf node
    leaf: Option<*mut Leaf<A>>,
    // The range that is covered by the lowest part of the focus
    leaf_range: Range<usize>,
}

impl<'a, A: Clone + Debug> FocusMut<'a, A> {
    fn empty(&mut self) -> Self {
        FocusMut {
            origin: Rc::clone(&self.origin),
            nodes: vec![],
            len: 0,
            root: None,
            path: Vec::new(),
            leaf: None,
            leaf_range: 0..0,
        }
    }

    pub(crate) fn from_vector(origin: Rc<&'a mut Vector<A>>, nodes: Vec<BorrowedNode<A>>) -> Self {
        let mut len = 0;
        for node in nodes.iter() {
            len += node.len();
        }

        FocusMut {
            origin,
            nodes,
            len,
            root: None,
            path: Vec::new(),
            leaf: None,
            leaf_range: 0..0,
        }
    }

    /// Gets the length of the focus.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// let mut focus_1 = v.focus_mut();
    /// let mut focus_2 = focus_1.split_at(1);
    /// assert_eq!(focus_1.len(), 1);
    /// assert_eq!(focus_2.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Tests whether the focus represents no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// let mut focus_1 = v.focus_mut();
    /// let mut focus_2 = focus_1.split_at(0);
    /// assert!(focus_1.is_empty());
    /// assert!(!focus_2.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Splits the focus into two foci. This focus is replaced with a focus that represents
    /// everything up to (excluding) the index. The return result is a focus that represents
    /// everything after (including) the index.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// let mut focus_1 = v.focus_mut();
    /// let mut focus_2 = focus_1.split_at(1);
    /// assert_eq!(focus_1.get(0), Some(&mut 1));
    /// assert_eq!(focus_1.get(1), None);
    /// assert_eq!(focus_2.get(0), Some(&mut 2));
    /// assert_eq!(focus_2.get(1), Some(&mut 3));
    /// ```
    /// # Panics
    ///
    /// Panics if the given index is greater than the focus' length.
    ///
    pub fn split_at(&mut self, index: usize) -> Self {
        // We split the vector in two at the position, we need to find the two positions that denote
        // the last of this vector and the first of the next vector.
        let original_len = self.len;
        if index == 0 {
            // This vector becomes empty and the returned one is self.
            let empty = self.empty();
            let result = mem::replace(self, empty);
            debug_assert!(self.assert_invariants());
            debug_assert!(result.assert_invariants());
            return result;
        } else if index == self.len {
            // This vector is unchanged and the returned one is empty.
            let result = self.empty();
            debug_assert!(self.assert_invariants());
            debug_assert!(result.assert_invariants());
            return result;
        }
        // index is now 1..self.len()
        let (self_child_position, mut subindex) = self.find_node_info_for_index(index).unwrap();
        self.len = index;
        let mut right_nodes = self.nodes.split_off(self_child_position);
        right_nodes.reverse();

        loop {
            if subindex == 0 {
                break;
            }
            let node = right_nodes.pop().unwrap();
            match node {
                BorrowedNode::Internal(mut internal) => {
                    let (child_idx, new_subindex) = internal.position_info_for(subindex).unwrap();
                    subindex = new_subindex;
                    let mut new_internal = internal.split_at_child(child_idx);
                    // Know how to do this, we borrow the child node.
                    let child = match new_internal.children {
                        BorrowedChildList::Internals(ref mut children) => BorrowedNode::Internal(
                            Rc::make_mut(children.front_mut().unwrap()).borrow(),
                        ),
                        BorrowedChildList::Leaves(ref mut children) => {
                            BorrowedNode::Leaf(Rc::make_mut(children.front_mut().unwrap()).borrow())
                        }
                    };
                    new_internal.children.range_mut().start += 1;
                    if !internal.is_empty() {
                        self.nodes.push(BorrowedNode::Internal(internal));
                    }
                    if !new_internal.is_empty() {
                        right_nodes.push(BorrowedNode::Internal(new_internal));
                    }
                    right_nodes.push(child);
                }
                BorrowedNode::Leaf(mut leaf) => {
                    let new_leaf = leaf.split_at(subindex);
                    if !leaf.is_empty() {
                        self.nodes.push(BorrowedNode::Leaf(leaf));
                    }
                    if !new_leaf.is_empty() {
                        right_nodes.push(BorrowedNode::Leaf(new_leaf));
                    }
                    break;
                }
            }
        }
        right_nodes.reverse();

        self.root.take();
        self.path.clear();
        self.leaf = None;
        self.leaf_range = 0..0;

        let result = FocusMut::from_vector(Rc::clone(&self.origin), right_nodes);
        assert_eq!(self.len + result.len, original_len);
        debug_assert!(self.assert_invariants());
        debug_assert!(result.assert_invariants());
        result
    }

    /// Narrows the focus so it only represents the given subrange of the focus.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// let mut focus = v.focus_mut();
    /// focus.narrow(1..3);
    /// assert_eq!(focus.get(0), Some(&mut 2));
    /// assert_eq!(focus.get(1), Some(&mut 3));
    /// assert_eq!(focus.get(2), None);
    /// ```
    pub fn narrow<R: RangeBounds<usize>>(&mut self, range: R) {
        let range_start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(x) => *x,
            Bound::Excluded(x) => x + 1,
        };
        let range_end = match range.end_bound() {
            Bound::Unbounded => self.len(),
            Bound::Included(x) => x + 1,
            Bound::Excluded(x) => *x,
        };
        if range_start != 0 {
            let new_focus = self.split_at(range_start);
            mem::replace(self, new_focus);
        }

        self.split_at(range_end - range_start);
    }

    fn move_focus(&mut self, mut idx: usize) {
        if self.leaf_range.contains(&idx) {
            // Nothing needs to move here
            return;
        }
        if let Some((root_id, ref mut range)) = self.root {
            // println!(
            //     "Checking root range validity {:?} for {} sz {}",
            //     range,
            //     idx,
            //     self.nodes[root_id].len()
            // );
            if !range.contains(&idx) {
                // println!(
                //     "resetting root {} {} {:?} {}",
                //     self.nodes.len(),
                //     root_id,
                //     range,
                //     idx
                // );
                self.root.take(); //.unwrap();
                self.path.clear();
                self.leaf = None;
                self.leaf_range = 0..0;
                // if let Some((node_position, new_idx)) =
                //     self.find_new_node_info_for_index(idx, root_id, range)
                // {
                //     let node_start = idx - new_idx;
                //     let node_len = self.nodes[node_position].len();
                //     self.root = Some((node_position, node_start..node_start + node_len));
                //     // println!(
                //     //     "{} {:?} vs {} {:?} with {}",
                //     //     root_id,
                //     //     range,
                //     //     node_position,
                //     //     node_start..node_start + node_len,
                //     //     idx
                //     // );
                // }
            }
        }
        // println!("Checking root {:?}", self.root);
        if self.root.is_none() {
            // If the root is unassigned we can potentially find a new one.
            if idx < self.len {
                let (node_position, new_idx) = self.find_node_info_for_index(idx).unwrap();
                let node_start = idx - new_idx;
                let node_len = self.nodes[node_position].len();
                self.root = Some((node_position, node_start..node_start + node_len));
            } else {
                // No match for the root we just go ahead and retun as we have no hope of finding
                // the correct leaf
                return;
            }
            // if  {
            //     // println!("Found new {:?} going to {}", node_position, new_idx);
            // } else {
            // }
        }
        // Resets the path so only correct items remain
        // println!("Checking path {} {}", idx, self.path.len());
        while let Some((_, range)) = self.path.last() {
            // println!("Check {:?}", range);
            if range.contains(&idx) {
                break;
            }
            // println!("Popping path");
            self.path.pop();
        }
        // println!("Done checking path");
        if self.path.is_empty() {
            if let Some((ref root, ref range)) = self.root {
                // Over here we are guaranteed only the root and the elements in path are correct
                // Our job now is to refresh the remainder of the path and the leave.
                let root = &mut self.nodes[*root];
                match root {
                    BorrowedNode::Internal(internal) => {
                        // Root has children
                        // println!("derp {} {:?} {:?}", idx, range, internal.len());
                        let (child_idx, new_idx) =
                            internal.position_info_for(idx - range.start).unwrap();
                        let range_start = idx - new_idx;
                        match internal.children {
                            BorrowedChildList::Internals(ref mut children) => {
                                let child = Rc::make_mut(children.get_mut(child_idx).unwrap());
                                self.path
                                    .push((child, range_start..range_start + child.len()));
                            }
                            BorrowedChildList::Leaves(ref mut children) => {
                                let leaf = Rc::make_mut(children.get_mut(child_idx).unwrap());
                                let leaf_len = leaf.len();
                                let leaf: *mut Leaf<A> = leaf;
                                self.leaf = Some(leaf);
                                self.leaf_range = range_start..range_start + leaf_len;
                                return;
                            }
                        }
                    }
                    BorrowedNode::Leaf(_) => {
                        // Root is a leaf so the only thing we need to do here is just set the leaf
                        // range
                        self.leaf_range = range.clone();
                        return;
                    }
                }
            }
        }
        idx -= self
            .path
            .last()
            .map(|x| &x.1)
            .unwrap_or(&self.root.as_ref().unwrap().1)
            .start;

        let mut skipped_items = self
            .path
            .last()
            .map(|x| &x.1)
            .unwrap_or(&self.root.as_ref().unwrap().1)
            .clone();
        loop {
            let (parent, parent_range) = self.path.last_mut().unwrap();
            let parent = unsafe { &mut **parent };
            let (child_idx, new_idx) = parent
                .position_info_for(idx /* - parent_range.start*/)
                .unwrap();
            let this_skipped_items = idx - new_idx;
            idx = new_idx;
            match parent.children {
                ChildList::Internals(ref mut children) => {
                    let new_root = Rc::make_mut(children.get_mut(child_idx).unwrap());
                    skipped_items.start += this_skipped_items;
                    skipped_items.end = skipped_items.start + new_root.len();
                    self.path.push((new_root, skipped_items.clone()));
                }
                ChildList::Leaves(ref mut children) => {
                    let leaf = Rc::make_mut(children.get_mut(child_idx).unwrap());
                    let leaf_len = leaf.len();
                    let leaf: *mut Leaf<A> = leaf;
                    skipped_items.start += this_skipped_items;
                    skipped_items.end = skipped_items.start + leaf_len;
                    self.leaf = Some(leaf);
                    self.leaf_range = skipped_items;
                    break;
                }
            }
        }
    }

    /// Gets a mutable reference to the element at the given index of the focus. The index is
    /// relative to the start of the focus. If the index does not exist this will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// let mut focus = v.focus_mut();
    /// assert_eq!(focus.get(0), Some(&mut 1));
    /// assert_eq!(focus.get(1), Some(&mut 2));
    /// assert_eq!(focus.get(2), Some(&mut 3));
    /// assert_eq!(focus.get(3), None);
    /// ```
    pub fn get(&mut self, idx: usize) -> Option<&mut A> {
        self.move_focus(idx);
        if self.leaf_range.contains(&idx) {
            if let Some(leaf) = self.leaf {
                let leaf = unsafe { &mut *leaf };
                leaf.get_mut(idx - self.leaf_range.start)
            } else {
                let root_index = self.root.as_ref().unwrap().0;
                self.nodes[root_index]
                    .leaf_mut()
                    .get_mut(idx - self.leaf_range.start)
            }
        } else {
            None
        }
    }

    /// Gets a mutable reference to the element at the given index of the focus. The index is
    /// relative to the start of the focus. If the index does not exist this will panic.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// let mut focus = v.focus_mut();
    /// assert_eq!(focus.index(0), &mut 1);
    /// assert_eq!(focus.index(1), &mut 2);
    /// assert_eq!(focus.index(2), &mut 3);
    /// ```
    pub fn index(&mut self, idx: usize) -> &mut A {
        self.get(idx).expect("Index out of range.")
    }

    /// Returns the spine position and subindex corresponding the given index.
    fn find_node_info_for_index(&self, index: usize) -> Option<(usize, usize)> {
        if index >= self.len {
            None
        } else {
            let mut forward_end = 0;

            for (idx, node) in self.nodes.iter().enumerate() {
                if index < forward_end + node.len() {
                    return Some((idx, index - forward_end));
                }
                forward_end += node.len();
            }
            unreachable!();
        }
    }

    /// Returns the spine position and subindex corresponding the given index.
    fn find_new_node_info_for_index(
        &self,
        index: usize,
        old_position: usize,
        old_range: Range<usize>,
    ) -> Option<(usize, usize)> {
        if index >= self.len {
            None
        } else if index >= old_range.end {
            let mut forward_end = old_range.end;

            for (idx, node) in self.nodes.iter().enumerate().skip(old_position + 1) {
                if index < forward_end + node.len() {
                    return Some((idx, index - forward_end));
                }
                forward_end += node.len();
            }
            unreachable!();
        } else {
            let mut forward_start = old_range.start;

            for (idx, node) in self.nodes.iter().enumerate().take(old_position).rev() {
                forward_start -= node.len();
                if index >= forward_start {
                    return Some((idx, index - forward_start));
                }
            }
            unreachable!()
        }
    }

    fn assert_invariants(&self) -> bool {
        let mut cumulative = 0;
        for node in self.nodes.iter() {
            cumulative += node.assert_size_invariants();
        }
        cumulative == self.len
    }
}
