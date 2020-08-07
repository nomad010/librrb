//! A focus for a vector.
//!
//! A focus tracks the last leaf and positions which was read. The path down this tree is saved in
//! the focus and is used to accelerate lookups in nearby locations.
use crate::node_traits::{
    BorrowedInternalTrait, BorrowedLeafTrait, BorrowedNode, InternalTrait, LeafTrait, NodeMut,
    NodeRc, NodeRef,
};

use crate::vector::InternalVector;
use crate::Side;
use std::fmt::Debug;
use std::mem;
use std::ops::{Bound, Range, RangeBounds};
use std::rc::Rc;

use archery::{SharedPointer, SharedPointerKind};

// #[derive(Debug, Clone)]
// pub(crate) struct RefMutRcByPtr<'a, A>(pub(crate) Rc<&'a mut A>);

// impl<'a, A> RefMutRcByPtr<'a, A> {}

// impl<'a, A> std::hash::Hash for RefMutRcByPtr<'a, A> {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         std::ptr::hash(*self.0, state);
//     }
// }

// impl<'a, A> Eq for RefMutRcByPtr<'a, A> {}

// impl<'a, A> PartialEq for RefMutRcByPtr<'a, A> {
//     fn eq(&self, other: &Self) -> bool {
//         Rc::ptr_eq(&self.0, &other.0)
//     }
// }

/// A focus for a particular node in the spine.
///
/// This tracks the path down to a particular leaf in the tree.
#[derive(Clone, Debug)]
struct PartialFocus<P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    path: Vec<(SharedPointer<Internal, P>, Range<usize>)>,
    leaf: SharedPointer<Leaf, P>,
    leaf_range: Range<usize>,
}

impl<'a, P, Internal, Leaf> PartialFocus<P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    /// A helper method to compute the remainder of a path down a tree to a particular index.
    fn tree_path(
        nodes: &mut Vec<(SharedPointer<Internal, P>, Range<usize>)>,
        mut idx: usize,
    ) -> (Range<usize>, SharedPointer<Leaf, P>) {
        while let Some((previous_root, range)) = nodes.last() {
            let (next_node, next_range) = if let Some((subchild, subchild_range)) =
                previous_root.get_child_ref_for_position(idx)
            {
                let subrange_len = subchild_range.end - subchild_range.start;
                let absolute_subrange_start = range.start + subchild_range.start;
                let absolute_subrange =
                    absolute_subrange_start..absolute_subrange_start + subrange_len;
                if let NodeRef::Internal(internal) = subchild {
                    // println!(
                    //     "Absolute subrange {:?} from {:?} derp {} {} to {}",
                    //     absolute_subrange,
                    //     nodes.iter().map(|x| x.1.clone()).collect::<Vec<_>>(),
                    //     idx,
                    //     previous_root.level(),
                    //     internal.len()
                    // );
                    idx -= subchild_range.start;
                    (SharedPointer::clone(internal), absolute_subrange)
                } else {
                    // println!("Done {:?}", absolute_subrange);
                    return (absolute_subrange, SharedPointer::clone(subchild.leaf()));
                }
            } else {
                panic!("Attempt to move a focus to an out of bounds location.")
            };
            nodes.push((next_node, next_range));
        }
        unreachable!()
    }

    /// Constructs the focus from a tree node. This will focus on the first element in the node.
    fn from_tree(tree: &'a NodeRc<P, Internal, Leaf>) -> Self {
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
    fn get(&mut self, idx: usize) -> Option<&Leaf::Item> {
        if self.path.is_empty() {
            self.leaf.get(idx)
        } else if idx >= self.path[0].0.len() {
            None
        } else {
            self.move_focus(idx);
            self.leaf.get(idx - self.leaf_range.start)
        }
    }
}

/// A focus for the entire the tree. Like a `PartialFocus`, but this also takes the position in the
/// spine into account.
#[derive(Clone, Debug)]
pub struct Focus<'a, P, Internal, Leaf, BorrowedInternal>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Borrowed = BorrowedInternal>,
    BorrowedInternal: BorrowedInternalTrait<P, Leaf, InternalChild = Internal> + Debug,
    Leaf: LeafTrait,
{
    tree: &'a InternalVector<P, Internal, Leaf, BorrowedInternal>,
    spine_position: Option<(Side, usize)>,
    spine_node_focus: PartialFocus<P, Internal, Leaf>,
    focus_range: Range<usize>,
    range: Range<usize>,
}

impl<'a, P, Internal, Leaf, BorrowedInternal> Focus<'a, P, Internal, Leaf, BorrowedInternal>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Borrowed = BorrowedInternal>,
    BorrowedInternal: BorrowedInternalTrait<P, Leaf, InternalChild = Internal> + Debug,
    Leaf: LeafTrait,
{
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
    pub fn new(tree: &'a InternalVector<P, Internal, Leaf, BorrowedInternal>) -> Self {
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
    pub fn narrowed_tree(
        tree: &'a InternalVector<P, Internal, Leaf, BorrowedInternal>,
        mut range: Range<usize>,
    ) -> Self {
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
    pub fn get(&mut self, idx: usize) -> Option<&Leaf::Item> {
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

    /// Derp
    pub fn index(&mut self, idx: usize) -> &Leaf::Item {
        self.get(idx).unwrap()
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
        if range_end > self.range.end {
            panic!("Range must be inside the parent range");
        }
        let new_start = self.range.start + range_start;
        let new_end = new_start + range_end - range_start;
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

// derp
// #[derive(Debug)]
// pub struct PartialFocusMut<A, P, Internal, Leaf>
// where
//     A: Clone + Debug,
//     P: SharedPointerKind,
//     Internal: InternalTrait<P, Leaf, Item = A>,
//     Leaf: LeafTrait<Item = A>,
// {
//     pub(crate) node: BorrowedNode<A, P, Internal, Leaf>,

//     root: Option<(usize, Range<usize>)>,
//     // The listing of internal nodes below the borrowed root node along with their associated ranges
//     path: Vec<(*mut Internal, Range<usize>)>,
//     // The leaf of the focus part, might not exist if the borrowed root is a leaf node
//     leaf: Option<*mut Leaf>,
//     // The range that is covered by the lowest part of the focus
//     leaf_range: Range<usize>,
// }

/// A focus of the elements of a vector. The focus allows mutation of the elements in the vector.
// #[derive(Debug)]
pub struct FocusMut<'a, P, Internal, Leaf, BorrowedInternal>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Borrowed = BorrowedInternal>,
    BorrowedInternal: BorrowedInternalTrait<P, Leaf, InternalChild = Internal> + Debug,
    Leaf: LeafTrait,
{
    origins: Vec<Rc<&'a mut InternalVector<P, Internal, Leaf, BorrowedInternal>>>,
    pub(crate) nodes: Vec<BorrowedNode<P, Internal, Leaf>>,
    len: usize,
    // Focus part
    // This indicates the index of the root in the node list and the range of that is covered by it
    root: Option<(usize, Range<usize>)>,
    // The listing of internal nodes below the borrowed root node along with their associated ranges
    path:
        Vec<(
            *mut <<Internal as InternalTrait<P, Leaf>>::Borrowed as BorrowedInternalTrait<
                P,
                Leaf,
            >>::InternalChild,
            Range<usize>,
        )>,
    // The leaf of the focus part, might not exist if the borrowed root is a leaf node
    leaf: Option<*mut Leaf>,
    // The range that is covered by the lowest part of the focus
    leaf_range: Range<usize>,
}

impl<'a, P, Internal, Leaf, BorrowedInternal> FocusMut<'a, P, Internal, Leaf, BorrowedInternal>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Borrowed = BorrowedInternal>,
    BorrowedInternal: BorrowedInternalTrait<P, Leaf, InternalChild = Internal> + Debug,
    Leaf: LeafTrait,
{
    fn empty(&mut self) -> Self {
        FocusMut {
            origins: self.origins.clone(),
            nodes: vec![],
            len: 0,
            root: None,
            path: Vec::new(),
            leaf: None,
            leaf_range: 0..0,
        }
    }

    pub(crate) fn from_vectors(
        origins: Vec<Rc<&'a mut InternalVector<P, Internal, Leaf, BorrowedInternal>>>,
        nodes: Vec<BorrowedNode<P, Internal, Leaf>>,
    ) -> Self {
        let mut len = 0;
        for node in nodes.iter() {
            len += node.len();
        }
        FocusMut {
            origins,
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
    /// focus_1.split_at_fn(1, |focus_1, focus_2| {
    ///     assert_eq!(focus_1.len(), 1);
    ///     assert_eq!(focus_2.len(), 2);
    /// });
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
    /// focus_1.split_at_fn(0, |focus_1, focus_2| {
    ///     assert!(focus_1.is_empty());
    ///     assert!(!focus_2.is_empty());
    /// });
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Splits the focus into two foci. This focus is replaced with a focus that represents
    /// everything up to (excluding) the index. The return result is a focus that represents
    /// everything after (including) the index.
    ///
    /// # Panics
    ///
    /// Panics if the given index is greater than the focus' length.
    ///
    fn split_at(&mut self, index: usize) -> Self {
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
        println!(
            "index = {} {:#?}",
            index,
            self.nodes.iter().map(|x| x.len()).collect::<Vec<_>>()
        );
        let (self_child_position, mut subindex) = self.find_node_info_for_index(index).unwrap();
        self.len = index;
        println!("After find node info {} {}", self_child_position, subindex);
        let mut right_nodes = self.nodes.split_off(self_child_position);
        println!(
            "right nodes = {} {:#?}",
            subindex,
            right_nodes.iter().map(|x| x.len()).collect::<Vec<_>>()
        );
        right_nodes.reverse();

        loop {
            if subindex == 0 {
                break;
            }
            let node = right_nodes.pop().unwrap();
            match node {
                BorrowedNode::Internal(mut internal) => {
                    let (new_subindex, mut new_internal) = internal.split_at_position(subindex);
                    println!(
                        "New subindex {} {} {:?} {}",
                        subindex,
                        new_subindex,
                        new_internal.range(),
                        new_internal.len()
                    );
                    subindex = new_subindex;
                    let child = new_internal.pop_child(Side::Front).unwrap();
                    // let (child_idx, new_subindex) = internal.position_info_for(subindex).unwrap();
                    // subindex = new_subindex;
                    // let mut new_internal = internal.split_at_child(child_idx);
                    // Know how to do this, we borrow the child node.
                    // let child = match new_internal.children {
                    //     BorrowedChildList::Internals(ref mut children) => BorrowedNode::Internal(
                    //         SharedPointer::make_mut(children.front_mut().unwrap()).borrow_node(),
                    //     ),
                    //     BorrowedChildList::Leaves(ref mut children) => BorrowedNode::Leaf(
                    //         SharedPointer::make_mut(children.front_mut().unwrap()).borrow_node(),
                    //     ),
                    // };
                    // new_internal.children.range_mut().start += 1;
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

        let result = FocusMut::from_vectors(self.origins.clone(), right_nodes);
        assert_eq!(self.len + result.len, original_len);
        debug_assert!(self.assert_invariants());
        debug_assert!(result.assert_invariants());
        result
    }

    /// Splits the focus into two foci. Then calls the given function with the two foci as
    /// arguments. This focus is replaced with a focus that represents everything up to
    /// (excluding) the index. The return result is a focus that represents everything after
    /// (including) the index. Afterwards, the original focus is reconstituted.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// let mut focus_1 = v.focus_mut();
    /// focus_1.split_at_fn(1, |focus_1, focus_2| {
    ///     assert_eq!(focus_1.get(0), Some(&mut 1));
    ///     assert_eq!(focus_1.get(1), None);
    ///     assert_eq!(focus_2.get(0), Some(&mut 2));
    ///     assert_eq!(focus_2.get(1), Some(&mut 3));
    /// });
    /// ```
    /// # Panics
    ///
    /// Panics if the given index is greater than the focus' length.
    ///
    pub fn split_at_fn<F: FnMut(&mut Self, &mut Self)>(&mut self, index: usize, mut f: F) {
        println!("Splitting at {} of {}", index, self.len());
        let mut result = self.split_at(index);
        f(self, &mut result);
        self.combine(result);
    }

    fn combine(&mut self, mut other: Self) {
        // Recombine both side FocusMuts into back into the original FocusMut
        if self.is_empty() {
            *self = other;
            return;
        } else if other.is_empty() {
            return;
        }

        other.nodes.reverse();

        while !self.nodes.is_empty() && !other.nodes.is_empty() {
            let mut left_node = self.nodes.pop().unwrap();
            let right_node = other.nodes.pop().unwrap();
            let right_level = right_node.level();

            if !left_node.from_same_source(&right_node) {
                self.nodes.push(left_node);
                other.nodes.push(right_node);
                break;
            }

            left_node.combine(right_node);
            if let Some(right_parent) = other.nodes.last_mut() {
                // If we have a borrowed Leaf as the root then we could have exhausted the list.
                if right_level + 1 == right_parent.level() {
                    // The parent range must be updated to include the whole borrowed node now.
                    right_parent.internal_mut().unpop_child(Side::Front);
                } else {
                    other.nodes.push(left_node);
                }
            } else {
                other.nodes.push(left_node);
            }
        }
        other.nodes.reverse();
        self.nodes.append(&mut other.nodes);
        self.len += other.len;
    }

    /// Derp
    pub fn append(&mut self, other: Self) {
        // The focus part remains the same, but update other bits.
        self.origins.extend(other.origins);
        self.nodes.extend(other.nodes);
        self.len += other.len;
    }

    /// Derp
    pub fn prepend(&mut self, mut other: Self) {
        // The focus part must be updated to point to the new place before the other bits
        if let Some((ref mut root_id, ref mut root_range)) = self.root {
            *root_id += other.nodes.len();
            root_range.end += other.len;
            root_range.start += other.len;
            for (_, path_range) in self.path.iter_mut() {
                path_range.end += other.len;
                path_range.start += other.len;
            }
            self.leaf_range.end += other.len();
            self.leaf_range.start += other.len();
        }
        self.origins.reverse();
        other.origins.reverse();
        self.origins.extend(other.origins);
        self.origins.reverse();
        mem::swap(&mut self.nodes, &mut other.nodes);
        self.nodes.extend(other.nodes);
        self.len += other.len;
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
    /// focus.narrow(1..3, |focus| {
    ///     assert_eq!(focus.get(0), Some(&mut 2));
    ///     assert_eq!(focus.get(1), Some(&mut 3));
    ///     assert_eq!(focus.get(2), None);
    /// });
    /// ```
    pub fn narrow<
        R: RangeBounds<usize>,
        F: FnMut(&mut FocusMut<P, Internal, Leaf, BorrowedInternal>),
    >(
        &mut self,
        range: R,
        mut f: F,
    ) {
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
        self.split_at_fn(range_end, |left, _right| {
            left.split_at_fn(range_start, |_left, right| f(right))
        });
    }

    fn move_focus(&mut self, mut idx: usize) {
        // unimplemented!()
        if self.leaf_range.contains(&idx) {
            // Nothing needs to move here
            return;
        }
        if let Some((_, ref mut range)) = self.root {
            if !range.contains(&idx) {
                self.root.take();
                self.path.clear();
                self.leaf = None;
                self.leaf_range = 0..0;
            }
        }
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
        }
        // Resets the path so only correct items remain
        while let Some((_, range)) = self.path.last() {
            if range.contains(&idx) {
                break;
            }
            self.path.pop();
        }
        if self.path.is_empty() {
            if let Some((ref root, ref range)) = self.root {
                // Over here we are guaranteed only the root and the elements in path are correct
                // Our job now is to refresh the remainder of the path and the leave.
                let root = &mut self.nodes[*root];
                match root {
                    BorrowedNode::Internal(internal) => {
                        println!(
                            "{} - {} = {} out of {}",
                            idx,
                            range.start,
                            idx - range.start,
                            internal.len()
                        );
                        let (subchild, subchild_range) = internal
                            .get_child_mut_for_position(idx - range.start)
                            .unwrap();
                        let absolute_subchild_range = (subchild_range.start + range.start)
                            ..(subchild_range.end + range.start);

                        match subchild {
                            NodeMut::Internal(subchild) => {
                                self.path.push((
                                    SharedPointer::make_mut(subchild),
                                    absolute_subchild_range,
                                ));
                            }
                            NodeMut::Leaf(subchild) => {
                                let leaf: *mut Leaf = SharedPointer::make_mut(subchild);
                                self.leaf = Some(leaf);
                                self.leaf_range = absolute_subchild_range;
                                return;
                            }
                        }
                        // internal.
                        // Root has children
                        // let (child_idx, new_idx) =
                        //     internal.position_info_for(idx - range.start).unwrap();
                        // let range_start = idx - new_idx;
                        // match internal.children {
                        //     BorrowedChildList::Internals(ref mut children) => {
                        //         let child =
                        //             SharedPointer::make_mut(children.get_mut(child_idx).unwrap());
                        //         self.path
                        //             .push((child, range_start..range_start + child.len()));
                        //     }
                        //     BorrowedChildList::Leaves(ref mut children) => {
                        //         let leaf =
                        //             SharedPointer::make_mut(children.get_mut(child_idx).unwrap());
                        //         let leaf_len = leaf.len();
                        //         let leaf: *mut Leaf<A> = leaf;
                        //         self.leaf = Some(leaf);
                        //         self.leaf_range = range_start..range_start + leaf_len;
                        //         return;
                        //     }
                        // }
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
        loop {
            let (parent, parent_subrange) = self.path.last_mut().unwrap();
            let parent = unsafe { &mut **parent };
            let (child_node, child_subrange) = parent.get_child_mut_for_position(idx).unwrap();
            idx -= child_subrange.start;
            let child_subrange = (parent_subrange.start + child_subrange.start)
                ..(parent_subrange.start + child_subrange.end);
            // let (child_idx, new_idx) = parent.position_info_for(idx).unwrap();
            // let this_skipped_items = idx - new_idx;
            // idx = new_idx;
            match child_node {
                NodeMut::Internal(internal) => {
                    let new_root = SharedPointer::make_mut(internal);
                    self.path.push((new_root, child_subrange));
                }
                NodeMut::Leaf(leaf) => {
                    // skipped_items.start += this_skipped_items;
                    // skipped_items.end = skipped_items.start + leaf_len;
                    self.leaf = Some(SharedPointer::make_mut(leaf));
                    self.leaf_range = child_subrange;
                    break;
                }
            }
            // match parent.children {
            //     ChildList::Internals(ref mut children) => {
            //         let new_root = SharedPointer::make_mut(children.get_mut(child_idx).unwrap());
            //         skipped_items.start += this_skipped_items;
            //         skipped_items.end = skipped_items.start + new_root.len();
            //         self.path.push((new_root, skipped_items.clone()));
            //     }
            //     ChildList::Leaves(ref mut children) => {
            //         let leaf = SharedPointer::make_mut(children.get_mut(child_idx).unwrap());
            //         let leaf_len = leaf.len();
            //         let leaf: *mut Leaf = leaf;
            //         skipped_items.start += this_skipped_items;
            //         skipped_items.end = skipped_items.start + leaf_len;
            //         self.leaf = Some(leaf);
            //         self.leaf_range = skipped_items;
            //         break;
            //     }
            // }
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
    pub fn get(&mut self, idx: usize) -> Option<&mut Leaf::Item> {
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
    pub fn index(&mut self, idx: usize) -> &mut Leaf::Item {
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

    fn assert_invariants(&self) -> bool {
        let mut cumulative = 0;
        for node in self.nodes.iter() {
            cumulative += node.len();
        }
        cumulative == self.len
    }
}

/*
trait Container<A: Clone + Debug>: DerefMut<Target = A> {
    fn into_inner(inner: Self) -> A;

    fn swap(left: &mut Self, right: &mut Self);
}

trait FocusLike<A: Clone + Debug> {
    type Item: Container<A>;

    fn get(&self) -> Self::Item;

    fn split_at<F: Fn(&mut Self, &mut Self)>(&mut self, f: F);
}

struct DualFocusMut<'a, 'b, A: Clone + Debug, B: Clone + Debug> {
    first: &'b mut FocusMut<'a, A>,
    second: &'b mut FocusMut<'a, B>,
}

impl<'a, 'b, A: Clone + Debug, B: Clone + Debug> FocusLike<(A, B)> for DualFocusMut<'a, 'b, A, B> {
    type Item = Dual<'a, A, B>;

    // fn get(&mut self)
}

struct Single<A: Clone + Debug> {
    item: A,
}

impl<A: Clone + Debug> Deref for Single<A> {
    type Target = A;

    fn deref(&self) -> &A {
        &mut self.item
    }
}

impl<A: Clone + Debug> DerefMut for Single<A> {
    fn deref_mut(&mut self) -> &mut A {
        &mut self.item
    }
}

impl<A: Clone + Debug> Container<A> for Single<A> {
    fn into_inner(inner: Self) -> A {
        inner.item
    }

    fn swap(left: &mut Self, right: &mut Self) {
        mem::swap(left, right)
    }
}

struct Dual<'a, A: Clone + Debug, B: Clone + Debug> {
    item: (&'a mut A, &'a mut B),
}

impl<'a, A: Clone + Debug, B: Clone + Debug> Deref for Dual<'a, A, B> {
    type Target = (A, B);

    fn deref(&self) -> &(A, B) {
        &self.item
    }
}

impl<A: Clone + Debug, B: Clone + Debug> DerefMut for Dual<A, B> {
    fn deref_mut(&mut self) -> &mut (A, B) {
        &mut self.item
    }
}

impl<A: Clone + Debug, B: Clone + Debug> Container<(A, B)> for Dual<A, B> {
    fn into_inner(inner: Self) -> (A, B) {
        inner.item
    }

    fn swap(left: &mut Self, right: &mut Self) {
        mem::swap(left, right)
    }
}
*/
#[cfg(test)]
mod test {
    use crate::Vector;

    #[test]
    pub fn single_focus_mut() {
        let mut v = Vector::new();
        const N: isize = 1_000;
        for i in 0..N {
            v.push_back(i);
        }

        let mut focus = v.focus_mut();
        for i in 0..N {
            let thing = focus.index(i as usize);
            *thing = -i;
        }
        for (i, v) in v.iter().enumerate() {
            let r = -(i as isize);
            assert_eq!(v, &r);
        }
    }

    #[test]
    pub fn split_focus_mut() {
        let mut v = Vector::new();
        const N: usize = 1_000;
        for i in 0..N {
            v.push_back(i);
        }

        const S: usize = N / 2;
        let mut focus = v.focus_mut();
        focus.split_at_fn(S, |left, right| {
            for i in 0..S {
                let thing = left.get(i);
                if let Some(thing) = thing {
                    *thing = 0;
                }
            }

            for i in 0..N - S {
                let thing = right.get(i);
                if let Some(thing) = thing {
                    *thing = 1;
                }
            }
        });
        for i in 0..N {
            if i < S {
                assert_eq!(focus.get(i), Some(&mut 0));
            } else {
                assert_eq!(focus.get(i), Some(&mut 1));
            }
        }
        for (i, v) in v.iter().enumerate() {
            if i < S {
                assert_eq!(v, &0);
            } else {
                assert_eq!(v, &1);
            }
        }
    }
}
