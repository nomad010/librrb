//! A container for representing a sequence of elements.
//!
//! # Terminology
//!
//! * RRB Tree
//!
//! A relaxed radix B-tree. An M-ary tree where the data is stored only in the leaves. The leaves
//! are all the same distance from the root.
//!
//! * Level of a Node
//!
//! A node's distance from its descendant leaves. A leaf node's level is 0.
//!
//! * Height of a Tree
//!
//! The tree root's level.
//!
//! * Len/Size of a Node
//!
//! The number of data elements that are accessible through the node. The size of a node T is
//! denoted as |T|
//!
//! * Slots of a Node
//!
//! The number of direct children assocciated with a node. For a leaf node, this is equivalent its
//! size. The slots of a node T is denoted as ||T||.
//!
//! * Left/Right Spine of a Tree
//!
//! The path that is formed by accessing the first/last data element from the root. This is
//! equivalent to the collection of nodes that are accessed by repeatedly following the first/last
//! children from the root.
//!
//! * A Node being Fully Dense
//!
//! A node is Fully Dense if it has no free slots and all children of that node are Fully Dense. If
//! the node is a leaf it has exactly M data elements. If the node is internal, then it has exactly
//! M children which are all Fully Dense.
//!
//! The size of a Fully Dense node T of level L may be calculated as:
//! |T| = M^(L + 1).
//!
//! * A Node being Left/Right Dense
//!
//! A node is Left/Right Dense if all but the left/right most and last child are Fully Dense while
//! the left/right most node is Left/Right Dense. A leaf node is always Left/Right Dense. The number
//! of children of a Left/Right Dense node is allowed to vary, but may not be empty.
//!
//! The size of a Left/Right Dense node T of level L may be calculated as:
//! |T| = (||T|| - 1) * M ^ L + |T_left/right|
//!
//! * A Node being Inner Dense
//!
//! A node is Inner Dense if all but the first and last children are Fully Dense and the first and
//! last children are Left/Right Dense. A leaf node is always Inner Dense. The number of children of an
//! Inner Dense node is allowed to vary, but must contain at least two children.
//!
//! The size of an Inner Dense node T of level L may be calculated as:
//! |T| = (||T|| - 2) * M ^ L + |T_left| + |T_right|
//!
//! # Implementation
//!
//! We keep the tree in a format that allows quick access to its spines. Each node in the spine is
//! stored in a vector (one for the left spine, and one for the right spine). The root is
//! technically part of both spines, but we'd rather not have the root be in two places so we keep
//! the root in a separate node in the tree.
//!
//! This implementation is similar to the Display model of other implementations, however there are
//! two differences. Instead of tracking a single, flexible position, we always track both the first
//! and last positions within the tree. Since we are tracking two positions, the root requires
//! special handling. Like the Display model, the tracked nodes of the spine are removed from the
//! other nodes of the tree. This is explained in the Invariants section.
//!
//! The implementation provides operations for push_back, push_front, pop_front, pop_back,
//! concatenate, slice_from_start and slice_to_end. Additionaly, there are fast operations to get
//! the front and back elements.
//!
//! # Invariants
//!
//! ## Invariant 1
//!
//! Spines must be of equal length. This should never be broken by an operation.
//!
//! ## Invariant 2
//!
//! All nodes in the spine except for the first one (the leaf) must have at least 1 slot free.
//!
//! This is equivalent to the process of keeping a Display focused on the first element. On the left
//! spine, this is the missing child is the leftmost node, while the missing child is the rightmost
//! node on the right spine. This missing child logically points to the spine element below.
//!
//! ## Invariant 3
//!
//! The first node(the leaf) in the spine must have at least 1 element, but may be full.
//!
//! Leaves, in general, cannot be empty. There is one case in which there could be an empty leaf in
//! the tree being the empty tree. In this case the leaf would be the root node and the spines would
//! be empty.
//!
//! By ensuring leaves in the spine cannot be empty, we can handle queries for the front/back
//! element (or elements near it) quickly.
//!
//! ## Invariant 4
//!
//! If the root is a non-leaf, it must always have at least 2 slot free, but may be empty.
//!
//! The two missing slots occur on either end of the root. These two missing slots logically point
//! to the the tops of both spines. This explains why this invariant if the root is a leaf as both
//! spines are empty.
//!
//! ## Invariant 5
//!
//! If the root is an empty non-leaf node then the last two nodes of both spines:
//! 1) Must not be able to be merged into a node of 1 less height.
//! 2) Must differ in slots by at most one node.
//!
//! This invariant is important to ensuring the pop* operations complete in a timely fashion. These
//! operations must occassionally traverse the tree to find the next non-empty leaf. Typically,
//! finding the leaf doesn't require more than a few hops, however in the worst case we would end up
//! going all the way up to the root and back down the other side of the tree. Although not strictly
//! necessary to keep performance guarantees we want to avoid this situation.
//!
//! Invariant 5 puts a bound on the size of the top of the spine in the case that the root is empty.
//! Let L and R be the number of slots in the left and right spine tops respectively. If the root is
//! empty, then L + R > M - 2 and |R - L| <= 1.
//!
//! Operations may cause the this invariant to be broken, a certain method is employed to fix up
//! these violations.
//!
//! This invariant is experimental and is likely to change.
//!
//! # Operations
//!
//! ## push_front/push_back
//!
//! Adds an element to the front or back of the collection. This operation will not create or remove
//! unbalanced nodes in the tree.
//!
//! ## pop_front/pop_back
//!
//! Removes an element to the front or back of the collection. This operation will not create
//! additional unbalanced nodes in the tree, however it may remove them.
//!
//! ## slice_from_start
//!
//! Slices a tree from the start to the given index. The elements after this index are discarded.
//! Additionally, this shrinks the tree if required. This happens if the LCA of the first index and
//! the given index is not the root. This LCA node will become the new root. This new tree will be
//! Inner Dense. Slicing a tree cannot increase the number of unbalanced nodes in the tree.
//!
//! ## slice_to_end
//!
//! Slices a tree from the given index to the end. The elements before this index are discarded.
//! Additionally, this shrinks the tree if required. This happens if the LCA of the last index and
//! the given index is not the root. This LCA node will become the new root. This new tree will be
//! Inner Dense. Slicing a tree cannot increase the number of unbalanced nodes. in the tree.
//!
//! ## concatenate
//!
//! Appends one tree onto the other, combining the sequences into one. The operation starts by
//! equalizing the height of both trees by add single child parents to the shorter tree. This
//! will break Invariant 5, but we do not worry about it till the end. Internally, concatenating
//! requires getting of the right spine of the first tree and the left spine of the tree.
//! Ideally, we want the two trees be Left/Right Dense so that they can be snapped together and the
//! result is Inner Dense.
//!
//! Unfortunately, this is usually not possible. Recall from the terminology section that Left/Right
//! Dense nodes have but one node Fully Dense. In this case, we need the spines we are removing to
//! be Fully Dense. The size of a Fully Dense node is always a power of M, however the size of these
//! spines might not be. As it is not possible to make this Fully Dense, the root may also not be
//! Inner Dense. Clearly, we have to compromise by allowing unbalanced nodes. We will describe 3
//! algorithms.
//!
//! ### Summing over two levels
//!
//! The algorithm proceeds by walking both spines to the level just above the leaves and merging
//! these node's children as far left as possible. If the right node becomes completely empty then
//! it is discarded, dropping the node. This reduces the height of the unbalanced part by two
//! levels. Correspondingly, any unbalanced part of at most 2 levels becomes Fully Dense afterwards.
//! Finally, this works its way up to the roots of both trees. The tree might have to be grown if
//! there are still two roots left or Invariant 5 is broken. The downside to this algorithm is that
//! it needs O(M^2 * H) time to complete the operation. It is presumed that this maintains a height
//! of O(logN + log(C/M^2)) where N and C are the total elements and the total concatenations used
//! to build up the tree.
//!
//! ### Summing over one level
//!
//! Similar to the above algorithm, this one proceeds by walking both spines to the leaves and
//! merging these node's children as far left as possible. If the right node becomes completely
//! empty then it is discarded, dropping the node. This reduces the height of the unbalanced part by
//! two levels. Any unbalanced part of at most 1 level becomes Fully Dense. Finally, this works its
//! way up to the roots of both trees. The tree might have to be grown if there are still two roots
//! left or Invariant 5 is broken. This algorithm runs in O(M * H) time, but it balances the tree
//! less than the first algorithm. This maintains a height of O(logN + log(C/M)) where N and C are
//! the total elements and the total concatenations used to build up the tree.
//!
//! ### Concatenating roots
//!
//! Finally, this algorithm only merges the roots of both trees. If both roots fit into a single
//! node, then this just needs to merge the nodes and return the merged node. If it is not possible
//! to merge into a single node then we simple grow the tree to encompass both nodes. This algorithm
//! runs in O(M) time, but it doesn't balance the tree at all. There is an advantage to not doing
//! any balancing, the tree structure is able to make better use of structural sharing as only the
//! roots of the original tree need to change. Like the above, this maintains a height of
//! O(logN + log(C/M)) where N and C are the total elements and the total concatenations used to
//! build up the tree. The difference between this algorithm and the one listed above is that the
//! worst case is much more likely to happen here. However, the plus side there is better structural
//! sharing.
//!
//! # Performance
//!
//! Assume the height of the tree is H and the number of elements in the tree is N.
//!
//! | Operation | Average case | Worst case |
//! | --- | --- | --- |
//! | [`Push front`][Vector::push_front] | O(1) | O(H) |
//! | [`Push back`][Vector::push_back] | O(1) | O(H) |
//! | [`Pop front`][Vector::pop_front] | O(1) | O(H) |
//! | [`Pop back`][Vector::pop_back] | O(1) | O(H) |
//! | [`Slice from Start`][Vector::slice_from_start] | O(MH) | O(MH) |
//! | [`Slice to Back`][Vector::slice_to_end] | O(MH) | O(MH) |
//! | [`Concatenate`][Vector::append] | O(MH) | O(MH) |
//! | [`Clone`][Vector::clone] | O(H) | O(H) |
//! | [`Front`][Vector::front] | O(1) | O(1) |
//! | [`Back`][Vector::back] | O(1) | O(1) |
//! | [`New empty`][Vector::new] | O(M) | O(M) |
//! | [`New singleton`][Vector::singleton] | O(M) | O(M) |
//!
//! We currently choose the `Summing over two levels` algorithm for concatenation, however in the
//! future this may switch to other algorithms. This means that H is bounded by O(logN + logC/M^2).
//! An example of what to expect from this algorithm can be found by understanding the inserts test.
//! The test inserts 0..N repetitively into the middle of a Vector. The insert operation is done by
//! splitting the original vector into two pieces before and after the split point. The new element
//! is inserted and the two pieces are concatenated back together. When N is roughly 200,000, the
//! height of the tree reaches 26. For comparison a full tree of height 4 can hold 16,777,216 or
//! over 800 times this amount and a full tree of 26 levels could hold over 10^44 elements.
//!
//! [Vector::push_front]: ./struct.InternalVector.html#method.push_front
//! [Vector::push_back]: ./struct.InternalVector.html#method.push_back
//! [Vector::pop_front]: ./struct.InternalVector.html#method.pop_front
//! [Vector::pop_back]: ./struct.InternalVector.html#method.pop_back
//! [Vector::slice_from_start]: ./struct.InternalVector.html#method.slice_from_start
//! [Vector::slice_to_end]: ./struct.InternalVector.html#method.slice_to_end
//! [Vector::append]: ./struct.InternalVector.html#method.append
//! [Vector::clone]: ./struct.InternalVector.html#method.clone
//! [Vector::front]: ./struct.InternalVector.html#method.front
//! [Vector::back]: ./struct.InternalVector.html#method.back
//! [Vector::new]: ./struct.InternalVector.html#method.new
//! [Vector::singleton]: ./struct.InternalVector.html#method.singleton

use crate::focus::{Focus, FocusMut};
use crate::node_traits::{
    BorrowedInternalTrait, BorrowedLeafTrait, InternalTrait, LeafTrait, NodeRc,
};
// use crate::node_impls::basic::;.
use crate::node_impls::basic::{ChildList, Internal, Leaf};
use crate::sort::{do_dual_sort, do_single_sort};
use crate::{Side, RRB_WIDTH};
use archery::{ArcK, RcK, SharedPointer, SharedPointerKind};
use rand_core::SeedableRng;
use std::borrow::Borrow;
use std::cmp;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::{self, FromIterator, FusedIterator};
use std::mem;
use std::ops::{Bound, Range, RangeBounds};
use std::rc::Rc;

/// Construct a vector.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate librrb;
/// # use librrb::Vector;
/// let first = vector![1, 2, 3];
/// let mut second = Vector::new();
/// second.push_back(1);
/// second.push_back(2);
/// second.push_back(3);
/// assert_eq!(first, second);
/// ```
#[macro_export]
macro_rules! vector {
    () => { $crate::vector::Vector::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::vector::Vector::new();
        $(
            l.push_back($x);
        )*
            l
    }};

    ( $($x:expr ,)* ) => {{
        let mut l = $crate::vector::Vector::new();
        $(
            l.push_back($x);
        )*
            l
    }};
}

/// Construct a vector.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate librrb;
/// # use librrb::ThreadSafeVector;
/// let first = vector_ts![1, 2, 3];
/// let mut second = ThreadSafeVector::new();
/// second.push_back(1);
/// second.push_back(2);
/// second.push_back(3);
/// assert_eq!(first, second);
/// ```
#[macro_export]
macro_rules! vector_ts {
    () => { $crate::vector::ThreadSafeVector::new() };

    ( $($x:expr),* ) => {{
        let mut l = $crate::vector::ThreadSafeVector::new();
        $(
            l.push_back($x);
        )*
            l
    }};

    ( $($x:expr ,)* ) => {{
        let mut l = $crate::vector::ThreadSafeVector::new();
        $(
            l.push_back($x);
        )*
            l
    }};
}

// Keep the left and right spines in a deconstructed form in the vector
// This allows us quicker access to the spines of the tree which are most commonly accessed.
// We have the following invariants on the spine though
// 1) The height of the tree is |tree|/2 and the length of the tree is always odd. The root of the
// tree is the middle element and the leaves are at the end. Each element closer to the root is a
// tree of one taller. Each element further from the root is a tree of one shallower.
// 2) All nodes must be non-empty to ensure that they are truly the spine. The only exception to
// this being empty trees (everything empty) and trees of size 1 which will have one of the
// back/front leaves empty.
// 3) The root must always have at least 2 slot free, but may be empty.
// 4) The leaves must have at least 1 element taken except in the case of the empty vector. The
// leaves may be full.
// 5) The other trees must have at least 1 slot free, but may be empty.
// 6) Descendant nodes of these trees are non-empty.
// 7) If the root is empty, the tops of the spines must have at least SIZE - 1 elements between
// them. If the this invariant is broken, we can shrink the tree by combing both the tops of the
// spines into one root element.

// Appending an element:
// We check the back of the tree's leaf. If the leaf is full BEFORE we add the new item, we append
// the leaf onto the next level up the tree. We continue until we can insert the a node into the
// tree. Once we successfully insert, the lower levels are replaced with 0 tree elements that
// represent owning only 1 element. If we eventually get to the root and break its size invariant
// when we insert we need to grow the tree. Growing the root involves inserting 2 elements after the
// root in the tree list. The first being the same height as the root and the second being one
// higher. the children of the old root get shared evenly between the first created node.
// Finally, we insert the element into the new leaf.
// A similar process works for the prepending elements
//
// Popping an element:
// If the leaf becomes empty after the item is popped, we must replace it the right most leaf from
// the level above. If the level above becomes empty we continue upwards. In a non-empty tree this
// procedure will always terminate. If we encounter an empty root, we replace the root with the
// opposite spine and then remove the node we came from out of the trees list. The right spine
// gets replaced with right most part of the left spine. A similar process exists for the
// popping from the left.
//
// Concatenating two trees(A + B):
// The right spine of A and the left spine of B must be merged into either A or B. We will assume
// two trees of equal height for now. We denote the Ar as A's right spine ad Bl as B's left spine.
// Ar[n] or Bl[n] denote the spine tree n levels above the leaf. We start at the bottom of both
// spines and shift Bl[0] into Ar[0]. If Bl[0] cannot be fully merged into Ar[0] then we have to
// prepend a node to Bl[1]. This node represent's the spine from Bl[1] down. It is always possible
// to add this node since each there is always 1 or 2 slots free. In either case, we must always
// push_right a node to Al[1] for the spine of Al[0] downwards. The process continues all the way
// up to the root. If the two roots cannot be fully merged into one node, or that node has less
// than 2 slots free we add a new root in much the same process as in appending.
// Splitting a tree from start a position
// There a 4 cases to consider
// 1) The position lies within the left of the root by traversing down the left spine at least once.
// 1) The position lies within the left of the root by traversing down the left spine at least once.
//
/// A container for representing sequence of elements
#[derive(Debug)]
pub struct InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    pub(crate) left_spine: Vec<NodeRc<A, P, Internal, Leaf>>,
    pub(crate) right_spine: Vec<NodeRc<A, P, Internal, Leaf>>,
    pub(crate) root: NodeRc<A, P, Internal, Leaf>,
    len: usize,
}

impl<A, P, Internal, Leaf> Clone for InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    fn clone(&self) -> Self {
        InternalVector {
            left_spine: self.left_spine.clone(),
            right_spine: self.right_spine.clone(),
            root: self.root.clone(),
            len: self.len,
        }
    }
}

impl<A, P, Internal, Leaf> InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    /// Constructs a new empty vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let v: Vector<u64> = Vector::new();
    /// assert_eq!(v, vector![]);
    /// ```
    pub fn new() -> Self {
        InternalVector {
            left_spine: vec![],
            right_spine: vec![],
            root: NodeRc::Leaf(SharedPointer::new(Leaf::empty())),
            len: 0,
        }
    }

    /// Constructs a new vector with a single element.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let v = Vector::singleton(1);
    /// assert_eq!(v, vector![1]);
    /// ```
    pub fn singleton(item: A) -> Self {
        InternalVector {
            left_spine: vec![],
            right_spine: vec![],
            root: NodeRc::Leaf(SharedPointer::new(Leaf::with_item(item))),
            len: 1,
        }
    }

    /// Derp
    pub fn constant_vec_of_length(item: A, len: usize) -> Self {
        let mut store = InternalVector::new();
        let mut accumulator = InternalVector::singleton(item);
        while accumulator.len() <= len {
            if len & accumulator.len() != 0 {
                store.append(accumulator.clone());
            }
            accumulator.append(accumulator.clone());
        }
        store
    }

    /// Returns the length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let v: Vector<u64> = Vector::new();
    /// assert_eq!(v.len(), 0);
    /// assert_eq!(Vector::singleton(1).len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Tests whether the vector is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let v: Vector<u64> = Vector::new();
    /// assert!(v.is_empty());
    /// assert!(!Vector::singleton(1).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the spines and root. The leaf spines are at the end of the
    /// iterator, while the root is in the middle.
    pub(crate) fn spine_iter(
        &self,
    ) -> impl Iterator<Item = (Option<(Side, usize)>, &NodeRc<A, P, Internal, Leaf>)> + DoubleEndedIterator
    {
        let left_spine_iter = self
            .left_spine
            .iter()
            .enumerate()
            .map(|(i, x)| (Some((Side::Front, i)), x));
        let root_iter = iter::once((None, &self.root));
        let right_spine_iter = self
            .right_spine
            .iter()
            .enumerate()
            .map(|(i, x)| (Some((Side::Back, i)), x))
            .rev();
        left_spine_iter.chain(root_iter).chain(right_spine_iter)
    }

    /// Completes a leaf on a side of the tree. This will push the leaf in toward the tree and
    /// bubble up full nodes. This will also handle expandung the tree.
    fn complete_leaf(&mut self, side: Side) {
        debug_assert_eq!(self.left_spine.len(), self.right_spine.len());
        debug_assert_eq!(self.leaf_ref(side).free_space(), 0);
        let (spine, other_spine) = match side {
            Side::Back => (&mut self.right_spine, &mut self.left_spine),
            Side::Front => (&mut self.left_spine, &mut self.right_spine),
        };

        for idx in 0..spine.len() {
            let node = &mut spine[idx];

            if node.free_slots() != 0 {
                // Nothing to do here
                break;
            }

            let full_node = mem::replace(node, node.new_empty());
            let parent_node = SharedPointer::make_mut(
                spine
                    .get_mut(idx + 1)
                    .unwrap_or(&mut self.root)
                    .internal_mut(),
            );
            parent_node.push_child(side, full_node);
        }

        if self.root.slots() >= RRB_WIDTH - 1 {
            // This root is overfull so we have to raise the tree here, we add a new node of the
            // same height as the old root. We decant half the old root into this new node.
            // Finally, we create a new node of height one more than the old root and set that as
            // the new root. We leave the root empty
            let new_root = NodeRc::Internal(SharedPointer::new(Internal::empty_internal(
                self.root.level() + 1,
            )));
            let mut new_node = mem::replace(&mut self.root, new_root);
            let mut other_new_node = new_node.new_empty();
            new_node.share_children_with(&mut other_new_node, side.negate(), RRB_WIDTH / 2);
            spine.push(new_node);
            other_spine.push(other_new_node);
        } else if self.root.slots() == 0 {
            // We have e have enough space in the root but we have balance the top of the spines
            self.fixup_spine_tops();
        }
    }

    /// Pushes an item into a side leaf of the tree. This fixes up some invariants in the case that
    /// the root sits directly above the leaves.
    fn push_side(&mut self, side: Side, item: A) {
        if self.leaf_ref(side).free_space() == 0 {
            self.complete_leaf(side);
        }

        SharedPointer::make_mut(self.leaf_mut(side)).push(side, item);
        self.len += 1;

        if self.spine_ref(side).len() == 1 {
            self.fixup_spine_tops();
        }
    }

    /// Appends a single item to the back of the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = Vector::new();
    /// v.push_back(1);
    /// assert_eq!(Vector::singleton(1), v);
    /// v.push_back(2);
    /// v.push_back(3);
    /// assert_eq!(vector![1, 2, 3], v);
    /// ```
    pub fn push_back(&mut self, item: A) {
        self.push_side(Side::Back, item);
    }

    /// Prepends a single item to the front of the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = Vector::new();
    /// v.push_back(1);
    /// assert_eq!(Vector::singleton(1), v);
    /// v.push_front(2);
    /// v.push_front(3);
    /// assert_eq!(vector![3, 2, 1], v);
    /// ```
    pub fn push_front(&mut self, item: A) {
        self.push_side(Side::Front, item);
    }

    /// Signals that a leaf is empty. This process the replace the leaf with the next leaf.
    fn empty_leaf(&mut self, side: Side) {
        // Invariants
        // 1) If the root is empty, the top spines have at least SIZE - 1 children between them.
        // 2) If the root is empty, each of top spines are within 2 size of each other
        // As a consequence, if any of the top spines are empty, the root is non-empty.
        // If we break invariant 1) then we shrink the tree, as we can make a new root from the
        // children of the tops of the spine.
        // If we break invariant 2), we call balance spine tops to correct it.
        debug_assert_eq!(self.left_spine.len(), self.right_spine.len());
        // println!("LUL {:#?}", self);
        // Try and pop the node into the level above
        let spine = match side {
            Side::Back => &mut self.right_spine,
            Side::Front => &mut self.left_spine,
        };
        // println!(
        //     "LUL {:?} {} {} {}",
        //     side,
        //     spine.len(),
        //     self.root.slots(),
        //     self.root.slots() != 0 || spine.last().unwrap().slots() != 0
        // );
        debug_assert_eq!(spine.first().unwrap().slots(), 0);
        debug_assert!(self.root.slots() != 0 || spine.last().unwrap().slots() != 0);

        let mut last_empty = spine.len() - 1;
        for (i, v) in spine.iter().enumerate().skip(1) {
            if v.slots() != 0 {
                last_empty = i - 1;
                break;
            }
        }

        // We can just grab from the the level above in the spine, this may leave that level
        // empty, The only things we need to worry about is making the root empty or making
        // the top of the spines unbalanced. Both of these we'll handle later.
        // let spine_len = spine.len();
        for level in (0..=last_empty).rev() {
            let node = spine
                .get_mut(level + 1)
                .unwrap_or(&mut self.root)
                .internal_mut();
            let child = SharedPointer::make_mut(node).pop_child(side);
            spine[level] = child;
        }

        self.fixup_spine_tops();
    }

    /// Fixes up the top of the spines. Certain operations break some invariants that we use to keep
    /// the tree balanced. This repeatedly fixes up the tree to fulfill the invariants.
    fn fixup_spine_tops(&mut self) {
        // The following invariant is fixed up here

        // Invariant 5
        // If the root is an empty non-leaf node then the last two nodes of both spines:
        // 1) Must not be able to be merged into a node of 1 less height
        // 2) Must differ in slots by at most one node

        // The invariant must be checked in a loop as slicing may break the invariant multiple times
        // println!(
        //     "LUL {} {} {}",
        //     self.root.slots(),
        //     self.root.level(),
        //     self.root.is_leaf()
        // );
        while self.root.slots() == 0 && !self.root.is_leaf() {
            // println!("gar");
            let left_spine_top = self.left_spine.last_mut().unwrap();
            let right_spine_top = self.right_spine.last_mut().unwrap();
            let left_spine_children = left_spine_top.slots();
            let right_spine_children = right_spine_top.slots();

            let total_children = left_spine_children + right_spine_children;
            let difference = if left_spine_children > right_spine_children {
                left_spine_children - right_spine_children
            } else {
                right_spine_children - left_spine_children
            };
            let min_children = if self.root.level() == 1 {
                // The root is one above a leaf and a new leaf root could be completely full
                RRB_WIDTH
            } else {
                // New non-leaf roots must contain at least 2 empty slots
                RRB_WIDTH - 2
            };

            if total_children < min_children {
                // Part 1) of invariant 5 is broken, we merge into a single node decreasing the
                // tree's height by 1. Invariant 5 might be broken with the new root so we need to
                // continue checking.
                let mut left_spine_top = self.left_spine.pop().unwrap();
                let mut right_spine_top = self.right_spine.pop().unwrap();
                left_spine_top.share_children_with(&mut right_spine_top, Side::Back, RRB_WIDTH);
                self.root = right_spine_top;
            } else if difference >= 2 {
                // Part 2) of invariant 5 is broken, we might need to share children between the
                // left and right spines. We are also guaranteed to have invariant 5 fulfilled
                // afterwards.
                let (source, destination, side) = if left_spine_children > right_spine_children {
                    (left_spine_top, right_spine_top, Side::Back)
                } else {
                    (right_spine_top, left_spine_top, Side::Front)
                };
                source.share_children_with(destination, side, difference / 2);
                break;
            } else {
                // No invariant is broken. We can stop checking here
                break;
            }
        }
    }

    /// Pops an item from a side leaf of the tree. This fixes up some invariants in the case that
    /// the root sits directly above the leaves.
    fn pop_side(&mut self, side: Side) -> Option<A> {
        debug_assert_eq!(self.left_spine.len(), self.right_spine.len());
        if self.spine_ref(side).is_empty() {
            if !self.root.is_empty() {
                self.len -= 1;
                Some(SharedPointer::make_mut(self.root.leaf_mut()).pop(side))
            } else {
                None
            }
        } else {
            // Can never be none as the is of height at least 1
            let leaf = self.leaf_mut(side);
            let item = SharedPointer::make_mut(leaf).pop(side);

            if leaf.is_empty() {
                self.empty_leaf(side);
            } else if self.spine_ref(side).len() == 1 {
                self.fixup_spine_tops();
            }

            self.len -= 1;
            Some(item)
        }
    }

    /// Removes and returns a single item from the back of the sequence. If the tree is empty this
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// assert_eq!(v.pop_back(), Some(3));
    /// assert_eq!(v, vector![1, 2]);
    /// assert_eq!(v.pop_back(), Some(2));
    /// assert_eq!(v, vector![1]);
    /// assert_eq!(v.pop_back(), Some(1));
    /// assert_eq!(v, vector![]);
    /// assert_eq!(v.pop_back(), None);
    /// assert_eq!(v, vector![]);
    /// ```
    pub fn pop_front(&mut self) -> Option<A> {
        self.pop_side(Side::Front)
    }

    /// Removes and returns a single item from the front of the sequence. If the tree is empty this
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// assert_eq!(v.pop_front(), Some(1));
    /// assert_eq!(v, vector![2, 3]);
    /// assert_eq!(v.pop_front(), Some(2));
    /// assert_eq!(v, vector![3]);
    /// assert_eq!(v.pop_front(), Some(3));
    /// assert_eq!(v, vector![]);
    /// assert_eq!(v.pop_front(), None);
    /// assert_eq!(v, vector![]);
    /// ```
    pub fn pop_back(&mut self) -> Option<A> {
        self.pop_side(Side::Back)
    }

    /// Returns a reference to the item at the front of the sequence. If the tree is empty this
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let v = vector![1, 2, 3];
    /// assert_eq!(v.front(), Some(&1));
    /// assert_eq!(v, vector![1, 2, 3]);
    /// ```
    pub fn front(&self) -> Option<&A> {
        let leaf = self.left_spine.first().unwrap_or(&self.root);
        leaf.leaf_ref().front()
    }

    /// Returns a mutable reference to the item at the front of the sequence. If the tree is empty
    /// this returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// assert_eq!(v.front_mut(), Some(&mut 1));
    /// assert_eq!(v, vector![1, 2, 3]);
    /// ```
    pub fn front_mut(&mut self) -> Option<&mut A> {
        let leaf = self.left_spine.first_mut().unwrap_or(&mut self.root);
        SharedPointer::make_mut(leaf.leaf_mut()).front_mut()
    }

    /// Returns a reference to the item at the back of the sequence. If the tree is empty this
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// assert_eq!(v.back(), Some(&3));
    /// assert_eq!(v, vector![1, 2, 3]);
    /// ```
    pub fn back(&self) -> Option<&A> {
        let leaf = self.right_spine.first().unwrap_or(&self.root);
        leaf.leaf_ref().back()
    }

    /// Returns a mutable reference to the item at the back of the sequence. If the tree is empty
    /// this returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// assert_eq!(v.back_mut(), Some(&mut 3));
    /// assert_eq!(v, vector![1, 2, 3]);
    /// ```
    pub fn back_mut(&mut self) -> Option<&mut A> {
        let leaf = self.right_spine.first_mut().unwrap_or(&mut self.root);
        SharedPointer::make_mut(leaf.leaf_mut()).back_mut()
    }

    /// Derp
    pub fn get(&self, idx: usize) -> Option<&A> {
        if let Some((spine_info, subindex)) = self.find_node_info_for_index(idx) {
            let node = match spine_info {
                Some((Side::Front, spine_idx)) => &self.left_spine[spine_idx],
                Some((Side::Back, spine_idx)) => &self.right_spine[spine_idx],
                None => &self.root,
            };
            Some(node.get(subindex).unwrap())
        } else {
            None
        }
    }

    /// Derp
    pub fn index(&self, idx: usize) -> &A {
        self.get(idx).expect("Index out of bounds.")
    }

    /// Derp
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut A> {
        if let Some((spine_info, subindex)) = self.find_node_info_for_index(idx) {
            let node = match spine_info {
                Some((Side::Front, spine_idx)) => &mut self.left_spine[spine_idx],
                Some((Side::Back, spine_idx)) => &mut self.right_spine[spine_idx],
                None => &mut self.root,
            };
            Some(node.get_mut(subindex).unwrap())
        } else {
            None
        }
    }

    /// Derp
    pub fn index_mut(&mut self, idx: usize) -> &mut A {
        self.get_mut(idx).expect("Index out of bounds.")
    }

    #[cfg(feature = "level-concatenations")]
    /// Appends the given vector onto the back of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// v.append(vector![4, 5, 6]);
    /// assert_eq!(v, vector![1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn append(&mut self, mut other: Self) {
        // Don't merge too thoroughly
        // 1) Walk down to the leaves
        // 2a) If the number of occupied slots on both sides is more than M than leave both sides
        // untouched
        // 2b) If the number of occupied slots on both sides is less than or equal to M than leave
        // merge the left into the right (or the right into the left)
        // 3) If left or right is non-empty, pack them up into their respective parent
        // 4) Go up 1 level to the node parents and repeat the process from step 2 (unless this is the root)
        // 5a) If there is one root that has at most M - 2 children then we are DONE
        // 5b) If there is one root with more than M - 2 children, then we split the root up into the
        // two evenly sized nodes (these nodes will have at most M/2 children)
        // 5c) If there are two roots we share the the nodes evenly between them
        // 6) The two roots from the previous step become the new left and right spine tops and a new
        // empty root is set.
        //
        // Analysis: This algorithm is O(M.H) as in the worst case for each pair of spine nodes
        // there could be a merge of the left and right children. Each merge simply shifts elements
        // out of one node and adds them tp another.
        // In the best case, this algorithm is O(H)

        if self.is_empty() {
            mem::replace(self, other);
            return;
        }
        if other.is_empty() {
            return;
        }

        // Make the spines the same length
        while self.right_spine.len() < other.left_spine.len() {
            // The root moves to either the left or right spine and the root becomes empty
            // We replace the left with root here and right with an empty node
            let new_root = match &self.root {
                NodeRc::Leaf(_) => Rc::new(Internal::empty_leaves()).into(),
                NodeRc::Internal(root) => {
                    Rc::new(Internal::empty_internal(root.level() + 1)).into()
                }
            };
            let mut new_left = mem::replace(&mut self.root, new_root);
            let mut new_right = new_left.new_empty();
            new_left.share_children_with(&mut new_right, Side::Back, new_left.slots() / 2);
            self.left_spine.push(new_left);
            self.right_spine.push(new_right);
        }

        while other.left_spine.len() < self.right_spine.len() {
            // The root moves to either the left or right spine and the root becomes empty
            // We replace the right with root here and left with an empty node
            let new_root = match &other.root {
                NodeRc::Leaf(_) => Rc::new(Internal::empty_leaves()).into(),
                NodeRc::Internal(root) => {
                    Rc::new(Internal::empty_internal(root.level() + 1)).into()
                }
            };
            let mut new_right = mem::replace(&mut other.root, new_root);
            let mut new_left = new_right.new_empty();
            new_right.share_children_with(&mut new_left, Side::Front, new_right.slots() / 2);
            other.left_spine.push(new_left);
            other.right_spine.push(new_right);
        }

        while other.left_spine.len() < self.right_spine.len() {
            // The root moves to either the left or right spine and the root becomes empty
            // We replace the right with root here and left with an empty node
            let new_root = match &other.root {
                NodeRc::Leaf(_) => Rc::new(Internal::empty_leaves()).into(),
                NodeRc::Internal(root) => {
                    Rc::new(Internal::empty_internal(root.level() + 1)).into()
                }
            };
            let new_right = mem::replace(&mut other.root, new_root);
            let new_left = new_right.new_empty();
            other.left_spine.push(new_left);
            other.right_spine.push(new_right);
        }

        debug_assert_eq!(self.right_spine.len(), self.right_spine.len());
        debug_assert_eq!(other.left_spine.len(), other.right_spine.len());
        debug_assert_eq!(self.right_spine.len(), other.left_spine.len());

        // More efficient to work from front to back here, but we need to remove elements
        // We reverse to make this more efficient
        self.right_spine.reverse();
        other.left_spine.reverse();
        while !self.right_spine.is_empty() {
            let mut left_node = self.right_spine.pop().unwrap();
            let mut right_node = other.left_spine.pop().unwrap();

            right_node.share_children_with(&mut left_node, Side::Front, RRB_WIDTH);
            let parent_node = self.right_spine.last_mut().unwrap_or(&mut self.root);
            let parent = Rc::make_mut(parent_node.internal_mut());
            Rc::make_mut(&mut parent.sizes).push_child(Side::Back, left_node.len());
            match parent.children {
                ChildList::Internals(ref mut children) => children.push_back(left_node.internal()),
                ChildList::Leaves(ref mut children) => children.push_back(left_node.leaf()),
            }
            if !right_node.is_empty() {
                let parent_node = other.left_spine.last_mut().unwrap_or(&mut other.root);
                let parent = Rc::make_mut(parent_node.internal_mut());
                Rc::make_mut(&mut parent.sizes).push_child(Side::Front, right_node.len());
                match parent.children {
                    ChildList::Internals(ref mut children) => {
                        children.push_front(right_node.internal());
                    }
                    ChildList::Leaves(ref mut children) => children.push_front(right_node.leaf()),
                }
            }
        }
        debug_assert!(self.right_spine.is_empty());
        debug_assert!(other.left_spine.is_empty());
        mem::replace(&mut self.right_spine, other.right_spine);

        other
            .root
            .share_children_with(&mut self.root, Side::Front, RRB_WIDTH);

        if self.root.free_slots() < 2 {
            self.root
                .share_children_with(&mut other.root, Side::Back, 1);
        }

        if !other.root.is_empty() {
            other
                .root
                .share_children_with(&mut self.root, Side::Front, RRB_WIDTH);
            let new_root = match &self.root {
                NodeRc::Leaf(_) => Rc::new(Internal::empty_leaves()).into(),
                NodeRc::Internal(root) => {
                    Rc::new(Internal::empty_internal(root.level() + 1)).into()
                }
            };
            let old_root = mem::replace(&mut self.root, new_root);
            self.left_spine.push(old_root);
            self.right_spine.push(other.root);
        }
        self.len += other.len;
        self.fixup_spine_tops();
    }

    #[cfg(not(feature = "level-concatenations"))]
    /// Appends the given vector onto the back of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// v.append(vector![4, 5, 6]);
    /// assert_eq!(v, vector![1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn append(&mut self, mut other: Self) {
        // Don't merge too thoroughly
        // 1) Walk down to the leaves
        // 2a) If the number of occupied slots on both sides is more than M than leave both sides
        // untouched
        // 2b) If the number of occupied slots on both sides is less than or equal to M than leave
        // merge the left into the right (or the right into the left)
        // 3) If left or right is non-empty, pack them up into their respective parent
        // 4) Go up 1 level to the node parents and repeat the process from step 2 (unless this is the root)
        // 5a) If there is one root that has at most M - 2 children then we are DONE
        // 5b) If there is one root with more than M - 2 children, then we split the root up into the
        // two evenly sized nodes (these nodes will have at most M/2 children)
        // 5c) If there are two roots we share the the nodes evenly between them
        // 6) The two roots from the previous step become the new left and right spine tops and a new
        // empty root is set.
        //
        // Analysis: This algorithm is O(M.H) as in the worst case for each pair of spine nodes
        // there could be a merge of the left and right children. Each merge simply shifts elements
        // out of one node and adds them tp another.
        // In the best case, this algorithm is O(H)

        // For each level we want to combine in the tree
        // 1) Pack the children in the left node. O(M^2)
        // 2) Pop grandchildren off the right node until the left node is full. O(M)
        // 3) Pack the children in the right node. O(M^2)
        // 4) Move the children off right node until the left node is full. O(M)
        // 5) If at any stage the right node becomes empty we stop and left is the result, otherwise
        //    we have to add represent it as a root with height: level + 1 and 'return' the result.
        if self.is_empty() {
            *self = other;
            return;
        }
        if other.is_empty() {
            return;
        }

        if self.len() == 1 {
            other.push_front(self.pop_back().unwrap());
            *self = other;
            return;
        }
        if other.len() == 1 {
            self.push_back(other.pop_back().unwrap());
            return;
        }

        println!("lens {} {}", self.len(), other.len());

        let new_len = self.len + other.len();
        // println!("Roflpi {:#?}", other);

        // Make the spines the same length
        while self.right_spine.len() < other.left_spine.len() {
            // The root moves to either the left or right spine and the root becomes empty
            // We replace the left with root here and right with an empty node
            // println!("Adding to self");
            let new_root = NodeRc::Internal(SharedPointer::new(Internal::empty_internal(
                self.root.level() + 1,
            )));
            let mut new_left = mem::replace(&mut self.root, new_root);
            let mut new_right = new_left.new_empty();
            new_left.share_children_with(&mut new_right, Side::Back, new_left.slots() / 2);
            self.left_spine.push(new_left);
            self.right_spine.push(new_right);
        }

        while other.left_spine.len() < self.right_spine.len() {
            // The root moves to either the left or right spine and the root becomes empty
            // We replace the right with root here and left with an empty node
            // println!("Adding to other");
            let new_root = NodeRc::Internal(SharedPointer::new(Internal::empty_internal(
                other.root.level() + 1,
            )));
            let mut new_right = mem::replace(&mut other.root, new_root);
            let mut new_left = new_right.new_empty();
            new_right.share_children_with(&mut new_left, Side::Front, new_right.slots() / 2);
            other.left_spine.push(new_left);
            other.right_spine.push(new_right);
        }

        // println!("heights {} {}", self.height(), other.height());

        // debug_assert_eq!(self.right_spine.len(), self.right_spine.len());
        debug_assert_eq!(other.left_spine.len(), other.right_spine.len());
        debug_assert_eq!(self.right_spine.len(), other.left_spine.len());

        let packer = |new_node: NodeRc<A, P, Internal, Leaf>,
                      parent_node: &mut SharedPointer<Internal, P>,
                      side| {
            if !new_node.is_empty() {
                let parent = &mut SharedPointer::make_mut(parent_node);
                // println!("gar {} vs {}", parent.level(), new_node.level());
                parent.push_child(side, new_node);
            }
        };

        // More efficient to work from front to back here, but we need to remove elements
        // We reverse to make this more efficient
        self.right_spine.reverse();
        other.left_spine.reverse();
        if let Some(left_child) = self.right_spine.pop() {
            let parent_node = self
                .right_spine
                .last_mut()
                .unwrap_or(&mut self.root)
                .internal_mut();
            packer(left_child, parent_node, Side::Back);
        }
        if let Some(right_child) = other.left_spine.pop() {
            // println!(
            //     "Gar {} {} {}",
            //     other.left_spine.len(),
            //     other.root.level(),
            //     right_child.level()
            // );
            // for spine_node in other.left_spine.iter() {
            //     println!(
            //         "Roflderp gar {} vs {}",
            //         spine_node.level(),
            //         right_child.level()
            //     );
            // }

            let parent_node = other
                .left_spine
                .last_mut()
                .unwrap_or(&mut other.root)
                .internal_mut();
            packer(right_child, parent_node, Side::Front);
        }
        // let sls: usize = self.left_spine.iter().map(|x| x.len()).sum();
        // let srs: usize = self.right_spine.iter().map(|x| x.len()).sum();
        // let ols: usize = other.left_spine.iter().map(|x| x.len()).sum();
        // let ors: usize = other.right_spine.iter().map(|x| x.len()).sum();
        // println!(
        //     "AAAAAG DERO GABLE GAR {} {} {} = {}, {} {} {} = {}, {}",
        //     sls,
        //     self.root.len(),
        //     srs,
        //     sls + self.root.len() + srs,
        //     ols,
        //     other.root.len(),
        //     ors,
        //     ols + other.root.len() + ors,
        //     sls + self.root.len() + srs + ols + other.root.len() + ors,
        // );
        // Error in this loop
        while !self.right_spine.is_empty() {
            let mut left_node = self.right_spine.pop().unwrap();
            let mut right_node = other.left_spine.pop().unwrap();

            let left = SharedPointer::make_mut(left_node.internal_mut());
            let right = SharedPointer::make_mut(right_node.internal_mut());

            left.pack_children();
            println!(
                "OOH {} {} {:?} {} --- {} {} {:?} {} LEL {} {}",
                left.len(),
                left.slots(),
                self.right_spine
                    .iter()
                    .map(|x| x.level())
                    .collect::<Vec<_>>(),
                self.root.level(),
                right.len(),
                right.slots(),
                other
                    .left_spine
                    .iter()
                    .map(|x| x.level())
                    .collect::<Vec<_>>(),
                other.root.level(),
                left.level(),
                right.level()
            );
            let mut left_right_most = left.pop_child(Side::Back);
            while !left_right_most.is_full() && !right.is_empty() {
                let mut right_left_most = right.pop_child(Side::Front);
                println!("roflpi {} {}", left_right_most.len(), right_left_most.len());
                right_left_most.share_children_with(
                    &mut left_right_most,
                    Side::Front,
                    right_left_most.slots(),
                );
                println!(
                    "roflpi2 {} {}",
                    left_right_most.len(),
                    right_left_most.len()
                );
                if !right_left_most.is_empty() {
                    right.push_child(Side::Front, right_left_most);
                }
            }
            left.push_child(Side::Back, left_right_most);
            right.pack_children();
            right.share_children_with(left, Side::Front, right.slots());

            // packer(
            //     left_node,
            //     self.right_spine
            //         .last_mut()
            //         .unwrap_or(&mut self.root)
            //         .internal_mut(),
            //     Side::Back,
            // );
            // packer(
            //     right_node,
            //     other
            //         .left_spine
            //         .last_mut()
            //         .unwrap_or(&mut other.root)
            //         .internal_mut(),
            //     Side::Front,
            // );

            // if !left.ch
            // while left.get
            // loop {
            //     // if left.
            // }
            // while !right.is_empty() && left.
            /*

            if !left.is_empty() {
                left.pack_children();

                let left_position = left.slots() - 1;
                while left
                    .get_child_at_slot(left_position)
                    .unwrap()
                    .0
                    .free_slots()
                    != 0
                    && !right.is_empty()
                {
                    match left.children {
                        ChildList::Internals(ref mut children) => {
                            let destination_node =
                                SharedPointer::make_mut(children.get_mut(left_position).unwrap());
                            let source_node = SharedPointer::make_mut(
                                right.children.internals_mut().front_mut().unwrap(),
                            );
                            let shared = source_node.share_children_with(
                                destination_node,
                                Side::Front,
                                RRB_WIDTH,
                            );
                            SharedPointer::make_mut(&mut left.sizes)
                                .increment_side_size(Side::Back, shared);
                            SharedPointer::make_mut(&mut right.sizes)
                                .decrement_side_size(Side::Front, shared);
                            if source_node.is_empty() {
                                right.children.internals_mut().pop_front();
                                SharedPointer::make_mut(&mut right.sizes).pop_child(Side::Front);
                            }
                            assert_eq!(right.sizes.len(), right.children.internals_mut().len());
                        }
                        ChildList::Leaves(ref mut children) => {
                            let destination_node =
                                SharedPointer::make_mut(children.get_mut(left_position).unwrap());
                            let source_node = SharedPointer::make_mut(
                                right.children.leaves_mut().front_mut().unwrap(),
                            );
                            let shared = source_node.share_children_with(
                                destination_node,
                                Side::Front,
                                RRB_WIDTH,
                            );
                            SharedPointer::make_mut(&mut left.sizes)
                                .increment_side_size(Side::Back, shared);
                            SharedPointer::make_mut(&mut right.sizes)
                                .decrement_side_size(Side::Front, shared);
                            if source_node.is_empty() {
                                right.children.leaves_mut().pop_front();
                                SharedPointer::make_mut(&mut right.sizes).pop_child(Side::Front);
                            }
                            assert_eq!(right.sizes.len(), right.children.leaves_mut().len());
                        }
                    }
                }
            }

            if !right.is_empty() {
                right.pack_children();
                right.share_children_with(left, Side::Front, RRB_WIDTH);
            }
            */

            packer(
                left_node,
                self.right_spine
                    .last_mut()
                    .unwrap_or(&mut self.root)
                    .internal_mut(),
                Side::Back,
            );
            if !right.is_empty() {
                packer(
                    right_node,
                    other
                        .left_spine
                        .last_mut()
                        .unwrap_or(&mut other.root)
                        .internal_mut(),
                    Side::Front,
                );
            }
            let left = self.right_spine.last_mut().unwrap_or(&mut self.root);
            let right = other.left_spine.last_mut().unwrap_or(&mut other.root);
            // println!(
            //     "OOH {} {} {:?} {} --- {} {} {:?} {}",
            //     left.len(),
            //     left.slots(),
            //     self.right_spine.iter().map(|x| x.len()).collect::<Vec<_>>(),
            //     self.root.len(),
            //     right.len(),
            //     right.slots(),
            //     other.left_spine.iter().map(|x| x.len()).collect::<Vec<_>>(),
            //     other.root.len()
            // );
        }

        // Bug is between DERO println and here
        debug_assert!(self.right_spine.is_empty());
        debug_assert!(other.left_spine.is_empty());
        self.right_spine = other.right_spine;

        other
            .root
            .share_children_with(&mut self.root, Side::Front, RRB_WIDTH);

        if self.root.free_slots() < 2 {
            self.root
                .share_children_with(&mut other.root, Side::Back, 1);
        }

        if !other.root.is_empty() {
            let new_root = NodeRc::Internal(SharedPointer::new(Internal::empty_internal(
                self.root.level() + 1,
            )));
            let old_root = mem::replace(&mut self.root, new_root);
            self.left_spine.push(old_root);
            self.right_spine.push(other.root);
        }
        self.len = new_len;
        self.fixup_spine_tops();
        // let ls: usize = self.left_spine.iter().map(|x| x.len()).sum();
        // let rs: usize = self.right_spine.iter().map(|x| x.len()).sum();
        // println!(
        //     "GAR GABLE GAR {} {} {} = {}",
        //     ls,
        //     self.root.len(),
        //     rs,
        //     ls + self.root.len() + rs
        // );
    }

    /// Prepends the given vector onto the front of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// v.prepend(vector![4, 5, 6]);
    /// assert_eq!(v, vector![4, 5, 6, 1, 2, 3]);
    /// ```
    pub fn prepend(&mut self, other: Self) {
        let other = mem::replace(self, other);
        self.append(other)
    }

    /// Slices the vector from the start to the given index exclusive.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// v.slice_from_start(2);
    /// assert_eq!(v, vector![1, 2]);
    /// ```
    pub fn slice_from_start(&mut self, len: usize) {
        // We find a path from the root down to the leaf in question
        // 1) If the leaf has an ancestor in the left spine, we pop (from the root) the until the
        // lowest ancestor in the spine, the right spine becomes the path from the left spine to
        // this leaf
        // 2) If the leaf has an ancestor on the right spine, we pop from the leaf up to this
        // node, but we add back the spine by replacing it with path to this leaf
        // 3) If the leaf only has the root as an ancestor it proceeds as 2) with the path becoming
        // the entire right spine.
        // if len == 0 {
        //     self.left_spine.clear();
        //     self.right_spine.clear();
        //     self.root = NodeRc::Leaf(SharedPointer::new(Leaf::empty()));
        //     self.len = 0;
        //     return;
        // }
        // let index = len;
        // unimplemented!();
        // // self.make_index_side(index - 1, Side::Back);
        // self.fixup_spine_tops();
        self.split_off(len);
    }

    /// Slices the vector from the given index inclusive to the end.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// v.slice_to_end(2);
    /// assert_eq!(v, vector![3]);
    /// ```
    pub fn slice_to_end(&mut self, start: usize) {
        // This proceeds the same as above, except that we instead recompute the left spine instead
        // of the right spine.
        // 1) If the leaf has an ancestor on the left spine, we pop from the leaf up to this
        // node, but we add back the spine by replacing it with path to this leaf
        // 2) If the leaf has an ancestor in the right spine, we pop (from the root) the until the
        // lowest ancestor in the spine, the left spine becomes the path from the right spine to
        // this leaf
        // 3) If the leaf only has the root as an ancestor it proceeds as 2) with the path becoming
        // the entire left spine.
        // if start >= self.len {
        //     self.left_spine.clear();
        //     self.right_spine.clear();
        //     self.root = NodeRc::Leaf(SharedPointer::new(Leaf::empty()));
        //     self.len = 0;
        //     return;
        // }
        // let index = start;
        // unimplemented!();
        // // self.make_index_side(index, Side::Front);
        // self.fixup_spine_tops();
        // println!("before {}", self.len());
        let result = self.split_off(start);
        // println!("after {}", result.len());
        *self = result;
    }

    /// Derp
    pub fn extract_slice<R: RangeBounds<usize> + Debug>(&mut self, range: R) -> Self {
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

        let last_bit = self.split_off(range_end);
        let middle_bit = self.split_off(range_start);
        self.append(last_bit);
        middle_bit
    }

    /// Returns the spine position and subindex corresponding the given index.
    fn find_node_info_for_index(&self, index: usize) -> Option<(Option<(Side, usize)>, usize)> {
        if index >= self.len {
            None
        } else {
            let mut forward_end = 0;
            let mut backward_start = self.len;

            for (idx, (left, right)) in self
                .left_spine
                .iter()
                .zip(self.right_spine.iter())
                .enumerate()
            {
                if index < forward_end + left.len() {
                    return Some((Some((Side::Front, idx)), index - forward_end));
                }
                forward_end += left.len();
                backward_start -= right.len();
                if index >= backward_start {
                    return Some((Some((Side::Back, idx)), index - backward_start));
                }
            }
            debug_assert!(index >= forward_end && index < backward_start);
            Some((None, index - forward_end))
        }
    }

    /// Slices the tree so that the element is first at the requested side of the tree.
    // fn make_index_side(&mut self, index: usize, side: Side) {
    // if let Some((node_position, mut node_index)) = self.find_node_info_for_index(index) {
    //     // We need to make this node the first/last in the tree
    //     // This means that this node will become the part of the spine for the given side
    //     match node_position {
    //         None => {
    //             // The root is where the spine starts.
    //             // This means all of the requested side's spine must be discarded
    //             while let Some(node) = self.spine_mut(side).pop() {
    //                 self.len -= node.len();
    //             }
    //         }
    //         Some((spine_side, spine_position)) if side != spine_side => {
    //             // The new end comes from the opposite spine
    //             // This means the root AND the requested spine must be discarded
    //             // The node we are pointing to becomes the new root and things above it in the
    //             // opposite spine are discarded. Since higher points come first we can do this
    //             // efficiently without reversing the spine
    //             while spine_position + 1 != self.spine_ref(side.negate()).len() {
    //                 self.len -= self.spine_mut(side.negate()).pop().unwrap().len();
    //             }
    //             self.len -= self.root.len();
    //             let new_root = self.spine_mut(side.negate()).pop().unwrap();
    //             self.root = new_root;
    //             while let Some(node) = self.spine_mut(side).pop() {
    //                 self.len -= node.len();
    //             }
    //         }
    //         Some((spine_side, spine_position)) if side == spine_side => {
    //             // The new end comes from the same spine.
    //             // Only the elements below the node in the spine need to be discarded
    //             // The root and left spine remain untouched
    //             // To the spine discarding efficiently we reverse, pop and reverse again
    //             self.spine_mut(side).reverse();
    //             for _ in 0..spine_position {
    //                 self.len -= self.spine_mut(side).pop().unwrap().len();
    //             }
    //             self.spine_mut(side).reverse();
    //         }
    //         _ => unreachable!(),
    //     }
    //     // We need to complete the spine here, we do this by cutting a side off of
    //     // nodes going down the spine
    //     // We need to do this in reverse order again to make this more efficient
    //     let spine = match side {
    //         Side::Front => &mut self.left_spine,
    //         Side::Back => &mut self.right_spine,
    //     };
    //     spine.reverse();
    //     while let NodeRc::Internal(ref mut internal) =
    //         spine.last_mut().unwrap_or(&mut self.root)
    //     {
    //         assert!(node_index < internal.len());
    //         let num_slots = internal.slots();
    //         let (child_position, new_index) = internal.position_info_for(node_index).unwrap();
    //         let internal_mut = SharedPointer::make_mut(internal);
    //         // let children = &mut internal_mut.children;
    //         let sizes = SharedPointer::make_mut(&mut internal_mut.sizes);
    //         let range = match side {
    //             Side::Back => child_position + 1..num_slots,
    //             Side::Front => 0..child_position,
    //         };
    //         for _ in range {
    //             match children {
    //                 ChildList::Internals(children) => {
    //                     children.pop(side);
    //                 }
    //                 ChildList::Leaves(children) => {
    //                     children.pop(side);
    //                 }
    //             }
    //             self.len -= sizes.pop_child(side);
    //         }
    //         let next_node = internal_mut.pop_child(side);
    //         sizes.pop_child(side);
    //         spine.push(next_node);
    //         node_index = new_index;
    //     }

    //     // The only thing to be fixed here is the leaf spine node
    //     let leaf =
    //         SharedPointer::make_mut(spine.last_mut().unwrap_or(&mut self.root).leaf_mut());
    //     let range = match side {
    //         Side::Back => node_index + 1..leaf.len(),
    //         Side::Front => 0..node_index,
    //     };
    //     assert!(node_index < leaf.len());
    //     for _ in range {
    //         leaf.pop(side);
    //         self.len -= 1;
    //     }

    //     // Now we are done, we can reverse the spine here to get it back to normal
    //     spine.reverse();
    // }
    // }

    /// Splits the vector at the given index into two vectors. `self` is replaced with every element
    /// before the given index and a new vector containing everything after and including the index.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// let last_half = v.split_off(1);
    /// assert_eq!(v, vector![1]);
    /// assert_eq!(last_half, vector![2, 3]);
    /// ```
    pub fn split_off(&mut self, at: usize) -> InternalVector<A, P, Internal, Leaf> {
        if at == 0 {
            // We can early out because the result is self and self becomes empty.
            mem::replace(self, InternalVector::new())
        } else if at >= self.len() {
            // We can early out because the result is empty and self remains unchanged.
            InternalVector::new()
        } else {
            // We know now that the split position lies within the vector allowing us to make some
            // simplifications.
            let original_len = self.len();
            // println!("Slicing at {} out of {}", at, self.len);
            // println!("Roflskates {:#?}", self);
            let (position, subposition) = self.find_node_info_for_index(at).unwrap();
            let mut result = InternalVector::new();
            // println!(
            //     "Split off position {} {:?}, {} {}",
            //     at, position, subposition, original_len
            // );
            // println!(
            //     "derp should be here {} {} {} {}",
            //     self.left_spine.iter().map(|x| x.len()).sum::<usize>(),
            //     self.root.len(),
            //     self.right_spine.iter().map(|x| x.len()).sum::<usize>(),
            //     self.height()
            // );
            match position {
                Some((Side::Front, node_position)) => {
                    // The left spine is has the node getting split
                    // println!("LEL {}", self.left_spine.len());
                    result.left_spine = self.left_spine.split_off(node_position + 1);
                    let mut split_node = self.left_spine.pop().unwrap();
                    result
                        .left_spine
                        .insert(0, split_node.split_at_position(subposition));
                    mem::swap(&mut self.root, &mut result.root);
                    mem::swap(&mut self.right_spine, &mut result.right_spine);
                    self.root = split_node;
                }
                None => {
                    // The root node is getting split
                    // println!("RAER {}", self.root.len());
                    mem::swap(&mut self.right_spine, &mut result.right_spine);
                    result.root = self.root.split_at_position(subposition);
                    // println!("preroflskates {:#?}", result);
                }
                Some((Side::Back, node_position)) => {
                    // The right spine is getting split
                    // println!("preroflskates {:#?}", self);

                    // Derp I think this is reversed completely
                    result.right_spine = self.right_spine.split_off(node_position + 1);
                    let mut split_node = self.right_spine.pop().unwrap();
                    // println!("preroflskates {:#?}", split_node.len());
                    mem::swap(&mut result.right_spine, &mut self.right_spine);
                    let split_right = split_node.split_at_position(subposition);
                    // println!("roflskates {:#?}", self);

                    // Problem is here I think and potentially with the split off above.
                    self.right_spine.insert(0, split_node);
                    result.right_spine.push(split_right);

                    // println!(
                    //     "now derp should be here {} {}",
                    //     self.right_spine.iter().map(|x| x.len()).sum::<usize>(),
                    //     result.right_spine.iter().map(|x| x.len()).sum::<usize>(),
                    // );
                }
            }

            // println!(
            //     "should be here {} {}",
            //     self.root.slots(),
            //     self.root.len()
            //         + self.right_spine.iter().map(|x| x.len()).sum::<usize>()
            //         + self.left_spine.iter().map(|x| x.len()).sum::<usize>()
            // );

            result.len = original_len - at;
            self.len = at;
            // println!("Lol {} vs {}", self.root.slots(), result.root.slots());
            // println!("roflskates {:#?}", self);

            self.right_spine.reverse();
            while self
                .right_spine
                .last()
                .map(|x| x.is_empty())
                .unwrap_or(false)
            {
                // println!("lul");
                self.right_spine.pop().unwrap();
            }
            self.right_spine.reverse();

            while self.root.is_empty()
                && (self.left_spine.is_empty() || self.right_spine.is_empty())
            {
                // Basically only the left branch has nodes - we pop a node from the top of the
                // left spine to serve as root. We know the left spine cannot be empty because in
                // that case we would fallen into one of the first two ifs.
                let node = if self.left_spine.is_empty() {
                    self.right_spine.pop()
                } else {
                    self.left_spine.pop()
                }
                .unwrap();
                self.root = node;
                // println!("lol a {} {}", self.root.level(), self.left_spine.len());
            }

            // println!("roflskates2 {:#?}", self);
            if self.fill_spine(Side::Back) {
                self.fixup_spine_tops();
                self.empty_leaf(Side::Back);
            }
            // if self.fill_spine(Side::Front) {
            //     self.fixup_spine_tops();
            //     self.empty_leaf(Side::Front);
            // }
            self.fixup_spine_tops();

            // println!("here");

            result.left_spine.reverse();
            while result
                .left_spine
                .last()
                .map(|x| x.is_empty())
                .unwrap_or(false)
            {
                // println!("lul");
                result.left_spine.pop().unwrap();
            }
            result.left_spine.reverse();

            while result.root.is_empty()
                && (result.left_spine.is_empty() || result.right_spine.is_empty())
            {
                // Basically only the right branch has nodes - we pop a node from the top of the
                // left spine to serve as root. We know the right spine cannot be empty because in
                // that case we would fallen into one of the first two ifs.
                let node = if result.left_spine.is_empty() {
                    result.right_spine.pop()
                } else {
                    result.left_spine.pop()
                }
                .unwrap();
                result.root = node;
                // println!("lol a {} {}", result.root.level(), result.left_spine.len());
            }

            if result.fill_spine(Side::Front) {
                result.fixup_spine_tops();
                // println!("leldongs");
                result.empty_leaf(Side::Front);
            }
            // if result.fill_spine(Side::Back) {
            //     result.fixup_spine_tops();
            //     result.empty_leaf(Side::Back);
            // }
            result.fixup_spine_tops();

            // for part in self.left_spine.iter() {
            //     println!("LS Gargablegar {}", part.level());
            // }
            // println!("RO Gargablegar {}", self.root.level());
            // for part in self.right_spine.iter() {
            //     println!("RS Gargablegar {}", part.level());
            // }

            // self.assert_invariants();
            // result.assert_invariants();

            result
        }
    }

    fn fill_spine(&mut self, side: Side) -> bool {
        let spine = match side {
            Side::Front => &mut self.left_spine,
            Side::Back => &mut self.right_spine,
        };
        // println!("Gablegar {}", spine.len());
        spine.reverse();
        let result = loop {
            match spine.last_mut().unwrap_or(&mut self.root) {
                NodeRc::Internal(internal) => {
                    //
                    // println!("Gargablegar {} {}", internal.slots(), internal.level(),);
                    let child = SharedPointer::make_mut(internal).pop_child(side);
                    spine.push(child);
                }
                NodeRc::Leaf(leaf) => {
                    // println!("Derp {}", leaf.len());
                    break leaf.is_empty();
                }
            }
        };
        spine.reverse();
        result && !spine.is_empty()
    }

    /// Inserts the item into the vector at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// v.insert(1, 4);
    /// assert_eq!(v, vector![1, 4, 2, 3]);
    /// ```
    pub fn insert(&mut self, index: usize, element: A) {
        // TODO: This is not really the most efficient way to do this, specialize this function.
        let last_part = self.split_off(index);
        // println!("RAWWWWWWWWWWWWWWWR result {:#?}", last_part);
        // println!("RAWWWWWWWWWWWWWWWR self {:#?}", self);
        self.push_back(element);
        self.append(last_part);
    }

    /// Derp
    pub fn reverse_range<R: RangeBounds<usize>>(&mut self, range: R) {
        let range = self.arbitrary_range_to_range(range);
        let mut focus = self.focus_mut();

        focus.narrow(range, |focus| {
            let half_len = focus.len() / 2;
            focus.split_at_fn(half_len, |left, right| {
                let right_len = right.len();
                for i in 0..half_len {
                    let left = left.index(i);
                    let right = right.index(right_len - 1 - i);
                    mem::swap(left, right);
                }
            })
            // let mut split_focus = focus.split_at(half_len);
        })
        // focus.get(0);
        // {
        //     let split = focus.split_at(range.end);
        // }
        // focus.get(0);
        // focus.narrow(range);
    }

    /// Derp
    pub fn reverse(&mut self) {
        self.reverse_range(..)
    }

    fn arbitrary_range_to_range<R: RangeBounds<usize>>(&self, range: R) -> Range<usize> {
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
        range_start..range_end
    }

    /// Finds the range in the given subrange of the vector that corresponds to Ordering::Equal.
    /// The given index must corresponds to a single element that compares equal. For this method to
    /// work the subrange must be sorted with respect to the given comparator.
    ///
    /// If the index does not compare equal this will return an empty range. Otherwise, this will
    /// return the range that covers the equal elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use std::cmp::Ordering;
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// let f = |x: &i32| {
    ///     if *x < 0 {
    ///         Ordering::Less
    ///     } else if *x > 5 {
    ///         Ordering::Greater
    ///     } else {
    ///         Ordering::Equal
    ///     }
    /// };
    /// assert_eq!(v.equal_range_for_index_in_range_by(1, &f, 1..), 1..6);
    /// ```
    pub fn equal_range_for_index_in_range_by<K, F, R>(
        &self,
        index: usize,
        f: &F,
        range: R,
    ) -> Range<usize>
    where
        A: Borrow<K>,
        K: ?Sized,
        F: Fn(&K) -> cmp::Ordering,
        R: RangeBounds<usize>,
    {
        let mut start = match range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => x + 1,
            Bound::Unbounded => 0,
        };

        let mut end = match range.end_bound() {
            Bound::Included(x) => x - 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.len,
        };

        if index < start || index >= end {
            return 0..0;
        }

        let mut focus = self.focus();
        if f(focus.index(index).borrow()) != cmp::Ordering::Equal {
            return 0..0;
        }

        let avg = |x: usize, y: usize| (x / 2) + (y / 2) + (x % 2 + y % 2) / 2;

        // We know that there is at least one equal that lies at index in [start, end). Our
        // goal now is to expand that midpoint to cover the entire equal range. We can only return
        // Ok from here on out.
        let mut equals_start = index;
        let mut equals_end_inclusive = index;
        // First, we'll find the first less point in the range.
        if equals_start != start {
            loop {
                // println!("rawr start {} {} {}", start, equals_start, end);
                if start + 1 == equals_start {
                    let comparison = f(focus.index(start).borrow());
                    if comparison == cmp::Ordering::Equal {
                        equals_start = start;
                    }
                    break;
                } else {
                    let mid = avg(start, equals_start);
                    let comparison = f(focus.index(mid).borrow());
                    if comparison == cmp::Ordering::Equal {
                        equals_start = mid;
                    } else {
                        start = mid;
                    }
                }
            }
        }

        // Second, we'll find the first greater point in the range.
        loop {
            // println!("rawr end {} {} {}", start, equals_end_inclusive, end);
            if equals_end_inclusive + 1 == end {
                break;
            } else {
                let mid = avg(equals_end_inclusive, end);
                let comparison = f(focus.index(mid).borrow());
                if comparison == cmp::Ordering::Equal {
                    equals_end_inclusive = mid;
                } else {
                    end = mid;
                }
            }
        }

        equals_start..equals_end_inclusive + 1
    }

    /// Finds the range in the vector that corresponds to Ordering::Equal. The given index must
    /// corresponds to a single element that compares equal. For this method to work the subrange
    /// must be sorted with respect to the given comparator.
    ///
    /// If the index does not compare equal this will return an empty range. Otherwise, this will
    /// return the range that covers the equal elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use std::cmp::Ordering;
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// let f = |x: &i32| {
    ///     if *x < 0 {
    ///         Ordering::Less
    ///     } else if *x > 5 {
    ///         Ordering::Greater
    ///     } else {
    ///         Ordering::Equal
    ///     }
    /// };
    /// assert_eq!(v.equal_range_for_index_by(1, &f), 0..6);
    /// ```
    pub fn equal_range_for_index_by<K, F>(&self, index: usize, f: &F) -> Range<usize>
    where
        A: Borrow<K>,
        K: ?Sized,
        F: Fn(&K) -> cmp::Ordering,
    {
        self.equal_range_for_index_in_range_by(index, f, ..)
    }

    /// Finds the range in the given subrange of the vector that corresponds to Ordering::Equal. For
    /// this method to work the subrange must be sorted with respect to the given comparator. If
    /// there is a range which corresponds to Ordering::Equal, the answer will be Ok(range),
    /// otherwise the result is Err(position) corresponding to the position that the maintains the
    /// sorted order of the Vector with respect to the given comparator.  
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use std::cmp::Ordering;
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// let f = |x: &i32| {
    ///     if *x < 0 {
    ///         Ordering::Less
    ///     } else if *x > 5 {
    ///         Ordering::Greater
    ///     } else {
    ///         Ordering::Equal
    ///     }
    /// };
    /// assert_eq!(v.equal_range_in_range_by(&f, 1..), Ok(1..6));
    /// ```
    pub fn equal_range_in_range_by<K, F, R>(&self, f: &F, range: R) -> Result<Range<usize>, usize>
    where
        A: Borrow<K>,
        K: ?Sized,
        F: Fn(&K) -> cmp::Ordering,
        R: RangeBounds<usize>,
    {
        let mut start = match range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => x + 1,
            Bound::Unbounded => 0,
        };

        let mut end = match range.end_bound() {
            Bound::Included(x) => x - 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.len,
        };

        let avg = |x: usize, y: usize| (x / 2) + (y / 2) + (x % 2 + y % 2) / 2;
        let mut focus = self.focus();

        // Find an equal item, we maintain an invariant that [start, end) should be the only
        // possible equal elements. We binary search down to narrow the range. We stop if we find
        // an equal item for the next step.
        loop {
            if start + 1 == end {
                let comparison = f(focus.index(start).borrow());
                match comparison {
                    cmp::Ordering::Less => {
                        return Err(start + 1);
                    }
                    cmp::Ordering::Greater => {
                        return Err(start);
                    }
                    cmp::Ordering::Equal => {
                        return Ok(start..end);
                    }
                }
            } else {
                let mid = avg(start, end);
                let comparison = f(focus.index(mid).borrow());
                match comparison {
                    cmp::Ordering::Less => {
                        start = mid;
                    }
                    cmp::Ordering::Greater => {
                        end = mid;
                    }
                    cmp::Ordering::Equal => {
                        // We know that there is at least one equal that lies in the midpoint of [start, end). Our
                        // goal now is to expand that midpoint to cover the entire equal range. We can only return
                        // Ok from here on out.
                        return Ok(self.equal_range_for_index_in_range_by(mid, f, start..end));
                    }
                }
            }
        }
    }

    /// Finds the range in the given subrange of the vector that corresponds to Ordering::Equal. For
    /// this method to work the subrange must be sorted with respect to the given comparator. If
    /// there is a range which corresponds to Ordering::Equal, the answer will be Ok(range),
    /// otherwise the result is Err(position) corresponding to the position that the maintains the
    /// sorted order of the Vector with respect to the given comparator.  
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use std::cmp::Ordering;
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// let f = |x: &i32| {
    ///     if *x < 0 {
    ///         Ordering::Less
    ///     } else if *x > 5 {
    ///         Ordering::Greater
    ///     } else {
    ///         Ordering::Equal
    ///     }
    /// };
    /// assert_eq!(v.equal_range_by(&f), Ok(0..6));
    /// ```
    pub fn equal_range_by<K, F>(&self, f: &F) -> Result<Range<usize>, usize>
    where
        A: Borrow<K>,
        K: ?Sized,
        F: Fn(&K) -> cmp::Ordering,
    {
        self.equal_range_in_range_by(f, ..)
    }

    /// Finds the range in the given subrange of the vector that is equal to the given value. The
    /// given index must corresponds to a single element that compares equal. For this method to
    /// work the subrange must be sorted with respect to the natural ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use std::cmp::Ordering;
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// assert_eq!(v.equal_range_in_range(&1, 0..3), Ok(1..3));
    /// ```
    pub fn equal_range_in_range<K, R>(&self, value: &K, range: R) -> Result<Range<usize>, usize>
    where
        A: Borrow<K>,
        K: Ord + ?Sized,
        R: RangeBounds<usize>,
    {
        let f = |x: &K| x.cmp(value);
        self.equal_range_in_range_by(&f, range)
    }

    /// Finds the range in the given subrange of the vector that corresponds to Ordering::Equal.
    /// The given index must corresponds to a single element that compares equal. For this method to
    /// work the subrange must be sorted with respect to the natural ordering.
    ///
    /// If the index does not compare equal this will return an empty range. Otherwise, this will
    /// return the range that covers the equal elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use std::cmp::Ordering;
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// assert_eq!(v.equal_range(&1), Ok(1..3));
    /// ```
    pub fn equal_range<K>(&self, value: &K) -> Result<Range<usize>, usize>
    where
        A: Borrow<K>,
        K: Ord + ?Sized,
    {
        let f = |x: &K| x.cmp(value);
        self.equal_range_in_range_by(&f, ..)
    }

    /// Finds the range in the subbrange of the vector that corresponds that is between the two
    /// given bounds. For this method to work the subrange must be sorted with respect to the
    /// natural ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use std::cmp::Ordering;
    /// # use std::ops::Bound;
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// assert_eq!(v.between_range_in_range(Bound::Included(&1), Bound::Unbounded, 0..3), Ok(1..3));
    /// ```
    pub fn between_range_in_range<K, R>(
        &self,
        start: Bound<&K>,
        end: Bound<&K>,
        range: R,
    ) -> Result<Range<usize>, usize>
    where
        A: Borrow<K>,
        K: Ord + ?Sized,
        R: RangeBounds<usize>,
    {
        let f = |x: &K| {
            match start {
                Bound::Excluded(b) => {
                    if x <= b {
                        return cmp::Ordering::Less;
                    }
                }
                Bound::Included(b) => {
                    if x < b {
                        return cmp::Ordering::Less;
                    }
                }
                _ => {}
            }
            match end {
                Bound::Excluded(b) => {
                    if x >= b {
                        return cmp::Ordering::Greater;
                    }
                }
                Bound::Included(b) => {
                    if x > b {
                        return cmp::Ordering::Greater;
                    }
                }
                _ => {}
            }
            cmp::Ordering::Equal
        };
        self.equal_range_in_range_by(&f, range)
    }

    /// Finds the range in the vector that corresponds that is between the two given bounds. For
    /// this method to work the subrange must be sorted with respect to the natural ordering.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use std::cmp::Ordering;
    /// # use std::ops::Bound;
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// assert_eq!(v.between_range(Bound::Included(&3), Bound::Unbounded), Ok(4..9));
    /// ```
    pub fn between_range<K>(&self, start: Bound<&K>, end: Bound<&K>) -> Result<Range<usize>, usize>
    where
        A: Borrow<K>,
        K: Ord + ?Sized,
    {
        self.between_range_in_range(start, end, ..)
    }

    /// Sorts a range of the sequence by the given comparator.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// v.sort_range_by(&|x: &i32, y: &i32 | (-x).cmp(&(-y)), ..);
    /// assert_eq!(v, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    /// v.sort_range_by(&Ord::cmp, 2..);
    /// assert_eq!(v, vector![9, 8, 0, 1, 2, 3, 4, 5, 6, 7]);
    /// ```
    pub fn sort_range_by<F, R>(&mut self, f: &F, range: R)
    where
        F: Fn(&A, &A) -> cmp::Ordering,
        R: RangeBounds<usize>,
    {
        let mut focus = self.focus_mut();
        focus.narrow(range, |focus| {
            let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
            do_single_sort(focus, &mut rng, f);
        });
    }

    /// Sorts a range of the sequence by the given comparator. Any swap that occurs will be made in
    /// the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort_range_by(&|x: &i32, y: &i32 | (-x).cmp(&(-y)), .., &mut secondary);
    /// assert_eq!(v, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    /// assert_eq!(secondary, vector!['j', 'i','h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']);
    /// v.dual_sort_range_by(&Ord::cmp, 2.., &mut secondary);
    /// assert_eq!(v, vector![9, 8, 0, 1, 2, 3, 4, 5, 6, 7]);
    /// assert_eq!(secondary, vector!['j', 'i', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']);
    /// ```
    pub fn dual_sort_range_by<F, R, T, Q, Internal2, Leaf2>(
        &mut self,
        f: &F,
        range: R,
        secondary: &mut InternalVector<T, Q, Internal2, Leaf2>,
    ) where
        F: Fn(&A, &A) -> cmp::Ordering,
        R: RangeBounds<usize> + Clone,
        T: Clone + Debug,
        Q: SharedPointerKind,
        Internal2: InternalTrait<Q, Leaf2, Item = T>,
        Leaf2: LeafTrait<Item = T>,
    {
        let mut focus = self.focus_mut();
        focus.narrow(range.clone(), |focus| {
            let mut dual = secondary.focus_mut();
            dual.narrow(range.clone(), |dual| {
                let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
                do_dual_sort(focus, dual, &mut rng, &f);
            });
        });
    }

    /// Sorts a range of the sequence by the given comparator.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// v.sort_range_by_key(&|&x| -x, ..);
    /// assert_eq!(v, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    /// v.sort_range_by_key(&|&x| x, 2..);
    /// assert_eq!(v, vector![9, 8, 0, 1, 2, 3, 4, 5, 6, 7]);
    /// ```
    pub fn sort_range_by_key<F: Fn(&A) -> K, K: Ord, R: RangeBounds<usize>>(
        &mut self,
        f: &F,
        range: R,
    ) {
        let mut focus = self.focus_mut();
        focus.narrow(range, |focus| {
            let comp = |x: &A, y: &A| f(x).cmp(&f(y));
            let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
            do_single_sort(focus, &mut rng, &comp);
        });
    }

    /// Sorts a range of the sequence by the given comparator. Any swap that occurs will be made in
    /// the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort_range_by_key(&|&x| -x, .., &mut secondary);
    /// assert_eq!(v, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    /// assert_eq!(secondary, vector!['j', 'i','h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']);
    /// v.dual_sort_range_by_key(&|&x| x, 2.., &mut secondary);
    /// assert_eq!(v, vector![9, 8, 0, 1, 2, 3, 4, 5, 6, 7]);
    /// assert_eq!(secondary, vector!['j', 'i', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']);
    /// ```
    pub fn dual_sort_range_by_key<F, K, R, T, Q, Internal2, Leaf2>(
        &mut self,
        f: &F,
        range: R,
        secondary: &mut InternalVector<T, Q, Internal2, Leaf2>,
    ) where
        F: Fn(&A) -> K,
        K: Ord,
        R: RangeBounds<usize> + Clone,
        T: Debug + Clone,
        Q: SharedPointerKind,
        Internal2: InternalTrait<Q, Leaf2, Item = T>,
        Leaf2: LeafTrait<Item = T>,
    {
        let comp = |x: &A, y: &A| f(x).cmp(&f(y));
        let mut focus = self.focus_mut();
        focus.narrow(range.clone(), |focus| {
            let mut dual = secondary.focus_mut();
            dual.narrow(range.clone(), |dual| {
                let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
                do_dual_sort(focus, dual, &mut rng, &comp);
            });
        });
    }

    /// Sorts the entire sequence by the given comparator.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// v.sort_by(&|x: &i32, y: &i32 | (-x).cmp(&(-y)));
    /// assert_eq!(v, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    /// v.sort_by(&Ord::cmp);
    /// assert_eq!(v, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// ```
    pub fn sort_by<F: Fn(&A, &A) -> cmp::Ordering>(&mut self, f: &F) {
        self.sort_range_by(f, ..);
    }

    /// Sorts the entire sequence by the given comparator. Any swap that occurs will be made in
    /// the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort_by(&|x: &i32, y: &i32 | (-x).cmp(&(-y)), &mut secondary);
    /// assert_eq!(v, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    /// v.dual_sort_by(&Ord::cmp, &mut secondary);
    /// assert_eq!(v, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// ```
    pub fn dual_sort_by<F, T, Q, Internal2, Leaf2>(
        &mut self,
        f: &F,
        secondary: &mut InternalVector<T, Q, Internal2, Leaf2>,
    ) where
        F: Fn(&A, &A) -> cmp::Ordering,
        T: Debug + Clone,
        Q: SharedPointerKind,
        Internal2: InternalTrait<Q, Leaf2, Item = T>,
        Leaf2: LeafTrait<Item = T>,
    {
        self.dual_sort_range_by(f, .., secondary);
    }

    /// Removes item from the vector at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// v.remove(1);
    /// assert_eq!(v, vector![1, 3]);
    /// ```
    pub fn remove(&mut self, index: usize) -> Option<A> {
        // TODO: This is not really the most efficient way to do this, specialize this function
        if index < self.len {
            let mut last_part = self.split_off(index);
            let item = last_part.pop_front();
            self.append(last_part);
            item
        } else {
            None
        }
    }

    /// Returns the height of the tree. This is only used for debugging purposes.const
    #[allow(dead_code)]
    pub(crate) fn height(&self) -> usize {
        debug_assert_eq!(self.left_spine.len(), self.right_spine.len());
        self.left_spine.len()
    }

    /// Returns a reference the spine of the requested side of the tree.
    fn spine_ref(&self, side: Side) -> &Vec<NodeRc<A, P, Internal, Leaf>> {
        match side {
            Side::Front => &self.left_spine,
            Side::Back => &self.right_spine,
        }
    }

    /// Returns a mutable reference the spine of the requested side of the tree.
    fn spine_mut(&mut self, side: Side) -> &mut Vec<NodeRc<A, P, Internal, Leaf>> {
        match side {
            Side::Front => &mut self.left_spine,
            Side::Back => &mut self.right_spine,
        }
    }

    /// Returns a reference to the leaf on the requested side of the tree.
    fn leaf_ref(&self, side: Side) -> &SharedPointer<Leaf, P> {
        self.spine_ref(side)
            .first()
            .unwrap_or(&self.root)
            .leaf_ref()
    }

    /// Returns a mutable reference to the leaf on the requested side of the tree.
    fn leaf_mut(&mut self, side: Side) -> &mut SharedPointer<Leaf, P> {
        match side {
            Side::Front => self
                .left_spine
                .first_mut()
                .unwrap_or(&mut self.root)
                .leaf_mut(),
            Side::Back => self
                .right_spine
                .first_mut()
                .unwrap_or(&mut self.root)
                .leaf_mut(),
        }
    }

    /// Returns a focus over the vector. A focus tracks the last leaf and positions which was read.
    /// The path down this tree is saved in the focus and is used to accelerate lookups in nearby
    /// locations.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let v = vector![1, 2, 3];
    /// let mut f = v.focus();
    /// assert_eq!(f.get(0), Some(&1));
    /// assert_eq!(f.get(1), Some(&2));
    /// assert_eq!(f.get(2), Some(&3));
    /// ```
    pub fn focus(&self) -> Focus<A, P, Internal, Leaf> {
        Focus::new(self)
    }

    /// Returns a mutable focus over the vector. A focus tracks the last leaf and positions which
    /// was read. The path down this tree is saved in the focus and is used to accelerate lookups in
    /// nearby locations.
    pub fn focus_mut(&mut self) -> FocusMut<A, P, Internal, Leaf> {
        let mut nodes = Vec::new();
        for node in self.left_spine.iter_mut() {
            if !node.is_empty() {
                nodes.push(node.borrow_node());
            }
        }
        if !self.root.is_empty() {
            nodes.push(self.root.borrow_node());
        }
        for node in self.right_spine.iter_mut().rev() {
            if !node.is_empty() {
                nodes.push(node.borrow_node());
            }
        }
        FocusMut::from_vectors(vec![Rc::new(self)], nodes)
    }

    /// Returns a mutable focus over the vector. A focus tracks the last leaf and positions which
    /// was read. The path down this tree is saved in the focus and is used to accelerate lookups in
    /// nearby locations.
    pub fn focus_mut_fn<F>(&mut self, f: &F)
    where
        F: Fn(&mut FocusMut<A, P, Internal, Leaf>),
    {
        let mut nodes = Vec::new();
        for node in self.left_spine.iter_mut() {
            nodes.push(node.borrow_node());
        }
        nodes.push(self.root.borrow_node());
        for node in self.right_spine.iter_mut().rev() {
            nodes.push(node.borrow_node());
        }
        let mut focus = FocusMut::from_vectors(vec![Rc::new(self)], nodes);
        f(&mut focus);
    }

    /// Returns an iterator over the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let v = vector![1, 2, 3];
    /// let mut iter = v.iter();
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<A, P, Internal, Leaf> {
        Iter {
            front: 0,
            back: self.len(),
            focus: self.focus(),
        }
    }

    /// Returns a mutable iterator over the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// let mut iter = v.iter_mut();
    /// assert_eq!(iter.next(), Some(&mut 1));
    /// assert_eq!(iter.next(), Some(&mut 2));
    /// assert_eq!(iter.next(), Some(&mut 3));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<A, P, Internal, Leaf> {
        IterMut {
            front: 0,
            back: self.len(),
            focus: self.focus_mut(),
        }
    }

    /// Checks the internal invariants that are required by the Vector.
    #[allow(dead_code)]
    pub(crate) fn assert_invariants(&self) -> bool {
        // Invariant 1
        // Spines must be of equal length
        assert_eq!(self.left_spine.len(), self.right_spine.len());

        // Invariant 2
        // All nodes in the spine except for the first one (the leaf) must have at least 1 slot free.
        for spine in self.left_spine.iter().skip(1) {
            assert!(spine.free_slots() >= 1);
        }
        for spine in self.right_spine.iter().skip(1) {
            assert!(spine.free_slots() >= 1);
        }

        // Invariant 3
        // The first node(the leaf) in the spine must have at least 1 element, but may be full
        if let Some(leaf) = self.left_spine.first() {
            assert!(leaf.slots() >= 1);
        }
        if let Some(leaf) = self.right_spine.first() {
            assert!(leaf.slots() >= 1);
        }

        // Invariant 4
        // If the root is a non-leaf, it must always have at least 2 slot free, but may be empty
        if !self.root.is_leaf() {
            assert!(self.root.free_slots() >= 2);
        }

        // Invariant 5
        // If the root is an empty non-leaf node then the last two nodes of both spines:
        // 1) Must not be able to be merged into a node of 1 less height
        // 2) Must differ in slots by at most one node
        if self.root.is_empty() && !self.root.is_leaf() {
            let left_children = self.left_spine.last().unwrap().slots();
            let right_children = self.right_spine.last().unwrap().slots();

            let difference = if left_children > right_children {
                left_children - right_children
            } else {
                right_children - left_children
            };

            assert!(difference <= 1);

            let min_children = if self.root.level() == 1 {
                // The root is one above a leaf and a new leaf root could be completely full
                RRB_WIDTH
            } else {
                // New non-leaf roots must contain at least 2 empty slots
                RRB_WIDTH - 2
            };

            assert!(left_children + right_children >= min_children);
        }

        // Invariant 6
        // The spine nodes must have their RRB invariants fulfilled
        for (level, spine) in self.left_spine.iter().enumerate() {
            spine.debug_check_invariants(spine.len(), level);
        }
        self.root
            .debug_check_invariants(self.root.len(), self.left_spine.len());
        for (level, spine) in self.right_spine.iter().enumerate() {
            spine.debug_check_invariants(spine.len(), level);
        }

        // Invariant 7
        // The tree's `len` field must match the sum of the spine's lens
        let left_spine_len = self.left_spine.iter().map(|x| x.len()).sum::<usize>();
        let root_len = self.root.len();
        let right_spine_len = self.right_spine.iter().map(|x| x.len()).sum::<usize>();

        // println!(
        //     "derpledoo {} {} {}",
        //     left_spine_len, root_len, right_spine_len
        // );
        assert_eq!(self.len, left_spine_len + root_len + right_spine_len);
        true
    }
}

impl<A, P, Internal, Leaf> InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug + Ord,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    /// Sorts the entire sequence by the natural comparator on the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    /// v.sort();
    /// assert_eq!(v, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// ```
    pub fn sort(&mut self) {
        self.sort_by(&Ord::cmp)
    }

    /// Sorts the entire sequence by the natural comparator on the sequence. Any swap that occurs
    /// will be made in the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort(&mut secondary);
    /// assert_eq!(v, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    /// assert_eq!(secondary, vector!['j', 'i','h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']);
    /// ```
    pub fn dual_sort<T, Q, Internal2, Leaf2>(
        &mut self,
        secondary: &mut InternalVector<T, Q, Internal2, Leaf2>,
    ) where
        T: Debug + Clone,
        Q: SharedPointerKind,
        Internal2: InternalTrait<Q, Leaf2, Item = T>,
        Leaf2: LeafTrait<Item = T>,
    {
        self.dual_sort_by(&Ord::cmp, secondary)
    }

    /// Sorts the range of the sequence by the natural comparator on the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    /// v.sort_range(5..);
    /// assert_eq!(v, vector![9, 8, 7, 6, 5, 0, 1, 2, 3, 4]);
    /// ```
    pub fn sort_range<R>(&mut self, range: R)
    where
        R: RangeBounds<usize>,
    {
        self.sort_range_by(&Ord::cmp, range)
    }

    /// Sorts the range of the sequence by the natural comparator on the sequence. Any swap that
    /// occurs will be made in the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort_range(5.., &mut secondary);
    /// assert_eq!(v, vector![9, 8, 7, 6, 5, 0, 1, 2, 3, 4]);
    /// assert_eq!(secondary, vector!['a', 'b', 'c', 'd', 'e', 'j', 'i','h', 'g', 'f']);
    /// ```
    pub fn dual_sort_range<R, T, Q, Internal2, Leaf2>(
        &mut self,
        range: R,
        secondary: &mut InternalVector<T, Q, Internal2, Leaf2>,
    ) where
        R: RangeBounds<usize> + Clone,
        T: Debug + Clone,
        Q: SharedPointerKind,
        Internal2: InternalTrait<Q, Leaf2, Item = T>,
        Leaf2: LeafTrait<Item = T>,
    {
        self.dual_sort_range_by(&Ord::cmp, range, secondary)
    }
}

impl<A, P, Internal, Leaf> InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug + PartialEq,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    /// Tests whether the node is equal to the given vector. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) fn equal_vec(&self, v: &Vec<A>) -> bool {
        if self.len() == v.len() {
            let mut iter = v.iter();
            for spine in self.left_spine.iter() {
                if !spine.equal_iter_debug(&mut iter) {
                    println!("Left: {:?} {:?}", self, v);
                    return false;
                }
            }
            if !self.root.equal_iter_debug(&mut iter) {
                println!("Root: {:?} {:?}", self, v);
                return false;
            }
            for spine in self.right_spine.iter().rev() {
                if !spine.equal_iter_debug(&mut iter) {
                    println!("Right: {:?} {:?}", self, v);
                    return false;
                }
            }
            true
        } else {
            false
        }
    }
}

impl<A, P, Internal, Leaf> Default for InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, P, Internal, Leaf> PartialEq for InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug + PartialEq,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<A, P, Internal, Leaf> Eq for InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug + Eq,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
}

impl<A, P, Internal, Leaf> PartialOrd for InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug + PartialOrd,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<A, P, Internal, Leaf> Ord for InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug + Ord,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<A, P, Internal, Leaf> Hash for InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug + Hash,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in self {
            i.hash(state)
        }
    }
}

impl<A, P, Internal, Leaf> FromIterator<A> for InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    fn from_iter<I: IntoIterator<Item = A>>(iter: I) -> Self {
        let mut result = InternalVector::default();
        for item in iter {
            result.push_back(item);
        }
        result
    }
}

/// derp
pub type Vector<A> = InternalVector<A, RcK, Internal<A, RcK, Leaf<A>>, Leaf<A>>;
/// derp
pub type ThreadSafeVector<A> = InternalVector<A, ArcK, Internal<A, ArcK, Leaf<A>>, Leaf<A>>;

/// An iterator for a Vector.
#[derive(Clone, Debug)]
pub struct Iter<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    front: usize,
    back: usize,
    focus: Focus<'a, A, P, Internal, Leaf>,
}

impl<'a, A, P, Internal, Leaf> Iterator for Iter<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    type Item = &'a A;

    fn next(&mut self) -> Option<&'a A> {
        // This focus is broken
        if self.front != self.back {
            let focus: &'a mut Focus<A, P, Internal, Leaf> =
                unsafe { &mut *(&mut self.focus as *mut _) };
            let result = focus.get(self.front).unwrap();
            self.front += 1;
            Some(result)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl<'a, A, P, Internal, Leaf> IntoIterator for &'a InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    type Item = &'a A;
    type IntoIter = Iter<'a, A, P, Internal, Leaf>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, A, P, Internal, Leaf> DoubleEndedIterator for Iter<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    fn next_back(&mut self) -> Option<&'a A> {
        if self.front != self.back {
            self.back -= 1;
            let focus: &'a mut Focus<A, P, Internal, Leaf> =
                unsafe { &mut *(&mut self.focus as *mut _) };
            focus.get(self.back)
        } else {
            None
        }
    }
}

impl<'a, A, P, Internal, Leaf> ExactSizeIterator for Iter<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
}

impl<'a, A, P, Internal, Leaf> FusedIterator for Iter<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
}

/// An iterator for a Vector.
#[derive(Debug)]
pub struct IterMut<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    front: usize,
    back: usize,
    focus: FocusMut<'a, A, P, Internal, Leaf>,
}

impl<'a, A, P, Internal, Leaf> Iterator for IterMut<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    type Item = &'a mut A;

    fn next(&mut self) -> Option<&'a mut A> {
        if self.front != self.back {
            let focus: &'a mut FocusMut<A, P, Internal, Leaf> =
                unsafe { &mut *(&mut self.focus as *mut _) };
            let result = focus.get(self.front).unwrap();
            self.front += 1;
            Some(result)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl<'a, A, P, Internal, Leaf> IntoIterator for &'a mut InternalVector<A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    type Item = &'a mut A;
    type IntoIter = IterMut<'a, A, P, Internal, Leaf>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, A, P, Internal, Leaf> DoubleEndedIterator for IterMut<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
    fn next_back(&mut self) -> Option<&'a mut A> {
        if self.front != self.back {
            self.back -= 1;
            let focus: &'a mut FocusMut<A, P, Internal, Leaf> =
                unsafe { &mut *(&mut self.focus as *mut _) };
            focus.get(self.back)
        } else {
            None
        }
    }
}

impl<'a, A, P, Internal, Leaf> ExactSizeIterator for IterMut<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
}

impl<'a, A, P, Internal, Leaf> FusedIterator for IterMut<'a, A, P, Internal, Leaf>
where
    A: Clone + Debug + 'a,
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf, Item = A>,
    Leaf: LeafTrait<Item = A>,
{
}

#[allow(clippy::cognitive_complexity)]
#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;
    use proptest::proptest;
    use proptest_derive::Arbitrary;

    const MAX_EXTEND_SIZE: usize = 1000;

    #[derive(Arbitrary)]
    enum Action<A: Clone + Debug + Arbitrary + 'static> {
        PushFront(A),
        PushBack(A),
        #[proptest(
            strategy = "prop::collection::vec(any::<A>(), 0..MAX_EXTEND_SIZE).prop_map(Action::ExtendFront)"
        )]
        ExtendFront(Vec<A>),
        #[proptest(
            strategy = "prop::collection::vec(any::<A>(), 0..MAX_EXTEND_SIZE).prop_map(Action::ExtendBack)"
        )]
        ExtendBack(Vec<A>),
        SplitLeft(usize),
        SplitRight(usize),
        #[proptest(
            strategy = "prop::collection::vec(any::<A>(), 0..MAX_EXTEND_SIZE).prop_map(Action::ConcatFront)"
        )]
        ConcatFront(Vec<A>),
        #[proptest(
            strategy = "prop::collection::vec(any::<A>(), 0..MAX_EXTEND_SIZE).prop_map(Action::ConcatBack)"
        )]
        ConcatBack(Vec<A>),
    }

    impl<A: Clone + Debug + Arbitrary + 'static> std::fmt::Debug for Action<A> {
        fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
            match self {
                Action::PushFront(item) => {
                    fmt.write_str(&format!("vector.push_front({:?});\n", item))?;
                }
                Action::PushBack(item) => {
                    fmt.write_str(&format!("vector.push_back({:?});\n", item))?;
                }
                Action::ExtendFront(items) => {
                    fmt.write_str(&format!(
                        "let v: Vec<u64> = vec!{:?}; for item in v.into_iter() {{ vector.push_front(item); }}\n",
                        items
                    ))?;
                }
                Action::ExtendBack(items) => {
                    fmt.write_str(&format!(
                        "let v: Vec<u64> = vec!{:?}; for item in v.into_iter() {{ vector.push_back(item); }}\n",
                        items
                    ))?;
                }
                Action::SplitLeft(index) => {
                    fmt.write_str(&format!(
                        "let index = {:?} % (1 + vector.len()); vector.slice_from_start(index);\n",
                        index
                    ))?;
                }
                Action::SplitRight(index) => {
                    fmt.write_str(&format!(
                        "let index = {:?} % (1 + vector.len()); vector.slice_to_end(index);\n",
                        index
                    ))?;
                }
                Action::ConcatFront(items) => {
                    fmt.write_str(&format!(
                        "let v: Vec<u64> = vec!{:?}; let mut new_vector = Vector::new(); for item in v.into_iter() {{ new_vector.push_front(item); }} new_vector.append(vector); vector = new_vector; \n",
                        items
                    ))?;
                }
                Action::ConcatBack(items) => {
                    fmt.write_str(&format!(
                        "let v: Vec<u64> = vec!{:?}; let mut new_vector = Vector::new(); for item in v.into_iter() {{ new_vector.push_back(item); }} vector.append(new_vector); \n",
                        items
                    ))?;
                }
            }
            Ok(())
        }
    }

    #[derive(Arbitrary)]
    struct ActionList<A: Clone + Debug + Arbitrary + 'static> {
        actions: Vec<Action<A>>,
    }

    impl<A: Clone + Debug + Arbitrary + 'static> std::fmt::Debug for ActionList<A> {
        fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
            fmt.write_str("let mut vector = Vector::new();\n")?;
            for action in &self.actions {
                fmt.write_str(&format!(
                    "{:?};\nassert!(vector.assert_invariants());\n",
                    action
                ))?;
            }
            Ok(())
        }
    }

    proptest! {
        #[test]
        fn random_u64(actions: ActionList<u64>) {
            let mut vec: Vec<u64> = Vec::new();
            let mut vector: Vector<u64> = Vector::new();

            for action in &actions.actions {
                match action {
                    Action::PushFront(item) => {
                        vec.insert(0, item.clone());
                        vector.push_front(item.clone());
                        assert_eq!(vec.len(), vector.len());
                        assert!(vector.equal_vec(&vec));
                    },
                    Action::PushBack(item) => {
                        vec.push(item.clone());
                        vector.push_back(item.clone());
                        assert_eq!(vec.len(), vector.len());
                        assert!(vector.equal_vec(&vec));
                    },
                    Action::ExtendFront(items) => {
                        for item in items {
                            vec.insert(0, item.clone());
                            vector.push_front(item.clone());
                        }
                        assert_eq!(vec.len(), vector.len());
                        assert!(vector.equal_vec(&vec));
                    }
                    Action::ExtendBack(items) => {
                        for item in items {
                            vec.push(item.clone());
                            vector.push_back(item.clone());
                        }
                        assert_eq!(vec.len(), vector.len());
                        assert!(vector.equal_vec(&vec));
                    }
                    Action::SplitLeft(index) => {
                        let index = index % (1 + vec.len());
                        vec.truncate(index);
                        vector.slice_from_start(index);
                        assert_eq!(vec.len(), vector.len());
                        assert!(vector.equal_vec(&vec));
                    }
                    Action::SplitRight(index) => {
                        let index = index % (1 + vec.len());
                        vec = vec.split_off(index);
                        vector.slice_to_end(index);
                        assert_eq!(vec.len(), vector.len());
                        assert!(vector.equal_vec(&vec));
                    }
                    Action::ConcatFront(items) => {
                        let mut new_vector = Vector::new();
                        for item in items {
                            vec.insert(0, item.clone());
                            new_vector.push_front(item.clone());
                        }
                        new_vector.append(vector);
                        vector = new_vector;
                        assert_eq!(vec.len(), vector.len());
                        assert!(vector.equal_vec(&vec));
                    }
                    Action::ConcatBack(items) => {
                        let mut new_vector = Vector::new();
                        for item in items {
                            vec.push(item.clone());
                            new_vector.push_back(item.clone());
                        }
                        vector.append(new_vector);
                        assert_eq!(vec.len(), vector.len());
                        assert!(vector.equal_vec(&vec));
                    }
                }
                assert!(vector.assert_invariants());
            }

            assert_eq!(
                vector.iter().cloned().collect::<Vec<_>>(),
                vec
            );
            assert_eq!(
                vector.iter().rev().cloned().collect::<Vec<_>>(),
                vec.iter().rev().cloned().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    pub fn empty() {
        let empty: Vector<usize> = Vector::new();
        // let empty_vec: Vec<usize> = Vec::new();
        // let empty_ref_vec: Vec<&usize> = Vec::new();
        // let empty_ref_mut_vec: Vec<&mut usize> = Vec::new();

        // Len
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        // Back
        assert_eq!(empty.back(), None);
        assert_eq!(empty.front(), None);

        // Concat
        let mut empty_concat = empty.clone();
        empty_concat.append(empty.clone());
        assert!(empty_concat.is_empty());
        assert_eq!(empty_concat.len(), 0);

        // Slice
        let mut empty_slice_left = empty.clone();
        empty_slice_left.slice_from_start(10);
        assert!(empty_slice_left.is_empty());
        assert_eq!(empty_slice_left.len(), 0);

        let mut empty_slice_right = empty.clone();
        empty_slice_right.slice_to_end(10);
        assert!(empty_slice_right.is_empty());
        assert_eq!(empty_slice_right.len(), 0);

        // Iter
        // assert_eq!(empty.iter().collect::<Vec<_>>(), empty_ref_vec);
        // assert_eq!(empty.iter_mut().collect::<Vec<_>>(), empty_ref_mut_vec);
        // assert_eq!(empty.into_iter().collect::<Vec<_>>(), empty_vec);
    }

    #[test]
    pub fn single() {
        let mut item = 9;
        let mut single = Vector::new();
        single.push_back(item);

        // Len
        assert!(!single.is_empty());
        assert_eq!(single.len(), 1);

        // Back
        assert_eq!(single.back(), Some(&item));
        assert_eq!(single.back_mut(), Some(&mut item));
        let mut back = single.clone();
        assert_eq!(back.pop_back(), Some(item));
        assert_eq!(back.pop_back(), None);
        assert_eq!(back.back(), None);
        assert_eq!(back.back_mut(), None);

        // Front
        assert_eq!(single.front(), Some(&item));
        assert_eq!(single.front_mut(), Some(&mut item));
        let mut front = single.clone();
        assert_eq!(front.pop_front(), Some(item));
        assert_eq!(front.pop_front(), None);

        // Iter
        // assert_eq!(
        //     single.iter().collect::<Vec<_>>(),
        //     vec.iter().collect::<Vec<_>>()
        // );
        // assert_eq!(
        //     single.iter_mut().collect::<Vec<_>>(),
        //     vec.iter_mut().collect::<Vec<_>>()
        // );
        // assert_eq!(single.into_iter().collect::<Vec<_>>(), vec);
    }

    #[test]
    pub fn large() {
        const N: usize = 10000;
        let mut vec = Vector::new();
        for i in 0..N {
            vec.push_back(i);
        }

        // Len
        assert!(!vec.is_empty());
        assert_eq!(vec.len(), N);

        // Back
        assert_eq!(vec.back(), Some(&(N - 1)));
        assert_eq!(vec.back_mut(), Some(&mut (N - 1)));
        assert_eq!(vec.clone().pop_back(), Some(N - 1));

        // Front
        assert_eq!(vec.front(), Some(&0));
        assert_eq!(vec.front_mut(), Some(&mut 0));
        assert_eq!(vec.clone().pop_front(), Some(0));

        // Iter
        assert_eq!(
            vec.iter().collect::<Vec<_>>(),
            (0..N).collect::<Vec<_>>().iter().collect::<Vec<_>>()
        );
        // assert_eq!(
        //     vec.iter_mut().collect::<Vec<_>>(),
        //     (0..N).collect::<Vec<_>>()
        // );
        // assert_eq!(
        //     vec.into_iter().collect::<Vec<_>>(),
        //     (0..N).collect::<Vec<_>>()
        // );

        assert_eq!(
            vec.iter().rev().collect::<Vec<_>>(),
            (0..N).rev().collect::<Vec<_>>().iter().collect::<Vec<_>>()
        );
    }

    #[test]
    pub fn inserts() {
        let mut v = Vector::new();
        const N: usize = 1_000;
        for i in 0..N {
            // println!("{} inserted {}", i, v.height());
            v.insert(v.len() / 2, i);
            v.assert_invariants();

            let first_half = (1..i + 1).step_by(2);
            let second_half = (0..i + 1).step_by(2).rev();

            let mut vector = Vec::new();
            vector.extend(first_half);
            vector.extend(second_half);

            assert_eq!(v.iter().copied().collect::<Vec<usize>>(), vector);
            // println!("\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
        }
        let first_half = (1..N).step_by(2);
        let second_half = (0..N).step_by(2).rev();

        let mut vector = Vec::new();
        vector.extend(first_half);
        vector.extend(second_half);

        assert_eq!(v.iter().copied().collect::<Vec<usize>>(), vector);

        println!("{} {}", v.len(), v.height());
    }

    // #[test]
    // pub fn inserts_2() {
    //     let mut v = Vector::new();
    //     v.push_front(N);
    //     const N: usize = 1_000;
    //     for i in 0..N {
    //         v.insert(v.len() - 1, i);
    //         // println!("{} inserted\nDerplcakes {:#?}", i, v);
    //         v.assert_invariants();

    //         let mut vector = Vec::new();
    //         vector.extend(0..=i);
    //         vector.push(N);

    //         assert_eq!(v.iter().copied().collect::<Vec<usize>>(), vector);
    //         // println!("\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    //     }

    //     let mut vector = Vec::new();
    //     vector.extend(0..=N);

    //     assert_eq!(v.iter().copied().collect::<Vec<usize>>(), vector);

    //     println!("{} {}", v.len(), v.height());
    // }

    // #[test]
    // pub fn inserts_3() {
    //     let mut v = Vector::new();
    //     v.push_front(N);
    //     const N: usize = 100_000;
    //     for i in 0..N {
    //         v.insert(v.len()/2 - 1 - v.len() * 3 / 10, i);
    //         // println!("{} inserted\nDerplcakes {:#?}", i, v);
    //         v.assert_invariants();

    //         // println!("\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
    //     }
    // }

    #[test]
    fn test_equal_range() {
        let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
        assert_eq!(v.equal_range_in_range(&1, 0..3), Ok(1..3));
    }

    #[test]
    fn derp() {
        let mut vector = Vector::new();
        let v: Vec<u64> = vec![
            3782376438472279830,
            3782922383321040222,
            9282467348645230316,
            3099682068311206051,
            1579867689742461368,
            15763854684231622942,
            8478031543850908874,
            9771177525789698552,
            17071481029433700912,
            16495450046226436928,
            1435452230470457810,
            3459434357523641823,
            10912806810360342487,
            6388890661507589310,
            11220771608699319407,
            14962476607357223507,
            4044052869563950522,
            12100703115645442783,
            3061627102666813574,
            1303660860479587040,
            18052049599201723469,
            7790155925008150298,
            6497210909027768307,
            10724298534206028198,
            12553041629362007148,
            2985382166385362587,
            17322771116397830763,
            17815584259001579191,
            2407529284001510763,
            8165109969070529034,
            12299271562744609494,
            4468670366244300222,
            12885545298687542679,
            17196077174293631528,
            12922178449888656013,
            13460258177945758240,
            11756304873585121972,
            11277106263818164467,
            16289993095261915322,
            16961839671877147001,
            15522352186349813080,
            12004579060504449554,
            2822696551551401203,
            11769167359475095742,
            1643734768596063982,
            8781974533696821872,
            11926634024207784591,
            16810180787088795528,
            2412379648959361,
            10941374325817662326,
            5517590926889916921,
            2822183863384122029,
            12966841574061428326,
            5471014506630285366,
            14855495498756446035,
            4854315042952101030,
            12087088905599049308,
            11448212720402121029,
            814851498915682294,
            17712272321578261325,
            10559746817774371805,
            2751288828716945314,
            15304468296258813137,
            13123800798738361748,
            1565361455893387832,
            9666046116143062250,
            15020603692772545401,
            4104204252855814172,
            7091168475130325164,
            8121305411212429683,
            1696157524897060888,
            8071440656714601733,
            8143865656991994026,
            15143489779775240070,
            1643322682458618830,
            2748715743596354897,
            18215759204414345910,
            6133350227839325531,
            2959259052004406129,
            5633983209238118081,
            17791121286219326104,
            10679824508945380246,
            4916523836370472463,
            4817118761377560525,
            853371412129095278,
            14278057659042286409,
            10052019162383264060,
            503097587294928496,
            6007024108111107884,
            16305217383707291787,
            15640708511508090130,
            132058931224418773,
            10783099465462396073,
            9959356657155635224,
            17984576300613834771,
            17007512907884295233,
            3160944148312269081,
            1702579551530164333,
            6352485010677663861,
            8108554129076006496,
            7307500194835459126,
            8890265486292586318,
            548298931175779794,
            6215219229782960558,
            3097959753416923588,
            11681588253486930492,
            10619415920008073443,
            18017505358027603571,
            464182480690816516,
            14711854035002053501,
            17041303977383776320,
            2509502456926675618,
            14052603766058698387,
            8283656906377566215,
            11252063869667845247,
            6651285654450917056,
            12429874562808927854,
            3740074772011824820,
            17031909195597759584,
            16650529860575592628,
            10158820219924664926,
            8118855999389216094,
            1594578878439969586,
            16696259986100499316,
            12073286198253977802,
            9390269734638805854,
            14418497895046952710,
            2893773741461195371,
            5501935428940100539,
            6475858917714420179,
            4539251328336683234,
            3246956049926672289,
            12942748780269417333,
            5184113339870467058,
            2157677096938683950,
            9519029142747719102,
            8913577883105455040,
            2619873782609286945,
            6529017713091421761,
            4973972653569687158,
            2027130133477790830,
            3292601449199309005,
            11754043973224876347,
            5062387157598573474,
            15245233824791372597,
            16215158932157101938,
            16987906708608369685,
            2565888053518665369,
            14129843420214176692,
            8734975249583675726,
            15552893264903468913,
            6393381157685382362,
            17748816613274659558,
            11497069080618096135,
            11513019017550400669,
            10008120605087927393,
            12002088813575470527,
            15490809513483580988,
            10070622692746797435,
            15901971336399947513,
            8787768835203547385,
            1588494274210529485,
            48343600836584298,
            9481867381664702461,
            5331771206927416880,
            1633012265364294048,
            5791378921304401739,
            1664101071627373828,
            3280585592547456834,
            3009017445054562602,
            9291386294035679580,
            1427628055706610084,
            3599270500426517441,
            4536652015335756140,
            10708025700759991459,
            8345672092343042817,
            4404773657859975456,
            4906040932555227342,
            13215687674139885180,
            14616692885678685967,
            3248896832748476097,
            9965694985629988610,
            2404589415050438428,
            248984427643802056,
            13293257229312124690,
            14718267802829825579,
            1823885236015743263,
            6744238965871293502,
            17650430673779723661,
            5140526133450964624,
            10122752221861123653,
            3475108347221467557,
            4148388014359801872,
            16418912754668967381,
            13374655991551259962,
            6011167696416176378,
            14871460903518980569,
            8304762689570960719,
            11748433051276145753,
            3710679614872378133,
            8319553819724450921,
            5696742022196112073,
            12542884821347139597,
            16694789535804174369,
            7384585492335091525,
            10730033083560985231,
            6507954285378690192,
            17354627209016795244,
            17928190375262985259,
            17927954536998425765,
            6365832244212973651,
            4719564616655178371,
            12587620708066923952,
            13804896495477757007,
            14031938082755600371,
            707329641929719957,
            3334832893503441765,
            13334367348984900572,
            15478508038784457424,
            11086197077342469826,
            7967534256442193482,
            17446577318267990324,
            16748704092948352898,
            9869123496825264781,
            3406492985349421808,
            1517966561160674502,
            5775923768107397771,
            3805424921387300629,
            5451897756243538750,
            16459652630846489887,
            11389604915846540512,
            16919000466051824364,
            8277366558872647000,
            5220714013256195064,
            12065855448899325598,
            14265383866857981348,
            10488261572099883121,
            1199664225605024069,
            11357609003118436227,
            14945622315433343830,
            8950586039268808586,
            656313335572769677,
            10039246036343283079,
            13510834317074095715,
            10049209905234229903,
            16314871767425323623,
            1350447828025263712,
            8686603842667953820,
            14941394901520314326,
            6329776644598104355,
            5016728057468729371,
            15223625649639840872,
            13893738578231991575,
            13030498319652639060,
            9400251214610533656,
            8199175920607390459,
            9869307682752024443,
            10875172481931942482,
            16699254606567929010,
            14243755604167426258,
            1331170194598108110,
            15587629985365814714,
            14179000381137628427,
            4208057643110237259,
            8276896023142811696,
            10194444132323498224,
            16937292574420650944,
            6200942384474907973,
            15483212967298865764,
            17038514676364690837,
            17392635806741962563,
            9709495803924554142,
            1152695705125183284,
            11043061349208711677,
            11993318855290401700,
            13988016296923983415,
            9030492901754419224,
            4435695684988866369,
            848702110994333167,
            12868458703951307552,
            9923893786708599094,
            10626907417123490301,
            16567638710074136065,
            15331893201350769745,
            7806059116444378137,
            13707614865543378295,
            14214520459986989460,
            6035477577880016226,
            10277099529705946298,
            18372986873326501283,
            4338244300386537139,
            1885583709719807446,
            16845517839678220839,
            16947210250545253480,
            6670494832117124389,
            6036041367440494307,
            229754666723435825,
            15823804790946779523,
            15265177722210061081,
            6868267055223603923,
            9226621643685330148,
            18006094505761810706,
            18221081278841054189,
            12127857529373977986,
            16194103438356446493,
            8996172278454408544,
            16380076914543975707,
            6152245826487669630,
            4068010411743113672,
            9528640481646565540,
            17068727453522557633,
            16929070848025272575,
            9525905790714983636,
            16745442156906544418,
            16966301396161058106,
            1280761563218258815,
            9886487117634597825,
            9643016182903155690,
            12333756274736517538,
            7789295763714648757,
            4634531325244732527,
            18094172949027900335,
            12813762416139899734,
            13216145789920058862,
            16192490788986896738,
            13489576127803573018,
            10757280652278072407,
            7400982980219314289,
            7274050102340521837,
            3714038967780534003,
            5650849656184811207,
            4369648122029950440,
            17191757296173728994,
            12955856375209661885,
            15989806839955330599,
            14529483301767469755,
            13939990818245732786,
            10986964739469606485,
            9098204684779117224,
            11521797117261889257,
            10642640078248189706,
            8196455427569592906,
            3763241903052488220,
            8559932097290250553,
            13133701464452284906,
            12606299037676909148,
            10148049858280060453,
            6730193676409107344,
            11298714080289609839,
            9685196974497878148,
            10052923705387506217,
            8255157126125094936,
            12119155404921048391,
            1389060467248980855,
            12957275050321622167,
            16964936652210647234,
            6812955159678763438,
            2669438233099941043,
            7792956676522636845,
            9444261680616788673,
            4781369009293371008,
            7027196956351871592,
            3783100517254230504,
            2136161159738484041,
            17579329565989954385,
            8331765788382922696,
            10719857952209575230,
            10622507865356360405,
            2043448614250743598,
            12585531832778317158,
            5851147022117082392,
            4234555448179321347,
            777382462829122485,
            9486147678176198767,
            14044023215658818072,
            10103884993965162367,
            4347545313789506310,
            17590907353044704338,
            5430755417200540653,
            5792680144656904883,
            9447597628871135596,
            2715434771641147741,
            2742356519588552197,
            2751677969637528176,
            449392785856725010,
            14014303161624740192,
            3933929173967729663,
            15082414626689925195,
            3446225870720019366,
            8674938857455818676,
            13246336934069043946,
            12692634838318515989,
            2385236641926129336,
            16529553732703121776,
            1244091608272751159,
            1709487870760621905,
            4112102451625574537,
            5469425608206832391,
            1299457315756940460,
            1522412367018450525,
            5571010125892125646,
            1471132187168529530,
            15147176398648355744,
            12840139872297631533,
            3004698746756908432,
            9318617557635528100,
            15944620123105027853,
            17226293982169068409,
            5219106403624896929,
            9001634172856867452,
            11838460773580650269,
            17984888835688416507,
            13624057344533318355,
            8184817256059810237,
            15016701626621795924,
            16525024835192554364,
            13989010135258338668,
            4067910607725865104,
            15826918974312249358,
            1804177036621463452,
            11216080402129500484,
            18146976969666845903,
            687866287427144951,
            3475876164989039379,
            5372474020520578243,
            2362477830180649,
            6719880551638257926,
            12725511880795871455,
            17297750639313056351,
            15596247519114553563,
            14114544315301661791,
            3176636657146941803,
            11502114774192494105,
            14247630581120465340,
            9832875730466343564,
            14410889235485370623,
            4850627980131859397,
            2783985723875994117,
            1782009044283436158,
            16960537035757064968,
            7714917051643937790,
            1300947411233819506,
            7906598347965753718,
            2628099897634125474,
            8856452004171768493,
            13190475740156660866,
            8097477735965983987,
            12979842338488025797,
            11924302341648514299,
            9966328740325868830,
            10182736954701710105,
            12880474893988133819,
            2833087078442762163,
            9894656770255213073,
            3490636447083805869,
            11326295550835752455,
            7959561488346447711,
            13118808153470300340,
            739690466903322115,
            12467985274403658226,
            4468292083569796672,
            8390321831688957271,
            5668667140378830341,
            8093051522574100521,
            8962754439140657603,
            17303544467727048030,
            7899105858509275965,
            8730893133711266515,
            2852257307054784889,
            17719232358461466683,
            13625161347004766895,
            8914633611725785572,
            1758030471124114531,
            18405868732235833945,
            11515396015562478754,
            12728028496222661926,
            740843742250233035,
            15872682029467178869,
            14677346255214593961,
            13913074428586625918,
            6240704548519511927,
            12076094899693040279,
            5958309379840634042,
            8380114100373060248,
            14132336696535391580,
            3403989673216406659,
            7690345060797587143,
            16153234119113691715,
            11234651296024406654,
            10756958548627904636,
            1706022322170277820,
            13725735896604018784,
            16829620102915683357,
            17957037373129248858,
            3286468152283974619,
            2476451294380908916,
            1014194200833912426,
            15421295638788209550,
            638730749985828335,
            15569301647386060088,
            4884102721161105606,
            16118586026087858242,
            4141261105807827041,
            18246672595653795867,
            12702502449771781396,
            4770185548223620771,
            2319183180812440981,
            16950786289954407464,
            5692915144176797166,
            4230577438974974594,
            7854469962714771631,
            14766337872209772349,
            9192401624349718356,
            11921636439000291579,
            9779648577722152419,
            10275646925497320278,
            6582960130229329490,
            201482259046623037,
            14271002214942882042,
            8348014788972934310,
            8278369830727772546,
            10692261141716216399,
            431084338924724152,
            7760476181784124760,
            16791561282033735907,
            14937856555118346794,
            4520040927172355992,
            11301788356161356156,
            13653070290638861258,
            7364919761188842900,
            7220731169179578899,
            12559465807860841432,
            5733734986877931214,
            1419703392489500548,
            183407053241233255,
            7731215593389956691,
            16010277108643924188,
            7369991919745824642,
            15153966335645577923,
            1567719057034369444,
            6541819355038671624,
            5343885383844460770,
            11323748498604900666,
            10870611512409692663,
            2417728012159276468,
            6058250586784685679,
            12746650459158680982,
            14167892274904348676,
            9210555560286163801,
            10870945444315264979,
            7461974907254744909,
            12998415470087206286,
            5367601803528657741,
            13391189142981065105,
            12600867989191890048,
            15252543321131363775,
            3621562439365965098,
            3008136055947900085,
            5914590023909386884,
            2581263292695205452,
            4363569856957661778,
            10480854181905480190,
            7813627311394142224,
            13597747327879806302,
            8664393282792122328,
            4682041653649870196,
            8107754461855926356,
            6025764401698197373,
            18196287767001958392,
            8892602566284402201,
            5457119809494453325,
            16592005418933449765,
            11691122915008134570,
            13248170395133672365,
            5664546303010860390,
            9021895872558146858,
            10847780458538586521,
            7186149656886288658,
            18147281673151320759,
            9399848659191762041,
            4410371892196828876,
            2269944740767963762,
            7130630424464773331,
            6614485391929732220,
            11696934127503813176,
            11179203777508075734,
            5816954395742828897,
            2233750736991678621,
            2250548227260778039,
            3311810150147659718,
            15534180898229626840,
            844478144211496028,
            1630018398457951631,
            427951298078224985,
            16816640300547217534,
            4872149723405343636,
            2098539037731708065,
            3865985023792737568,
            7766914645561852426,
            16289388416892983797,
            302674028646936708,
            4004256602714201161,
            1944783077047610022,
            4261070330826218515,
            17121350012834071062,
            14838083964121848601,
            17905706352044386137,
            11806947281378184643,
            14084925998353055848,
            16146605218958202410,
            15981514469312601088,
            2057822052156189348,
            6908998195754168349,
            8930174042604898124,
            9807646585148908863,
            9401907824844674310,
            6747593364832664587,
            16762291701349027097,
            3827952680057863444,
            12574764531067931236,
            5771450607999641314,
            8847455200009169918,
            3381906471651615306,
            5924358206245292918,
            6232679967466356454,
            7111753117599109603,
            8545509509566822233,
            4750683220084005874,
            1797265195584927155,
            3566734674639978301,
            13091876679739109311,
            2730013642900689330,
            9158246586954364745,
            7552775034881486686,
            11562038520744779719,
            1642419170836502608,
            8365501154108037308,
            12228975771477197439,
            16172856196454821345,
            4532751461546415645,
            12655201687631845929,
            1611188306921857920,
            12135835943783216924,
            15111119901354352689,
            15065229054786985781,
            17563388092314472915,
            1981247557350032676,
            17844562502326398286,
            9187136374029800174,
            8849346800286670188,
            10269823864856644455,
            407851424007536050,
            11743876351251550334,
            919734798665473086,
            6792245985990113393,
            3715331894017583847,
            5785127628706274890,
            15394473277583202322,
            5309207364421503541,
            12932307830761802555,
            3152617457706590230,
            17751507033762733983,
            7379073407417862692,
            4149511830927344171,
            10950557958033284974,
            15370188461544349423,
            4873892910859989143,
            10530058905707434853,
            2827155088400094798,
            6634254327068383796,
            10025635359213696738,
            8403573578689131217,
            10249687701009097240,
            1442836509575869658,
            7096553297238381724,
            8284980693981920765,
            8210360064157518844,
            3240062377886427542,
            4899542889357880785,
            15191664955132664833,
            14727002093370238803,
            6787473557488468774,
            1559387775654289141,
            4847813749549467069,
            16524355076144296765,
            8636372289332824954,
            5112729447295030564,
            13904745551046937877,
            15072791111677729434,
            16310208154499351208,
            2925870205186752576,
            9778928955967196508,
            3194589392232032622,
            8642893949374728666,
            3085193482559686752,
            15609998079350579625,
            10961647725315330002,
            5873055276445667442,
            11819437232174631439,
            12597499124132882001,
            10310559810794209612,
            17532244473426418671,
            9366514517157686434,
            14114443438038705584,
            6441641252169911503,
            6313677318222253600,
            1490224382573500589,
            6409275930748395984,
            14474777896566606562,
            1396024470173694791,
            11150164864381276717,
            8619549547186273630,
            17085833301611544711,
            2076990937255843378,
            1388788410021734513,
            8352621373027994726,
            16764871056541089134,
            5038605981178122661,
            4940763045017437396,
            492964260323892563,
            4458563142240584397,
            5644829012803266040,
            8858699491843533546,
            4474498441974411889,
            9862771833140050450,
            11323163327787064605,
            16363080760942319922,
            5313783434730282254,
            7220767298663979735,
            12730815602857442111,
            4829156765787124529,
            3022161212716369519,
            3961783351565011942,
            8387064223446862787,
            5031406490469860658,
            14777298142338032930,
            4453282970239612374,
            3009494503459022660,
            10390954437830398595,
            13198354033367167762,
            18360859425698280933,
            14623610202962849239,
            17801193490995324881,
            14550444905591220098,
            9521874106042766672,
            16281889716611818354,
            10389558288632373410,
            13107481307139327786,
            16290372994385281025,
            10387981742575613943,
            18377269881993047408,
            15259913662672161309,
            16030005620751432317,
            11331214518698399463,
            12322838637394144472,
            3466359660087361121,
            1896557687542263390,
            6596788539731420830,
            11491979523772203478,
            2430136875471351115,
            6343383556700332930,
            10731127253572851665,
            3211229361050472159,
            992962522771374254,
            17106586641662980338,
            6032823707456483253,
            5093089231548918278,
            11199833458562482877,
            9925601966935761121,
        ];
        let mut new_vector = Vector::new();
        for item in v.into_iter() {
            new_vector.push_front(item);
        }
        new_vector.append(vector);
        vector = new_vector;
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            16161929035698470588,
            13732231361154761713,
            4529944023134301809,
            17449442698167537693,
            5156808784382666932,
            9318854299933671745,
            12607758430127192377,
            15382724415346476027,
            15115679632310445687,
            15431217659231250267,
            2970849795539367019,
            2749054445265153971,
            638769332411867707,
            1501024223638746163,
            1056647837633277556,
            13053589021610348915,
            1865029847540961539,
            6242908334315094584,
            11843981408440213591,
            2780826482414369588,
            10607577743206954193,
            3933018359131741552,
            8555775259381489766,
            11760900548018663002,
            11236386077703054547,
            17467365226594887655,
            13187862202861043626,
            7324905908332932662,
            2817347578398272810,
            11999935248146638723,
            4946546334154559164,
            15686390599464990676,
            5876127136537188450,
            14668014462419544747,
            16752442102536306616,
            4672405782684286745,
            5858154418134386980,
            13784701127634591403,
            12708552378701656130,
            5097865401661523190,
            6830554600514698280,
            16669355123325864680,
            10725879776589222969,
            7891406593515975695,
            14550798236454345230,
            5310079030091226554,
            12241478596092510208,
            870794849012633105,
            18385652809330484421,
            7381577976014567821,
            8007004555384781407,
            18216495661529427004,
            13628131389097558786,
            14933127624964139366,
            8060388526690441460,
            572645248120662131,
            9221372415035745609,
            13031512340041362903,
            15033477616550519220,
            9019301545867592222,
            15504675588614480571,
            7857987626237410965,
            5437121054597852279,
            17196096221069998406,
            10734390491301379745,
            7601781033701899717,
            1362214871392156218,
            5208131960650235560,
            17129117685361808276,
            17639719680358387833,
            3301849985349952537,
            17042052668812395161,
            8082936534154294900,
            4249318641404500334,
            1632978191509936954,
            5900779927842262022,
            10888477722616906826,
            2173089208873696651,
            5708654243370261118,
            5947316931786365747,
            73478704857274962,
            15530266271167891335,
            9528789450948351746,
            2415489674853731880,
            18161406444331084611,
            6909922859735498284,
            14593914349421090365,
            17728121435377051799,
            7413677289937193591,
            2504724566668482284,
            3832099382962668595,
            11924553391396558912,
            197259850941474495,
            2281123926023906303,
            3555592002317355544,
            7541955389529431487,
            12583737956362175704,
            7879637570421937175,
            17644875861713547481,
            9559095194806219538,
            1253370098693747197,
            7540181485781173752,
            14552233869860854370,
            1657670894078863057,
            11387190348329754948,
            7827333029770901944,
            1051281412499663656,
            3933820377861521891,
            3050421819890514198,
            1771013795499664770,
            1796108567465571875,
            8189222703419126608,
            17619268048090702855,
            4711324048919191361,
            6408362391539775712,
            11518227710364139126,
            11401187158221692148,
            8086740748195119232,
            6280309240540518494,
            13066136800252173586,
            16369466201506107885,
            17959416472848758449,
            879779326271087474,
            8380074114636354155,
            1083214282501643116,
            17618748511153262695,
            12533090825526586155,
            9484775143267822684,
            9172813635188284554,
            1282570609652409248,
            547578925704255817,
            9802206698115405869,
            7242409997816696801,
            13593391981657703748,
            8812288143679833969,
            15576852811636008286,
            12891393921069219473,
            1893624296752152149,
            16211437433080725084,
            7558709092209492748,
            9284022614684884082,
            10133269078530251240,
            1159516569820894066,
            18357180016677880296,
            17802414282693373079,
            6945311673745283536,
            10416513873155814003,
            16187246343517562944,
            14925575538358557005,
            14678390733640098580,
            6899056377419774255,
            8389958286259266153,
            4949804732916952493,
            3297364790603538337,
            16007187883872098846,
            6198852984973835596,
            10296631713560968421,
            15551688302677071495,
            14252478139150244321,
            8020419676774051490,
            14239761062866670975,
            11562734765756033820,
            351916092947424637,
            4429985235088843746,
            2885045963456869935,
            613114037276721259,
            9806951999977120171,
            13487584695737155671,
            7435600493157790453,
            18359541864686444079,
            11345752449389931652,
            7643253389822280766,
            4196265114868191576,
            2120040292072823816,
            3750015156014279108,
            7975926050021898015,
            9588220616968099955,
            7248074010293433291,
            15882176639550917724,
            5711769049027748210,
            695068726470443658,
            10634230493009380776,
            9412541844805511269,
            6935207839507849960,
            13226018195016880628,
            9250109880399377902,
            2893186874534093770,
            1052570131280622187,
            3545143646094726801,
            15400014019529241525,
            5553958808560550303,
            16905716902897930630,
            9910918242900682282,
            13338501952567464661,
            8962780125795709286,
            5662640728175394149,
            9103087762210915938,
            6486824826612788951,
            1737083691921895854,
            153092214174500545,
            2867062107840793987,
            6063808340668742118,
            3964007195849993659,
            17981544424600193965,
            11517221737061034292,
            7732324371303956192,
            4606207611633139401,
            12861237891290768187,
            10664327170150335192,
            10719442821800818989,
            10649412652964869663,
            13609327990930975629,
            4761747566109631669,
            3114506466152535804,
            838873174179868249,
            12668001359640273605,
            8694539886838682210,
            11031021930653956061,
            5619666932338374491,
            16229969193606137816,
            4205577457844043259,
            1813713935421772497,
            6311608989699684734,
            11445795194768717004,
            12732706119223123192,
            12229845125557797944,
            16978339594737521655,
            16414350070916237438,
            898732831287963265,
            4167013972463949826,
            9417322127193477289,
            9611777349182366197,
            2210308409165433830,
            13864460818530503661,
            1305653796881315068,
            3635519172109003144,
            9742427091654696345,
            7265054873685773617,
            3224643001152933214,
            6570056703411330264,
            6698903594336340441,
            1130218437770091282,
            8864459036544459057,
            6087681166004718053,
            86728188971729609,
            2302629519022678616,
            18155455396353525211,
            14293873800026243813,
            4309108231586909133,
            11086982655677791869,
            8988695014278792955,
            5652173867748736251,
            565713988366460997,
            1875944321383544098,
            8791421480314790164,
            1166865294317386997,
            1103051208349694820,
            10124659744836638627,
            1601399137569731856,
            13156405042909319400,
            3086210298794243606,
            4019364402778223192,
            16226599726779109663,
            7162201455484126200,
            9587076539205819347,
            5228340843169678401,
            8636781763327587416,
            110289303999158439,
            18130594399899303990,
            3091093165463606222,
            15555106228290051251,
            2234502407416015068,
            14541891633369164702,
            7096314448482586260,
            14000540116852577286,
            987995986336289568,
            1355208711968548577,
            4016979273929041683,
            10521891894068253395,
            50928353064908291,
            5494946100631331231,
            9054977831869035342,
            441425813150486990,
            13627452011347749466,
            7289899059428123305,
            4599628237314585425,
            1010343204274937496,
            5984744470925372118,
            7672379922953215050,
            10927510003336644671,
            4014066369109339959,
            15487886211670711268,
            15211795532373574830,
            1307420048130602364,
            15639591113617533523,
            1154370173536871754,
            1499614111365064331,
            7250766473353825805,
            17572710821788830051,
            5069940561270485845,
            978945046958142862,
            1011665765587999034,
            5972062611674456560,
            11180192774165753444,
            4537649813370482506,
            2175705370444849185,
            5103057574648202575,
            1046996850756638767,
            14747744317307249132,
            13127362741810002811,
            17820527843607694812,
            2407352849109521234,
            12323698483177514722,
            12097308754501353013,
            6602722720209454998,
            5208323680612507037,
            7609968256756032124,
            16985470941112939545,
            14527822743136002966,
            333534029523119955,
            1696471421680266321,
            8954830855538285284,
            1207427192245805214,
            2830629027876733883,
            15633412640685473771,
            17013671580418668928,
            2926454756002155987,
            11935732198944187886,
            15655700645887934729,
            7269177985767140643,
            16412754665357681683,
            9682209512702945095,
            7370625355280474854,
            12752168508174276462,
            5248270772764185627,
            13442704073883311580,
            7537095241360085725,
            12616247534462060405,
            16552054085988322792,
            17732362645829107762,
            4086653933512271279,
            670501760836752882,
            3668706254838500969,
            8436497314056722263,
            65404607690805364,
            17370879870384095102,
            4008247980479711565,
            508966698791921984,
            13989353594587607112,
            17776981512540537180,
            8817985061818222519,
            8779206226167870356,
            7455937818783755673,
            6151557903644163900,
            17126036616831690676,
            1403899264931192083,
            5529669962979086658,
            15588068159470684628,
            9034604375505650125,
            16006535298788765293,
            14134598241114967791,
            2975483964233307374,
            1053250770200329967,
            3177051851863938068,
            678000653292656687,
            2458193683909421443,
            2090488815147752759,
            12504321229698533910,
            13198734285639019851,
            550660206044821955,
            1904298958753194204,
            1917261481513598736,
            13385337473318321850,
            6026968950123325038,
            13050517845450106613,
            14859087199145697056,
            16826434768625944996,
            14542865018169015866,
            10313702362678693924,
            14441847508042588617,
            1927850490540696752,
            8302065150431810192,
            10842084071589167822,
            9403700246289893297,
            18046290607384841218,
            12021169429362383551,
            14466242127902903726,
            14819265465272792648,
            11870877484039751832,
            3637375600624589269,
            7161236198262257172,
            399249673553089680,
            7856143059215069230,
            18309748450361926829,
            16761433101242109358,
            3937030119279388903,
            14140842186840285941,
            3640778329223124873,
            15241581368766215049,
            15725854050865829560,
            10354391746919032763,
            4978468336671620732,
            17754309530769583792,
            11533660830659436862,
            10069753364019018992,
            10693330398150824156,
            18166914202669570844,
            9434584582156223044,
            10584196945953307556,
            14022101083213121144,
            18158748816927927067,
            7727836265337342575,
            11753022075286898661,
            12533947640680176781,
            13097460844796528744,
            872396191300077890,
            11398327086418666703,
            9253378549369266419,
            9444685683531344065,
            12630314416374392847,
            12123965809388298093,
            2691063152544075427,
            13508737655271419715,
            15087273309763773949,
            4929307065971446609,
            8627967870347944950,
            18308143060795019068,
            15701381544897278207,
            14692524645016141818,
            7047311444928460920,
            7920849416308357929,
            7417608012230403616,
            18022720540092274299,
            17808312140540228413,
            10467433359478189358,
            2126879918620018953,
            7749019746796772417,
            3327200357253464451,
            8207593597370957156,
            1653171339217737759,
            14603455137027424300,
            967172482386198178,
            13414085447054535479,
            6836827346526602709,
            1334960165641210454,
            17308657400931745680,
            6965370343785952654,
            17711632442165951684,
            8439899061370348114,
            8213004793942556810,
            7444119595067022627,
            5156069583096174585,
            8258501387653708894,
            6615866124259745764,
            15793250554785644129,
            8960252491306661321,
            18362286662368310284,
            3799462805623563769,
            4862984488664090326,
            3833158884562747125,
            5358003703757447234,
            15651170884639673748,
            861535777377429766,
            5495450085680355020,
            7932456890763879058,
            4558931492908298094,
            12067324208079092441,
            17650140194846053270,
            658708648678177388,
            2662431941511791635,
            14005907944935572304,
            9736650669232610972,
            13372359920333477748,
            2825775765756422957,
            11999323170354507207,
            7590886027990267852,
            389925874848495691,
            3171528520947294873,
            14924500760148830059,
            17806421036308422301,
            933046091652429343,
            5730706816708306934,
            2424900245652275082,
            581611813344433907,
            15274343054562497913,
            1043705168752328840,
            14834878015553638179,
            11183619480640419780,
            3597133923412671844,
            7010229019320564939,
            1234467785671471661,
            1665996142833294481,
            9089211108028629558,
            14065702142770035511,
            2722431988925184956,
            5584424670039481032,
            13648436988425310836,
            6686903156772946621,
            2794543206768595862,
            1031997637552846289,
            16745121649697021808,
            3807093933895243125,
            5734822293938858876,
            11831285842214958708,
            12483088087887188457,
            16450327645151469713,
            16464005988825324559,
            6791443154411809355,
            4416974434752169852,
            7997295467689034078,
            5399792659687246962,
            3303738031079858342,
            12672039416282767996,
            17408305427611781017,
            15150787285244445155,
            9036036372435877344,
            16538812103565692328,
            15146529042447337196,
            8469362246548412320,
            3906063196625842645,
            17547003443783546262,
            11967710828546566991,
            14649175140527425727,
            10197511296044743079,
            549664131847517893,
            4235300763017895382,
            15207047627179546146,
            4146812255439075475,
            6939608009296001923,
            6503393815783685443,
            2883578296827196425,
            1474413062548500699,
            13149308766944069760,
            10329162032141018056,
            15733934216326910416,
            8505299486569407690,
            5383395982924508718,
            15778041285086794427,
            14284854444013486023,
            14341924610510855360,
            3359485321872037957,
            5444759505588667226,
            18084931906996054182,
            17509970316377667427,
            15685828828845134592,
            1012364064235899961,
            13043848720331458721,
            14464050824092642056,
            12548596800294056857,
            11773568509875063649,
            2841895765890432681,
            10918869373956513396,
            17492117791136493148,
            5518468411989764080,
            6801156645168047107,
            16052696009865325450,
            11876316797674174992,
            1154835286289596594,
            5052605791911880852,
            244562862166634575,
            17752450105130317295,
            4620204176208910534,
            11271470486093994891,
            1992889398274591819,
            13290159186827416385,
            4173893959386213181,
            17757240455989105007,
            16433539282058561524,
            6543951671796933921,
            13898562239616407421,
            6930617253971014621,
            10646686679606261645,
            4570612270554401762,
            14248086323986848619,
            814846724364552411,
            2701886858504680698,
            13583805733998700970,
            12894351808163835614,
            5871028648087077944,
            9270874676717222081,
            18290010081537257127,
            6056877715989744024,
            11033805475255077005,
            15315055395620108471,
            12017511181094222909,
            487263099289236825,
            16658155359086878207,
            4114817921713878330,
            1616181350593503988,
            5924398604067472715,
            15998122491862159205,
            6172618224246407,
            7536019313963486194,
            2366984742498493186,
            17316749888982549545,
            2192337818925395567,
            17251033860346595041,
            13887968553037453683,
            7866014166210718732,
            16540733201696896622,
            15304084380005367253,
            5860257516284489064,
            8586471173277012849,
            4137283455552645779,
            403413714080139115,
            8505953362453759408,
            10049623544897679596,
            11249359194717190126,
            3704894209884349370,
            5642376744536683698,
            15741945266247779039,
            9430404652727664576,
            3733390955268769243,
            17118228598634085191,
            9617163595523029550,
            2039665056313629071,
            7495423973203845028,
            4801871853495940864,
            5555440798208981356,
            15366780392744576632,
            2926764823254076862,
            12706149623348208712,
            12634722171653425897,
            1696180916183996821,
            12373339573655601320,
            16922357088361129839,
            7562139935367705923,
            6236040146898046694,
            11828755712961179130,
            3500748886723777582,
            13438409617796997299,
            15999989164687088861,
            3315186344735195107,
            12303146888450937015,
            7626040245699606740,
            17938617789283902992,
            5157343605645414116,
            656481563230137415,
            14312530216689830122,
            10393354529763233559,
            15806197222084046962,
            6807083122890382179,
            16132795895317086278,
            13237238146897440356,
            12254672700284523234,
            17248632913613611230,
            6660730377286328169,
            1986511574993779178,
            4718193718243545874,
            7184760182498561263,
            3503863421809440166,
            8017582373580069317,
            612690519136707233,
            2996730654565066057,
            304176594473397697,
            5903011286084045019,
            16856424905639626072,
            13566811326690768177,
            1136400936966714152,
            7675494612053038671,
            18315059827479998868,
            3067560508807337169,
            13902949236328764200,
            9370861088378194568,
            15978803877667321473,
            16985226868515914393,
            2005914053125629165,
            16628060026772312529,
            2068810083994473005,
            10347842038637714473,
            3083136699594152449,
            4728885924830062881,
            8939035835375144322,
            16310177184379978511,
            2821953113386182819,
            15423572841432783706,
            17316778377148457642,
            7920984260929109932,
            12177350311698331219,
            14899013647363828085,
            10216732243688685310,
            12243971115637763426,
            5588873355549552569,
            3548400258533562517,
            11974517637326006495,
            10379029353948722272,
            4700236135267156270,
            3266938820439451458,
            2500344442205446704,
            255497244345143351,
            13813540557040522231,
            8235683525250944076,
            3620861764971607513,
            14742555495542663443,
            11585231116945819235,
            3808010319058639701,
            1776343767429962341,
            10002957714220135574,
            4248891750409296336,
            17222661370873172962,
            7691419636380271736,
            6187897403158773607,
            5940304106906816051,
            3942082051848777880,
            10325191781543187660,
            2461324149398378106,
            3557758294842723879,
            6686738959465146999,
            6949944152704027875,
            12957387508655799734,
            15426428888353045468,
            10969251609892299880,
            12684821069867561260,
            15107056207007456465,
            3903357914014996006,
            15982856357393407246,
            10104856993417802900,
            11180435042388179656,
            10896488776925088882,
            16972414531546388434,
            13932275324066304422,
            17825627146198327263,
            16188097986572853006,
            6077792166994314083,
            9154182292609069169,
            13488075977144224726,
            372470561552069271,
            4863709893905413987,
            3786332422155750701,
            9981829636132508033,
            170450224656464600,
            13105998387770296806,
            10020919785090687414,
            6359202487181458003,
            9045522445522744275,
            13958318158468168671,
            13513872394191878089,
            14159398270070167739,
            6585353860351330095,
            12434431340714766122,
            14580126653921378683,
            10373509244287776278,
            14654000917177096065,
            6938772615855661152,
            4089103891911416757,
            18426469823097624579,
            14517694514516101153,
            3826087908219747089,
            6666909812926506304,
            6688331542042793486,
            17658505872182032148,
            17949339430815654025,
            2369984750983767092,
            7589032472381186801,
            7574432770172553884,
            14520039659016671201,
            9547234849095245384,
            3942954305727227676,
            2139093081013156163,
            12660708714047001901,
            2076451470761156963,
            9373661916453279409,
            14912848662576067846,
            6613989234436227515,
            16431076151408494592,
            3806230087506272096,
            14098437713676167102,
            8544159529554056823,
            5479768225343120483,
            3992993380688244514,
            15670800099175452350,
            5956163132856254576,
            17248413422472533769,
            14814312275105238893,
            9024397419476185555,
            12229801554051906369,
            13273655538496314032,
            14101328513126879885,
            17179167900660537679,
            11486086406410666940,
            12512450619856133023,
            10131526193785863998,
            11898313958800063854,
            5841918358112177834,
            9961094012889773283,
            546781874073067888,
            3309409759573285474,
            16573448753305486640,
            13119906426354546993,
            10697322706295761003,
            4112460243926742257,
            11833845309153534460,
            5322922931913485800,
            1902722196262843086,
            12888914758512270643,
            7070305021169714295,
            639356209185427333,
            4709749233831413897,
            14792517280119118840,
            5856401108537314393,
            12955499683274530035,
            14637037586987271349,
            10435591867685566721,
            1283159194805463307,
            8234948147033802673,
            3447328887484982937,
            15771026864777551935,
            7622070303151126356,
            9500889728117867972,
            12572253205320414247,
            5891461382703182241,
            9835889694587572693,
            11701224386015505680,
            17330055627599108440,
            13762863244695430394,
            7180644142539814609,
            15799316570258602446,
            4359207083102548029,
            15270909852964658871,
            12117168066149955258,
            5102408530314077582,
            56812752599075157,
            1630993708357145773,
            18250946477458252605,
        ];
        for item in v.into_iter() {
            vector.push_front(item);
        }
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            12339401187885714932,
            18003753312457419922,
            7008253201816165931,
            11034850637780250512,
            13305074914213676312,
            14592697212514753651,
            3871748062146601668,
            6656971110580844172,
            18300185743844939241,
            17149436827831872933,
            13956628612997617892,
            6610750771184026274,
            16812338705809723653,
            18171262666538695258,
            11048501314219999340,
            9602012590516857956,
            12640651665651421543,
            14422752285597167264,
            17028997679882210003,
            7681944225981992510,
            8287488325854488127,
            5057381923023902333,
            15046928733862602947,
            11455224391431615100,
            12077330893640783439,
            9540144770571026823,
            6680531352138483891,
            8643772251982778107,
            1563297026585987352,
            6306126781788601541,
            3715455411098292945,
            2203132667496255356,
            11723625713750380260,
            17843888341408008649,
            6251736412571952461,
            6589752145482831391,
            8039486692459023020,
            15315186380412977600,
            11395931695710313001,
            7620310675638813532,
            16124029438759849309,
            2769158969173021744,
            15853153185955139696,
            7288093681192786446,
            5896410361505138067,
            1593894575832866032,
            4753771350027563802,
            1264069111388297407,
            16460915183790224176,
            11420534226024074819,
            11967398719510278510,
            1785075326037184119,
            11422382188086139661,
            2815179800657376185,
            18003818194648721074,
            15633468783494664806,
            6127877954008967100,
            272403502462097569,
            17974667185407068318,
            1760387307968109031,
            11536708890970804304,
            6733668043754968581,
            1667410657592487468,
            9245501930097578105,
            7642485794353279665,
            17570613225447356065,
            14271277955675742007,
            2569532871235685171,
            18288093158636816393,
            14767238833153446161,
            14934044070055366683,
            3824982909750343192,
            5435650610223426340,
            8866398686193654196,
            4331572139730016292,
            13189984615608980104,
            6565228243738751595,
            2838407344519318626,
            16596393599809395674,
            3178197989317994304,
            14266302636264820593,
            9048757158926405891,
            17239200202274187260,
            11323129896440224638,
            106602299212762089,
            4528567577169528784,
            16776841078007832389,
            14294174797451141227,
            32146844526753607,
            3692602082865309874,
            625334032638551443,
            13315257089014951045,
            16146105979509049039,
            5774509995366494358,
            3025983471149725661,
            7876533203810753643,
            12843069035472168517,
            16102778759354816868,
            15930101302271789430,
            15042588182148307328,
            15775762915304367518,
            11738135312356621871,
            16670592027072183171,
            8529671932430905972,
            8786680543271622609,
            5793035765212293780,
            1765420642243885272,
            10976285635547414312,
            448100704525382581,
            9624161324973649457,
            12087786982560127693,
            2143864690181773270,
            9281103196556170744,
            7109794354466957153,
            5884623003174632881,
            18342270638574420679,
            6366103027591636159,
            4074135627734720869,
            17597362388507690028,
            14352306627562376566,
            8093888424099933858,
            18201169253234108308,
            1493161238483887953,
            4690931458795374127,
            14466330607082850071,
            15848443334842174968,
            16944828328438716270,
            15644331208584274345,
            14870178607134210135,
            4066450952295053663,
            17080791101027085329,
            5377835141769795326,
            437249989358205900,
            6469015794987796322,
            6009276879521045389,
            958702333087310753,
            6430175862700673298,
            6099855537680469627,
            17902591454200710512,
            16464050558822144973,
            6226390649724597796,
            5697613832836662659,
            13102985670964200138,
            12619748288573533828,
            13588525889831183000,
            2450192823492908028,
            9581426793701517533,
            13912190259460663398,
            16616364209874942301,
            9166620617176424607,
            5440778077060119125,
            16768463376873962631,
            10391802733501230328,
            3637584237992975400,
            6465868497134345207,
            6381864894644934775,
            14017750399818028941,
            10461142546586078943,
            7435544894571840064,
            7077207816341528426,
            10492840781290622517,
            13182758800108656848,
            2559127930858477126,
            16411077653053407167,
            8719114780061595761,
            5957711326197565288,
            9877323431012299758,
            14976145098372091348,
            11403797607689553644,
            452803197512962289,
            2145372430776698108,
            2284164873225043049,
            7357696541656743618,
            18183318114461810244,
            3576877547138556201,
            1736394620497422724,
            16096378868072016332,
            2182927913540400624,
            10886366431545890120,
            7366507081682799299,
            2542205934795016584,
            5610804219208572861,
            3695982253096610961,
            4413078201079182662,
            10062618064048970526,
            6638467557903554263,
            11763901388920848982,
            12001090240387792736,
            4572342412711293085,
            6536621394725676275,
            7008273873674708652,
            7705486605675771399,
            6738650737229770030,
            7310347241484980733,
            1798316290405789108,
            7848286005869541029,
            17761831346723603139,
            3380187332072008547,
            7333626060219056594,
            8335735037035105048,
            10960682412914116559,
            17559741580648256108,
            9215586764395147210,
            14213338531326378706,
            14606438950033116050,
            12511263767899960804,
            2179003033562712047,
            216655863682490159,
            10896539460930258250,
            139828728719333795,
            7223557295826038396,
            2284291203318907214,
            5819910848559923926,
            7492231490828814763,
            11394606614106202703,
            8421604097195600419,
            18323523630788814571,
            1064987892761484633,
            747815896137588392,
            15889239238258791016,
            5381032867169861237,
            6164285325389974759,
            13971377619859168719,
            12722644613638264161,
            5089428041239236931,
            7821471770383489468,
            7154430573748277400,
            15034155403184673,
            17527819573125869033,
            9719855296201805009,
            4166636431502815047,
            6693623685728873416,
            11523958388069540818,
            4416079596625347875,
            3030975268251230619,
            5492949428139279972,
            6731186038062517996,
            3420169136225902573,
            14539186099139773311,
            4538330237127090222,
            12414539277903182523,
            4611366522177080818,
            14020484805970221029,
            11446078156255699635,
            10800132638198460111,
            17685488498437828219,
            9145598170838150190,
            10064608478474157414,
            281562468213524152,
            10730872832607822173,
            14636098490107188320,
            11399874015684318388,
            10232640391827507445,
            5526814462564451175,
            17447068719142619153,
            47309154043554753,
            17557861491309090817,
            1802479067917380683,
            7795289160321371325,
            13747389071901830616,
            8226772336715673740,
            8550008115017732940,
            3219396420182944395,
            16546960573395193569,
            4394061358827405362,
            6765948940181311516,
            13057398982455473561,
            6561450230557146819,
            9693699325292982401,
            13673027463945469194,
            2900621270589449027,
            3590262319450748848,
            1307104716827050377,
            12285362739741160723,
            7476788584791979183,
            17933824268669143361,
            13576572161457284386,
            14604310630964340550,
            1982924605645903214,
            16496696430065037075,
            16520018876384737850,
            918563842138916897,
            11459264509685113426,
            6440861123646035991,
            6525012379602695480,
            14433244037962510337,
            10897152805333775299,
            9149871048397577687,
            7608446109530494463,
            13021452046819193526,
            9410553060252541348,
            13910750285911627447,
            5539809766094640132,
            8090134220919074625,
            8930012614621724966,
            1112942657824721862,
            9757323347755319517,
            9541729660675631091,
            5850626651384539209,
            12480242877259616703,
            15598722890330810077,
            13803554285115862018,
            1711338582948864112,
            16970700075043038768,
            7398118966801512661,
            5077848393277589915,
            6152386168075064111,
            2093291805612244387,
            8550943200126787183,
            11749335203379406982,
            4687687208324174015,
            5127894486104273800,
            18406556512966052334,
            13511879713456626873,
            4525497559509240822,
            16529088465149756751,
            2908630910219254164,
            11785215365799117100,
            3168113499945583138,
            8238614423485028211,
            162779924329150583,
            12926867203538420227,
            9117689461608481150,
            9895596836120212192,
            7873900131804959051,
            651926654400141787,
            2943738778326216405,
            2379544927005224585,
            17109261797849289314,
            4423128677704345626,
            6926036716101284748,
            13463127113935153518,
            14017307736320982780,
            3921708854004418689,
            2389045329352246173,
            13182867528276223155,
            3127345921721318035,
            6533550470802695469,
            13498898469683869579,
            10493642741056280180,
            14827369433853167691,
            11591946651436066427,
            1148477241914873503,
            12931716978350149979,
            12515019378977527986,
            16362553548072243756,
            9352280650646153190,
            5293733285232783450,
            3793765618068049720,
            18137235687868192991,
            5003057890862942773,
            16445634523547897615,
            16173374633406338721,
            5039558104276203224,
            15750426566207423292,
            13669722570197209360,
            16847208118060652211,
            13120807783678206078,
            8121873008684190215,
            4420503094238356458,
            17975057875743419824,
            18288923404731749154,
            2231932688150141835,
            17763649898192230699,
            11103709039266176010,
            12778020659493241807,
            17523858621967978436,
            572997210381373849,
            5444952425248164523,
            9415581444975119518,
            14208070360131156457,
            14264218896645510173,
            12749381301344611366,
            15409191044983101957,
            16479374487280435640,
            2471409659903774588,
            2524640623383032109,
            1729694350261449929,
            8496106721362488318,
            7724608387385410787,
            5619157147667964528,
            10865359298747288230,
            10140594162198192347,
            7109487595169543140,
            13746559933424921068,
            1767444878811144727,
            863315162598150340,
            9074421951361394236,
            16098046464619230579,
            15530962748454056854,
            6271139727981043172,
            15807966376059925992,
            10143154580149267296,
            2752060147332486141,
            13766776862455626779,
            12619762250281964790,
            10637318104185009644,
            15368065191523539046,
            496894997675410577,
            3399556733856908256,
            4542440091370011750,
            847270569335503094,
            6292627710419174238,
            14454244745347687010,
            11377695408290037949,
            12507071271451888861,
            11201583082188099353,
            12514750349526117278,
            3810638888418427927,
            15516597832484386499,
            2991607614379589214,
            15382534272199972344,
            6846497816937309418,
            14960577158118886016,
            16669692115381546175,
            18348110787231633250,
            15568829469507953037,
            16272827136306992148,
            14109035322321577332,
            14657206280041819459,
            12827116864437261590,
            3354552533671522559,
            17389736034699176043,
            314828105604555792,
            2507153389695238204,
            12634850553775743295,
            18083113462377612519,
            3630837013665094409,
            11435454596250875732,
            7247627277295283319,
            9117053232592851419,
            13237144934204212478,
            11538809381027186728,
            1496990172524432178,
            5138228089030984572,
            12169434184041541151,
            1958218063891190581,
            16850666101103357508,
            15508783087325500646,
            8932485753155835520,
            2672890735191127900,
            16826567089435904421,
            17983491704756195576,
            4495957992521747200,
            1651670734137376137,
            9664140613648976611,
            8623836988276729375,
            17274092182660817373,
            10866271478524645894,
            10696466138425435212,
            9173023447673514341,
            4316642457083391067,
            8954526569677686163,
            9447964319705212997,
            16516728469424687540,
            1191527867512131939,
            7816293025181180246,
            13693395424591063160,
            10476366276052908657,
            11718520459002260926,
            4727244621165360700,
            16374962482159059544,
            15418207186485293467,
            13682364771566776979,
            2320721886064435427,
            1874758457419048874,
            5320037541762392049,
            10703610691791368548,
            15273557619251652381,
            543263203813386985,
            4652650932968319012,
            1019171949384415078,
            11044304494290441941,
            13588708391428320530,
            1399484698903314034,
            15959152216817586786,
            7839518293048969695,
            16953018186516921639,
            17618116366269410612,
            16263672289791169214,
            12654288430436109561,
            10346138202077550632,
            14355545114950898407,
            17356466570155978958,
            10866923870062247454,
            4796439362218737540,
            7929939665690352883,
            15212909351457147990,
            18034610731244141102,
            17427796563054993814,
            5340266235852823426,
            3313027461506348447,
            11844263763966027560,
            11387941587382039753,
            17354154009164757184,
            16172162620133361862,
            14241618435725453986,
            638724666594799525,
            4549131066616491580,
            17707273884901853907,
            13589797297133020046,
            4400897816413074683,
            15016984970797420151,
            11584754694783860269,
            10615612810570253995,
            13997216949976365905,
            17856791682913290297,
            17497496184519293559,
            5888098244689189191,
            8417426089493764829,
            17921335580151466519,
            13562926026129760643,
            3722027933333867666,
            720068287559387033,
            2953972330189506456,
            557945695813420507,
            5762140940601371387,
            7818817594101638335,
            15553847311246186501,
            3562121266756910768,
            5359175464692970152,
            12778425208188776177,
            14196870121576528606,
            10257056278836752154,
            15323661021996742502,
            16467219196938848195,
            17777443996569249480,
            5894043041013081201,
            10433235973817736942,
            16953285857397314479,
            2631771332112844224,
            7672568004832177829,
            11449181353854311586,
            18432506702149768058,
            15263850550860129819,
            1331100853910871352,
            2780562016487943549,
            17650178717250057447,
            4701453098665720318,
            10246734922692369836,
            5061864408708350030,
            890932097011482657,
            3961270485105662938,
            11059944039231402404,
            3206228516353851993,
            4448681425395837103,
            1178527581858170780,
            13873511296465359531,
            10582922029023638892,
            12068363610010511163,
            11852060519320797126,
            16403763674728202876,
            16840865538492198527,
            3428944159797117194,
            2879462820795314201,
            9864861884616694879,
            2825056028778593713,
            3171529565746615094,
            9768068332502697302,
            6579292156636659391,
            4077575904776897217,
            2480622245149894874,
            17333469956597597559,
            14638470696492755763,
            11679517872115173346,
            6604673080870806548,
            13600570462587431853,
            2777605977248616112,
            2998102728222079438,
            9391474567142329064,
            3828109074252789760,
            2858374924094310025,
            5386090972497905484,
            1676771902156673940,
            17565570052730792327,
            16833582580723059630,
            16829548284800046429,
            5406655081001287129,
            15906583714493165568,
            6457805882873333445,
            9264026606607255567,
            5661385612709440532,
            6336534519548615471,
            3627224503599667898,
            8041150832093296998,
            17297426873182975620,
            5424155578468934811,
            8481622367838399058,
            18071207143905957221,
            13207470843657332157,
            3307561075154280299,
            3433365185300383819,
            2338408959499754643,
            8803066607580542845,
            8251785991901796970,
            8917888247734908543,
            14864781702357404734,
            16923360879644045622,
            9401086027475385425,
            3031110610751791661,
            2681571422936265594,
            14498538209246895460,
            5863218716423708023,
            7445609619885283304,
            4713976769614030644,
            12000023057496721749,
            15826363928397787601,
            15628891205777778468,
            12469938649776304215,
            6583845492230015144,
            7356683705050007169,
            2367537501785421913,
            84949991744259258,
            10798827821302321512,
            1844174710161073932,
            9956712161868892386,
            16037841796610846152,
            4849428283686398204,
            16836391207147088566,
            7161293151808877759,
            16386235194018269479,
            480852531500857256,
            9092916415707061776,
            3547896090157707592,
            7791935174335839719,
            4143513408047454072,
            10989651629800639133,
            4022062904301431194,
            13683464012449881051,
            3671721629428439363,
            6210790641993894797,
            3004559992190975887,
            4581624809637726342,
            15480219221586551346,
            13299852905190458157,
            7012253738436341261,
            7779997060957511251,
            3918059842899883187,
            65068744210768679,
            14654376987202799389,
            7594192932054921299,
            14371470108979899343,
            10463833074540137184,
            12605870491601101468,
            2915415380894273784,
            694845969851993871,
            15784005139428892299,
            5466191968964682392,
            4330633216330496364,
            15833999786793844681,
            1722682772408189235,
            13463756154187773642,
            6964644695092981406,
            10423398122467503942,
            8020130946876007799,
            12975496394472419596,
            7102078488628559646,
            10169298321819443437,
            4025737750965201006,
            15245491547942471870,
            8437959934133081442,
            10078290769623134685,
            4405699421321025280,
            8531060269065767580,
            8123005881366957692,
            7345864402026418121,
            14453043194454694950,
            8366332675743052734,
            8814217858162666700,
            7994304794541380348,
            889523148431137842,
            12372535449743938727,
            5593585044110345047,
            7408926041675447751,
            15881895897141837271,
            15297011819116778612,
            9612060008973119390,
            2630979951675526218,
            12030135924177604387,
            16635232055198429054,
            1289326023081914512,
            17935937918000376625,
            2630462394306268333,
            1130194465126742521,
            14846576835393901269,
            17705782020736054768,
            3004695526413034818,
            1098621706954606993,
            11664555459866082588,
            17099633472014763720,
            7785686754593729066,
            8002965001324592087,
            10666303835622557635,
            14216263979991511307,
            11198730393149771138,
            14958152544429714194,
            4487579442817450669,
            7078906691642194083,
            2448914124815146363,
            479069383226390989,
            7558416146727142656,
            6040055854081747468,
            7136846915686281300,
            16102827969065019639,
            14822611424767759748,
            8235856354338640743,
            8996876904129191208,
            16928098861337302994,
            15049956268890599500,
            83593070355964376,
            496237151516236051,
            4778777936570995278,
            18186500916537703964,
            10943395081708003888,
            10725359091624142678,
            6562048119316950478,
            11874140005578633157,
            18373419681301360169,
            4331151142141221458,
            8379175411928564980,
            6669060017200103147,
            12262090250732696557,
            8731324873398379715,
            7178297724416842337,
            4680187466093996423,
            9340070788853636693,
            7813221393636046168,
            6753981736519889767,
            1327363091505078083,
            9707009524387834115,
            6513717507022464199,
            3479995898189670918,
            10496450252660939631,
            8918333074473442754,
            16894155879920970093,
            11594482479017181373,
            1013264861101072481,
            15138042115433148760,
            17076666935667723824,
            1965016727825804547,
            15211915652340206243,
            10723952195856891523,
            16571731554090521099,
            15234711471110183538,
            17593122463489854112,
            13765391561338635470,
            10758834374706002501,
            15755351302129539587,
            1929542122296365569,
            14990460886693399126,
            7255715500122990037,
            348866711017577218,
            8986433949975853592,
            17369669629832972534,
            5222885281904554469,
            10384746766959687237,
            5899310985953493294,
            3094122382077795080,
            15797145615396298402,
            43042220101694254,
            821573003629861569,
            13941684374255860562,
            8782224688562551212,
            8850480589737605488,
            5607547750233135331,
            15363895624951650094,
            7139775678419238234,
            5948443002702530713,
            10328097795809958974,
            8449761063176633954,
            1802956076600324370,
            10749013393051658891,
            2082525655472294498,
            8535252035291700212,
            4005521314771830557,
            7067222379557866365,
            4830880138560465266,
            15625460678599478139,
            4275738095932320268,
            2242572045009375571,
            4900295638013831403,
            5401906183193547448,
            4476288964569490957,
            12104031712231500198,
            1619647828455941690,
            15093368831833650047,
            8059747839907472312,
            6980525344823939160,
            1621914843783738510,
            7714377030806267660,
            18140052050091171580,
            5874999795926798456,
            13909852932530513648,
            15314994087127184633,
            11984898718092350537,
            10153151039040666746,
            7801306679350164149,
            18248439322478991559,
            17094961014126073238,
            13592776112668236996,
            5109604213741072123,
            11134257539206368109,
            13602967446384791564,
            8256354250634399306,
            13039943388684369163,
            12659329877243060508,
            16144383951090397828,
            2909726958247468982,
            4425117558618367600,
            3060430377717116768,
            15113305174110052739,
            12136666271090076865,
            11627695918718011718,
            9812717733916150131,
            12803863244853704959,
            5868717770474884688,
            13831450542419062501,
            7512833908487128329,
            17097670329605193800,
            17907220709943170408,
            13938234187495358002,
            10232263328471213326,
            13528206592783775833,
            5662453852256248769,
            2598144542033351072,
            11502753020357150517,
            10478081061647119936,
            16828809220269996186,
            7848077553772945457,
            3319060205448833129,
            6968200304521460990,
            12817349334255451711,
            4622957535423248748,
            10692118907540091282,
            11321487811279695600,
            1897795539161945432,
            1908365082875526523,
            4943634704969953067,
            7383367560036755077,
            16351484354395448715,
            14135621800671382981,
            6418018767555789959,
            12895503019759128927,
            16600389076014783798,
            12586783550616269497,
            7584644730970206115,
            15604146389747427296,
            7642118800791995429,
            2851511496237347488,
            1279679737847009595,
            12654923212732082959,
            14928057250838769819,
            15688707392537063906,
            4355032756504950228,
            12212200555432748760,
            5919195350128355739,
            2343647372419138215,
            4144094081962893482,
            17087818518085057362,
            1878194547478679623,
            12346553311990078346,
            8542350382911051833,
            18093405426930851033,
            13241980987497859839,
            706736986444209782,
            3461335337130015472,
            17225396041635108472,
            2261108715061027706,
        ];
        let mut new_vector = Vector::new();
        for item in v.into_iter() {
            new_vector.push_back(item);
        }
        vector.append(new_vector);
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            18109323956941351451,
            5372473040573237141,
            10618968779971592669,
            6714263554747653349,
            16105443874938514573,
            13853075557974782046,
            495365312195052045,
            13138362806598688052,
            12108928168437543774,
            3083524506060132812,
            6173537675097874830,
            11244911116739490191,
            356376427723112133,
            18188842492957579193,
            3059331468338895833,
            7195852665838600624,
            11715631320038589758,
            4899877219026658874,
            547522874683814821,
            1358806843693637242,
            4430716441800160690,
            2220660263048960099,
            2726740199365650155,
            7221882922118228471,
            13566013843239511823,
            15303967188750531686,
            17964928728663670800,
            17920694175671454759,
            8495058302716262533,
            10306239205704136651,
            14124247292863555295,
            3454335984288787824,
            5032970031578354533,
            12093755969967172713,
            2856346350417780460,
            1345195619424875206,
            11952866564329708738,
            5826892279156755216,
            10470793023239900219,
            11434981463775184068,
            16462889152755215264,
            6746457490651254781,
            7037970407346305460,
            2715223121450558440,
            17694003886118484429,
            12896615741391677234,
            15197179366090299100,
            12798031984522977730,
            3857885692564673917,
            11899616843927934171,
            13127583907733905176,
            489203260894671924,
            12533629119663027517,
            8864329869281258062,
            5194308888420051015,
            11936637421999337409,
            10396536117839680117,
            14830291209329154064,
            9111793205139114920,
            2651112373491707595,
            502755214914117739,
            2021606510954876772,
            11301972408201922159,
            4229512524468640445,
            5439115626155383528,
            13413019677908941306,
            6411317330637806719,
            15921697369675200674,
            13708601903870061033,
            6126260978959392734,
            8018349111915500931,
            518945132474068630,
            12716310125407534642,
            13363725633839166899,
            7114105821018476213,
            12815442092035095826,
            14700858794923107966,
            12572879784767349261,
            10857045844822029228,
            9808316474885188076,
            17948681296032025659,
            3405462978222797499,
            12693503343431768760,
            10871128743443095634,
            250183297915136427,
            12128189149407916252,
            17778358587042792567,
            12485496152633828438,
            8745837721591584601,
            17957822513503614881,
            7861953108458859312,
            9167120592972211852,
            4538043111646578079,
            8202776123940823089,
            6961642375761621507,
            9022474901344594455,
            15073645758341207767,
            13199474933191403756,
            18416138117471599825,
            10011050317875363590,
            1777916536953644805,
            13054268175698985448,
            15781807654775063367,
            10955385994147484191,
            8256224871911567764,
            9678474722017613501,
            12287939138123412443,
            8838387199450118889,
            15867915592701659701,
            13875796768546611681,
            12720494412415057989,
            12194628403727034395,
            11045038241580099232,
            1431911245019811435,
            5831437527385018439,
            3268975267262710465,
            15848140450189700326,
            17054326462062064732,
            13338777496444656598,
            8029986898562259072,
            8841709355514360501,
            14396838353445654712,
            10051823804073381148,
            15755468608218160441,
            12265079450449309464,
            6729561425100941548,
            4737250889488024427,
            16731750066494402999,
            12434395312042228619,
            3937754838810101554,
            15041177924769111300,
            13433965819840790728,
            3499990288024812265,
            17832416904696655242,
            15713128785082196770,
            4414388940804571605,
            6109333932915515524,
            7166306236445680003,
            14663884431431181581,
            11061976455246642229,
            2278015760224454190,
            3836212650781580972,
            1833102427686726899,
            2926199611896551928,
            4125127386804793779,
            12314628984101536501,
            12128997759432233982,
            5820030158405610400,
            790706820840712868,
            7189287558186298777,
            7458615632189901369,
            2800426311149595534,
            6794698924256778918,
            9735384697481133484,
            1170651192054935796,
            14931269406605828790,
            1568611057001442506,
            12102977173523775232,
            15211301019547709474,
            16470799347713764114,
            11740039799722080035,
            10855533224484680936,
            16341551921799233752,
            12175168831762589679,
            17367511028893461904,
            7325651926206017136,
            11863795414606617302,
            16445997137666734149,
            17646754118923171339,
            14710935648426219261,
            14539736707721216674,
            15119120518278778304,
            7656316799345612627,
            9338384489421180151,
            9231886191356997216,
            5212732062032109089,
            18137274196682642895,
            9922417639357933885,
            8074303922629326746,
            7893288950697856383,
            16928520868244996045,
            1395996076400333518,
            11739385116261364685,
            14012689519653330617,
            2037954686398580021,
            12447185506132495397,
            5526471520924982721,
            1106508627368783365,
            8469994371170493096,
            3861108631580990707,
            9269375538383086126,
            12803590308578569007,
            8327199022473073979,
            7820268547366126131,
            2834280438043921346,
            3397345266489967598,
            12438000741314696598,
            12735815693440417884,
            12535897534062395505,
            4949965341199130270,
            488938691084724537,
            7128657990636142511,
            12260107246406190565,
            9628194158809655648,
            2958718047570801251,
            15636609360932146981,
            13994738974138587361,
            17197512308660922324,
            11588562012621527648,
            1745610680258081941,
            17713338450971493581,
            3124837718303499278,
            8034826069943935459,
            13527208650462834115,
            17550484285030552302,
            15614391053585003719,
            14992101330559622759,
            3918205227015503346,
            14884806059450219169,
            9118003853590691327,
            12507243237497942652,
            8052529686424694356,
            8588426668059169860,
            3503718163065143711,
            11911558792062377063,
            3630173609811567558,
            4633170892817785776,
            17912030238717169502,
            1312490078284196580,
            10961126900065030868,
            8222005470117753430,
            10812938867918260213,
            5092293988834440701,
            846695410919529542,
            437597572698309519,
            13845349731018962545,
            777464005433629972,
            3244588044268845254,
            8420006226028592978,
            14901944116961633397,
            2964249172170802094,
            7236428919970578659,
            4699736168772455927,
            6086751814479150089,
            11697659269592254860,
            4839865581784597127,
            9802530394770063791,
            3931932854744154376,
            13681862490819645180,
            3981325570644034381,
            10110943859782542602,
            5291574352771074913,
            16679580625015847435,
            9659439817957970430,
            12550991877146259995,
            786209797591484032,
            12535967329448413259,
            12887352531550844730,
            29173578954134636,
            4744870652779425728,
            10772719362853204784,
            11452281300200093011,
            5304995157610838502,
            8031945519017035699,
            4162042615190208377,
            18248622884974959814,
            9548700890345561413,
            9726069172230062063,
            11385348753153854186,
            711083829382910344,
            124367158289321442,
            18012334444632017862,
            13300836494087236891,
            17406504470112031511,
            3552887719201259876,
            16157809046368849247,
            2930064351535125775,
            4854194863538547398,
            14949503338676231572,
            417184137265530995,
            6372891802445133334,
            12282994079740590199,
            17359541428539570074,
            3357711451430026251,
            8198337511469313160,
            15679903792697880065,
            7537990759834023832,
            4688970837853509915,
            16849254683948594085,
            11827913489008909471,
            1721147547953904410,
            9724669257593321941,
            4357605745045910061,
            14194362651930077582,
            1856582883087230372,
            11789788108939720141,
            4803603772164319955,
            14306990648062508343,
            17953317697406125615,
            9282324780680194771,
            14187653089624932542,
            1090869041311456624,
            8259085160355795416,
            8584238751248450520,
            4414710714012183195,
            5754450687650378268,
            17620061987543262228,
            13581913951407239010,
            6968227878045744579,
            16527431591290480173,
            11514334549289178422,
            7981034012708122214,
            9588030035643838867,
            11805605757841398581,
            17547992409812977071,
            4730071292276643506,
            929186189937976279,
            15483972501016371134,
            14836023455794108152,
            349130906105753342,
            18324453107713497484,
            14549596751421372858,
            15600355261952243627,
            7150107384834672824,
            10349400113263802855,
            62421448743000590,
            9260279957398845750,
            9185720242704751786,
            5774250910578677136,
            13791857248491206628,
            14621379694086483225,
            13797834512337056291,
            17251310410697047435,
            4694073484905174765,
            789971810188090334,
            12189372008851771703,
            4657839657934099195,
            7104454388292160862,
            922943650746334572,
            13135727773308580440,
            14438109480449917072,
            4020101659624740059,
            2177281037485393518,
            10881639032436059464,
            684790205484970956,
            360888033785906651,
            12633601108563977200,
            1590196075746720241,
            1159472982105835474,
            9534531374810702464,
            1025106728612591098,
            18029832562029659214,
            17569748541291431743,
            17592360737667026387,
            15182260152894421423,
            13732078005887920897,
            14756027965085679757,
            8692236928365487472,
            6421699069375179762,
            10794821822258172140,
            16037167708090102593,
            3784844604473485285,
            2265006812994389400,
            17243228331907082675,
            17238556889415730726,
            14286233765592567142,
            6812597278134440689,
            18283944290459007538,
            8627228843910307316,
            10763237768926347428,
            18049841402015757133,
            15795600045878028929,
            4846921241702559346,
            12201933322838125559,
            9348656383737029087,
            1216866241105357659,
            8458034721520934661,
            11490319516903303649,
            3154686153997539015,
            18199514304062977637,
            5359016406794506374,
            12704236697676965064,
            13641206715249045311,
            10601611890596236871,
            14324580539327258677,
            15360855473603131333,
            3931962832931420404,
            7226178609115256136,
            6176060371300230003,
            11580119638109087110,
            9607666944814027641,
            15225079136487574725,
            12746094803582297258,
            11157512625410199453,
            9306782753132701466,
            17364080964477654000,
            10333328437689724989,
            1089159162187812157,
            2837123934094331693,
            12601791555510660344,
            2241748351056581305,
            9425494596133921472,
            14007269504091576738,
            12439848709982525176,
            6270427567125085635,
            11384261821375018518,
            5931420706012817673,
            6452099205397239179,
            4439563027699584912,
            61665021442002775,
            17181265875202082865,
            6102124328311631877,
            3653604921004236811,
            12847174066411044529,
            8272184530698858957,
            2270751891826165736,
            12472543085810650741,
            3506823619610560742,
            9096420555988983901,
            3910973987289139461,
            5454929140977561708,
            6928086918845533429,
            17415663044190097477,
            15011982844710665154,
            9039905477419747133,
            17784049401642770452,
            6003763442495524381,
            11803931364672644073,
            8928426939173114926,
            13307388719505026326,
            17743908119648038725,
            2309204971658963993,
            18356250114587272603,
            2011875131055022786,
            3260236020672418372,
            4615238776375449472,
            6547865563568891886,
            664370507522293493,
            14937936427569838155,
            5587487389768711261,
            1432664306500953057,
            9748422589841317288,
            650054187743406199,
            6513120062691005948,
        ];
        let mut new_vector = Vector::new();
        for item in v.into_iter() {
            new_vector.push_back(item);
        }
        vector.append(new_vector);
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            16948188768394490071,
            7483397615827071156,
            5742156617882565718,
            1951974232093484068,
            4912195790289629927,
            13254988813351656146,
            9035389666059780450,
            1800400031346471137,
            16298025475154576950,
            6052747419663586730,
            8277780921489456992,
            17846378238864922054,
            16140397944101937484,
            240494039618946364,
            8956614847119964726,
            10901225888166038551,
            8685779038691260522,
            18368197466179284071,
            12779428454941036057,
            14394042849209447522,
            7857908909040554665,
            463569630042207205,
            3926064257469688880,
            15637214268619962693,
            1192192832687825961,
            1129694564124690699,
            14508202846646372517,
            15052235903420044241,
            9065948282959317934,
            7449235251434233273,
            5813860805169072842,
            7576382319769002733,
            16980611439280369080,
            11122968570147689821,
            17396134947006455869,
            14888898024314279162,
            14059241690621314186,
            14196652549788558687,
            16938678091377048164,
            11317593546357668482,
            14287339380008562728,
            9922191929630260627,
            4843815640520512742,
            4064317841936684721,
            622677772390970085,
            17048770759396194125,
            3791426783918603527,
            8875237495055529906,
            14799823882285267893,
            12731752721145526995,
            17645243187473453171,
            15459652312989024709,
            9588274984956183479,
            1494692853064681845,
            16879941155920682707,
            1270311621496265384,
            5131069673474640077,
            12778502189709211624,
            11379405768911986421,
            2437453271962729661,
            4246916290324862781,
            8024174981152984471,
            1732863694873830604,
            1905792954976701294,
            17301581985000986300,
            13773523640396043981,
            3728796200786998811,
            10605767486263024797,
            4371266045122552310,
            4724354505507910203,
            1395436312111757597,
            7409336140612983994,
            10341042524851744550,
            6239949491263554515,
            6834885440880936182,
            3992890204030251918,
            4644469260757239403,
            7354738161270772526,
            11764816096132359271,
            3419040546706131335,
            5532459317054399316,
            1349189340191978692,
            16363886736711932133,
            4368150622210160918,
            14282063037349146657,
            2915816389559065885,
            6689685572342114572,
            16516781398664907224,
            5326688401730855989,
            15015022052543360536,
            37141660582648749,
            3359639580885919021,
            14262414168339042222,
            16733260568845836916,
            17038830621435570095,
            4624509982879216205,
            532577823994778322,
            15490837770718813195,
            10793650322650638871,
            3091452830885673994,
            4957610662146163522,
            7153124992552446356,
            1418965745275107236,
            17220570317397644770,
            7206719092700126267,
            5385899006714734374,
            15082063881603834750,
            3592221206724411542,
            11340403037124079784,
            14489099668734528555,
            13536072618690988926,
            17731437805114217883,
            15278038057162236731,
            9959866965387916897,
            17543646284931067031,
            8844053599121744225,
            7134482234607331259,
            13554578312576900580,
            7958509903486963436,
            9278274072664848373,
            6275565747124208250,
            5790844772551950235,
            12903781024016842133,
            15745695546915409966,
            10588244871178003803,
            8367835768450623982,
            2252499676036578592,
            18045794395182311169,
            4536185175561560885,
            3854099689955442246,
            16852196569560514281,
            10787021052955084202,
            12509488337155895551,
            12841640513544816007,
            1425463924567478056,
            977621049543848297,
            7755961414178943202,
            3978516544510821310,
            16605945548575107532,
            5156562981990102120,
            3242632345027723677,
            12538500735752038016,
            7786580905833201484,
            576091159176312370,
            16118409598440057938,
            4240335085571601416,
            9135715208616932976,
            1680742748236157004,
            5104178925985407486,
            9259407084355551427,
            12042500412380685271,
            12130767040487335232,
            10993961725350284248,
            14631565198937602750,
            10075799436755140829,
            12273657084407912809,
            6049442733863716963,
            3767250291409697189,
            11175193811908560113,
            16281237210784956123,
            5467824433724445765,
            193068627173641818,
            3677727527883552955,
            735032094435259242,
            10227024467365741944,
            6117853780760403058,
            11144054565078885512,
            7497417101236769545,
            13850712453739734601,
            14454425549468235610,
            544079850131832388,
            989625665531868242,
            14712145589844407561,
            15054635509260658402,
            11634779820591403568,
            15175414012554154405,
            388619808464791513,
            9784271255229261268,
            14115055259472952106,
            9358448268008929763,
            8718286031125324389,
            7326527087655296387,
            17243451121458038828,
            1385199007942327471,
            13750376658328152180,
            5278274551999417867,
            3813165916017359026,
            9204330998340715464,
            2042681588603950149,
            9963420320145845770,
            9128409540768484403,
            379078091210069495,
            5231560324319978779,
            13156394432068360711,
            6824678028249904832,
            1759946709216484718,
            15465986570047577690,
            12982155861438503137,
            16411014474339246416,
            8645635548058595339,
            15391098534334173263,
            11584627877828200070,
            12250296856505295292,
            4567858449877937438,
            15448173743940932761,
            91491659109874980,
            4922927446733818957,
            12338631150562508907,
            7419897977074504286,
            4513946344528404760,
            2355667293666220186,
            7330597621803985599,
            533513619279043441,
            7945888175978586417,
            12714053212661608041,
            12983904739343784574,
            17450960926107872955,
            17831882425885765775,
            4729226334469704961,
            13283037637896308816,
            14437718797684404592,
            14500686189284550040,
            15681188070322697930,
            7964768545314630303,
            14173551689638119024,
            9008668295999649776,
            4410222922910690949,
            10012097023402553479,
            2896187985263661015,
            4529735626913382351,
            16072119690924221791,
            17633885908101171539,
            11146775513820426020,
            879326361335423532,
            16008303565394844420,
            1320329533744727536,
            8390537345753463428,
            13752686685337120161,
            10352713399628913729,
            16056498654091267762,
            9217042307594391883,
            13417114550027674995,
            13534621443542531997,
            7468557489781046940,
            11730842825055544267,
            16761824131386482290,
            731793992132607733,
            4021668578596287315,
            8287016649432873993,
            2190941676761263438,
            10324542527121053509,
            3475415200964735561,
            12526223841675152481,
            7392854042964184800,
            177449618246198,
            2686292646623871066,
            1226161613686071811,
            14159532067152668601,
            3153549005353286165,
            3595595166054002719,
            17376347882190997438,
            1152283134474190188,
            10147610780627569129,
            12747812245965034702,
            15452141492836670383,
            17414139788282168429,
            7071107158864890622,
            1093543014453829636,
            14107087835550750884,
            1337339628649995799,
            5047351178595877745,
            3680075596651727214,
            14666841990240847736,
            12346140850260920856,
            6964046638783001849,
            1179358433063891465,
            11693951287206120986,
            17256568827492553371,
            17106353457574899512,
            9394112203415000498,
            10584314649183022891,
            7238190211589418663,
            17693313817532798976,
            11718198626266049879,
            12053914497749782093,
            10760628518667416825,
            10142025953211725754,
            18301795396118568828,
            4717992172743449479,
            742501904649057068,
            14288039371757338241,
            11509704410341769161,
            11469001267241085478,
            13083673977985800103,
            6894792456766043532,
            2383016254443291664,
            5052785106794202059,
            997174143404354861,
            15329587980536067285,
            9443983416140805164,
            16118879369957521631,
            13713575802121580163,
            15955462685980869480,
            198131923323886284,
            7233350270105923681,
            8405469595222893468,
            16281080488196725886,
            14570386880453637295,
            2308913385125473715,
            12018858513095782,
            13689200591655198598,
            11976406699285269461,
            685896708702306181,
            13970130691645747887,
            37265337005544707,
            4754600062805616469,
            16637450354003977870,
            4389993808194047232,
            7538151622236361947,
            4247936063192665404,
            1255769822142077980,
            13009816961727760211,
            10675841837500026104,
            15354488881877415202,
            14252957450354240479,
            11673207694227598556,
            8734013134868685321,
            17560653038946307926,
            18056892869459938269,
            4960028377255682794,
            12069212699126377974,
            6586178095429460557,
            12604733936010808559,
            4463903151152757917,
            1245869410098285568,
            13498000932209834455,
            17240792325260304456,
            7140727332074527141,
            2683973889533162889,
            4986963360986548106,
            8102861621378811579,
            16276831877117830578,
            7230429187266771425,
            1890935229017041859,
            5667685832174483210,
            1757386947222656703,
            15004641064215768149,
            5068828141684065544,
            13884068609370894742,
            16889234665770122647,
            3684610464892797058,
            3578887207051281367,
            303556744507789335,
            18005824990639359593,
            7519158659713523814,
            6121376826296452386,
            15418977238910103727,
            5323304941238660707,
            4432283338993960969,
            15244697809704189826,
            7194656221114746303,
            11650548119987804770,
            6690363590637555595,
            7090923293136180068,
            993236763586466053,
            12147907121527996897,
            2712308230232429911,
            3071258738053246321,
            8152160579254563769,
            6513881724440334518,
            2995672696854431994,
            10892648009880988483,
            4202115906111300909,
            12091581514205900919,
            11329256229182034007,
            16842833317005019857,
            11370056483870139452,
            5166703632711514094,
            16419620996782216415,
            7657612832720893036,
            1697128184024708370,
            11977432772495961674,
            15197008519132979483,
            6476773189643496432,
            13246461770819080083,
            16617647800528458647,
            10617736674691667117,
            17140493526045010236,
            16616741406234815609,
            13930210332531513025,
            12418533098509765311,
            1168347077585569511,
            6685575126830905834,
            11019858271755690237,
            5917243677005929497,
            3293096398083261601,
            13896800979110686369,
            14967348157450158125,
            16201993722917278845,
            12726368963739083135,
            3796218359738258835,
            16428303672364781819,
            15487956547616799274,
            8006814999036283600,
            15467931249363083901,
            724348067578061554,
            18014734905180611996,
            1380450269040266205,
            5850145838142962537,
            8926601502695929458,
            3135659671305756448,
            7297379073882193986,
            15481095256647113140,
            3673626745246912445,
            10292973900035147913,
            10093827919401682835,
            1847937563445310911,
            122990134010572106,
            9465347519101652713,
            1780574265533613726,
            10431457972294585899,
            15595620516813624822,
            14640236690203908656,
            18004688087958745080,
            1357354861898772601,
            1008555570067540551,
            5153861566920802468,
            12394926909324490992,
            15370011619294614286,
            13748120742473722183,
            9766478499029599266,
            1017865772161158293,
            15302040603826841843,
            9470727105613808302,
            14655888001975390987,
            1643026342061413402,
            5386696526182987626,
            6847165530941512659,
            7707702395135130625,
            5386603349173177678,
            12318756976077510058,
            6087027522434666422,
            814298577387628083,
            9074473926528149043,
            1181505244285317608,
            8430470369642201922,
            6899179683814747279,
            5915312854331817109,
            16197648669156554138,
            6382868055050686857,
            14831296625279461561,
            9946783901285407351,
            10912497023587388235,
            5175276546051231838,
            9419783644560427794,
            4769692297142628657,
            13679254919142928002,
            16754590566055948958,
            8438782193611964941,
            1206992546677912867,
            5576720672083627302,
            5351855511180942865,
            2021987904800538363,
            9659423085407583853,
            17271011713336891525,
            7943444908205954246,
            3750320106792306987,
            11748257287511633570,
            17429255525986317750,
            10327162322780950914,
            14481609890727833927,
            9762217172175458206,
            6285869546960648099,
            17873633035810021874,
            11860593889361417021,
            18376834603254159096,
            10076507281819732477,
            8331958279680679791,
            2419540525020509539,
            18313070143393995293,
            2211532208432461309,
            5034134855143443870,
            2359581360666456734,
            16653824210158833131,
            2666660501889515058,
            16644137040716021283,
            11051812728093919827,
            4571908749904555491,
            936954795904610097,
            14432128638050552949,
            11247984480000547591,
            17733356122139174230,
            634478801463766789,
            13939470770324990914,
            2130179514102729399,
            10593640399904200416,
            17022122177340334188,
            11254181755565400582,
            15921201060653848104,
            11042164775195414310,
            10932348384805297216,
            1354468458305597951,
            327124166109045518,
            15085640465269688462,
            9754267033981213646,
            6033594947994825654,
            1266428622033360597,
            7626762910016080597,
            11520759858425994534,
            16777841684114262160,
            6432404447537433514,
            15167784764169121026,
            7245392797832258513,
            13777339875296496597,
            3500341724448111195,
            2410758222735001787,
            11488998528493758481,
            10135299310155736948,
            13642346775468490809,
        ];
        for item in v.into_iter() {
            vector.push_back(item);
        }
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            13866074553085241000,
            12647304708430078345,
            1637025332794988334,
            7053022271778843336,
            1123332855740210904,
            12197942513692021622,
            15359447363984676590,
            4384248352236410315,
            16583779584662021477,
            9628863481658168098,
            5744622026068542174,
            5014454094221201047,
            1839397506803296648,
            18180237014332128809,
            15953663587643745789,
            5775821819607792461,
            1603861444030784904,
            11465209250508075054,
        ];
        for item in v.into_iter() {
            vector.push_back(item);
        }
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            1445633002330512032,
            979437831613277354,
            1261136482228695152,
            9098648006959257280,
            1736670847652268192,
            5958379859097802746,
            17141216890642394064,
            17147212961982925910,
            8640290757585361728,
            13021434806540163836,
            11585065903112546884,
            12044623679599947521,
            8155783111481412320,
            3302481165642897763,
            13408264063700975536,
            18442520363316073169,
            13379460973798716836,
            1498817092281869023,
            16186888316917610478,
            16227177779867392196,
            16664661995153042447,
            4451134493425583440,
            16909317189300227161,
            6101857068006786380,
            14341085181613528828,
            1388437364113204166,
            11195888299151633680,
            14899978120860077486,
            14307103773615529862,
            4356685174733272641,
            2653792313906923708,
            4513268471651176438,
            1436340562219314202,
            17853816688664714970,
            17673467787063282783,
            6313431720702961327,
            8839218746749267103,
            15178215975846533212,
            7285434478416943543,
            10101697968084138036,
            10506139816716070337,
            10126097192253213141,
            14558434037919852933,
            16306697908063503829,
            17205927770150357723,
            7048324586836461901,
            416677693832294673,
            17549636501318048263,
            11167793641137684353,
            12122615504296172972,
            14763616057993606398,
            9784931006808431907,
            13954379314023797535,
            14973012497273520230,
            3105273062649841627,
            11610248039011937290,
            10001442746035032019,
            9206943288916923930,
            7334984326219483757,
            3701415912815438146,
            6358963732768952990,
            15059749181992342414,
            13635590448873801071,
            6357894995155327928,
            6276599881245938984,
            18048130541744057446,
            13690807524237234151,
            14167882217269365464,
            10168849567364862503,
            858725738355215439,
            12135696503952352821,
            9064761509241229240,
            12381208920046880773,
            17982222135967830486,
            1064582928508558072,
            8290582132694373997,
            17064827544866728083,
            6905578828977380947,
            4313139915438014726,
            15930028928519705476,
            6703566559722552829,
            7110777033686844958,
            1937714856412078377,
            15561434840128521770,
            10409168130874692731,
            12298391255172554151,
            14599348498617136090,
            10863256136953230254,
            3665501809519006312,
            11779358074027103115,
            4428155290364907583,
            15493930817119574257,
            714421348064588445,
            13305871117197258276,
            9586232663045396132,
            7922436930407265270,
            18159009101310871839,
            3251230841358085881,
            9380671011714925929,
            4111569559319377676,
            12701732342079097496,
            17551440440833303729,
            1609923385254508549,
            6329070591170330277,
            5185230662559470203,
            1299405141520193962,
            10211296915166559987,
            7642110332186554615,
            16415503273019441198,
            2799327583645533166,
            5032713411853781027,
            9650891976923399651,
            14704437213635875553,
            9753241279837994762,
            15812400265132433921,
            14302098099504306001,
            8718992297974556916,
            16066012340216834230,
            7636513567109143952,
            15909455480852322939,
            13887126082358226850,
            5646503991821129390,
            14086873406626427967,
            12065532384166144235,
            9564327561899094559,
            11094293411462725439,
            6691988304604925998,
            9050364273776668160,
            375349219223770062,
            17971874426382120870,
            8898301613055400429,
            4140319427175225778,
            4849118348598321963,
            3235065555402989067,
            728280171664356575,
            431857954395602711,
            9286716527845955143,
            2309698032044686957,
            1028643405629859699,
            8432256948786060691,
            9029313078384181836,
            15492217007308809797,
            16681339050305404779,
            15446266241386742925,
            7339092060739478430,
            9556316028402843144,
            18042713384096042934,
            7053770711110316864,
            15920043631252399266,
            14221835576954454180,
            8994295971712527451,
            16948251505593396343,
            10208669101936244855,
            5740960605568873442,
            9975992476528398264,
            13715907841120088150,
            4039726820556953825,
            15632048655948896003,
            2706861230948707304,
            2424793749864683444,
            6304634078825298470,
            7650605559161099396,
            15555225303257356673,
            14765947040597808379,
            5255839966125900255,
            7172400249561926488,
            2741453418783269971,
            17315850782798176158,
            16957068748855706788,
            16781668113015530776,
            6630892897065077394,
            17582575578027239558,
            4870482854287842676,
            2663544148034313950,
            9414105401981980721,
            13092490117660748295,
            13654405747325884148,
            12241876860768315265,
            12885032929551633765,
            2923859768983598671,
            5653371808691952251,
            6114739073743583185,
            13713992966118673690,
            5642518918487174114,
            14698968591915693843,
            10297126948119879568,
            17777126364380804580,
            12939957032625596458,
            4313497627508126035,
            5706871734954464201,
            6153492032663321945,
            939562729793807043,
            13396604881877722593,
            12299509005510136304,
            16482715109355097637,
            11572696046123103980,
            6959393542249013312,
            10684006856386807067,
            11818569712228452631,
            10658006337287067750,
            3361060608948384678,
            12589831587407589333,
            3401601570266071900,
            15242496580754150156,
            7384995246223203950,
            1900461754426011649,
            5555038363040554726,
            4177221332847871683,
            11527083759308847308,
            9527138894959065592,
            7690306898802747523,
            340158765504401702,
            7343792920521430544,
            8165533529680942924,
            16305736741966145766,
            6270185653311058460,
            4351088398430985520,
            7957870094371837122,
            4715649672749745401,
            5823850981936730114,
            11257270770227703204,
            17623187060665781106,
            7303863498545972888,
            5111022808381973512,
            2833921595810749858,
            11966357519235956019,
            199059481393934874,
            7558839225420524027,
            9232316030404345079,
            8979438242768706124,
            1119721885319006306,
            4330403878708378629,
            1053119267257131572,
            2385754191530374809,
            2896468263769813564,
            5415761987487515918,
            5781289732580455415,
            2095267726379005907,
            5679190635024876028,
            3428357134292043807,
            9905882758434325099,
            9100046709707762307,
            17923931059962813752,
            5005476051003402350,
            11688893745578869366,
            10024383411369630219,
            8423668216790657960,
            357042254814682115,
            5663948019397078927,
            4020542734700130642,
            1627049790866410124,
            10480768385078691675,
            1628438349195611834,
            6857167586752833943,
            17178326956817788453,
            7625876819293156457,
            4650685329705923005,
            9243265251981318916,
            6796223075682630776,
            10419175387482905790,
            13264276576076362507,
            16291680423979159732,
            11780729755873421622,
            3832904469939793824,
            6240341544592480268,
            15241732066790374902,
            10390985203670859122,
            8359360783990272816,
            11110859194231346394,
            1740660341575022975,
            2992010463923838895,
            9988480966975067968,
            5636932625306969103,
            142158800796637197,
            8500538385945364036,
            16324415458511496098,
            9434355288216782206,
            10274373726671661330,
            16067966644032547969,
            12753647994714874376,
            11380199850056747008,
            16503707280735775721,
            18226767217499556516,
            8884400353428778606,
            17594122495497447688,
            16108485268915606798,
            16958959631079955254,
            11467988138681177919,
            12070047464744832036,
            8569512805357255267,
            14970629013640566208,
            6971942654215832371,
            12586590602240595059,
            12297887366773064296,
            6622325245046364520,
            14867336454392445191,
            8035720967295674286,
            5057443545453424053,
            18397723075676780507,
            11719565385921373196,
            8000648333020396216,
            4970564721971971436,
            3595440844954854316,
            10243548729884490471,
            13721306954805991304,
            17539624252853444408,
            5112113809875627417,
            7077622474221876389,
            3246834272426958864,
            8742288619614443009,
            17757631312744842893,
            169253518253846732,
            14030112500885812698,
            7245797005570900571,
            5327195707762596298,
            17258203212533693789,
            12737000274030739164,
            9498617254888082077,
            11048348167825516371,
            6564057511812213490,
            925120688899667167,
            3420271226464780452,
            1704926600400278501,
            16115617751831290581,
            3054008672811744851,
            17439381125299348187,
            9610820609879578847,
            6241608283893141692,
            14327236250234744248,
            8496729826071288041,
            10846691589859308613,
            13379607060963330514,
            12983346441355918085,
            10245409671651890614,
            13387811439933929386,
            7911124565152550419,
            854826428232209522,
            7989611805030640814,
            608713362785518645,
            11471259287662689836,
            14438958498503650086,
            17855785370464278920,
            12527151929134476644,
            5828330307172110696,
            7308078765758682712,
            13497048346369306269,
            16411522945396873787,
            7274348291378062400,
            16374002292048440174,
            16180164982002924212,
            3786317577179814315,
            13899563547616261800,
            15974708074806550943,
            8653540078416740333,
            10675722211236072869,
            14368044849144647451,
            4848448497030176677,
            10754930184484380619,
            11429829496601731581,
            13797941106858627779,
            16324018897131465946,
            5967017226960988367,
            7099189723652322754,
            18087276027350174104,
            536879552515350681,
            17085956919163098065,
            5445585256020628301,
            348120515879394061,
            5999864303020144614,
            11258187862330293504,
            7102183942272397897,
            18045327341539735328,
            889554934995386498,
            3129924879480820511,
            14354888717937846634,
            3974956792010185492,
            6431746839682134608,
            2912494043473177212,
            6931570934569921449,
            5790524853856862338,
            7932912494495217099,
            3574128368523495966,
            8153022615984265090,
            8041827190810488311,
            7218239778061890624,
            17470165800907201541,
            6738536330433660096,
            6475442116181274209,
            11400923321845533048,
            1127749066112792439,
            5707165161944498471,
            14081282063503686687,
            7289862115895901587,
            17351696412729613057,
            9909363889637542086,
            9960466025479112320,
            8990204599660260216,
            16191860691400872182,
            2620498144985173133,
            427501405564867119,
            12309460904396636462,
            5746201081345181861,
            16700131504546257733,
            13713177132825925797,
            6474018339229783823,
            16218306396620539790,
            16306747690058840007,
            7430949948052410570,
            4593792319023041560,
            10842324885585100880,
            8170956003171239592,
            5743312765742274439,
            15956465978730503687,
            13801322574908624958,
            12248788458038778511,
            11036931141585153887,
            17467876701382247856,
            11016040983110445413,
            10966923304567539559,
            5171778413858545976,
            4770733719479965070,
            3984200460631407529,
            1080628645225199701,
            14981883099372345540,
            8521094894169243614,
            14379800845506000648,
            14008545855546721581,
            8758757657085027467,
            7873858242848786883,
            16104248990104473353,
            13229747762005424970,
            9206860662315502890,
            2626472327983144381,
            517776124318396394,
            1712793217977162221,
            8737596992389894719,
            16505164875692770335,
            3874648156408869553,
            17723500982233715860,
            17664302766257403328,
            4307847234743404630,
            12126940240707352486,
            13982863769078073147,
            9950236653974795892,
            3844180064888110881,
            17888921567287710140,
            207226638081742059,
            12388268070222174740,
            9625868498561657193,
            9385394453457242166,
        ];
        for item in v.into_iter() {
            vector.push_front(item);
        }
        assert!(vector.assert_invariants());
        let index = 9483910130229642250 % (1 + vector.len());
        vector.slice_from_start(index);
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            4789121114009272615,
            7887678443774873295,
            7804034960481459199,
            7543430069588584962,
            12335729318571681037,
            5884241189299925055,
            8253109481854550635,
            12977217512241890094,
            8198265522051100755,
            12977600502630994349,
            16522304771495857263,
            7488954114691727760,
            9567835521456793837,
            9274344868043540557,
            4609484298691232950,
            17677802539516814265,
            17782865265907835293,
            7672789926397181593,
            7957302195480226539,
            2112484694653216051,
            10547353730555029330,
            9930660185706948482,
            16166897663978217772,
            5819595071821316686,
            10477857426109668540,
            15304407451160866581,
            2554329510539239646,
            6239114327835559857,
            10661499083988851280,
            1623419966557590347,
            9035021213981889308,
            11755977103796425887,
            1741512430420482916,
            14560797007519904513,
            11507694164170878121,
            11224602176732447619,
            7375563299857856636,
            9172116722187553515,
            7868797081229545044,
            5950813931561295431,
            718414087147474410,
            1095431931164645412,
            6118683225668077300,
            4117637304454730892,
            8814110184418366470,
            9379441882540014512,
            16539717835252328944,
            16146011658715492105,
            4943230027220771146,
            7007083918886257917,
            6195651747779944390,
            1045483787608931492,
            4646664971123770846,
            8961773651794005405,
            14677175897297680690,
            3853065568567891443,
            8377499057726210254,
            6428931623030023240,
            2881248050131132032,
            9658762428344141218,
            11663929381330463127,
            5457317625157011529,
            11853816165030023632,
            5435296917063464685,
            8067229007461080026,
            9222739182571349333,
            16236571609853281057,
            7215654968222070852,
            5449180099018947971,
            9294111190311567711,
            1039188408158301235,
            6147915874598838110,
            13576505258764682200,
            57186914466010573,
            4792496372855607505,
            10037838114363946035,
            10854711949839553034,
            11180030736261238215,
            16990401047663528732,
            13362679273181688894,
            13170480036089936777,
            634165928071774637,
            230316644011384727,
            5972490217795952284,
            17790454755554862670,
            1634245232365470989,
            6346360744353235509,
            15913421719898335666,
            1734820511007649590,
            5550829903381520325,
            5561834945147918144,
            1809178549749766216,
            951215242987487,
            14888169021413224507,
            14923949747385661806,
            13564291046256758674,
            3925896726477904612,
            12614569080657218892,
            16518903830178873489,
            11994892935147876684,
            2137656000145117916,
            18071008613722252969,
            6490574865058484731,
            14221643081765544187,
            980062198101425317,
            18058223883586224488,
            14780434645108161852,
            4226446071549148257,
            6088438255482387523,
            2634095647833036242,
            18298899974868458838,
            5590872591438643242,
            18125352212081625984,
            15893547524304376775,
            12597943871980606469,
            11472225609074876192,
            8881006246802039053,
            174817129683903468,
            11379536583839235003,
            10087895609541693823,
            10693952490604366854,
        ];
        for item in v.into_iter() {
            vector.push_front(item);
        }
        assert!(vector.assert_invariants());
        let index = 16074873107703469719 % (1 + vector.len());
        vector.slice_from_start(index);
        assert!(vector.assert_invariants());
        vector.push_front(4000471885859263874);
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            9505559548436360986,
            788838044899042349,
            7667086748794209812,
            10383045719983619857,
            7130765673375905027,
            5729438341741481576,
            10669173960402122865,
            10375870777724191487,
            10732252975372829969,
            5941742033492549542,
            14014384207414877334,
            9003828438159176457,
            1618176016311239076,
            12546497925926272026,
            1380711307767426943,
            2005047893181065161,
            17703666914056343526,
            4831231611227582750,
            1686038095374134561,
            11663880912017772807,
            3735743493733521440,
            17661527744417534580,
            11525519028273994893,
            14809162205962402306,
            6155720813948917988,
            9033720500089751342,
            12289102085093388733,
            14147672426848417079,
            11610665965685809787,
            14188739215108507186,
            4156676012628894745,
            14051103772352080317,
            9375713104175739174,
            16221172209919793378,
            8873112631961584129,
            4717329354306813595,
            18239964862269052092,
            9483722707727868021,
            16993073010203741059,
            9370523181800782234,
            9188059861098182919,
            13765573540452094366,
            5112169713585560442,
            1951703718099084196,
            16413537133539808381,
            3812943801711257607,
            4129516543722144055,
            5282270864600490080,
            3569412831084437189,
            16168600570019200876,
            16993965799184533918,
            4166555681068677880,
            10772658832362616832,
            14826699738616402254,
            8639936658116791082,
            13601199685416156130,
            4040268371108830182,
            9076270380056893536,
            15081228984853916318,
            11166847056465585234,
            15975907690304112511,
            14003084885213982697,
            16192282915921129545,
            5424999235585715786,
            7599466605544073686,
            14177783359717797705,
            677959864588869612,
            9715072823272467626,
            1780614671843287216,
            5185613650607776576,
            9610674130712947527,
            13991586950409655704,
            8534876470451622803,
            3889009302990790484,
            6518778749936879637,
            3467601168794689547,
            3810154646275657997,
            16350605536254234155,
            8804701767333627556,
            9817828767680188064,
            4244766560532967531,
            8891375242096648987,
            9941853441574376950,
            6679911067262963418,
            5339076406712268113,
            17414208436355393554,
            15653242750515757398,
            17319552018034221799,
            4155115980637922012,
            10285955734805187837,
            9567328006700858284,
            4149919615536349376,
            9230081041625689738,
            15566230262037322307,
            1590090738945223913,
            3755478449785591595,
            14518364492385826311,
            13827544540957420318,
            17405910809530450903,
            2276087616990650663,
            16180937236012726453,
            9981481856451793396,
            18429490524767617417,
            4268114448789973808,
            11271575233530023301,
            16875718574148142214,
            15290653087743332841,
            14802508275225826567,
            15897173455641180423,
            9697724823219089520,
            5073943387478922540,
            15157378341050003955,
            8662726690533453774,
            13083533163625346086,
            10189672319108901998,
            6761151017583884502,
            9832431232660673169,
            8267300407486719149,
            9742476017826224781,
            2761146688917533062,
            13133164592994711749,
            1934298328197989132,
            2835043472139048606,
            848871104502362639,
            14130776192195023958,
            4455446369849931319,
            5335394791668121451,
            10346135246950511119,
            5723293641924712759,
            8761992544295306996,
            4539631170413141946,
            8751212898526392004,
            6785549260305721543,
            14460604718866746271,
            9724198150606446974,
            5132377901415015338,
            431272070925679936,
            2108698405099601812,
            16490464456723751032,
            10920151446600555573,
            13458485963396956364,
            4471126811277800405,
            8309798715111372463,
            3195621141782440771,
            652114390059680910,
            10607905867905769438,
            2099232349753293121,
            471057570820192544,
            3943209388369574908,
            10987365036414210345,
            3495474591365150421,
            12310856013742746426,
            12546198130321320231,
            6632443333470526110,
            13845198895835653992,
            727371715155552272,
            7196356345078912023,
            7874477474570456263,
            13843401705391572247,
            2024760546766790124,
            6885857176798050420,
            6862676979531168966,
            10514642897442955861,
            3889817892352887896,
            2584072869299469317,
            10681715880928537435,
            16271830573060708451,
            10583175893616786150,
            8521714569179918171,
            13343426987675527759,
            5663182966172774272,
            16366343226664729735,
            8111368861440253539,
            2354878509352090172,
            7686941010748905973,
            7916925893739950219,
            4411730076227773119,
            7733849608811611770,
            10907689147868064672,
            295534134233948094,
            14697114824854533969,
            2236397178547694340,
            8255357001400850345,
            16070087839950795775,
            18260802243095957759,
            3220027649304818289,
            4743283422656349948,
            7959679753989910894,
            12760838895059635721,
            14950247986267964008,
            4856999289993764346,
            7305314442666923465,
            311428062299305143,
            14021081973204212503,
            11328960457252546542,
            13897215983113908125,
            9590430370796934302,
            1772490880758783320,
            13153998863143159608,
            16679965952364990739,
            5127440202936980457,
            2500843728753786445,
            4470564725295395450,
            18423358372335632352,
            16217726946078155434,
            13155616416675224232,
            8800835977067677407,
            15128366970363314672,
            3077088648607104222,
            77615433986801845,
            6660170893898622878,
            18354627428121447087,
            6729913556891888256,
            17007199356214754266,
            2690957931297977875,
            11751101413911723815,
            5906372182469066994,
            3705868199374502766,
            9846545747113262385,
            5911562549076085241,
            3317388183090171723,
            1609498807234234536,
            6925027365903009099,
            9776551689262648591,
            13348726230562205450,
            13217508446447420490,
            1762045903705531821,
            137112581535233019,
            14381764062005239536,
            8230372714460925294,
            2260251430623591583,
            6576076618970267303,
            198877860869609988,
            1250861276418379599,
            15494054776625176922,
            16468393257423845258,
            10914276518270099443,
            10238183516226310891,
            14600299251119032735,
            2946391755781356955,
            4581927736953471088,
            18351501822629766477,
            11781064915255004935,
            6273184190503206249,
            5558354768167696788,
            6918945839250750387,
            17015741143815676759,
            14513174275985652064,
            2097421601967164680,
            5291202113662722931,
            321595067130175381,
            4024277347299998012,
            10529154608053432313,
            1809959044080283009,
            1606400424520423208,
            1213177312181344340,
            11503848669716820744,
            12852317076347530158,
            11421241655418239050,
            2579709013649366592,
            13768186516813764914,
            14399355999068323125,
            2205385701277082352,
            10279449931818647206,
            7994796814496816876,
            12042172737556202455,
            16825039296915795096,
            17898822333200340683,
            11110680965571963883,
            7152746266042035843,
            9921256614205488454,
            10192669910371566260,
            16162240241132224771,
            4482440861025549141,
            14855105988436458813,
            9212444510643858229,
            4788520713531145878,
            683648453563377288,
            9599669474427903286,
            3408455322233356972,
            8981315911893163412,
            5122373682716534146,
            4706889822816502486,
            8543683078583636681,
            15702020746513490361,
            16956821424973632116,
            11633739725752859808,
            15444555703770926874,
            16668953856724659480,
            5244634418222474749,
            2896147773170894390,
            14673528809867598486,
            18277569833964399891,
            7311413246073764477,
            16201928523996928081,
            3080525076600566677,
            4564364282940060754,
            7761112087697801564,
            7018343309453834880,
            11661688895831832203,
            13630459410495445378,
            14170348280220115895,
            16431486929643393441,
            9984329260460537836,
            16429554778570188878,
            10406620654960945044,
            1870435529270890211,
            11413940564565760167,
            1398877774433381208,
            4171182067740316561,
            14446212357958136566,
            2710667196032906095,
            17678159607151514952,
            2032044418223377457,
            14046774913434122906,
            4532279682124142735,
            17499069008399748648,
            4791427359941339493,
            17819228800001864799,
            1144459113537821590,
            243279592427335754,
            8808389899690033066,
            975701477182707180,
            14177440187581922476,
            17961996321971513103,
            6513426480361242218,
            12764937122102116071,
            4281359250903664469,
            8615812656358997542,
            6004524164961597401,
            7889992187538697851,
            3785569851321795292,
            7472964736272181338,
            11900125583692921101,
            1338641333476074616,
            706387359218039989,
            563871150387915723,
            10183063328894047883,
            5290897026146419068,
            14260200703503545288,
            9215053470712704406,
            12582621392053645375,
            6628455228612382052,
            4054399275685541297,
            5787511073972289869,
            14774728771925610429,
            9499784222730683885,
            2953030061300978276,
            4667021628166778343,
            18046168456008951756,
            7592520540885419647,
            17547454786652717593,
            10031763743543082567,
            3637161789079977983,
            14249316830998581259,
            16666539153587439093,
            3165788891561229164,
            15796533069718993353,
            11340604951763752894,
            11248057946967518598,
            2225097541802991339,
            5981755534863939305,
            10194185867630368034,
            6502715542347361253,
            1354004409193895170,
            5592255411760350295,
            3183572890664741442,
            7206975080524438050,
            10539769613829538496,
            2085248108689748822,
            2538932345282324385,
            17040235256290042123,
            5935163733368623243,
            16035508714652281159,
            17877952007424913350,
            11131649534645742386,
            10971633804538736098,
            1852685092843709047,
            14090607523353269764,
            11770012156325296962,
            14501112255949105428,
            6418462056290979088,
            5129024128221743418,
            8408081931459632603,
            3666399778309769940,
            11430259872553887424,
            17984203006605143761,
            3251185897203831054,
            1722122014388089180,
            11494505470657046252,
            12875609851463672652,
            14276146048123745595,
            5833809323379984843,
            4463166478709332297,
            3276495856417618096,
            3274568134412025739,
            5006625154851941006,
            13561577670636707674,
            13761830390700847148,
            4038856264722472072,
            611409265251343188,
            18410761176191853415,
            9322759079879818050,
            3441609792820360398,
            3488756171458958813,
            5525451316143325798,
            15924018788583768170,
            355471840808282452,
            8551925549036640905,
            12399576345525198164,
            8947139866465175044,
            18172041362074774081,
            963326107155569559,
            3405422830563699751,
            17258543072285046999,
            9627708478554270435,
            2849292880961362755,
            5560045301271892424,
            11393753312536687212,
            9852093957921901532,
            15325405793764508565,
            5015454345810637798,
            16435113963607122257,
            15407998755382883377,
            8369524831793940995,
            17214929129214070981,
            256889330653273424,
            8680402425345263805,
            14012666394131710935,
            8859029745707798109,
            6751783698088404728,
            15824842298042252171,
            13556441433223049993,
            11082790236983896724,
            7889601455677052309,
            18195661333183890428,
            9868219309893860524,
            1624826256341116679,
            17731789693881207711,
            12517219042147671221,
            16704317173532403285,
            16526574853496313368,
            8638809078859811260,
            7145097959192794831,
            5198465982619624538,
            170417061343481804,
            12590435580675530522,
            17419592889008301063,
            4984952468439071080,
            6792038274210607619,
            17736751122760372729,
            11301406853570230557,
            15333181720592211761,
            14465713885270911578,
            12940579586023152252,
            12273976247738646388,
            10503389330977546748,
            3064701209859374694,
            12673588290795608818,
            11099388697976328892,
            3086095071460160410,
            16232306867661009621,
            5807481880308447556,
            10518181529937540920,
            17869838437099563342,
            18015432708728445762,
            12636257555330961135,
            14492545289286252172,
            8834248870008587738,
            18011000889192539372,
            9083472478337068377,
            5910576911005402998,
            8412530566831890718,
            11197048165667031132,
            11343094282212445004,
            4345086587815348377,
            6129284137031900674,
            16877982786746679289,
            9518184910869022896,
            11312099021669849944,
            8165973238061583716,
            17229140540813516719,
            11361434927290799240,
            8673328812349495833,
            3712288566652869957,
            14685716758098950562,
            6038437883534780000,
            12690062647976239642,
            11631902187847967407,
            11605969545451089615,
            2824922258697060438,
            6952506932525403129,
            14926237992337949115,
            2081125295542416458,
            5163764213390358325,
            8835155392400759715,
            13254952905081898541,
            9238752422662687270,
            1051969189754518705,
            18122298351109243316,
            1312739060360844124,
            8479299378683656948,
            17321691329755316764,
            9765796868793759420,
            13549120924194229550,
            10735100213329925647,
            4080374853835544682,
            16211787521158890001,
            403071775741862204,
            8950330493770440053,
            796569882312114526,
            13880380723083144454,
            7892257949771742928,
            2458239979673027000,
            10007486542642105614,
            62191425264181364,
            15885474363783903993,
            645180201779383258,
            12328153895990876643,
            13622953934648808872,
            12342258361513864096,
            7442229290460368004,
            12974238866170675958,
            16922759570975981540,
            13462913302398864685,
            16469212975863754964,
            14043516031043888189,
            5430214597196030502,
            13840096123753560688,
            7418508651277492159,
            6404433429484920344,
            16868313673537783982,
            9851647556322704373,
            16343721173957940006,
            5563831235089997929,
            2280862672366447021,
            10612548230534670055,
            8575082117919202532,
            17136798979899403724,
            7402944975590143953,
            1222705563897646609,
            18091938108182664017,
            10764863477106637490,
            8856036900673668621,
            15943801028822761175,
            7421033598813201519,
            16291842685759936771,
            7826639610597302021,
            12035220662980726279,
            15249220482773843380,
            16765461724794413847,
            3979751924298704849,
            15413255519847101785,
            13632076687671773025,
            10937156125248916262,
            1275573475299122531,
            5816022472061832327,
            6311877902950241122,
            4838852815336516417,
            16948388044058417204,
            10888439130689285088,
            1860647903885200345,
            15912941139770178223,
            5335374139346336769,
            11406866136778794406,
            3516380093284488876,
            8891010126461582350,
            10945333954973216041,
            5032952099679358609,
            14437451703854924502,
            18185761624889234233,
            14364187455420833795,
            7161583443372323454,
            16095961725528090548,
            13303287986060238039,
            3368631929107091631,
            15958087981510265579,
            18264940479818576139,
            9582305506116021242,
            7335296198775596167,
            6292692660822130553,
            2183033778861414983,
            14536851870631673602,
            668230914652168892,
            7882057306758870928,
            15586604976241160687,
            9951978050020386573,
            6092468080297284098,
            412186865169721558,
            16677770413940065168,
            15071076080482400209,
            17060772107418923084,
            200589335524485286,
            6760740834679775303,
            15113919403624372893,
            13720039629091659036,
            3771771732272005090,
            2593220742074077683,
            4388633656678245825,
            5998090689269797108,
            3710435110610515071,
            2077366398812842711,
            3440160194382519061,
            6939630222739051732,
            8002088499865771916,
            914558458421235785,
            1627669664712909676,
            1984025362861627549,
            13621962793169203308,
            6371852655110097223,
            318663643053915221,
            2417698420040731972,
            11359991948006303004,
            6165878485856092017,
            17669924338310657653,
            7521888963803319078,
            2824283788094155575,
            17630674954701905259,
            13222247888277460852,
            17392737029391500597,
            167681834512207991,
            17231262975460248458,
            86813439674598985,
            17154617870644833718,
            13961607867699250344,
            6249557252549307513,
            9893388211271327333,
            15729820218357150568,
            1187209024217952831,
            14874547278756626722,
            16755489766852270832,
            8442247521785379856,
            8470628624991392816,
            7094094801359427038,
            11195542394723878591,
            2106973274755587889,
            924280625048064777,
            772312969785934106,
            7285999749918312352,
            10905676139663330194,
            2270270321720189753,
            3919581366799688140,
            15985386888229204322,
            10455777570546368245,
            3122658698426230886,
            8983358825806697083,
            11564907253174295544,
            1337930798620186928,
            444140427188410434,
            8531934913484221989,
            8589663553185457155,
            2989790566256143563,
            4022607843063993041,
            12292014534231989250,
            5994844558532494927,
            30113659008093075,
            9291355878353951696,
            11363918361970065352,
            13694717304686684216,
            5906971099348333252,
            11274805769001821470,
            6562878788368902377,
            17304833267642236265,
            18433037666660437622,
            5849495253481090127,
            11384915731626726990,
            7215523455569410300,
            17472718889887345472,
            3973295607396769167,
            17006078917127568511,
            9515745456555445540,
            589488664259682418,
            6910820212553932177,
            2182029307250693512,
            16879375759898913345,
            2228917043077878616,
            2374223542168056268,
            8294927947147465927,
            15243480012915752768,
            18033312057650223239,
            7436534164700381575,
            11077285664136113501,
            9794873955929499985,
            2257995191936402767,
            17446029082183970965,
            7940661244834870654,
            13940150119612366575,
            6787377384935865767,
            14072772705322668334,
            1159172000086367752,
            14522272854107638144,
            5884166832974490058,
            5996722856104792392,
            9110657789325544936,
            3589195705608819497,
            10796292643572205486,
            4847001951949031312,
            2537738765554849119,
            2456865253179121470,
            17118104471420377754,
            3630107699072031054,
            6092273792572325592,
            5644227677041898957,
            2101836831802277598,
            17219313823417590120,
            1300073020688818634,
            8503654413847263276,
            6437652150885659128,
            1765630851967953614,
            3978830640382869328,
            355523266071980007,
            14117547673163352810,
            6215701093549675819,
            8597933454833374598,
            11162695641014329416,
            4489762562646713368,
            9544849987790125333,
            4614032060957641874,
            8615313626122783824,
            5489014433474844401,
            7239925042099928171,
            4157860388052764504,
            17561580155590505405,
            3851207135374996866,
            15622918921161098538,
            7891753511826416269,
            2321263898597629002,
            2379980063490127330,
            2063977598574084228,
            16618799545475225388,
            4349801832868331313,
            16995478575909556133,
            4166741842575629073,
            8999632437541584080,
            17329454482831679222,
            5947133715009549520,
            14506435071823625330,
            17309646462765479347,
            13713919393492250745,
            3845683876046837468,
            16496180362243743155,
            18443088593885914212,
            1601129329257264723,
            9164584812005805019,
            15490665523924578601,
            5242347203808717184,
            3017265641692260535,
            13442908986860253089,
            18437852012720471411,
            13072967666372416272,
            10331256027931815144,
            8891333783190031035,
            5966583601540149979,
            8522436755763823422,
            16885781161351617885,
            13422049467118583635,
            1134138611451637323,
            10440057652573065277,
            5629266905818341833,
            16156237006362627603,
            953203130947777878,
            3866456093368251525,
            11414060204500769082,
            10077413489795347786,
            8347244737556293803,
            10174470315246232294,
            15518995240246197794,
            12566403048133174706,
            10582032924475828299,
            16856638178078474420,
            5648775402321673952,
            11767837052687135029,
            892271846275521005,
            18063256709514107896,
            11576827784142678632,
            7636045135390885317,
            663348326589696424,
            15768318775905594652,
            4887890651258050104,
            7081414529901521982,
            14934590690459654504,
            14907286588591507488,
            6060472356353453210,
            15928546270042997954,
            13083192314743640391,
            10793320008527586588,
            3656531420970155858,
            14839320893738446834,
            12651200065028238192,
            3040174262662453338,
            8107098111374391860,
            6617490585747778985,
            3638175338906409751,
            4361695359796022204,
            1039682956051276925,
            7480377639487893316,
            5018138502355336008,
            4456694785262258388,
            7219828940555207470,
            16846615248078100114,
            10551732472240593638,
            7345789671201661312,
            14778552052839315797,
            947576415359207645,
            4937384188645857347,
            3700870917905574761,
            3512830409084406298,
            3072620922777095373,
            6770906195857264571,
            515887822081053312,
            15546690074953246741,
            12094939497132780861,
            8133626981279807654,
            2352213344000927365,
            7637689770855795783,
            7970689897764406739,
            9390025246965153295,
            5145922149332786207,
            9328934502697136529,
            1319593649456741040,
            6917027626039194690,
            10694254103160227139,
            7078673170027416990,
            17625188118518772252,
            1889573310795359580,
            1420013638019523164,
            5590957343330889439,
            10724249271911309887,
            6971383509076765009,
            13170122329951224380,
            13584930965583439156,
            7988876776689702153,
            15376851232575811927,
            16037016199331398008,
            12186945882684966231,
            6562788800021959875,
            5580688625135902021,
            1696703894950995639,
            17089766253736805581,
            14623389152614536343,
            11123300776687871019,
            11650601616803584675,
            11810198537537316871,
            9422815900016018403,
            1014385059659465867,
            13435508763088986994,
            1932940332018735395,
            10890374312187160486,
            823749163741770580,
            1807036936117265655,
            16908923166404495539,
            15600001486318723960,
            1934395997264011338,
            4435673719497460822,
            1940057522232375004,
            17759284835839501081,
            12500977780977818427,
            1721210243542202981,
            7565625964413018912,
            79048211700116706,
            2046752712939083942,
            412282893250995276,
            12995725615316950306,
            15571362906860462014,
            9671277349882738611,
            536503646790749114,
            1541939818593121465,
            12403245227014096173,
            4329365203390083327,
            6067550384795080410,
            5569940954367851470,
            18416605743159265240,
            8396278167041452879,
            741779778397699880,
            4178491922403708833,
            6153170572373182253,
            11984655320073415657,
            7714353196996478740,
            7607986434322682701,
            15046814086745518782,
            13679083201279375547,
            5789300974736156898,
            16088576176523305511,
            8538727390435219566,
            11765971010872403197,
            12126108636118193755,
            5932111105099557845,
            7980009740765297605,
            5044608033051030736,
            9314282142275902499,
            6159799101030580410,
            9562079454902063448,
            872286155517120774,
            2502789651625694455,
            6705465483611954441,
            6791320152745577607,
            15613713311642021687,
            15787752595337633424,
            12702646246192833064,
            14529410022771927864,
            16377988994345818532,
            784845685986231748,
            2516419032011722846,
            2333740273929856487,
            9264670482661717366,
            4340656482723177449,
            8748201710558361390,
            14345151921146333113,
            11882136446119763107,
            11774128102930064915,
            12295561171482816038,
            13279998522597301868,
            17725335956871448227,
            2798439978855481148,
            2249743610245605296,
            15554332735600055998,
            7947665921228223116,
            6775348058279930146,
            18198919754351961398,
            3960993918090045413,
            1180567478597547874,
            11947578415094271791,
            8679675973875233633,
            5308166334333175036,
            16186868933390041987,
            15385132542882294000,
            6809682685368910100,
            13208511205253566853,
            13545047702001135326,
            5620345609050908251,
            17378201187046913683,
            9253679733767530640,
            18216154376784693567,
            4923721093485826347,
            2733692057484145242,
            17193617220382752836,
            5499885575997596045,
            11267895246919834114,
            1721861679946000285,
            14189347675929111149,
            14018310920246567655,
            6188683999863774544,
            5969927965797822196,
            492685881189776146,
            869879350285436449,
            6189803333381131583,
            17583621091504867081,
            14396772343251511616,
            11847331976776302881,
            9906540937143530189,
            15789614161562700380,
            13499739335625342172,
            9105671979429767899,
            6865305979083927237,
            2849448497323950173,
            12445249321389475875,
            756994710694785983,
            7726003628751417630,
            17039787700370318367,
            14702395408367796102,
            1384971995588048901,
            10074723568713580789,
            14810257257168152076,
            6026417170814205358,
            8125936474840561217,
            10859430470999581529,
            3672738033489287269,
        ];
        let mut new_vector = Vector::new();
        for item in v.into_iter() {
            new_vector.push_front(item);
        }
        new_vector.append(vector);
        vector = new_vector;
        assert!(vector.assert_invariants());
        vector.push_front(1275213165495212831);
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            2312111520407079388,
            8362213592715922159,
            15471063913196137674,
            6093190216523199329,
            1187175992464236841,
            2724938186624612005,
            18287176153332794064,
            2733378377651329337,
            14358573756327861599,
            12616895741130057672,
            7709528502249135644,
            12352097687017955172,
            142919157431014919,
            13275597766937154614,
            6321020939053458296,
            11645069646756542391,
            16737985754780654059,
            18195902585239996919,
            17899052342514413767,
            17841798505615674819,
            2202351429763014813,
            7868245304405838285,
            17988116544283554567,
            17861323458663187285,
            11518442980785374148,
            11170264297289597884,
            13415379670261293536,
            3081277993154213357,
            6682893652425534014,
            12924098216706286016,
            15943085932804880185,
            8393631831213857546,
            7695152953858540648,
            9009720345250730643,
            7973865457057803436,
            8971356578637826545,
            18346609909324151657,
            13100372389296303788,
            17317214064994422981,
            5449471419823829692,
            17733241541089595719,
            13129835767545257222,
            6133921239605880088,
            5212182134759491004,
            6179424814262722147,
            16088633301849518354,
            13598838532409516021,
            16726119427014094717,
            14034475332080364206,
            16695691955179164277,
            10151772238755150554,
            1660438778200723246,
            14775358254951765467,
            17555542200471715587,
            10754633229822155312,
            909951607762995233,
            16522572007485146141,
            13294212744666742455,
            5536202395696267666,
            16943866364309676143,
            3077928309327897884,
            11535117945950421625,
            11254956411581098846,
            6651395334865299936,
            3675016030816939507,
            7955849733459013468,
            17880823588890277521,
            1097070191356076578,
            8931856750563471142,
            45228916797716192,
            11108985013954580703,
            7614019730683810831,
            13501717515712821403,
            6325346099123158352,
            7156115675092629156,
            12211779937120578822,
            12570950907040449962,
            12267515370115232131,
            6425664945026899586,
            13162424744262963549,
            14581756091231450904,
            12341922835676827716,
            13669892213019452434,
            5380620885577816564,
            14870496517131428224,
            14084722711971024172,
            11577647577547314173,
            10026733228199294465,
            17846190547678759102,
            15723659695670719203,
            12821734799548622674,
            12572847416365594628,
            8157223003057156480,
            11092191216848914337,
            6849071226179696334,
            11327737136868873657,
            683642628456025166,
            7890545451340509056,
            6373547744152891891,
            12991407829964778296,
            2588064501388901749,
            15660884592360619495,
            17313229597715660647,
            17156593971376212573,
            16917430893906308715,
            11574514109986664387,
            1004592909304634682,
            412764863324909087,
            855514643786666758,
            17040979823688573159,
            10606605808041817901,
            5483509044035348416,
            3453666231344619519,
            2477127814028971909,
            2856412221506962206,
            14557797339909333266,
            18307513219357200703,
            11491470456346047914,
            3739318793598134705,
            13304451453647444885,
            8454424464141505106,
            7630117724935333588,
            7936759794506890907,
            11535686144925297484,
            13819644694844390126,
            8729039580219385314,
            3624923701330589549,
            9962986411975666785,
            3022464016206233583,
            10482116547890302674,
            4577445010317750575,
            16445826445395110155,
            5002662171251903208,
            343633130343191819,
            16456895840579472203,
            8537811315196361196,
            3224351436820218142,
            9505014464037539072,
            16552987063487879633,
            10407499864602896093,
            9536581356800684520,
            6440883860652280035,
            3286695459331554530,
            8677288577921911263,
            14644189399809557653,
            4133986011408904178,
            8986533282520542181,
            3612464291310342055,
            715850315604605898,
            3747009676284047630,
            10952310077939164521,
            16161449417980766008,
            14051152726687588497,
            14579954445947802242,
            8651511347791532928,
            8738585594595610842,
            17950438230271877777,
            2149299923776489312,
            14910942012129781152,
            11170514967324029679,
            15732738534103932200,
            11538206949429985243,
            10315148295266244283,
            7312233859353714193,
            12865537496815195688,
            10223329338273198099,
            11039229086042225096,
            16637017657970144372,
            17693721416496051559,
            8031510040825250164,
            3215416118559746174,
            7254358930469407014,
            8987140083056004525,
            4287314007214077925,
            11019915642667487928,
            4715402730154322485,
            4253564125241210400,
            8493913835775985945,
            17353529851363645095,
            13496908661073509536,
            1620331715971776946,
            8411082855831384891,
            13941343073712270723,
            15252486776630108954,
            11399631739247349768,
            6270936143068600168,
            68414736628864897,
            7295511955063889773,
            1528993730661745529,
            4963807663521996618,
            14460108658243229057,
            11438648737328613801,
            11057342517618278425,
            6522449395072056208,
            8483057089432180617,
            7659057315397153731,
            2102907098476996677,
            5202560056093022237,
            15604325426627384757,
            10081712200912140045,
            621935236634566149,
            18358547310924716193,
            13921592217747389623,
            3431186381010050851,
            1060370415553983725,
            5743132010637400948,
            17478467635940364225,
            1404117670739465031,
            18321551576540014752,
            6810519315603735335,
            12469904931427636248,
            13259426383543921830,
            8435050949074252174,
            12466768865548997224,
            17429499250601690120,
            7053206934884926828,
            14912867500677203835,
            2045517021033591928,
            11261685101237185790,
            3125853893789831742,
            9964591468436040764,
            13506385382861310817,
            14436003366761905473,
            9149731612941497381,
            11334804931477679186,
            5761486689825421465,
            7552831836047629509,
            10637636973542484006,
            18395630669142105628,
            14996790684042516512,
            5715025254306341600,
            2988070358475711834,
            4309040547203320740,
            14252221251366872233,
            3520192043249264415,
            4628087430496133125,
            11450430814887335052,
            246716014434778481,
            11661225069871560632,
            16686385573546562386,
            4107885933989476625,
            687094905722571145,
            17096455952483160907,
            6861898461650648835,
            15836575199915640843,
            9683652189711445733,
            7915392038278284246,
            12343161990999732759,
            5158876761126709393,
            5532737548254907659,
            16251972255290018297,
            6492833151371304812,
            8135107468744480888,
            9028837765011995626,
            1714009962306632137,
            184518458563217431,
            9933370391627941249,
            9068217240084704252,
            6132313688054055015,
            2540905512006776726,
            15891121621942157002,
            1499774173441732854,
            251396483224880736,
            15818921459913052931,
            15297302327112802315,
            6386216807535176365,
            12433044233717348631,
            10229402431882841779,
            17407448206260192802,
            12436279332065072703,
            16127474352722782183,
            2356878999031244220,
            15681976073219472925,
            2861347013328364485,
            5380539338992305878,
            18100691919089422239,
            5022504881524287037,
            11882040048764026371,
            5725072234455237481,
            7731756731974902630,
            14944796103037752930,
            17511894607623522252,
            8960007803420917038,
            679278702691134345,
            78485657793959199,
            15464559967148670871,
            4726987948968707750,
            1459255634809124104,
            6421948307856405816,
            1889964348519833621,
            10296225033398627560,
            2895863838991620650,
            12830892277364990949,
            6093478941932343336,
            13491031812774651209,
            11177949588053526825,
            17160727426644598298,
            17377601443178424120,
            14395546528841858334,
            10697647494708093572,
            7345472175218873914,
            4815850417749611314,
            5620110318198223194,
            11653164431597748656,
            9896992814575294659,
            15052014546480085829,
            2635767549336838024,
            3002867044798098788,
            12946773914143893461,
            15242723023271342474,
            18000513572669122148,
            11192903127970649614,
            7145954461835660646,
            15165384697184668733,
            6218639902059608818,
            11826395880415010808,
            14027021024220912953,
            2680945988275341351,
            14841009453746267130,
            8454140656797535383,
            8963550430136890261,
            13506816331475468955,
            15213232739407484232,
            320893343696383145,
            5891672320889430364,
            3341287860331172572,
            3214289365987402154,
            1059304126929128000,
            6601305246463423890,
            1149079682525218829,
            502930147956991578,
            11209147255117641791,
            5713146439380720067,
            7689713614201172466,
            9663739605843756490,
            15528453372051640058,
            12004076707181013639,
            9417189048671823646,
            1463837761108765265,
            3568932416177701719,
            13046512797151479560,
            4817628121438107923,
            12308595784744777222,
            8000287674126935759,
            87742285910633190,
            13563293670947005032,
            5324508622101951310,
            4514724548125708179,
            7260843018440277431,
            5414977849314319145,
            7712648068921910762,
            6932489152624184209,
            7598923810161195497,
            13035457788188821882,
            7769643132521990496,
            2641446192941606263,
            3616129002245040839,
            5140696702888379056,
            13307621328855594920,
            11692962846193203663,
            9746961749423605419,
            12551789566826513542,
            613160649767497576,
            14226963218882556803,
            18056934613940658577,
            11877116426796305982,
            5352900296570486882,
            10751812927643426670,
            8096629693616588040,
            2708137189564558152,
            4623158877253700264,
            2428679401694510116,
            3058314754775856116,
            4991028088569998344,
            13707276567995311015,
            13074920039374052704,
            3352139205948939350,
            16633864540726113688,
            1402197331437109489,
            3366580165245681814,
            4519987014649380686,
            15848390723444352906,
            15769215800497906064,
            9293823635229308934,
            11312249312187508948,
            7734308948587106921,
            9216065250133475188,
            8958192484550220325,
            11507700129456596227,
            8570090830402577621,
            14195306684141535602,
            18395851358731873715,
            8522510253847451942,
            7841573477164246321,
            3559640405801237693,
            9929938075584324451,
            11934429619482963882,
            14837430325445815957,
            14945866624192145421,
            7585571435009790136,
            3563794189087200041,
            5081259172449144047,
            17973637430444466576,
            10955775256516115426,
            2422790738127353395,
            17308483388281429531,
            10156835345408935972,
            1054845288856597881,
            890457687662333716,
            1960552494278498470,
            11672008048901845131,
            15015961892418817567,
            2323894309200898078,
            4201359027777072207,
            14654506264644932658,
            14400481722834486718,
            15169195961617602088,
            17947108982826677958,
            13914271828881941510,
            3446401415679420078,
            5311710847313514009,
            17480543758894440687,
            3244933554170201849,
            6474019032541315006,
            14100300947257136189,
            12824280882504819845,
            5303908837365989520,
            2203723480835737697,
            4574286695484037275,
            1561238470928652779,
            7450576232681440983,
            5460834579085199297,
            6450887737117189376,
            10795428822964435999,
            4467564353013194578,
            4618425363655205537,
            8084004824737518706,
            1965221735852158039,
            5604999977224245693,
            11019982215989770423,
            2060604858877232577,
            1199799917391528480,
            8047344888910103761,
            8352076989397697229,
            15568076421824964757,
            1829000663337112639,
            9035502500970147491,
            3742109145487842960,
            13814667262821774668,
            14794673218634613190,
            10373978869816863108,
            13841434936574891598,
            5659613634263865205,
            86996461153596151,
            13665232181278756782,
            5189377086412131598,
            13506330796493666775,
            16565429254675273163,
            17166086145877721225,
            10792395060990266308,
            11792885345858970085,
            5762569335744172469,
            11769234336645776363,
            8097009977095111487,
            8706430207268063850,
            7508692077519484276,
            14081852437861545465,
            9518637132819703571,
            15591567540794173411,
            15307360798038275579,
            927090285209448782,
            16586512218484370134,
            1816020450835091655,
            9551877890216116381,
            11387228781716179757,
            16139479876959935451,
            5747451762442671623,
            6094856360897139415,
            1193033548204246787,
            14762020012087225649,
            16410130990782502418,
            2169650984957439340,
            2776117157389852719,
            10043889369562038841,
            8253120955698024535,
            3066128498462668790,
            8014121795791592521,
            14889580533075083850,
            973470663623838379,
            4617383287464435579,
            5498473887987281665,
            5799754734320468869,
            16918898347043980913,
            1470248286336789320,
            16205349176417477852,
            12759672524022450721,
            83208500089035868,
            15326815892662645507,
            13421564541811818841,
            7305524149609442902,
            5734530692110371664,
            9398488352861684304,
            12118999155103768645,
            9119608708205019400,
            11432419188305761734,
            12657150782939025997,
            4357607846669688529,
            8134159812529193784,
            8890868499045753316,
            15383937637115134320,
            4931876264450327405,
            4845233990206021896,
            9229481744508104007,
            11551225934930958721,
            5354109207161794884,
            16842223923414728357,
            12461449342036079112,
            17900696571456789371,
            15210598981949074869,
            15914865910851618250,
            17292143213960676244,
            16144248142869918404,
            17836882077240380131,
            8040433891737172167,
            560427929860028595,
            7705272884613026991,
            11282407894487264633,
            15134354556448692926,
            16094087255596120330,
            11038050891465397478,
            354205510386289586,
            8665426760909413569,
            1226084569757134194,
            15410329333156188968,
            1751523867416914831,
            14188443859589240705,
            5042391585871472648,
            14186815557061814111,
            606723005524680868,
            4738816016743411545,
            9701204240060265652,
            16023418252133515281,
            11824201059406386234,
            8865809540049583866,
            12019889577222717585,
            5814634445016123204,
            15388990929891167875,
            3484977701245200730,
            13128582556169345055,
            6974689382462879610,
            12318369631023652047,
            14013844541154065691,
            380656205079700124,
            16673203052723306497,
            12478551009219449573,
            3674142923929226987,
            7783005337793847366,
            6926202760249616095,
            6815099748669783936,
            16907368897351659619,
            15053170473098345738,
            6696581300062026506,
            9911696577769671460,
            2907230749680622460,
            17947531139401173912,
            7615621910546349787,
            2803288970606791950,
            10186338465585157158,
            4989121949075875265,
            6515821376231771426,
            16426504896722468823,
            8487525049638269559,
            1067589602144821456,
            9976815759233690216,
            2468288590792251496,
            2148466043749833757,
            1648824296088981529,
            4034060500411026023,
            18218390483038991940,
            7222052613356312536,
            7051638624424609133,
            13947781951056565270,
            9172849108150319125,
            8583064883923160803,
            1416042097782063537,
            14893812712441343672,
            2332805269566448550,
            16010050909473248669,
            5121287686009161876,
            301164132718804757,
            6585715339238416570,
            5546808774169479819,
            939010057760786611,
            14261895686801787629,
            14582216260967476119,
            10283332089525761387,
            16792489892494954641,
            1516689824570554992,
            17088519127666721115,
            13651542840493156452,
            17286681070137330076,
            11552753702641352519,
            9730898952234254679,
            18058456137426577106,
            2402626078568179634,
            11577958664767985398,
            5600074098211480182,
            14346521392408199261,
            6936043784926809114,
            13737568490195543917,
            10140554358747218580,
            6778587132095578084,
            9764431993239536356,
            18159378645651854765,
            3082766374665814288,
            5851509099263501070,
            9064997740398144234,
            1794595380637224608,
            15842569556596943886,
            15934278246507817572,
            2106839367060231349,
            14459838985462369674,
            18189336768882907090,
            4713507614874738875,
            14497062580740114674,
            16570840504856186634,
            11646539683381911609,
            15104958572269300388,
            18295328110325867928,
            13134922369040359288,
            11874595080768359843,
            5748548388432268318,
            17196234750355932456,
            3446797507012479998,
            330127026225797388,
            3286258222049651039,
            3419353473789945227,
            7777337659905342071,
            10061182279066318806,
            13834400053342922853,
            510871767288121178,
            12101997182217737663,
            2983313864107960672,
            1329563011219965368,
            3483218451411793169,
            11297180628091838667,
            8914166495261196936,
            13864117313428986990,
            13249474237530784224,
            12754990444176251702,
            4572674660767774596,
            6281315886343796024,
            5638599300375901040,
            1363102291038588615,
            6492830685179434038,
            7469818348365712215,
            6380287165599105256,
            13015532955807042153,
            4721947879814951616,
            2733194220410462833,
            15790914491048554585,
            12571114015323395133,
            14441514679015186247,
            9200917617954982500,
            9585816741499208163,
            12160675723380639769,
            8829882829547539149,
            17032443082070478725,
            14665658050157416681,
            11122463182074949035,
            10320736733791391698,
            8399242759574166466,
            16688005299862928368,
            15421844184316987870,
            10032134732314783422,
            15159419287173292583,
            9017441505644439683,
            7704543752508996276,
            3913976203618867269,
            1601341234745698362,
            3586535991127549112,
            14769011334772218440,
            13011217249469100689,
            14099485199549964788,
            18065619877492866742,
            18349209122714256576,
            13133949609260900342,
            9723169911203042926,
            1130505863375210077,
            7887172427010437790,
            9890183716514944447,
            11834587854987118482,
            16607279491865289377,
            16740447621105427111,
            3871935048298641222,
            17402241162436318506,
            3245948768959227532,
            13726454774220992560,
            5824896683795770060,
            713672493283898270,
            16894125557615225798,
            5531086394954681274,
            9033070755185197508,
            3550138350786636321,
            17350971849446529225,
            15241992692683160817,
            12382036991929937134,
            4891924363688403009,
            1098564266108127782,
            556395899040842554,
            2600859271857660071,
            4252715339085925365,
            16287010117493831215,
            4432889655254287227,
            17096664486926777450,
            2432959935392292675,
            6779231924339420012,
            9785146402098757470,
            10846103425286817799,
            5103168914490652373,
            2447716975901920188,
            5069355826195687141,
            7406289349279616361,
            925104185547218640,
            11380445164033896824,
            8651266857773976561,
            4204732618581859142,
            9454673554067571307,
            18014966063916347012,
            6105422976107200857,
            6680849631942531648,
            5512122030425664175,
            15024402014285088322,
            5294759348651142914,
            18367878651600387513,
            9915347002635443974,
            10794140241777595535,
            1234391112369721744,
            17840810371980192045,
            14548151408968740291,
            11564564826616109683,
            1842731120500863967,
            4634271285615936875,
            11975452555353950891,
            11046787150544428121,
            10191424411369969920,
            2033545864601927376,
            17157154531520093023,
            14913618340588823511,
            4975174575802488510,
            15491187817014920044,
            14766451333834927360,
            2795867543059102967,
            18301782347116321524,
            16202815690274287521,
            9222831030401547236,
            18037509351892701118,
            14385232901937307736,
            11832368916234135775,
            12851062627365705358,
            17555069550467992201,
            12066219058652099963,
            15329795133694315279,
            10777833851971881347,
            10172906988711223563,
            14880565380245645730,
            5192250681704614003,
            1966847103566260375,
            7213357720604136148,
            4768949191729901385,
            14482363987258940290,
            11858977017179363660,
            14711234430777137936,
            6039688781609092883,
            10788194373186383625,
            17138101604746891701,
            12008432004201182512,
            3967753884172426051,
            12094521730864652004,
            504506216845686182,
            4736145746406461104,
            10489677627276206992,
            9149058326035869299,
            10793391126177529469,
            362433134257960963,
            14314789126004804723,
            11779349658572874459,
            2186192934702142462,
            5723491553185071316,
            15444490171224620801,
            2541629343473463915,
            13711405172428258929,
            12191825616799006458,
            6718536743463096079,
            9133850103311060046,
            11443287403205006395,
            6033138038277741679,
            7582226562494124156,
            2861575180434561512,
            9212718150921875087,
            10324023115286556704,
            1717180955692113798,
            11145089992220524241,
            11525784111808960508,
            4412907413390701806,
            2740530470108523925,
            10543422990276962649,
            3490680562799957485,
            12840314323243041497,
            11628758750302488542,
            8987803285492481597,
            11873431417673627982,
            11836735497139788489,
            5433823656010846526,
            11927590500891274813,
            961258990702674799,
            172343546891165470,
            11529888134841060718,
            5015552183499448791,
            13509773714127786626,
            16301387581059804403,
            12677719928944264283,
            1236981698743650595,
            1392637680788568940,
            39380065213024120,
            5417030772725117641,
            17696980998038158405,
            11678116020570046980,
            9483114375532302674,
            2240609883665866480,
            16881635834391376935,
            6466422173158860634,
            10671375433508278099,
            7600451213907147142,
            16490816122880317818,
            5100331107632345152,
            7696098842529040090,
            5444380960755178678,
            14926874284002320487,
            6194551764676980572,
            8879639467685774349,
            9504213708847351842,
            16029037346337963621,
            11127604531016654221,
            664325519603989889,
            3229296187405655409,
            18330497271878149931,
            12230339262151465432,
            11434941774430283288,
            11319034905874564045,
            11128543221651429696,
            2298721093745496171,
            12673266861195885018,
            2484670317199458790,
        ];
        for item in v.into_iter() {
            vector.push_back(item);
        }
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            18097788401836456247,
            714768107848774786,
            2900793698635765115,
            15265877962471506675,
            13917172466949750995,
            14818721498395301320,
            4751270087814444862,
            2617380005498458493,
            11045795357498917543,
            1548456391310386934,
            1981342581390493053,
            285176094743584022,
            1774639620025702622,
            10496964374662463474,
            2136730575553735312,
            11701392856246860967,
            10870205191872088672,
            12563697913978186267,
            7495185840500509689,
            12477811704583515442,
            6551004188083804063,
            10162429036523716790,
            10474219339677019081,
            13833798166213470584,
            5685371917489424549,
            10349844645057596705,
            12542897578662272516,
            9457743733463887840,
            11109462305150159476,
            6443276140607760850,
            9111823159668720422,
            11644841228352491400,
            17786827697400814678,
            16896723473865185525,
            7373033268934337905,
            1538496908585423301,
            1491586795446994602,
            6203703098500053776,
            7283036921430095234,
            11910351036092443305,
            513799924612430355,
            379664288133054189,
            270965877958210874,
            8192109989561594792,
            5294104277902507529,
            10047463758185760751,
            8807777726026891126,
            16951930607545318949,
            11678071412845973598,
        ];
        for item in v.into_iter() {
            vector.push_back(item);
        }
        assert!(vector.assert_invariants());
        vector.push_front(15186554573722801336);
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            12070553229783924238,
            1540180477945899838,
            16026252530011627910,
            15889974627949970080,
            17517777689486480683,
            6359722177557656526,
            14227327077402773941,
            18302335286750481635,
            4289207885455645910,
            14849128213297710346,
            11253185785372294504,
            11225388504915961916,
            14229040400279547684,
            190932188680821499,
            16728553944685529667,
            2156554081206650130,
            18414678703360368287,
            12289308544092052660,
            4565659476214215975,
            7209203512615277743,
            14403839409000105002,
            10414002133291484459,
            15910514128233667084,
            2033225893245260131,
            10644901657020980608,
            7987618099309461573,
            5019022438001975439,
            13357055503131984384,
            14428373623804399480,
            13728906858580217312,
            16232381557890096211,
            10760741212931990397,
            14544267613042125115,
            7710545796757935660,
            18416984419635576166,
            161956591960187304,
            475232576846054521,
            7891467382835033725,
            15917849072731740400,
            12948540815371993547,
            7787309239718178859,
            303917822595356198,
            9937037997838720818,
            1962825125789078107,
            6493661391497630608,
            14161238292014045813,
            15626069160866286514,
            12252201106070524386,
            17774188714070006552,
            2341946902729633574,
            3878888030420405606,
            14378576828614375126,
            5118093669044726376,
            11518623421418140047,
            12379643777609715408,
            6345622355192114913,
            8853793426769109753,
            12670744964482624912,
            2362001900744357276,
            12115635181813424317,
            11508497090383787846,
            15278606625170379981,
            6195180316792306147,
            14676565834788788731,
            958444316665748170,
            2316103284350339553,
            7550131231536499368,
            3602252803973253676,
            14259026698907740781,
            13706937117942622872,
            296687751634116678,
            4750395823260744745,
            8658125584611681550,
            4458812807138846469,
            4376452137293356599,
            17906643353594810954,
            9225261219817966849,
            14036811066997076590,
            8430992892115156075,
            263805469268725545,
            12086354738993791962,
            16976584849168915554,
            567126435618966470,
            1584897015052798170,
            1452357820270071946,
            12642815088241713562,
            8633007365487567915,
            18096930422931958435,
            12446459681243459746,
            1703545035392170038,
            16971862952208076698,
            10288360348355527693,
            15769713583422047967,
            2570115628669631757,
            15959977197053286371,
            16853202041358164781,
            4777300786807925778,
            8332998428095680346,
            8401094054912058399,
            11340739544274395467,
            2256378912687640677,
            9656759299800187766,
            16686102502736672805,
            17522844188732344839,
            13366093263308129786,
            11257880617041342174,
            5675753198334278921,
            12982281744751263715,
            17952798891931048549,
            10511865605217377161,
            4145566617681790264,
            17424435027567792845,
            2331714420845478949,
            18274115595484296307,
            3121066938878917760,
            965924274378375295,
            5022224873549184612,
            10296678479164649307,
            15150859148917178278,
            5342366689515353400,
            5411381206656570375,
            14129273076808467608,
            5452466235122818697,
            330958243603129469,
            15286391257350519762,
            16131333546108925101,
            2121260307710578842,
            4207222236096993489,
            5577116078451044571,
            5437732697226296765,
            15432064197207831848,
            7362029626816510605,
            17416276820888941995,
            15579943374821860428,
            6253191956604859350,
            14347326769292139183,
            16537892445921893612,
            10699861413467812338,
            2754881661169330701,
            17675524305872952272,
            9203825624073374949,
            8613345988137160774,
            12806649558677603480,
            8010799058170715089,
            18422234444980402075,
            10043266447759568598,
            13184475170082355617,
            3364683293849233356,
            17115271303768752714,
            8815700423451590282,
            2745355800087582511,
            17345378205582849710,
            1241621717324305633,
            1047674243806303893,
            9420633373515094941,
            15812788856484791653,
            5124774452963990902,
            16580311325049758178,
            11119777892677276897,
            14049601063220602714,
            8447196931284456591,
            10798796624150403479,
            14463873583555314007,
            9492483968086576890,
            18209392371727121864,
            3138464380265671700,
            17978059868951542165,
            28480046538912488,
            8901223303407685543,
            2312380454943460938,
            3607315856681566200,
            6499483852823280089,
            12907404625465572203,
            536058910153856481,
            1074866889177604060,
            16086327797007849898,
            6653440009324246407,
            269647577509885357,
            9713805874174683276,
            1059921231610270194,
            17313423558070553997,
            2633477903299835502,
            7261627179523637434,
            11025957936270767745,
            15233428735240168756,
            2673839672108357764,
            10649736841158022666,
            4081993842508091661,
            8710235434899760087,
            14523388415546702969,
            9306695578574560358,
            10347174881905164056,
            3975212620824904495,
            2846204983608814492,
            7955804227898461639,
            10357097187656686033,
            4775638118631848748,
            12557275536214809752,
            17516378544663266010,
            17200905758509124054,
            12943117061033453516,
            14588962673737705181,
            9010893330254259618,
            10995449745313838410,
            5879487365536743589,
            15748623560347898456,
            14311745007759077514,
            3050908019161292783,
            6973435402090121419,
            15068281296359269268,
            7708905559338713940,
            17367953421336300344,
            9637968382029889795,
            8916742707753403431,
            1500954246805791512,
            15897697971028489333,
            11377148273260083488,
            10742958800608201577,
            10512849611299136141,
            10557448204864018295,
            16863637428684942461,
            4850196733840874106,
            3001307410135338516,
            11336792086914756763,
            7824808345821148326,
            10101722683148818172,
            7990295084578556808,
            1009860175501165970,
            11121077781036629333,
            17338987571558402406,
            5304902071582261727,
            10200999368838648946,
            10316584235689010791,
            12473888951574409728,
            1136718540817683829,
            2499283565169147061,
            12682976176784227296,
            10977857176061052447,
            10120971547598497205,
            1946891118814031694,
            11085638293305601963,
            9684707578415522044,
            9626424968002798599,
            6177677597816771003,
            13892006259398818194,
            14817745875641955246,
            3186495158828272903,
            3604888601783708814,
            1487734202241352287,
            4153936572875928291,
            4914775295327470796,
            16333575171811349951,
            2158320190501118944,
            16992697251935902406,
            15250925481490315800,
            15142072385179286964,
            9503086524940664827,
            16913525904212020582,
            11221154822816334563,
            7412111232183420148,
            12089744517719022256,
            4731694924201391762,
            1571043256637904290,
            11520450376920473398,
            600818104236468889,
            12900289051197994680,
            15985750280456364523,
            11435129979289793781,
            16623770165507472047,
            17193723641300598163,
            14480551168097584037,
            15780061107424643626,
            9276907648894927181,
            4208748158435443521,
            1311634343375535841,
            1246764143711932182,
            12953059841355975207,
            13067802869980252478,
            9275946125428708792,
            371624715490463348,
            17908784674078711472,
            10195202541450560164,
            7828825882549892681,
            5028483557555099227,
            4165315688403156362,
            4855426421191221334,
            6658334943085113630,
        ];
        let mut new_vector = Vector::new();
        for item in v.into_iter() {
            new_vector.push_front(item);
        }
        new_vector.append(vector);
        vector = new_vector;
        assert!(vector.assert_invariants());
        vector.push_back(7616830648676210553);
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            1627168444111853573,
            10507794866013241728,
            10201928648501537819,
            17753977293558210087,
            9505262524423585527,
            10489947799356674064,
            3120278255684922415,
            14013592607129981065,
            654978382460279222,
            2049059111887666163,
            15709038798960018233,
            16155702462330029558,
            2128291161015157320,
            10344963190268030144,
            710735992220381048,
            16506095167231957326,
            5485541212784484079,
            4425437855836357476,
            8272893373809405140,
            6846129363923748430,
            13532506672880321634,
            10769578556848876833,
            14542471491234024790,
            18416017430455295599,
            3082893248372035872,
            11284103867927383962,
            3503402843939194903,
            5958229135839572177,
            369311538972144962,
            12388069972465389285,
            414746387962144734,
            4196798925394993924,
            14186563785497897531,
            17492356836705728306,
            9600304699187879685,
            2769390061738691035,
            8683753695309987814,
            18094549527561951634,
            15575577599530277085,
            4644419550551397690,
            3152864244404828191,
            11816248069150605485,
            15675343911190213793,
            15378955476733523521,
            1120080369973617815,
            12726510137522269681,
            12731041730053708273,
            9143027438656572316,
            13685751512391244310,
            5018892659701618407,
            17708427039961223120,
            17211687910636778702,
            18309814437319365093,
            18379462165443904141,
            11707419742208386321,
            5628850561439616838,
            12024279677614818707,
            7580345912879581184,
            8763995388499414849,
            5740268331026916375,
            17569805795527611859,
            8037034993546963074,
            914724507021121324,
            908448590550266775,
            7736013308010853306,
            2451763463812599234,
            2187074378344581640,
            17806947983293095654,
            7781591497323384907,
            6159118654896048013,
            11514202064215072870,
            15062452650577837215,
            9869742954730064594,
            14460994374769207421,
            9531221292412700301,
            7617073638122415579,
            2123873387496639502,
            3688792537705300668,
            284849490798960552,
            14450797266300253977,
            18156466744949272828,
            10627113224128520259,
            8173172804239320030,
            17012799119690790747,
            16595907193859536660,
            6875967840418248987,
            3514969962408421537,
            11068074503167489441,
            5957647174590957448,
            16402162258442304254,
            18246594705437249334,
        ];
        for item in v.into_iter() {
            vector.push_back(item);
        }
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            9704486163441504615,
            15139623894309238430,
            11410364840599447339,
            5881536195880020590,
            17826114330750034983,
            9775324634361888938,
            13374558808197780657,
            17550621091207206798,
            12171961268757028266,
            8414498554555520883,
            9253476093801938510,
            594533237069690983,
            5558109306835934473,
            3181361077765600175,
            14751680575120295271,
            5750941482056615783,
            14589410677726297841,
            3555867778595518982,
            17908389224158380986,
            15212447992236378115,
            11035559581867874818,
            3160637508390107581,
            649172026792315827,
            3324853492913201603,
            5239230201602641403,
            6826892856508515394,
            1465377578614690151,
            17522068600622318328,
            11073322517737323462,
            9958585185164082208,
            594660300594039290,
            16249601735023724103,
            17189364947302930843,
            14678032181187968584,
            17550193910137621138,
            18178081212184433312,
            17288715779687621489,
            432970508279774241,
            16068204275835856034,
            4519138623074740916,
            7864069099509577415,
            1502458495707145640,
            17828640259752905667,
            1570407724012264821,
            11719501368176525738,
            14105359811028060332,
            1945582085688913626,
            6255646112192859294,
            14744609641635614471,
            12954539539741254283,
            6560126325700748882,
            9982653412308274679,
            3721226813199167812,
            13527005038201151967,
            10071336593050241989,
            15339788885715833140,
            2198740712109396946,
            12486397050357711736,
            16506382158074294083,
            1927960023451800939,
            16393338737920671184,
            17727227049048317724,
            9524238933802080849,
            11561721930584167238,
            13883389789714706844,
            9266148738335232470,
            14856315623099166220,
            17019571289668636999,
            5067892077860354590,
            15792916590189475661,
            15809659056212841557,
            613564357606725674,
            9614834675934576540,
            4240299028101616883,
            2049881127246746951,
            7107112910235619288,
            6827263749833291218,
            11982574219370512693,
            782657172521312516,
            2574948071368581905,
            4456582716896886485,
            9099375244610207592,
            5469033400984863153,
            11029406999582885441,
            10304251601586394431,
            10883145760352749200,
            5343839123188940709,
            3481120411547270676,
            10772069622135252820,
            14601543292332530002,
            3624570778273881406,
            1135903354190017883,
            11511962678542427350,
            13521488093022430264,
            8859511890367322636,
            5834069723557908936,
            11806855101321215607,
            16790092470296327269,
            8657950371355520456,
            17023521755666265710,
            1295863205081166994,
            5105035812145571956,
            6533355259246593199,
            1810626334952379951,
            2312475032388835672,
            12791220430061225007,
            8771195993096234840,
            13726519721245562581,
            15405289548186568192,
            5371725187246102333,
            5459023237610151396,
            4754626066887167055,
            10553745724478562727,
            10750369126169131368,
            7042377789679595247,
            1075892665433271169,
            1817645324817591975,
            15480336652305539093,
            12003123856386804527,
            14131223727718098062,
            2633715514291894131,
            17539132648195827096,
            5775729883769698773,
            5330935016825856276,
            11112297728266276703,
            3633298269054989325,
            5518267083359827291,
            1518429259806995935,
            4379241623241024020,
            7666272129296596629,
            2928187192139330388,
            14719856945344155521,
            8643890290732777353,
            1453556899623419315,
            14843590660399662517,
            15447260990387974243,
            3088836309639962941,
            3486119244302167248,
            6501683573410273053,
            11497945129869273953,
            11670599808768993585,
            11161628522805939211,
            12971852256485324683,
            15886906348907956911,
            8769864054526859582,
            126346435621872899,
            5818414415043723373,
            8365413614023901251,
            1873483170000893036,
            9239390081193140011,
            1089649472565034247,
            9346894976656339440,
            14501303395203372242,
            9210278106251527714,
            4400100529770410320,
            4398812857882084789,
            13875661141430599825,
            1596805984855745375,
            3004146105824483869,
            1799019174582308024,
            15670902873428649861,
            11788917574771032420,
            2327871746250619683,
            11618181963213945928,
            12061730256005859839,
            8716976047004235535,
            17859824359887984959,
            6718335110612125061,
            18188611695612041790,
            3515433827540787200,
            5352510382906763701,
            14361399425533541680,
            11363502177285409418,
            4319801388747928392,
            17132855842084187622,
            12391151953236825741,
            13937748615773762066,
            13074580717040110746,
            9536812762643689867,
            5719358657743740261,
            6615349808167237910,
            8378385113004640684,
            457955522722589805,
            14569981664685374135,
            1194962890002875564,
            17341889085828621338,
            1426116268826667711,
            4502304835408985742,
            8797879829133724241,
            1267485363512703017,
            11441589260428658421,
            3833462114409229880,
            3964108302563308580,
            5539395930610653883,
            17927450418255681526,
            3141286039119313155,
            337135722387414782,
            16077995702336593849,
            15442820545356269710,
            14388041692131351826,
            10622157867675764834,
            13458819759356587950,
            7966392769679702989,
            531543895266784859,
            1210259833390644495,
            5280161032882450625,
            16464434114199834019,
            14239763828874734176,
            1470384961263210510,
            12896704156501049352,
            6211641471270258240,
            13003912526278510476,
            12080794918999361807,
            2686623831895362780,
            3759720586248051973,
            1691890749628120117,
            14717431311223065044,
            2951716441671345254,
            7754932474791790967,
            4765768643857692431,
            14070298632449111713,
            18441638563605740994,
            16556473250817697436,
            10936529817997800441,
            4874387414394373931,
            14786837019802046352,
            11770533905390438406,
            17578885899095090624,
            302256119256576927,
            9345778080473584027,
            10364538967325983458,
            113272102096493370,
            2135327315075906213,
            2120767509889943026,
            9175263376061503108,
            2566788151515168948,
            5876349546167837453,
            6504082391673541042,
            4736939020209829416,
            15612831358532680215,
            8817934843398462295,
            11251789301031536379,
            13677230752573332108,
            13748319366756702701,
            12965101289878891139,
            1972775725153044369,
            9648585810282111163,
            5484009252406561721,
            14539141163727100343,
            4429533407632645956,
            13416745650254481481,
            13652643609474129561,
            4870241048042915468,
            8535549686899957851,
            11334085157721544138,
            15510587245620089084,
            14710556750706262939,
            4539032213754257311,
            2939255170374158131,
            15745571416228980061,
            25381711537408267,
            4067333522794211673,
            1355446929432903662,
            10169069613463111445,
            16889024685941555331,
            8794510527122745695,
            16620544914310300209,
            11934454309668238181,
            5215725479785828349,
            949137555159775698,
            16653067186035943197,
            6316092429356498265,
            6931667391944937731,
            6731543925158082410,
            4853779404783271644,
            16678611146502492619,
            14318652289818538278,
            17197812129054272305,
            12995027322825931020,
            13290834662283319945,
            18274687287055470384,
            2064724977119842961,
            4205407444854339298,
            16821907270733396114,
            1776772147501243682,
            4702978230048724378,
            12797801735168854463,
            14228352426161981865,
            11016340879781451656,
            1116407879026854786,
            8754479409453916369,
            551350549293141026,
            8276313364776857606,
            2399805992182977457,
            131916166434611888,
            1229225777233916011,
            12253713524192541412,
            16706897661980161185,
            176412422412740527,
            16430061180440095725,
            11970305743191338265,
            3442136237583446443,
            3740046641013764036,
            13055542657299413108,
            17163154049617457444,
            2930338090021304750,
            11003479020565222578,
            5503613274729873094,
            7453640074779603882,
            14770053799160607556,
            10211528079476250566,
            8557896235182952367,
            7350027863192287849,
            12023110553236327214,
            4518356102475455759,
            27604043643590726,
            13254405952809676461,
            13142756636504642518,
            10308950411146029933,
            2782980271818783873,
            16478021220606771481,
            1352704759833866264,
            8683065154244794583,
        ];
        for item in v.into_iter() {
            vector.push_back(item);
        }
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            970935386315995388,
            18239475438954116417,
            18319525909197517530,
            5162925263928126717,
            12424389124876194158,
            6215650422037904429,
            6443289037295216077,
            7244981864007799610,
            3486022540297263134,
            1663870450782715445,
            9251548039753592124,
            14185151115881369172,
            4150296165001565682,
            4556148598730485423,
            13358644987929045265,
            3685503422897011349,
            4834649696264152738,
            17637372143074167618,
            3938597277872623354,
            955991479805768298,
            18431994660347813156,
            4131166349164763058,
            11798221299214662470,
            1731048977555287897,
            13845422595167511434,
            11473653420741890309,
            12789696601547252256,
            2631729172059556215,
            18060875344733452247,
            1317120497089572098,
            17600898385384843689,
            13884772903281828319,
            13625007248262703807,
            6169029335717059885,
            12227663093358426708,
            2232700651410037571,
            10712446849608895173,
            8380961169497377883,
            3205441955069112205,
            18296745244066864840,
            10380605939963734543,
            2160822980343404361,
            4677243245983840978,
            3686691477506396942,
            9641180910177838629,
            10872617892636938824,
            16244343748867501810,
            1058387260846545694,
            7047021752681570469,
            740519976020168740,
            7165727032227901643,
            4204954094595031810,
            17004085252676789892,
            7144792060623779086,
            8378583229618574801,
            7021116039605474353,
            5099499867076600619,
            9678356275433439716,
            12630850730163274137,
            12901676306219431752,
            10063139042638839751,
            2576512348037330638,
            16907299744738711936,
            3184765676603604411,
            17033640479893886065,
            7848833747953282586,
            1287270357606345977,
            17277261238918539376,
            18190160636290798508,
            17053732215718638668,
            2316742946334825547,
            10306778286328861316,
            12578634382442072314,
            1748633526703980397,
            906838523219820054,
            3864828524447385362,
            16840012667657558312,
            9427296998301845939,
            1790848809513036078,
            18160757742139119290,
            16462632083975083605,
            1447764985217020596,
            10137783045874515735,
            14866684276370205807,
            8205502907074080823,
            12903314568275385159,
            14911837970881403272,
            10279170418695981971,
            3285799706781985823,
            6823093690339295425,
            12887611234761594324,
            4168983783243659247,
            12738591045602398938,
            11965325820523188223,
            3940447864651572712,
            13960439903000319808,
            17965351807548519260,
            4250964251015772043,
            16896906455573574858,
            5998330905563413582,
            16852350818450547928,
            3383163497544164563,
            224084692242547861,
            14954702992881583722,
            1146883321504212529,
            679076294425432140,
            8744296203306599661,
            4131579214899769883,
            7825171910643622288,
            5377435380137465279,
            18007061778127492837,
            2726724067155214283,
            1176535630787634897,
            2929890299504623967,
            6344363061491157720,
            450758922847441786,
            3418154556430678767,
            11712686380106187982,
            5975891565395184786,
            8446905565957501787,
            9274560366455939723,
            15300037870117227815,
            14760006430772960897,
            10094755557428530005,
            9940079261637492482,
            9382854834816350081,
            12828941080875174630,
            13205299803625818415,
            17767176317703696196,
            17682525402495590291,
            10264903767953330897,
            2235858406517005328,
            6766710884125758871,
            1577154431395278418,
            13120176457686077948,
            292168117918576470,
            3940966365856196691,
            249494066524465881,
            6299120969464068222,
            3441470471401372391,
            18168108187818262962,
            13878798821457805694,
            1880628925607492362,
            13523893914162446524,
            5027139623150422742,
            10173736654394109500,
            15160972811026932844,
            9670915231357856603,
            1532889077217066189,
            5502121797008211580,
            5978998357852221572,
            5107553663849798450,
            10716512461957574847,
            8976095702569951321,
            2395691470507647201,
            11592986358842396830,
            15413619618647235325,
            9424974956022879373,
            2375006408929718462,
            11928816305420374158,
            8079844981012661449,
            1665340933671902321,
            16962999528030132645,
            16991696988999150037,
            8278056160980717173,
            13235689719656284651,
            1532395845260653474,
            12542085048562974464,
            7971426204399566319,
            15843900430387859972,
            4028409654535342668,
            11742550756742854505,
            4268208848898725556,
            547082513730585611,
            12575519787657791034,
            5041704428680962935,
            9071043979831048298,
            9155319576592364918,
            16635614382895181537,
            1665743736075257375,
            960372124172359436,
            13219941914506506698,
            3456421230738149071,
            9772150236102415222,
            13033126645623317221,
            17105202577073546222,
            13106608737396570148,
            14176004658123988151,
            2193349270608871712,
            12423539205537904678,
            17965964491013597421,
            5656482045153967889,
            18015792325043336205,
            3285092904077177411,
            14442350163929415066,
            13840566550524985078,
            1368268153129296013,
            10857563700414027855,
            15144753205453989348,
            11330292028602099463,
            18080014018610771036,
            14604203161241981196,
            668747833345312040,
            8864227456878352885,
            1478910197132668643,
            3515541121116989216,
            16346920898590917117,
            15975772328085604818,
            531300076945099404,
            261184231619452678,
            5495446537395364839,
            13783234786051960409,
            14706938508820268122,
            3299528389607846331,
            17990706505022086706,
            5555052466464994367,
            15255767213905504900,
            15029767658578794091,
            7036359288031772309,
            7539708496094773777,
            1170210716718584067,
            12280167074461287492,
            16259242964876250693,
            8474856198466724844,
            10902712727687349659,
            5888138869809464150,
            16754324016418996770,
            2113222199842089470,
            16952307182169237670,
            1066585472261716948,
            9489712365486520217,
            16964102047117799679,
            4309964180496073889,
            7592634813416349233,
            12663730919912437416,
            11771032429815885001,
            13879441440967005806,
            5866125298887920102,
            7083629416934578859,
            9298151990863033247,
            12359038331754221456,
            3484417373397901210,
            6638617909523282315,
            1867893112915322127,
            8930722290736560815,
            15061277728127274295,
            14577827375147938335,
            8540031119812151333,
            13317460419124411675,
            7078606966338796086,
            14800582856146192115,
            8671541084220856754,
            11708875493256307741,
            4467655934242595763,
            12446060062716561981,
            15018061651851940007,
            7088641329242279865,
            952990586878309599,
            76409505425850152,
            1054049214285692788,
            4307725862771655223,
            8865955616725377950,
            4434443185502789963,
            5086320212274245130,
            13669783290496841488,
            11367326551534709366,
            16677366348355147423,
            15033621149792753699,
            6331547207599414240,
            5358644506702082442,
            13154903493652620201,
            15829529119879244108,
            6682290016639867503,
            13317691176304345714,
            9793281684787999550,
            14980846391657346379,
            904993860823025611,
            16718422814065784218,
            3595796811264727592,
            5228126030105740057,
            9312588280395899626,
            3363177095872772500,
            1080085145334089773,
            5265193865861247168,
            10185972525858799416,
            807703002400202140,
            8263835144168193674,
            11637479110315228924,
            4512427729376136970,
            7582472081747053961,
            10990578463392737029,
            17065553594560316942,
            14734335948248998037,
            3632187440003520503,
            850173260929204671,
            3853042373131604302,
            6352356124904652225,
            9937803464869163491,
            10038785302879554487,
            7904483618045388354,
            14958751076595230183,
            6629758289721779008,
            16202022896759309002,
            17032166003495846747,
            14270132320009866442,
            8133420281582846395,
            1429133740546819066,
            6768884837336728626,
            11494538082256481984,
            5432588214879417412,
            15803538893812817711,
            10814744964847107169,
            6207791246978623790,
            12622517096440100684,
            12512885497919921030,
            278374753060359606,
            15179097850321603920,
            7806838476282931238,
            7644083073287198023,
            426527099984004582,
            11939367539757579946,
            673416659596919137,
            15349366448653432198,
            584543106922557582,
            17384045482949244564,
            7521636989763798992,
            6546164590031491994,
            12964166245441729491,
            11076389118426000226,
            12741563420085283592,
            1857788424428683539,
            4741517826156033587,
            17950220485035559271,
            14992630042856036187,
            17604540996027951963,
            7774239679719099679,
            4541763558780864719,
            2031762308212592589,
            3797125212946897075,
            17968437032201220583,
            9845343376209132745,
            8556892907187090914,
            1874988543758946629,
            10045618566167137050,
            4496210928810782919,
            9053284887964365429,
            13506249999200118846,
            4063690903208309244,
            9299158214321374787,
            1846664106792982472,
            15585517993331456928,
            17765036713999992193,
            16433325812493682713,
            13900252830601680815,
            8076278298690626788,
            397531676548792027,
            18092031775730454146,
            4765656601076429945,
            9696759837427159675,
            118766624260230866,
            14038656159381301900,
            4766410748540791400,
            8989934917675559561,
            2026534559458237503,
            5151823718365472942,
            11271707092058473678,
            5915905395701048951,
            2373735694848644841,
            8352006982473587540,
            1392404155199882867,
            16286470416853783794,
            9094413033578737315,
            2232017884559508862,
            2068543962104961685,
            12164990712627833944,
            3149793659825255828,
            17593385326009967863,
            9281194005934693960,
            17373129108923946694,
            2562733283348140207,
            7310249838546084215,
            1415153972202678877,
            10637980769418867109,
            13413245082016538328,
            240361941200958741,
            3871985471697258904,
            11509480779782259473,
            2594090912784537516,
            5922819741791209506,
            1360398338448655729,
            8541712799221248984,
            11224428397107962963,
            6423112798171826332,
            12545590580258003838,
            5991565522039073560,
            6519422859681520448,
            17925061848910260170,
            216201401124292729,
            11121260531680537729,
            15883183149777143790,
            4732766883590037273,
            2813324161725068917,
            17772003672696933308,
            782798092097342469,
            2934505652532201011,
            11541482758552231735,
            6411829342199353196,
            1938412939310591867,
            13056260460537774996,
            17339626591809382938,
            10896788120633274286,
            10810880100911416295,
            1335176357026099123,
            2324578251120123014,
            12512419964131211587,
            13946738228022590264,
            5814708519434359272,
            3524474940317106616,
            10660252771622433300,
            5159326501864252661,
            15255726534088559465,
            4223034258054172365,
            16537181047026695731,
            12555663328890574089,
            5091217062907525153,
            4895005266333526250,
            18065118596912494345,
            9397037345805472758,
            11660218380719882977,
            2602408215964589057,
            5424815119243214646,
            15766479757163450383,
            12501635975269066717,
            9064678358547302377,
            783552068166127220,
            16890386535921841478,
            11687874056847264462,
            10229350117004653584,
            5381219722486263503,
            14728465844855369610,
            356296053818096199,
            18234007501009322298,
            18342921082884287169,
            8355765375656385215,
            925237697270775749,
            1382009822723819545,
            11565514923584761902,
            5345676016014134623,
            17632434763519988186,
            7034359126919022344,
            12940707861506815613,
            17001985253016397750,
            15975236423596767363,
            1700970537587805075,
            6847887058885525174,
            10636738992314009931,
            1267197753573320114,
            8318867032664392582,
            220894085744684389,
            11078033618100820399,
            17338429636517933646,
            11270031867533314080,
            4047085984198270888,
            7109447136754819035,
            16005387982275355202,
            16675705597236281786,
            12300537905716017979,
            7581706776540012077,
            16320196834361079311,
            6366323575333421438,
            2187640857204571047,
            5121134750491405892,
            7563782581437313874,
            323483345126262297,
            671357852845832275,
            13716471088175844893,
            13490296756388087051,
            10042606706481981383,
            2869626562754227863,
            14628318782550460783,
            15938482945122308426,
            4463904122535661017,
            355181386614707511,
            13537606997342942657,
            6432436488291563179,
            5462344331446405276,
            16191522220562417027,
            6004191088141075184,
            17104966483352295254,
            9158864459611898144,
            3654536371429800411,
            7431984816804826121,
            7168647048360233905,
            7900641114545088211,
            1931159185771253301,
            5237489967993523241,
            13696299977323936944,
            14524962759921568384,
            1399135786466243345,
            9920690356955845366,
            17152370489056925285,
            15330386506484169220,
            17282817267697010022,
            9767423880769413803,
            2190265334080404475,
            2745772546921923602,
            10479746594935082160,
            1004285227802680415,
            14809964040590190190,
            16358958722397522061,
            5536293708904944631,
            7537948737067455730,
            3511850083508523055,
            5332348593494675353,
            15977876635418358024,
            18412267680091231135,
            5460309471508535307,
            9289900262213272569,
            13286348651646230176,
            6522209658791982494,
            6075346246397034023,
            11973474871515360577,
            2412808281297246422,
            4539643891959764287,
            5615642801672163866,
            1861377058220409916,
            2032068329794422734,
            3972491300014073564,
            18424846340005485889,
            11151951397216506502,
            99083267429418761,
            2784649789939373104,
            13948239872598876396,
            8060623703270962317,
            1083710206737525531,
            11149566857153403773,
            359522570863389435,
            5047494622792329834,
            12683451233445476868,
            16525227250638010036,
            14956915791268323094,
            4134364362879686920,
            465915113333842616,
            15847805391031380573,
            10304262140191945670,
            2482266849593158978,
            6317592966217072969,
            8042347210745221615,
            18069555390922814001,
            2030301008249946291,
            17910646698416354088,
            12078628488286161604,
            11093094447928751961,
            11623237432262409196,
            8892019813188476112,
            8563976599540589574,
            3707026930109627301,
            10048551475926284474,
            390828895709920869,
            2863033712428045169,
            12676392878310372279,
            4614103896291853288,
            13677121043864051526,
            7603833336555690687,
            16793405178418447114,
            4207029990274301168,
            10332831766792102544,
            12263249654381460311,
            6058214514854565551,
            9586483725004202578,
            10306480318743575142,
            10309805938363497068,
            18056606945356481302,
            17557265018593690061,
            12165996509085881872,
            3656818056056954717,
            5428859980996759945,
            555647585525811904,
            16099957564285114116,
            1990310175914831970,
            1633325586903873298,
            2862415615580296290,
            4076621952396152827,
            6178936183122436759,
            10528953186396137266,
            16385950417200918807,
            5940650591077210503,
            2727054135958865246,
            927968473452611705,
            1283523786245589498,
            2160178327862764977,
            3710491568464158699,
            9105439700185863926,
            2748028767310340789,
            2248070231045169046,
            6573480320915444749,
            13505230706370483200,
            16832170361329550529,
            7973877415667750908,
            4396784018718944069,
            106471480049019828,
            6212526451782295972,
            14817956739240458636,
            11076597739099133405,
            1674938822871911116,
            2842352335671308344,
            6621201547328650176,
            6572041374890978863,
            16168038130956085102,
            11108561258084597231,
            3255621408879363230,
            2390072753685199075,
            2449281880777710473,
            13470370142695618001,
            7815112405679593498,
            11043760393309655793,
            5153265669705278532,
            10368492728325764789,
            6921740492034384980,
            13514192749519922805,
            6397527011205988808,
            337246354198001582,
            8252717981508988811,
            9957421388715734990,
            4867920059629452679,
            9732089391317127447,
            404651710614544155,
            4040837754413269294,
            14089854635649909368,
            8689130073186571828,
            8464054946005836724,
            1825821908208233113,
            7574359299529343680,
            14718406219095050752,
            11732428854013789284,
            8217099702439338555,
            12614474023124672510,
            7689458279874551065,
            15302799089435341456,
            3103015065282594319,
            14317588355828811790,
            10435845936742394132,
            8825210084446328347,
            17723347906298388826,
            4280201013500254061,
            17546736376306976741,
            10886919252028028638,
            1347279761838045587,
            976465627626592856,
            2673065074601947646,
            6861599634300102991,
            8829589996811322420,
            5740297437331545421,
            10018864397477041838,
            1594871220590215087,
            13805317630549364512,
            3208334280262560313,
            5096011366978881745,
            4078456272654377290,
            601108671885150437,
            10561396304628795619,
            2793931519585019701,
            17003973068220840334,
            13625570697683821845,
            8359899832863235843,
            14313616762406686563,
            4765797822386815431,
            11925982874804542246,
            4550660493541646726,
            9737812492111587112,
            8160071575200020598,
            3090933445982835749,
            7790017766482684204,
            3369804602959344083,
            5666911646369121316,
            16854762180484511165,
            15315328029007859614,
            15941953978848261142,
            5209787268385217681,
            1742961000907502016,
            12323446887611563836,
            1128076406290274329,
            12464623235549088830,
            6129344170548298048,
            9681986900566805125,
            1063878614722851495,
            14400879070748359438,
            16402355185219109138,
            5820075804054128323,
            11617701936322877895,
            17779960563798324887,
            15669660728595709193,
            2567056000055431723,
            4003710497352735483,
            3656433648214241409,
            15162672058339351696,
            13592989399321144647,
            4044072218289357967,
            12637296667862700721,
            7182176739776970529,
            9294617069928882236,
            11895862584749295164,
            11910787406333822145,
            9479866365093740793,
            6811682987703623940,
            3177135152567606601,
            11255589266935196705,
            13983834625167214489,
            15240358163750827475,
            17989680793173261236,
            15397106991008415493,
            8614490225748137337,
            3735734308447136588,
            12533329079878267576,
            5645445633567727893,
            3046624883284432856,
            6584308803865395301,
            4392949048373752371,
            7791768338771655463,
            249444847995374183,
            11760395270234257558,
            11248540399908619626,
            1251397685293026802,
            15015890928909701526,
            18109747303982907373,
            10154370843006151548,
            16893419127227914816,
            13423150087147682165,
            10746719339904477949,
            14498099794763129373,
            11256516953884252490,
            5540762621508419198,
            15449934900659225792,
            16209016329320121106,
            12813501817949021801,
            11883263215204609553,
            11680499573896858668,
            4169192082702521885,
            17961129717778224891,
            9080297577165948060,
            13521066463597568028,
            2950775345183731342,
            7109804752100600809,
            17186031772769075160,
            10119063638928414388,
            9719778720760645930,
            8854438578173121771,
            4201323475736587447,
            9875198500352867364,
            3158424219143144712,
            17526692212636487324,
            1092738682559830203,
            2463176061360951010,
            11072153364631905838,
            4427247801982067070,
            16310697878771005784,
            7055124627194291660,
            11177175741828046314,
            10190516719776078144,
            18062422414368345735,
            1372595521226428524,
            8708287087087565540,
            439502603080724564,
            16846325421539607823,
            4185915222704035397,
            15492993114529352290,
            11973554942280106848,
            4557419981167225754,
            13308740713393235800,
            6523045310951210137,
            9902574994610428899,
            10155731518569908193,
            8032898039699191985,
            14873938192541804009,
            829425646495242225,
            3806355026232415361,
            2440836690733476045,
            9334314793137562644,
            17666231249434837638,
            5367198724522718061,
            5690221396324014055,
            12848694436144921418,
            7894706307710608410,
            13425336693724328856,
            17724433104707406942,
            18298151012918099152,
            9007503298109492860,
            16626229726302175771,
            16170701057093670433,
            14771579467801547971,
            4772867062988863830,
            12442080962563675547,
            17077745604326452491,
            4907540227919826511,
            12796438412016105590,
            3523974959040741212,
            12909368810298423166,
            11055661857496654454,
            14892904661646685267,
            7843040161494715663,
            11596858248121427823,
            1529941072723021702,
            8724028260690775051,
            8625449939850789187,
            3254213028847926667,
            17685974342643997380,
            14129642900702454777,
            8358462655670775060,
            4434465244334825015,
            7713505610337412048,
            5315651703955700935,
            6494727565007270652,
            6811941830631023044,
            4701360122162282714,
            16654672640612622393,
            15467493939152109694,
            16548381956834017072,
            5048705340166051752,
            9082225281638587715,
            5778589268868820555,
            584639593668506876,
            13126242504995359190,
            292185952717939684,
            8430100398347968926,
            5982395399406272866,
            14850169301338715263,
            13042715785150391780,
            8204396918532682948,
            8102459596244614243,
            16604794152040138808,
            10897184311946203145,
            14717385396178437061,
            8429222974817735352,
            15592299898205423582,
            5449784579945580173,
            3466740037129807085,
            16187423828125085566,
            17723257136342196411,
            2624888668067989181,
            139249098190696101,
            8173037253708142614,
            9102186004775202046,
            9959923101705842871,
            16621406202329389348,
            2243148047535467791,
            3535214435310634872,
            16421677655742897882,
            9666517799299772724,
            4274783470810664856,
            4078847538031439435,
            991239997361959792,
            13912699450921474273,
            9859115065971422420,
            17977700666748047437,
            17664364533371692199,
            11567064561948855496,
            5635218810677067625,
            17818764373959601314,
            11180079375631680927,
            13493724095664599688,
            8995474278960471273,
            8096006232516668319,
            3757534043574594779,
            18047631443129845710,
            968775491777850155,
            2837928087100799397,
            6514661013234655149,
            1958785647947860629,
            18424174020609171099,
            14595369882988933727,
            8517104138865227696,
            11203330715881753115,
            12832431343892430059,
            6436165275623918632,
            9237554695638907758,
            13986148378775091898,
            2188729999107545365,
            42281586039949832,
            527755732284719038,
            582146973191012664,
            15606520478103413797,
            3562074809929056579,
            15179974226238408439,
            9843006530233843309,
            13371075380776938234,
            15355391593018979032,
            16431326118119511118,
            3265395740493580220,
            5081429451884316258,
            9848836621972846726,
            7419281351930387199,
            17086215758345641912,
            9330196458415374957,
            4089489645279309861,
            3617900430328777705,
            3809179367379720272,
            10302255027991421668,
            7875385966749939019,
            7917828339405682210,
            15157451036065719586,
            4060131170589992313,
            11170679364084169910,
            10113593188999467464,
            10371530081825415062,
            1973460803062646624,
            5574011404673874631,
            7027305659160135694,
            7937656739709494028,
            14380069779153093596,
            10230158321167102818,
            14689079134930006004,
            1109078465060096488,
            16997193474398069,
            12302335017211058443,
            14157801764678952627,
            7979028941970185561,
            7846520968775961274,
            9973860428715527091,
            13747154168574832256,
            6709660143208008924,
            11136554806869410310,
            6431651916310477212,
            8721403174717244714,
            11394862951980881853,
            13411596782563106743,
            4630304965213733094,
            6131473776786752743,
            5877271733091861194,
            6136628782665990739,
            13832497654892819588,
            2988987420353204457,
            5203013080233950426,
            5541332707115228970,
            1456709864682564910,
            12596234500852625161,
            10027429985326331168,
            6106640044166330303,
            16449758310371841976,
            5570426757015653863,
            3070109237277398548,
            11776997324472155763,
            13912409334050693696,
            8182591900533638364,
            1804786871230590612,
            2871290593361076864,
            8437760288865496032,
            15560303803488152855,
            10188778672082882358,
            17320739069311440224,
            17741129753547212833,
            3534140395395092668,
            17741613654372890180,
            2460465468806975984,
            1718625770937332572,
            6592263169095699475,
            4432072184506256096,
            17071177053723575340,
            15645165287174304299,
            4341196293753623550,
            13254940639694643829,
            9458901694698912240,
            1360720723215583848,
            3430338023691190733,
        ];
        let mut new_vector = Vector::new();
        for item in v.into_iter() {
            new_vector.push_front(item);
        }
        new_vector.append(vector);
        vector = new_vector;
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![
            8281448380107603213,
            8968199636762334080,
            14043553275947670154,
            4212027269671610822,
            13738804027707579245,
            3392306825303616899,
            10462858521978988589,
            1506873232916249903,
            9968310746280482628,
            7115333817943255279,
            9350517859354978439,
            16682681148622617969,
            16157437779490066910,
            2383048905413808799,
            5056780844586347384,
            4446819363127660120,
            14491187600951210195,
            12226634718895842532,
            70502194391851811,
            10049888871111205940,
            10923866070950377329,
            9042620219444720700,
            18063625744824465663,
            1771598478926461899,
            9521875949151158355,
            1035739548787970019,
            10410533907782681904,
            5014600274907552307,
            16887597940899090620,
            10001347463014446863,
            7272792700010831800,
            8011377019302503949,
            941834534617080427,
            7986819317774464961,
            7723443356361998997,
            2485158855313855675,
            6825467690420903093,
            9819797473464404155,
            1512753791813549867,
            6298100164541190573,
            13944179790671356575,
            10550671321464268497,
            5566332512743709805,
            18126974206091853716,
            5664908333583831217,
            5065360735893201245,
            10633180908122886637,
            3109480431953006355,
            15009241610517007873,
            12421205332049966030,
            2193741450779268243,
            14901551102983644270,
            1224685054977856826,
            18074838891225457789,
            11249150213118849621,
            1888651027587464542,
            5802302740311144555,
            4214604964890708873,
            15449507913849849740,
            2220398536114095230,
            17696367650171288080,
            7408498461500273656,
            1209695502667201073,
            11798650396657578687,
            8302492679464144270,
            9119215984506274431,
            7741612921764505508,
            8498238978041281964,
            7547317241141300541,
            12552510842547554145,
            14046212534143364824,
            9097616468253825736,
            10891823424325752527,
            13897735545304188126,
            11684956870657994275,
            6818633703998364287,
            6666669052643773237,
            14790181983151940583,
            8220060219326871152,
            17255317358253557619,
            15931708303011278048,
            9937547507564836281,
            5928876630260715639,
            16677429789088916476,
            9456941196583530367,
            9793178597466790856,
            587935318280162778,
            13806541518533519874,
            2566852685870925261,
            15697687295184361315,
            7137523130045282917,
            11297107930519842805,
            14554362261674337960,
            10877178659181616204,
            10032280433989940071,
            11422164126284125541,
            11912641379402334705,
            6875963126749328257,
            16693997568437786549,
            14073896492345980041,
            2295001157277773312,
            4902022132745971409,
            5297139507179134953,
            15126282524710191501,
            12733159955915797000,
            4788887554657883070,
            4770781532660077625,
            124870421682097781,
            644578782104361964,
            14136730623925455745,
            6459819182580742971,
            12796808974763678765,
            3156569143217152474,
            2425807434189597664,
            3723213790426241267,
            15534102526837505996,
            16456920119514092398,
            17871780623793210676,
            16268978989006982242,
            15584786530677392476,
            2363505800468492344,
            11798814526811951310,
            15223103386897595228,
            4992397422252464150,
            16807356264795211960,
            9415567864121710055,
            13899155575052544744,
            39800438722987095,
            15244590325086360873,
            15009802266439045523,
            4814411406842345488,
            5649425734956797179,
            9974550968093160526,
            9554596380537671651,
            17168508713637674016,
            5974122140334473648,
            868838807717072654,
            399057640941980368,
            12004427764765727812,
            1407729965586004998,
            8297215758641051809,
            13784483881381964336,
            15713952475955116512,
            7597817164083121152,
            12112196378080696685,
            8687431199900132372,
            8312636302348483601,
            11038801983932353593,
            10869306998034514478,
            3294055461809567545,
            6724289882700165198,
            3027960835947107636,
            1660220875821840750,
            9954769967009179761,
            15697189825734302305,
            2596398523841936175,
            5524759891122318720,
            5258863375854631393,
            18315673154803354336,
            16664047631787766390,
            8579350147668837942,
            10592296341846364424,
            7423826597279990793,
            13473670089319186842,
            14343204363630856625,
            3129648781050386822,
            15468986603271843871,
            11676876043214418849,
            4050461466209478379,
            9824268438354100059,
            16476000554976880613,
            4326589976796095850,
            5279827292147685613,
            6854544266595188923,
            2820325733076670249,
            15181590593356262930,
            3513053552958966377,
            2872048237836618893,
            4430038194754287309,
            12352466297109962137,
            4502612375090975857,
            12596952913141523677,
            3565998443879446883,
            12353864032113274225,
            12965018338639141822,
            4226956252590541697,
            5943336922337106584,
            12648512561013984281,
            3241473955614976345,
            17581610707264523431,
            6276878328376112780,
            8392563049190175900,
            2938569092916437739,
            10575537289882958425,
            9600472851361882529,
            8557376681472365146,
            12968804928615188537,
            9471932459874190383,
            11709226348385548847,
            5100087857952972277,
            17747687994311040941,
            1392733953328322116,
            11441457744411620985,
            5168018411136726022,
            7940218313456905295,
            9048223127927224785,
            4986077653102296852,
            10508047594951184618,
            5338421268486274477,
            7774250641964732736,
            12151808300672407748,
            10794451767505716034,
            7071368695278445984,
            16207069691455450941,
            12378031374015309592,
            6730887436213973148,
            15566010643764610093,
            6688882331103034551,
            14716232330925764282,
            18409452682550039458,
            4727308003898835393,
            6966347304159087221,
            4835586352650026355,
            15007911089847243259,
            11420422591608680478,
            16122889658541190928,
            10012692889180295131,
            7996395508751698037,
            10346807325847132863,
            18039339283708900583,
            3601212726968930358,
            4494939363102115585,
            16240702075755287066,
            4231800307957013381,
            14519745657520370228,
            14916978630742441568,
            4766464367483576587,
            10746806616050850086,
            9216616453315639534,
            5920051101783844662,
            11144223426962045802,
            16984164883350237236,
            16978162629562842709,
            8998047494371215777,
            14262048124322385424,
            821064209281419355,
            10822686698509545180,
            13479293025194295718,
            4659165614467704487,
            434129260286565446,
            16576857551627996034,
            12518000006146258835,
            10165647693985455600,
            16180795852653293742,
            2243624567975010112,
            10067276443238014587,
            4717438028041902832,
            2482428787083045423,
            13135612085704026742,
            1287616353316515536,
            6378291031422596348,
            4586741194863569119,
            12772338039633544531,
            15515403009851143166,
            1103817212777222716,
            17513630427097984895,
            16290061065993157212,
            16545536429178738641,
            3038665396401513041,
            16883232237325712146,
            5894496096019578883,
            14564728289066063965,
            10316309595355600390,
            5669892614956647371,
            7081399942560292028,
            7177937101352009793,
            14294819386388058876,
            17883364068077804309,
            13369755665740724872,
            9117370191009158630,
            2940962106847264863,
            11830440118525474390,
            15073828695951543066,
            15240708711064061978,
            1250420736930770500,
            3001971418677500455,
            8321493806507415406,
            2738853450968688033,
            17092499640294547135,
            17787064561559356357,
            13190425814862770993,
            14341675823709844040,
            1211066599032087789,
            13915524090626542588,
            14172126120892557652,
            13004070188257101840,
            12144924079867416876,
            2290251731610872155,
            14445606604417116673,
            9225784807264729133,
            12810629627951651473,
            1711093900875123980,
            7588940389341660490,
            6501917270485417989,
            10432160864283300378,
            1928422360518862616,
            13129569786207188858,
            8123463298773263850,
            16086116202748536307,
            17155072915310637423,
            2255383309691243440,
            3973076390739154207,
            1495164858896608419,
            4743799525367712296,
            2994349019446123285,
            9982507507464563271,
            14837595199674608871,
            11871305301553066836,
            2149267746658605852,
        ];
        for item in v.into_iter() {
            vector.push_front(item);
        }
        assert!(vector.assert_invariants());
        let v: Vec<u64> = vec![1236782927197526029];
        let mut new_vector = Vector::new();
        for item in v.into_iter() {
            new_vector.push_front(item);
        }
        new_vector.append(vector);
        vector = new_vector;
        assert!(vector.assert_invariants());
    }
}
