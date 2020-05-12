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
//! | [`Concatenate`][Vector::concatenate] | O(MH) | O(MH) |
//! | [`Clone`][Vector::clone] | O(H) | O(H) |
//! | [`Front`][Vector::front] | O(1) | O(1) |
//! | [`Back`][Vector::back] | O(1) | O(1) |
//! | [`New empty`][Vector::new] | O(M) | O(M) |
//! | [`New singleton`][Vector::singleton] | O(M) | O(M) |
//!
//! We currently choose the `Summing over two levels` algorithm for concatenation, however in the
//! future this may switch to other algorithms. This means that H is bounded by O(logN + logC/M^2).
//! An example of what to expect from this algorithm can be found by understanding the inserts test.
//! The insert operation is done by splitting the original vector into two pieces before and after
//! the split point. The new element is inserted and the two pieces are concatenated back together.
//! The test, itself, inserts 0..N repetitively into the middle of a Vector. When N is roughly
//! 200000, the height of the tree reaches 26. For comparison a full tree of height 4 can holds
//! 16_777_216 or over 800 times this amount and a full tree of 26 levels could hold over 10^44
//! elements.
//!
//! [Vector::push_front]: ./struct.Vector.html#method.push_front
//! [Vector::push_back]: ./struct.Vector.html#method.push_back
//! [Vector::pop_front]: ./struct.Vector.html#method.pop_front
//! [Vector::pop_back]: ./struct.Vector.html#method.pop_back
//! [Vector::slice_from_start]: ./struct.Vector.html#method.slice_from_start
//! [Vector::slice_to_end]: ./struct.Vector.html#method.slice_to_end
//! [Vector::concatenate]: ./struct.Vector.html#method.concatenate
//! [Vector::clone]: ./struct.Vector.html#method.clone
//! [Vector::front]: ./struct.Vector.html#method.front
//! [Vector::back]: ./struct.Vector.html#method.back
//! [Vector::new]: ./struct.Vector.html#method.new
//! [Vector::singleton]: ./struct.Vector.html#method.singleton

use crate::focus::{Focus, FocusMut};
use crate::nodes::{ChildList, Internal, Leaf, NodeRc};
use crate::sort::{do_dual_sort, do_single_sort};
use crate::{Side, RRB_WIDTH};
use rand_core::{RngCore, SeedableRng};
use std::borrow::Borrow;
use std::cmp;
use std::collections::HashSet;
use std::fmt::Debug;
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
#[derive(Clone, Debug)]
pub struct Vector<A: Clone + Debug> {
    pub(crate) left_spine: Vec<NodeRc<A>>,
    pub(crate) right_spine: Vec<NodeRc<A>>,
    pub(crate) root: NodeRc<A>,
    len: usize,
}

impl<A: Clone + Debug> Vector<A> {
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
        Vector {
            left_spine: vec![],
            right_spine: vec![],
            root: Rc::new(Leaf::empty()).into(),
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
    /// let v: Vector<u64> = Vector::singleton(1);
    /// assert_eq!(v, vector![1]);
    /// ```
    pub fn singleton(item: A) -> Self {
        Vector {
            left_spine: vec![],
            right_spine: vec![],
            root: Rc::new(Leaf::with_item(item)).into(),
            len: 1,
        }
    }

    /// Derp
    pub fn constant_vec_of_length(item: A, len: usize) -> Self {
        let mut store = Vector::new();
        let mut accumulator = Vector::singleton(item);
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
    ) -> impl Iterator<Item = (Option<(Side, usize)>, &NodeRc<A>)> + DoubleEndedIterator {
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
        debug_assert_eq!(self.leaf_ref(side).free_slots(), 0);
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
            let parent_node = Rc::make_mut(
                spine
                    .get_mut(idx + 1)
                    .unwrap_or(&mut self.root)
                    .internal_mut(),
            );
            Rc::make_mut(&mut parent_node.sizes).push_child(side, full_node.len());
            match parent_node.children {
                ChildList::Leaves(ref mut children) => children.push(side, full_node.leaf()),
                ChildList::Internals(ref mut children) => children.push(side, full_node.internal()),
            };
        }

        if self.root.slots() >= RRB_WIDTH - 1 {
            // This root is overfull so we have to raise the tree here, we add a new node of the
            // same height as the old root. We decant half the old root into this new node.
            // Finally, we create a new node of height one more than the old root and set that as
            // the new root. We leave the root empty
            let new_root = match &self.root {
                NodeRc::Internal(_) => {
                    Rc::new(Internal::empty_internal(self.root.level() + 1)).into()
                }
                NodeRc::Leaf(_) => Rc::new(Internal::empty_leaves()).into(),
            };
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
        if self.leaf_ref(side).free_slots() == 0 {
            self.complete_leaf(side);
        }

        Rc::make_mut(self.leaf_mut(side)).push(side, item);
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
        // 1) If the root is empty, the top spines have at least SIZE - 1 xhildren between them.
        // 2) If the root is empty, each of top spines are within 2 size of each other
        // As a consequence, if any of the top spines are empty, the root is non-empty.
        // If we break invariant 1) then we shrink the tree, as we can make a new root from the
        // children of the tops of the spine.
        // If we break invariant 2), we call balance spine tops to correct it.
        debug_assert_eq!(self.left_spine.len(), self.right_spine.len());
        // Try and pop the node into the level above
        let spine = match side {
            Side::Back => &mut self.right_spine,
            Side::Front => &mut self.left_spine,
        };
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
            let child: NodeRc<A> = match Rc::make_mut(node).children {
                ChildList::Internals(ref mut children) => children.pop(side).into(),
                ChildList::Leaves(ref mut children) => children.pop(side).into(),
            };
            Rc::make_mut(&mut Rc::make_mut(node).sizes).pop_child(side);
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
        while self.root.slots() == 0 && !self.root.is_leaf() {
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
            if let Ok(item) = Rc::make_mut(self.root.leaf_mut()).try_pop(side) {
                self.len -= 1;
                Some(item)
            } else {
                None
            }
        } else {
            // Can never be none as the is of height at least 1
            let leaf = self.leaf_mut(side);
            let item = Rc::make_mut(leaf).pop(side);

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
        Rc::make_mut(leaf.leaf_mut()).front_mut()
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
        Rc::make_mut(leaf.leaf_mut()).back_mut()
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
        if self.is_empty() {
            *self = other;
            return;
        }
        if other.is_empty() {
            return;
        }

        let new_len = self.len + other.len();

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

        debug_assert_eq!(self.right_spine.len(), self.right_spine.len());
        debug_assert_eq!(other.left_spine.len(), other.right_spine.len());
        debug_assert_eq!(self.right_spine.len(), other.left_spine.len());

        let packer = |new_node: NodeRc<A>, parent_node: &mut Rc<Internal<A>>, side| {
            if !new_node.is_empty() {
                let parent = &mut Rc::make_mut(parent_node);
                Rc::make_mut(&mut parent.sizes).push_child(side, new_node.len());
                match parent.children {
                    ChildList::Internals(ref mut children) => {
                        children.push(side, new_node.internal())
                    }
                    ChildList::Leaves(ref mut children) => children.push(side, new_node.leaf()),
                }
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
            let parent_node = other
                .left_spine
                .last_mut()
                .unwrap_or(&mut other.root)
                .internal_mut();
            packer(right_child, parent_node, Side::Front);
        }
        while !self.right_spine.is_empty() {
            let mut left_node = self.right_spine.pop().unwrap();
            let mut right_node = other.left_spine.pop().unwrap();

            let left = Rc::make_mut(left_node.internal_mut());
            let right = Rc::make_mut(right_node.internal_mut());

            if !left.is_empty() {
                left.pack_children();

                let left_position = left.slots() - 1;
                while left
                    .children
                    .get_child_node(left_position)
                    .unwrap()
                    .free_slots()
                    != 0
                    && !right.is_empty()
                {
                    match left.children {
                        ChildList::Internals(ref mut children) => {
                            let destination_node =
                                Rc::make_mut(children.get_mut(left_position).unwrap());
                            let source_node =
                                Rc::make_mut(right.children.internals_mut().front_mut().unwrap());
                            let shared = source_node.share_children_with(
                                destination_node,
                                Side::Front,
                                RRB_WIDTH,
                            );
                            Rc::make_mut(&mut left.sizes).increment_side_size(Side::Back, shared);
                            Rc::make_mut(&mut right.sizes).decrement_side_size(Side::Front, shared);
                            if source_node.is_empty() {
                                right.children.internals_mut().pop_front();
                                Rc::make_mut(&mut right.sizes).pop_child(Side::Front);
                            }
                            assert_eq!(right.sizes.len(), right.children.internals_mut().len());
                        }
                        ChildList::Leaves(ref mut children) => {
                            let destination_node =
                                Rc::make_mut(children.get_mut(left_position).unwrap());
                            let source_node =
                                Rc::make_mut(right.children.leaves_mut().front_mut().unwrap());
                            let shared = source_node.share_children_with(
                                destination_node,
                                Side::Front,
                                RRB_WIDTH,
                            );
                            Rc::make_mut(&mut left.sizes).increment_side_size(Side::Back, shared);
                            Rc::make_mut(&mut right.sizes).decrement_side_size(Side::Front, shared);
                            if source_node.is_empty() {
                                right.children.leaves_mut().pop_front();
                                Rc::make_mut(&mut right.sizes).pop_child(Side::Front);
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
        }

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
        self.len = new_len;
        self.fixup_spine_tops();
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
        if len == 0 {
            self.left_spine.clear();
            self.right_spine.clear();
            self.root = Rc::new(Leaf::empty()).into();
            self.len = 0;
            return;
        }
        let index = len;
        self.make_index_side(index - 1, Side::Back);
        self.fixup_spine_tops();
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
        if start >= self.len {
            self.left_spine.clear();
            self.right_spine.clear();
            self.root = Rc::new(Leaf::empty()).into();
            self.len = 0;
            return;
        }
        let index = start;
        self.make_index_side(index, Side::Front);
        self.fixup_spine_tops();
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
    fn make_index_side(&mut self, index: usize, side: Side) {
        if let Some((node_position, mut node_index)) = self.find_node_info_for_index(index) {
            // We need to make this node the first/last in the tree
            // This means that this node will become the part of the spine for the given side
            match node_position {
                None => {
                    // The root is where the spine starts.
                    // This means all of the requested side's spine must be discarded
                    while let Some(node) = self.spine_mut(side).pop() {
                        self.len -= node.len();
                    }
                }
                Some((spine_side, spine_position)) if side != spine_side => {
                    // The new end comes from the opposite spine
                    // This means the root AND the requested spine must be discarded
                    // The node we are pointing to becomes the new root and things above it in the
                    // opposite spine are discarded. Since higher points come first we can do this
                    // efficiently without reversing the spine
                    while spine_position + 1 != self.spine_ref(side.negate()).len() {
                        self.len -= self.spine_mut(side.negate()).pop().unwrap().len();
                    }
                    self.len -= self.root.len();
                    let new_root = self.spine_mut(side.negate()).pop().unwrap();
                    self.root = new_root;
                    while let Some(node) = self.spine_mut(side).pop() {
                        self.len -= node.len();
                    }
                }
                Some((spine_side, spine_position)) if side == spine_side => {
                    // The new end comes from the same spine.
                    // Only the elements below the node in the spine need to be discarded
                    // The root and left spine remain untouched
                    // To the spine discarding efficiently we reverse, pop and reverse again
                    self.spine_mut(side).reverse();
                    for _ in 0..spine_position {
                        self.len -= self.spine_mut(side).pop().unwrap().len();
                    }
                    self.spine_mut(side).reverse();
                }
                _ => unreachable!(),
            }
            // We need to complete the spine here, we do this by cutting a side off of
            // nodes going down the spine
            // We need to do this in reverse order again to make this more efficient
            let spine = match side {
                Side::Front => &mut self.left_spine,
                Side::Back => &mut self.right_spine,
            };
            spine.reverse();
            while let NodeRc::Internal(ref mut internal) =
                spine.last_mut().unwrap_or(&mut self.root)
            {
                assert!(node_index < internal.len());
                let num_slots = internal.slots();
                let (child_position, new_index) = internal.position_info_for(node_index).unwrap();
                let internal_mut = Rc::make_mut(internal);
                let children = &mut internal_mut.children;
                let sizes = Rc::make_mut(&mut internal_mut.sizes);
                let range = match side {
                    Side::Back => child_position + 1..num_slots,
                    Side::Front => 0..child_position,
                };
                for _ in range {
                    match children {
                        ChildList::Internals(children) => {
                            children.pop(side);
                        }
                        ChildList::Leaves(children) => {
                            children.pop(side);
                        }
                    }
                    self.len -= sizes.pop_child(side);
                }
                let next_node = match children {
                    ChildList::Internals(children) => children.pop(side).into(),
                    ChildList::Leaves(children) => children.pop(side).into(),
                };
                sizes.pop_child(side);
                spine.push(next_node);
                node_index = new_index;
            }

            // The only thing to be fixed here is the leaf spine node
            let leaf = Rc::make_mut(spine.last_mut().unwrap_or(&mut self.root).leaf_mut());
            let range = match side {
                Side::Back => node_index + 1..leaf.slots(),
                Side::Front => 0..node_index,
            };
            assert!(node_index < leaf.len());
            for _ in range {
                leaf.buffer.pop(side);
                self.len -= 1;
            }

            // Now we are done, we can reverse the spine here to get it back to normal
            spine.reverse();
        }
    }

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
    pub fn split_off(&mut self, at: usize) -> Vector<A> {
        // TODO: This is not really the most efficient way to do this, specialize this function.
        let mut result = self.clone();
        result.slice_to_end(at);
        self.slice_from_start(at);
        result
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
        F: Fn(&K) -> cmp::Ordering,
    {
        self.equal_range_in_range_by(f, ..)
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
    pub fn dual_sort_range_by<F, R, T>(&mut self, f: &F, range: R, secondary: &mut Vector<T>)
    where
        F: Fn(&A, &A) -> cmp::Ordering,
        R: RangeBounds<usize> + Clone,
        T: Clone + Debug,
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
    pub fn dual_sort_range_by_key<F, K, R, T>(&mut self, f: &F, range: R, secondary: &mut Vector<T>)
    where
        F: Fn(&A) -> K,
        K: Ord,
        R: RangeBounds<usize> + Clone,
        T: Debug + Clone,
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
    pub fn dual_sort_by<F, T>(&mut self, f: &F, secondary: &mut Vector<T>)
    where
        F: Fn(&A, &A) -> cmp::Ordering,
        T: Debug + Clone,
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

    /// Returns the height of the tree.
    pub(crate) fn height(&self) -> usize {
        debug_assert_eq!(self.left_spine.len(), self.right_spine.len());
        self.left_spine.len()
    }

    /// Returns a reference the spine of the requested side of the tree.
    fn spine_ref(&self, side: Side) -> &Vec<NodeRc<A>> {
        match side {
            Side::Front => &self.left_spine,
            Side::Back => &self.right_spine,
        }
    }

    /// Returns a mutable reference the spine of the requested side of the tree.
    fn spine_mut(&mut self, side: Side) -> &mut Vec<NodeRc<A>> {
        match side {
            Side::Front => &mut self.left_spine,
            Side::Back => &mut self.right_spine,
        }
    }

    /// Returns a reference to the leaf on the requested side of the tree.
    fn leaf_ref(&self, side: Side) -> &Rc<Leaf<A>> {
        self.spine_ref(side)
            .first()
            .unwrap_or(&self.root)
            .leaf_ref()
    }

    /// Returns a mutable reference to the leaf on the requested side of the tree.
    fn leaf_mut(&mut self, side: Side) -> &mut Rc<Leaf<A>> {
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
    pub fn focus(&self) -> Focus<A> {
        Focus::new(self)
    }

    /// Returns a mutable focus over the vector. A focus tracks the last leaf and positions which
    /// was read. The path down this tree is saved in the focus and is used to accelerate lookups in
    /// nearby locations.
    pub fn focus_mut(&mut self) -> FocusMut<A> {
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
        F: Fn(&mut FocusMut<A>),
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
    pub fn iter(&self) -> Iter<A> {
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
    pub fn iter_mut(&mut self) -> IterMut<A> {
        IterMut {
            front: 0,
            back: self.len(),
            focus: self.focus_mut(),
        }
    }

    /// Checks the internal invariants that are required by the Vector.
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
        for spine in &self.left_spine {
            spine.debug_check_invariants();
        }
        self.root.debug_check_invariants();
        for spine in &self.right_spine {
            spine.debug_check_invariants();
        }

        // Invariant 7
        // The tree's `len` field must match the sum of the spine's lens
        let left_spine_len = self.left_spine.iter().map(|x| x.len()).sum::<usize>();
        let root_len = self.root.len();
        let right_spine_len = self.right_spine.iter().map(|x| x.len()).sum::<usize>();

        assert_eq!(self.len, left_spine_len + root_len + right_spine_len);
        true
    }
}

impl<A: Clone + Debug + Ord> Vector<A> {
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
    pub fn dual_sort<T>(&mut self, secondary: &mut Vector<T>)
    where
        T: Debug + Clone,
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
    pub fn dual_sort_range<R, T>(&mut self, range: R, secondary: &mut Vector<T>)
    where
        R: RangeBounds<usize> + Clone,
        T: Debug + Clone,
    {
        self.dual_sort_range_by(&Ord::cmp, range, secondary)
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
    pub fn equal_range_in_range<R>(&self, value: &A, range: R) -> Result<Range<usize>, usize>
    where
        R: RangeBounds<usize>,
    {
        let f = |x: &A| x.cmp(value);
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
    pub fn equal_range(&self, value: &A) -> Result<Range<usize>, usize> {
        let f = |x: &A| x.cmp(value);
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
    pub fn between_range_in_range<R>(
        &self,
        start: Bound<&A>,
        end: Bound<&A>,
        range: R,
    ) -> Result<Range<usize>, usize>
    where
        R: RangeBounds<usize>,
    {
        let f = |x: &A| {
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
    pub fn between_range(&self, start: Bound<&A>, end: Bound<&A>) -> Result<Range<usize>, usize> {
        self.between_range_in_range(start, end, ..)
    }
}

impl<A: Clone + Debug + PartialEq> Vector<A> {
    /// Tests whether the node is equal to the given vector. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_vec(&self, v: &Vec<A>) -> bool {
        if self.len() == v.len() {
            let mut iter = v.iter();
            for spine in self.left_spine.iter() {
                if !spine.equal_iter(&mut iter) {
                    println!("Left: {:?} {:?}", self, v);
                    return false;
                }
            }
            if !self.root.equal_iter(&mut iter) {
                println!("Root: {:?} {:?}", self, v);
                return false;
            }
            for spine in self.right_spine.iter().rev() {
                if !spine.equal_iter(&mut iter) {
                    println!("Right: {:?} {:?}", self, v);
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Tests whether the node is equal to the given vector. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_vec_debug(&self, v: &Vec<A>) {
        if self.len() == v.len() {
            let mut iter = v.iter();
            println!("Left");
            for spine in self.left_spine.iter() {
                spine.equal_iter_debug(&mut iter);
            }
            println!("Root");
            self.root.equal_iter_debug(&mut iter);
            println!("Right");
            for spine in self.right_spine.iter().rev() {
                spine.equal_iter_debug(&mut iter);
            }
        } else {
            println!("incorrect lens {} {}", self.len(), v.len());
        }
    }
}

impl<A: Clone + Debug + Eq> PartialEq for Vector<A> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<A: Clone + Debug + Eq> Eq for Vector<A> {}

/*
trait SortFocus {
    fn split_at<S>(&mut self, index: usize, f: S)
    where
        for<'l, 'r> S: Fn(&'l mut Self, &'r mut Self);

    fn compare(
        first_focus: &mut Self,
        first_index: usize,
        second_focus: &mut Self,
        second_index: usize,
    ) -> cmp::Ordering;

    fn swap(
        first_focus: &mut Self,
        first_index: usize,
        second_focus: &mut Self,
        second_index: usize,
    );

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

struct SingleFocus<'a, 'b, 'f, A: Clone + Debug, F: Fn(&A, &A) -> cmp::Ordering> {
    first: &'b mut FocusMut<'a, A>,
    comp: &'f F,
}

impl<'a, 'b, 'f, A: Clone + Debug, F: Fn(&A, &A) -> cmp::Ordering> SortFocus
    for SingleFocus<'a, 'b, 'f, A, F>
{
    fn split_at<S>(&mut self, index: usize, f: S)
    where
        for<'l, 'r> S: Fn(&'l mut Self, &'r mut Self),
    {
        self.first.split_at_fn(index, |left, right| {
            let left = SingleFocus::<'a, 'b, 'f> {
                first: left,
                comp: self.comp,
            };
            let right = SingleFocus {
                first: right,
                comp: self.comp,
            };
            f(&mut left, &mut right);
        })
    }
    fn compare(
        first_focus: &mut Self,
        first_index: usize,
        second_focus: &mut Self,
        second_index: usize,
    ) -> cmp::Ordering {
        let first = first_focus.first.index(first_index);
        let second = second_focus.first.index(second_index);
        (first_focus.comp)(first, second)
    }
    fn swap(
        first_focus: &mut Self,
        first_index: usize,
        second_focus: &mut Self,
        second_index: usize,
    ) {
        mem::swap(
            first_focus.first.index(first_index),
            second_focus.first.index(second_index),
        );
    }
    fn len(&self) -> usize {
        self.first.len()
    }
}

struct DualFocus<'a, 'b, 'f, A: Clone + Debug, B: Clone + Debug, F: Fn(&A, &A) -> cmp::Ordering> {
    first: FocusMut<'a, A>,
    second: FocusMut<'b, B>,
    comp: &'f F,
}

impl<'a, 'b, 'f, A: Clone + Debug, B: Clone + Debug, F: Fn(&A, &A) -> cmp::Ordering> SortFocus
    for DualFocus<'a, 'b, 'f, A, B, F>
{
    fn split_at(&mut self, index: usize) -> Self {
        DualFocus {
            first: self.first.split_at(index),
            second: self.second.split_at(index),
            comp: self.comp,
        }
    }

    fn compare(
        first_focus: &mut Self,
        first_index: usize,
        second_focus: &mut Self,
        second_index: usize,
    ) -> cmp::Ordering {
        let first = first_focus.first.index(first_index);
        let second = second_focus.first.index(second_index);
        (first_focus.comp)(first, second)
    }

    fn swap(
        first_focus: &mut Self,
        first_index: usize,
        second_focus: &mut Self,
        second_index: usize,
    ) {
        mem::swap(
            first_focus.first.index(first_index),
            second_focus.first.index(second_index),
        );
        mem::swap(
            first_focus.second.index(first_index),
            second_focus.second.index(second_index),
        );
    }

    fn len(&self) -> usize {
        self.first.len()
    }
}

fn do_sort<A, R>(mut focus: A, rng: &mut R)
where
    A: SortFocus,
    R: RngCore,
{
    if focus.len() <= 1 {
        return;
    }

    // We know there are at least 2 elements here
    let pivot_index = rng.next_u64() as usize % focus.len();
    let mut rest = focus.split_at(1);
    let mut first = focus;

    if pivot_index > 0 {
        A::swap(&mut rest, pivot_index - 1, &mut first, 0);
    }
    // Pivot is now always in the first slice
    // let pivot_item = first.index(0);

    // Find the exact place to put the pivot or pivot-equal items
    let mut less_count = 0;
    let mut equal_count = 0;

    for index in 0..rest.len() {
        let comp = A::compare(&mut rest, index, &mut first, 0);
        match comp {
            cmp::Ordering::Less => less_count += 1,
            cmp::Ordering::Equal => equal_count += 1,
            cmp::Ordering::Greater => {}
        }
    }

    // If by accident we picked the minimum element as a pivot, we just call sort again with the
    // rest of the vector.
    if less_count == 0 {
        do_sort(rest, rng);
        return;
    }

    // We know here that there is at least one item before the pivot, so we move the minimum to the
    // beginning part of the vector. First, however we swap the pivot to the start of the equal
    // zone.
    less_count -= 1;
    equal_count += 1;
    A::swap(&mut first, 0, &mut rest, less_count);
    for index in 0..rest.len() {
        if index == less_count {
            // This is the position we swapped the pivot to. We can't move it from its position, and
            // we know its not the minimum.
            continue;
        }
        // let rest_item = rest.index(index);
        if A::compare(&mut rest, index, &mut first, 0) == cmp::Ordering::Less {
            A::swap(&mut first, 0, &mut rest, index);
        }
    }

    // Split the vector up into less_than, equal to and greater than parts.
    let mut greater_focus = rest.split_at(less_count + equal_count);
    let mut equal_focus = rest.split_at(less_count);
    let mut less_focus = rest;

    let mut less_position = 0;
    let mut equal_position = 0;
    let mut greater_position = 0;

    while less_position != less_focus.len() || greater_position != greater_focus.len() {
        // At start of this loop, equal_position always points to an equal item
        let mut equal_swap_side = None;

        // Advance the less_position until we find an out of place item
        while less_position != less_focus.len() {
            match A::compare(
                &mut less_focus,
                less_position,
                &mut equal_focus,
                equal_position,
            ) {
                cmp::Ordering::Equal => {
                    equal_swap_side = Some(cmp::Ordering::Less);
                    break;
                }
                cmp::Ordering::Greater => {
                    break;
                }
                _ => {}
            }
            less_position += 1;
        }

        // Advance the greater until we find an out of place item
        while greater_position != greater_focus.len() {
            match A::compare(
                &mut greater_focus,
                greater_position,
                &mut equal_focus,
                equal_position,
            ) {
                cmp::Ordering::Less => break,
                cmp::Ordering::Equal => {
                    equal_swap_side = Some(cmp::Ordering::Greater);
                    break;
                }
                _ => {}
            }
            greater_position += 1;
        }

        if let Some(swap_side) = equal_swap_side {
            // One of the sides is equal to the pivot, advance the pivot
            let (focus, position) = if swap_side == cmp::Ordering::Less {
                (&mut less_focus, less_position)
            } else {
                (&mut greater_focus, greater_position)
            };

            // We are guaranteed not to hit the end of the equal focus
            while A::compare(focus, position, &mut equal_focus, equal_position)
                == cmp::Ordering::Equal
            {
                equal_position += 1;
            }

            // Swap the equal position and the desired side, it's important to note that only the
            // equals focus is guaranteed to have made progress so we don't advance the side's index
            A::swap(focus, position, &mut equal_focus, equal_position);
        } else if less_position != less_focus.len() && greater_position != greater_focus.len() {
            // Both sides are out of place and not equal to the pivot, this can only happen if there
            // is a greater item in the lesser zone and a lesser item in the greater zone. The
            // solution is to swap both sides and advance both side's indices.
            debug_assert_ne!(
                A::compare(
                    &mut less_focus,
                    less_position,
                    &mut equal_focus,
                    equal_position
                ),
                cmp::Ordering::Equal
            );
            debug_assert_ne!(
                A::compare(
                    &mut greater_focus,
                    greater_position,
                    &mut equal_focus,
                    equal_position
                ),
                cmp::Ordering::Equal
            );
            A::swap(
                &mut less_focus,
                less_position,
                &mut greater_focus,
                greater_position,
            );
            less_position += 1;
            greater_position += 1;
        }
    }

    // Now we have partitioned both sides correctly, we just have to recurse now
    do_sort(less_focus, rng);
    if !greater_focus.is_empty() {
        do_sort(greater_focus, rng);
    }
}
*/
impl<A: Clone + Debug> FromIterator<A> for Vector<A> {
    fn from_iter<I: IntoIterator<Item = A>>(iter: I) -> Self {
        let mut result = Vector::new();
        for item in iter {
            result.push_back(item);
        }
        result
    }
}

/// An iterator for a Vector.
#[derive(Clone, Debug)]
pub struct Iter<'a, A: Clone + Debug> {
    front: usize,
    back: usize,
    focus: Focus<'a, A>,
}

impl<'a, A: Clone + Debug + 'a> Iterator for Iter<'a, A> {
    type Item = &'a A;

    fn next(&mut self) -> Option<&'a A> {
        if self.front != self.back {
            let focus: &'a mut Focus<A> = unsafe { &mut *(&mut self.focus as *mut _) };
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

impl<'a, A: 'a + Clone + Debug> IntoIterator for &'a Vector<A> {
    type Item = &'a A;
    type IntoIter = Iter<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, A: Clone + Debug + 'a> DoubleEndedIterator for Iter<'a, A> {
    fn next_back(&mut self) -> Option<&'a A> {
        if self.front != self.back {
            self.back -= 1;
            let focus: &'a mut Focus<A> = unsafe { &mut *(&mut self.focus as *mut _) };
            focus.get(self.back)
        } else {
            None
        }
    }
}

impl<'a, A: Clone + Debug + 'a> ExactSizeIterator for Iter<'a, A> {}

impl<'a, A: Clone + Debug + 'a> FusedIterator for Iter<'a, A> {}

/// An iterator for a Vector.
#[derive(Debug)]
pub struct IterMut<'a, A: Clone + Debug> {
    front: usize,
    back: usize,
    focus: FocusMut<'a, A>,
}

impl<'a, A: Clone + Debug + 'a> Iterator for IterMut<'a, A> {
    type Item = &'a mut A;

    fn next(&mut self) -> Option<&'a mut A> {
        if self.front != self.back {
            let focus: &'a mut FocusMut<A> = unsafe { &mut *(&mut self.focus as *mut _) };
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

impl<'a, A: 'a + Clone + Debug> IntoIterator for &'a mut Vector<A> {
    type Item = &'a mut A;
    type IntoIter = IterMut<'a, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'a, A: Clone + Debug + 'a> DoubleEndedIterator for IterMut<'a, A> {
    fn next_back(&mut self) -> Option<&'a mut A> {
        if self.front != self.back {
            self.back -= 1;
            let focus: &'a mut FocusMut<A> = unsafe { &mut *(&mut self.focus as *mut _) };
            focus.get(self.back)
        } else {
            None
        }
    }
}

impl<'a, A: Clone + Debug + 'a> ExactSizeIterator for IterMut<'a, A> {}

impl<'a, A: Clone + Debug + 'a> FusedIterator for IterMut<'a, A> {}

#[allow(clippy::cognitive_complexity)]
#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;
    use proptest::proptest;
    use proptest_derive::Arbitrary;

    const MAX_EXTEND_SIZE: usize = 2000;

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
                        "let v = vec!{:?}; for item in v.into_iter() {{ vector.push_front(item); }}\n",
                        items
                    ))?;
                }
                Action::ExtendBack(items) => {
                    fmt.write_str(&format!(
                        "let v = vec!{:?}; for item in v.into_iter() {{ vector.push_back(item); }}\n",
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
                        "let v = vec!{:?}; let mut new_vector = Vector::new(); for item in v.into_iter() {{ new_vector.push_front(item); }} new_vector.concatenate(vector); vector = new_vector; \n",
                        items
                    ))?;
                }
                Action::ConcatBack(items) => {
                    fmt.write_str(&format!(
                        "let v = vec!{:?}; let mut new_vector = Vector::new(); for item in v.into_iter() {{ new_vector.push_back(item); }} vector.concatenate(new_vector); \n",
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
            v.insert(v.len() / 2, i);
            v.assert_invariants();
        }
        let first_half = (1..N).step_by(2);
        let second_half = (0..N).step_by(2).rev();

        let mut vector = Vec::new();
        vector.extend(first_half);
        vector.extend(second_half);

        assert_eq!(v.iter().copied().collect::<Vec<usize>>(), vector);

        println!("{} {}", v.len(), v.height());
    }

    #[test]
    fn test_equal_range() {
        let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
        assert_eq!(v.equal_range_in_range(&1, 0..3), Ok(1..3));
    }
}
