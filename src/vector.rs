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
use crate::node_traits::{Entry, InternalTrait, LeafTrait, NodeRc};
use crate::sort::{do_dual_sort, do_single_sort};
use crate::{Side, RRB_WIDTH};
use rand_core::SeedableRng;
use std::borrow::Borrow;
use std::cmp;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::iter::{self, FromIterator, FusedIterator};
use std::mem;
use std::ops::{Bound, Range, RangeBounds};

use futures::stream::Stream;

/// Construct a vector.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate librrb;
/// # use librrb::Vector;
/// # use futures::stream::StreamExt;
///
/// #[tokio::main]
/// async fn main() {
/// let first = vector![1, 2, 3];
/// let mut second = Vector::new().await;
/// second.push_back(1).await;
/// second.push_back(2).await;
/// second.push_back(3).await;
/// assert_eq!(first.iter().await.collect::<Vec<u32>>().await, second.iter().await.collect::<Vec<u32>>().await);
/// }
/// ```
#[macro_export]
macro_rules! vector {
    () => { $crate::Vector::new().await };

    ( $($x:expr),* ) => {{
        let mut l = $crate::Vector::new().await;
        $(
            l.push_back($x).await;
        )*
            l
    }};

    ( $($x:expr ,)* ) => {{
        let mut l = $crate::Vector::new().await;
        $(
            l.push_back($x).await;
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
/// # use futures::stream::StreamExt;
///
/// #[tokio::main]
/// async fn main() {
/// let first = vector_ts![1, 2, 3];
/// let mut second = ThreadSafeVector::new().await;
/// second.push_back(1).await;
/// second.push_back(2).await;
/// second.push_back(3).await;
/// assert_eq!(first.iter().await.collect::<Vec<u32>>().await, second.iter().await.collect::<Vec<u32>>().await);
/// }
/// ```
#[macro_export]
macro_rules! vector_ts {
    () => { $crate::ThreadSafeVector::new().await };

    ( $($x:expr),* ) => {{
        let mut l = $crate::ThreadSafeVector::new().await;
        $(
            l.push_back($x).await;
        )*
            l
    }};

    ( $($x:expr ,)* ) => {{
        let mut l = $crate::ThreadSafeVector::new().await;
        $(
            l.push_back($x).await;
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
pub struct InternalVector<Internal>
where
    Internal: InternalTrait,
{
    pub(crate) left_spine: Vec<NodeRc<Internal>>,
    pub(crate) right_spine: Vec<NodeRc<Internal>>,
    pub(crate) root: NodeRc<Internal>,
    pub(crate) context: Internal::Context,
    len: usize,
}

impl<Internal> Clone for InternalVector<Internal>
where
    Internal: InternalTrait,
{
    fn clone(&self) -> Self {
        InternalVector {
            context: Internal::Context::default(),
            left_spine: self.left_spine.clone(),
            right_spine: self.right_spine.clone(),
            root: self.root.clone(),
            len: self.len,
        }
    }
}

impl<Internal> InternalVector<Internal>
where
    Internal: InternalTrait,
{
    /// Constructs a new empty vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v: Vector<u64> = Vector::new().await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn new() -> Self {
        InternalVector {
            context: Internal::Context::default(),
            left_spine: vec![],
            right_spine: vec![],
            root: NodeRc::Leaf(Internal::LeafEntry::new(Internal::Leaf::empty()).await),
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = Vector::singleton(1).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn singleton(item: <Internal::Leaf as LeafTrait>::Item) -> Self {
        let mut result = Self::new().await;
        result.push_back(item).await;
        result
    }

    /// Derp
    pub async fn constant_vec_of_length(
        item: <Internal::Leaf as LeafTrait>::Item,
        len: usize,
    ) -> Self {
        let mut store = InternalVector::new().await;
        let mut accumulator = InternalVector::singleton(item).await;
        while accumulator.len() <= len {
            if len & accumulator.len() != 0 {
                store.append(accumulator.clone()).await;
            }
            accumulator.append(accumulator.clone()).await;
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v: Vector<u64> = Vector::new().await;
    /// assert_eq!(v.len(), 0);
    /// assert_eq!(Vector::singleton(1).await.len(), 1);
    /// }
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v: Vector<u64> = Vector::new().await;
    /// assert!(v.is_empty());
    /// assert!(!Vector::singleton(1).await.is_empty());
    /// }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the spines and root. The leaf spines are at the end of the
    /// iterator, while the root is in the middle.
    pub(crate) fn spine_iter(
        &self,
    ) -> impl Iterator<Item = (Option<(Side, usize)>, &NodeRc<Internal>)> + DoubleEndedIterator
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
    async fn complete_leaf(&mut self, side: Side) {
        debug_assert_eq!(self.left_spine.len(), self.right_spine.len());
        debug_assert_eq!(
            self.leaf_ref(side).load(&self.context).await.free_space(),
            0
        );
        let (spine, other_spine) = match side {
            Side::Back => (&mut self.right_spine, &mut self.left_spine),
            Side::Front => (&mut self.left_spine, &mut self.right_spine),
        };

        for idx in 0..spine.len() {
            let node = &mut spine[idx];

            if node.free_slots(&self.context).await != 0 {
                // Nothing to do here
                break;
            }

            let full_node = mem::replace(node, node.new_empty(&self.context).await);
            let mut parent_node = spine
                .get_mut(idx + 1)
                .unwrap_or(&mut self.root)
                .internal_mut()
                .load_mut(&self.context)
                .await;
            parent_node.push_child(side, full_node, &self.context).await;
        }

        if self.root.slots(&self.context).await >= RRB_WIDTH - 1 {
            // This root is overfull so we have to raise the tree here, we add a new node of the
            // same height as the old root. We decant half the old root into this new node.
            // Finally, we create a new node of height one more than the old root and set that as
            // the new root. We leave the root empty
            let new_root = NodeRc::Internal(
                Internal::InternalEntry::new(Internal::empty_internal(
                    self.root.level(&self.context).await + 1,
                ))
                .await,
            );
            let mut new_node = mem::replace(&mut self.root, new_root);
            let mut other_new_node = new_node.new_empty(&self.context).await;
            new_node
                .share_children_with(
                    &mut other_new_node,
                    side.negate(),
                    RRB_WIDTH / 2,
                    &self.context,
                )
                .await;
            spine.push(new_node);
            other_spine.push(other_new_node);
        } else if self.root.slots(&self.context).await == 0 {
            // We have e have enough space in the root but we have balance the top of the spines
            self.fixup_spine_tops().await;
        }
    }

    /// Pushes an item into a side leaf of the tree. This fixes up some invariants in the case that
    /// the root sits directly above the leaves.
    async fn push_side(&mut self, side: Side, item: <Internal::Leaf as LeafTrait>::Item) {
        if self.leaf_ref(side).load(&self.context).await.free_space() == 0 {
            self.complete_leaf(side).await;
        }
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
        .load_mut(&self.context)
        .await
        .push(side, item, &self.context);
        self.len += 1;

        if self.spine_ref(side).len() == 1 {
            self.fixup_spine_tops().await;
        }
    }

    /// Appends a single item to the back of the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = Vector::new().await;
    /// v.push_back(1).await;
    /// assert_eq!(Vector::singleton(1).await.iter().await.collect::<Vec<u64>>().await, v.iter().await.collect::<Vec<u64>>().await);
    /// v.push_back(2).await;
    /// v.push_back(3).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1, 2, 3].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn push_back(&mut self, item: <Internal::Leaf as LeafTrait>::Item) {
        self.push_side(Side::Back, item).await;
    }

    /// Prepends a single item to the front of the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = Vector::new().await;
    /// v.push_back(1).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, Vector::singleton(1).await.iter().await.collect::<Vec<u64>>().await);
    /// v.push_front(2).await;
    /// v.push_front(3).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![3, 2, 1].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn push_front(&mut self, item: <Internal::Leaf as LeafTrait>::Item) {
        self.push_side(Side::Front, item).await;
    }

    /// Signals that a leaf is empty. This process the replace the leaf with the next leaf.
    async fn empty_leaf(&mut self, side: Side) {
        // Invariants
        // 1) If the root is empty, the top spines have at least SIZE - 1 children between them.
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
        debug_assert_eq!(spine.first().unwrap().slots(&self.context).await, 0);
        debug_assert!(
            self.root.slots(&self.context).await != 0
                || spine.last().unwrap().slots(&self.context).await != 0
        );

        let mut last_empty = spine.len() - 1;
        for (i, v) in spine.iter().enumerate().skip(1) {
            if v.slots(&self.context).await != 0 {
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
            let child = node
                .load_mut(&self.context)
                .await
                .pop_child(side, &self.context);
            spine[level] = child;
        }

        self.fixup_spine_tops().await;
    }

    /// Fixes up the top of the spines. Certain operations break some invariants that we use to keep
    /// the tree balanced. This repeatedly fixes up the tree to fulfill the invariants.
    async fn fixup_spine_tops(&mut self) {
        // The following invariant is fixed up here

        // Invariant 5
        // If the root is an empty non-leaf node then the last two nodes of both spines:
        // 1) Must not be able to be merged into a node of 1 less height
        // 2) Must differ in slots by at most one node

        // The invariant must be checked in a loop as slicing may break the invariant multiple times
        while self.root.slots(&self.context).await == 0 && !self.root.is_leaf(&self.context).await {
            let left_spine_top = self.left_spine.last_mut().unwrap();
            let right_spine_top = self.right_spine.last_mut().unwrap();
            let left_spine_children = left_spine_top.slots(&self.context).await;
            let right_spine_children = right_spine_top.slots(&self.context).await;

            let total_children = left_spine_children + right_spine_children;
            let difference = if left_spine_children > right_spine_children {
                left_spine_children - right_spine_children
            } else {
                right_spine_children - left_spine_children
            };
            let min_children = if self.root.level(&self.context).await == 1 {
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
                left_spine_top
                    .share_children_with(&mut right_spine_top, Side::Back, RRB_WIDTH, &self.context)
                    .await;
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
                source
                    .share_children_with(destination, side, difference / 2, &self.context)
                    .await;
                break;
            } else {
                // No invariant is broken. We can stop checking here
                break;
            }
        }
    }

    /// Pops an item from a side leaf of the tree. This fixes up some invariants in the case that
    /// the root sits directly above the leaves.
    async fn pop_side(&mut self, side: Side) -> Option<<Internal::Leaf as LeafTrait>::Item> {
        debug_assert_eq!(self.left_spine.len(), self.right_spine.len());
        if self.spine_ref(side).is_empty() {
            if !self.root.is_empty(&self.context).await {
                self.len -= 1;
                Some(
                    self.root
                        .leaf_mut()
                        .load_mut(&self.context)
                        .await
                        .pop(side, &self.context),
                )
            } else {
                None
            }
        } else {
            // Can never be none as the is of height at least 1
            let leaf = match side {
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
            };
            let item = leaf.load_mut(&self.context).await.pop(side, &self.context);

            if leaf.load(&self.context).await.is_empty() {
                self.empty_leaf(side).await;
            } else if self.spine_ref(side).len() == 1 {
                self.fixup_spine_tops().await;
            }

            self.len -= 1;
            Some(item)
        }
    }

    /// Removes and returns a single item from the front of the sequence. If the tree is empty this
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v: Vector<u64> = vector![1, 2, 3];
    /// assert_eq!(v.pop_front().await, Some(1));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![2, 3].iter().await.collect::<Vec<u64>>().await);
    /// assert_eq!(v.pop_front().await, Some(2));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![3].iter().await.collect::<Vec<u64>>().await);
    /// assert_eq!(v.pop_front().await, Some(3));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![].iter().await.collect::<Vec<u64>>().await);
    /// assert_eq!(v.pop_front().await, None);
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn pop_front(&mut self) -> Option<<Internal::Leaf as LeafTrait>::Item> {
        self.pop_side(Side::Front).await
    }

    /// Removes and returns a single item from the back of the sequence. If the tree is empty this
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v: Vector<u64> = vector![1, 2, 3];
    /// assert_eq!(v.pop_back().await, Some(3));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1, 2].iter().await.collect::<Vec<u64>>().await);
    /// assert_eq!(v.pop_back().await, Some(2));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1].iter().await.collect::<Vec<u64>>().await);
    /// assert_eq!(v.pop_back().await, Some(1));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![].iter().await.collect::<Vec<u64>>().await);
    /// assert_eq!(v.pop_back().await, None);
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn pop_back(&mut self) -> Option<<Internal::Leaf as LeafTrait>::Item> {
        self.pop_side(Side::Back).await
    }

    /// Returns a reference to the item at the front of the sequence. If the tree is empty this
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// assert_eq!(v.front().await, Some(&1));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1, 2, 3].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn front(&self) -> Option<&<Internal::Leaf as LeafTrait>::Item> {
        let leaf = self.left_spine.first().unwrap_or(&self.root);
        unsafe { Some(&*leaf.leaf_ref().load(&self.context).await.front()?) }
    }

    /// Returns a mutable reference to the item at the front of the sequence. If the tree is empty
    /// this returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// assert_eq!(v.front_mut().await, Some(&mut 1));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1, 2, 3].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn front_mut(&mut self) -> Option<&mut <Internal::Leaf as LeafTrait>::Item> {
        let leaf = self.left_spine.first_mut().unwrap_or(&mut self.root);
        unsafe {
            Some(
                &mut *leaf
                    .leaf_mut()
                    .load_mut(&self.context)
                    .await
                    .front_mut(&self.context)?,
            )
        }
    }

    /// Returns a reference to the item at the back of the sequence. If the tree is empty this
    /// returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// assert_eq!(v.back().await, Some(&3));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1, 2, 3].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn back(&self) -> Option<&<Internal::Leaf as LeafTrait>::Item> {
        let leaf = self.right_spine.first().unwrap_or(&self.root);
        unsafe { Some(&*leaf.leaf_ref().load(&self.context).await.back()?) }
    }

    /// Returns a mutable reference to the item at the back of the sequence. If the tree is empty
    /// this returns `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// assert_eq!(v.back_mut().await, Some(&mut 3));
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1, 2, 3].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn back_mut(&mut self) -> Option<&mut <Internal::Leaf as LeafTrait>::Item> {
        let leaf = self.right_spine.first_mut().unwrap_or(&mut self.root);
        unsafe {
            Some(
                &mut *leaf
                    .leaf_mut()
                    .load_mut(&self.context)
                    .await
                    .back_mut(&self.context)?,
            )
        }
    }

    /// Derp
    pub async fn get(&self, idx: usize) -> Option<&<Internal::Leaf as LeafTrait>::Item> {
        if let Some((spine_info, subindex)) = self.find_node_info_for_index(idx).await {
            let node = match spine_info {
                Some((Side::Front, spine_idx)) => &self.left_spine[spine_idx],
                Some((Side::Back, spine_idx)) => &self.right_spine[spine_idx],
                None => &self.root,
            };
            unsafe { Some(&*node.get(subindex, &self.context).await.unwrap()) }
        } else {
            None
        }
    }

    /// Derp
    pub async fn index(&self, idx: usize) -> &<Internal::Leaf as LeafTrait>::Item {
        self.get(idx).await.expect("Index out of bounds.")
    }

    /// Derp
    pub async fn get_mut(
        &mut self,
        idx: usize,
    ) -> Option<&mut <Internal::Leaf as LeafTrait>::Item> {
        if let Some((spine_info, subindex)) = self.find_node_info_for_index(idx).await {
            let node = match spine_info {
                Some((Side::Front, spine_idx)) => &mut self.left_spine[spine_idx],
                Some((Side::Back, spine_idx)) => &mut self.right_spine[spine_idx],
                None => &mut self.root,
            };
            unsafe { Some(&mut *node.get_mut(subindex, &self.context).await.unwrap()) }
        } else {
            None
        }
    }

    pub async fn get_mut_guarded(&mut self, idx: usize) -> Option<MutBoundGuard<'_, Internal>> {
        if let Some((spine_info, subindex)) = self.find_node_info_for_index(idx).await {
            let node = match spine_info {
                Some((Side::Front, spine_idx)) => &mut self.left_spine[spine_idx],
                Some((Side::Back, spine_idx)) => &mut self.right_spine[spine_idx],
                None => &mut self.root,
            };
            Some(MutBoundGuard {
                guard: node.get_mut_guarded(subindex, &self.context).await?,
                vector: self,
            })
        } else {
            None
        }
    }

    /// Derp
    pub async fn index_mut(&mut self, idx: usize) -> &mut <Internal::Leaf as LeafTrait>::Item {
        self.get_mut(idx).await.expect("Index out of bounds.")
    }

    /// Appends the given vector onto the back of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// v.append(vector![4, 5, 6]).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1, 2, 3, 4, 5, 6].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn append(&mut self, mut other: Self) {
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
            other.push_front(self.pop_back().await.unwrap()).await;
            *self = other;
            return;
        }
        if other.len() == 1 {
            self.push_back(other.pop_back().await.unwrap()).await;
            return;
        }

        let new_len = self.len + other.len();

        // Make the spines the same length
        while self.right_spine.len() < other.left_spine.len() {
            // The root moves to either the left or right spine and the root becomes empty
            // We replace the left with root here and right with an empty node
            let new_root = NodeRc::Internal(
                Internal::InternalEntry::new(Internal::empty_internal(
                    self.root.level(&self.context).await + 1,
                ))
                .await,
            );
            let mut new_left = mem::replace(&mut self.root, new_root);
            let mut new_right = new_left.new_empty(&self.context).await;
            new_left
                .share_children_with(
                    &mut new_right,
                    Side::Back,
                    new_left.slots(&self.context).await / 2,
                    &self.context,
                )
                .await;
            self.left_spine.push(new_left);
            self.right_spine.push(new_right);
        }

        while other.left_spine.len() < self.right_spine.len() {
            // The root moves to either the left or right spine and the root becomes empty
            // We replace the right with root here and left with an empty node
            let new_root = NodeRc::Internal(
                Internal::InternalEntry::new(Internal::empty_internal(
                    other.root.level(&self.context).await + 1,
                ))
                .await,
            );
            let mut new_right = mem::replace(&mut other.root, new_root);
            let mut new_left = new_right.new_empty(&self.context).await;
            new_right
                .share_children_with(
                    &mut new_left,
                    Side::Front,
                    new_right.slots(&self.context).await / 2,
                    &self.context,
                )
                .await;
            other.left_spine.push(new_left);
            other.right_spine.push(new_right);
        }

        // debug_assert_eq!(self.right_spine.len(), self.right_spine.len());
        debug_assert_eq!(other.left_spine.len(), other.right_spine.len());
        debug_assert_eq!(self.right_spine.len(), other.left_spine.len());

        // More efficient to work from front to back here, but we need to remove elements
        // We reverse to make this more efficient
        self.right_spine.reverse();
        other.left_spine.reverse();
        if let Some(left_child) = self.right_spine.pop() {
            let mut parent_node = self
                .right_spine
                .last_mut()
                .unwrap_or(&mut self.root)
                .internal_mut()
                .load_mut(&self.context)
                .await;
            if !left_child.is_empty(&self.context).await {
                parent_node
                    .push_child(Side::Back, left_child, &self.context)
                    .await;
            }
        }
        if let Some(right_child) = other.left_spine.pop() {
            let mut parent_node = other
                .left_spine
                .last_mut()
                .unwrap_or(&mut other.root)
                .internal_mut()
                .load_mut(&self.context)
                .await;
            if !right_child.is_empty(&self.context).await {
                parent_node
                    .push_child(Side::Front, right_child, &self.context)
                    .await;
            }
        }
        while !self.right_spine.is_empty() {
            let mut left_node = self.right_spine.pop().unwrap();
            let mut right_node = other.left_spine.pop().unwrap();

            let mut left = left_node.internal_mut().load_mut(&self.context).await;
            let mut right = right_node.internal_mut().load_mut(&self.context).await;

            left.pack_children(&self.context).await;
            let mut left_right_most = left.pop_child(Side::Back, &self.context);
            while !left_right_most.is_full(&self.context).await && !right.is_empty() {
                let mut right_left_most = right.pop_child(Side::Front, &self.context);
                right_left_most
                    .share_children_with(
                        &mut left_right_most,
                        Side::Front,
                        right_left_most.slots(&self.context).await,
                        &self.context,
                    )
                    .await;
                if !right_left_most.is_empty(&self.context).await {
                    right
                        .push_child(Side::Front, right_left_most, &self.context)
                        .await;
                }
            }
            left.push_child(Side::Back, left_right_most, &self.context)
                .await;
            right.pack_children(&self.context).await;
            let slots = right.slots();
            right.share_children_with(&mut *left, Side::Front, slots, &self.context);

            if !left_node.is_empty(&self.context).await {
                self.right_spine
                    .last_mut()
                    .unwrap_or(&mut self.root)
                    .internal_mut()
                    .load_mut(&self.context)
                    .await
                    .push_child(Side::Back, left_node, &self.context)
                    .await;
            }
            if !right.is_empty() {
                other
                    .left_spine
                    .last_mut()
                    .unwrap_or(&mut other.root)
                    .internal_mut()
                    .load_mut(&self.context)
                    .await
                    .push_child(Side::Front, right_node, &self.context)
                    .await;
            }
        }

        debug_assert!(self.right_spine.is_empty());
        debug_assert!(other.left_spine.is_empty());
        self.right_spine = other.right_spine;

        other
            .root
            .share_children_with(&mut self.root, Side::Front, RRB_WIDTH, &self.context)
            .await;

        if self.root.free_slots(&self.context).await < 2 {
            self.root
                .share_children_with(&mut other.root, Side::Back, 1, &self.context)
                .await;
        }

        if !other.root.is_empty(&self.context).await {
            let new_root = NodeRc::Internal(
                Internal::InternalEntry::new(Internal::empty_internal(
                    self.root.level(&self.context).await + 1,
                ))
                .await,
            );
            let old_root = mem::replace(&mut self.root, new_root);
            self.left_spine.push(old_root);
            self.right_spine.push(other.root);
        }
        self.len = new_len;
        self.fixup_spine_tops().await;
    }

    /// Prepends the given vector onto the front of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// v.prepend(vector![4, 5, 6]).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![4, 5, 6, 1, 2, 3].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn prepend(&mut self, other: Self) {
        let other = mem::replace(self, other);
        self.append(other).await
    }

    /// Slices the vector from the start to the given index exclusive.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// v.slice_from_start(2).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1, 2].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn slice_from_start(&mut self, len: usize) {
        // We find a path from the root down to the leaf in question
        // 1) If the leaf has an ancestor in the left spine, we pop (from the root) the until the
        // lowest ancestor in the spine, the right spine becomes the path from the left spine to
        // this leaf
        // 2) If the leaf has an ancestor on the right spine, we pop from the leaf up to this
        // node, but we add back the spine by replacing it with path to this leaf
        // 3) If the leaf only has the root as an ancestor it proceeds as 2) with the path becoming
        // the entire right spine.
        self.split_off(len).await;
    }

    /// Slices the vector from the given index inclusive to the end.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// v.slice_to_end(2).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![3].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn slice_to_end(&mut self, start: usize) {
        // This proceeds the same as above, except that we instead recompute the left spine instead
        // of the right spine.
        // 1) If the leaf has an ancestor on the left spine, we pop from the leaf up to this
        // node, but we add back the spine by replacing it with path to this leaf
        // 2) If the leaf has an ancestor in the right spine, we pop (from the root) the until the
        // lowest ancestor in the spine, the left spine becomes the path from the right spine to
        // this leaf
        // 3) If the leaf only has the root as an ancestor it proceeds as 2) with the path becoming
        // the entire left spine.
        let result = self.split_off(start).await;
        *self = result;
    }

    /// Derp
    pub async fn extract_slice<R: RangeBounds<usize> + Debug>(&mut self, range: R) -> Self {
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

        let last_bit = self.split_off(range_end).await;
        let middle_bit = self.split_off(range_start).await;
        self.append(last_bit).await;
        middle_bit
    }

    /// Returns the spine position and subindex corresponding the given index.
    async fn find_node_info_for_index(
        &self,
        index: usize,
    ) -> Option<(Option<(Side, usize)>, usize)> {
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
                if index < forward_end + left.len(&self.context).await {
                    return Some((Some((Side::Front, idx)), index - forward_end));
                }
                forward_end += left.len(&self.context).await;
                backward_start -= right.len(&self.context).await;
                if index >= backward_start {
                    return Some((Some((Side::Back, idx)), index - backward_start));
                }
            }
            debug_assert!(index >= forward_end && index < backward_start);
            Some((None, index - forward_end))
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// let last_half = v.split_off(1).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1].iter().await.collect::<Vec<u64>>().await);
    /// assert_eq!(last_half.iter().await.collect::<Vec<u64>>().await, vector![2, 3].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn split_off(&mut self, at: usize) -> InternalVector<Internal> {
        if at == 0 {
            // We can early out because the result is self and self becomes empty.
            mem::replace(self, InternalVector::new().await)
        } else if at >= self.len() {
            // We can early out because the result is empty and self remains unchanged.
            InternalVector::new().await
        } else {
            // We know now that the split position lies within the vector allowing us to make some
            // simplifications.
            let original_len = self.len();
            let (position, subposition) = self.find_node_info_for_index(at).await.unwrap();
            let mut result = InternalVector::new().await;
            match position {
                Some((Side::Front, node_position)) => {
                    // The left spine is has the node getting split
                    result.left_spine = self.left_spine.split_off(node_position + 1);
                    let mut split_node = self.left_spine.pop().unwrap();
                    result.left_spine.insert(
                        0,
                        split_node
                            .split_at_position(subposition, &self.context)
                            .await,
                    );
                    mem::swap(&mut self.root, &mut result.root);
                    mem::swap(&mut self.right_spine, &mut result.right_spine);
                    self.root = split_node;
                }
                None => {
                    // The root node is getting split
                    mem::swap(&mut self.right_spine, &mut result.right_spine);
                    result.root = self
                        .root
                        .split_at_position(subposition, &self.context)
                        .await;
                }
                Some((Side::Back, node_position)) => {
                    // The right spine is getting split
                    result.right_spine = self.right_spine.split_off(node_position + 1);
                    let mut split_node = self.right_spine.pop().unwrap();
                    mem::swap(&mut result.right_spine, &mut self.right_spine);
                    let split_right = split_node
                        .split_at_position(subposition, &self.context)
                        .await;
                    self.right_spine.insert(0, split_node);
                    result.right_spine.push(split_right);
                }
            }

            result.len = original_len - at;
            self.len = at;

            self.right_spine.reverse();
            while let Some(x) = self.right_spine.last() {
                if x.is_empty(&self.context).await {
                    self.right_spine.pop().unwrap();
                } else {
                    break;
                }
            }
            // while self
            //     .right_spine
            //     .last()
            //     .map(async move |x| x.is_empty(&self.context).await)
            //     .unwrap_or(false)
            // {
            //     self.right_spine.pop().unwrap();
            // }
            self.right_spine.reverse();

            while self.root.is_empty(&self.context).await
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
            }

            if self.fill_spine(Side::Back).await {
                self.fixup_spine_tops().await;
                self.empty_leaf(Side::Back).await;
            }
            self.fixup_spine_tops().await;

            result.left_spine.reverse();
            // while result
            //     .left_spine
            //     .last()
            //     .map(|x| x.is_empty(&self.context))
            //     .unwrap_or(false)
            // {
            //     result.left_spine.pop().unwrap();
            // }
            while let Some(x) = result.left_spine.last() {
                if x.is_empty(&self.context).await {
                    result.left_spine.pop().unwrap();
                } else {
                    break;
                }
            }
            result.left_spine.reverse();

            while result.root.is_empty(&self.context).await
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
            }

            if result.fill_spine(Side::Front).await {
                result.fixup_spine_tops().await;
                result.empty_leaf(Side::Front).await;
            }
            result.fixup_spine_tops().await;
            result
        }
    }

    async fn fill_spine(&mut self, side: Side) -> bool {
        let spine = match side {
            Side::Front => &mut self.left_spine,
            Side::Back => &mut self.right_spine,
        };
        spine.reverse();
        let result = loop {
            match spine.last_mut().unwrap_or(&mut self.root) {
                NodeRc::Internal(internal) => {
                    let child = internal
                        .load_mut(&self.context)
                        .await
                        .pop_child(side, &self.context);
                    spine.push(child);
                }
                NodeRc::Leaf(leaf) => {
                    break leaf.load(&self.context).await.is_empty();
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// v.insert(1, 4).await;
    /// assert_eq!(v.iter().await.collect::<Vec<u64>>().await, vector![1, 4, 2, 3].iter().await.collect::<Vec<u64>>().await);
    /// }
    /// ```
    pub async fn insert(&mut self, index: usize, element: <Internal::Leaf as LeafTrait>::Item) {
        // TODO: This is not really the most efficient way to do this, specialize this function.
        let last_part = self.split_off(index).await;
        self.push_back(element).await;
        self.append(last_part).await;
    }

    /// Derp
    pub async fn reverse_range<R: RangeBounds<usize>>(&mut self, range: R) {
        let range = self.arbitrary_range_to_range(range);
        let mut focus = self.focus_mut().await;
        let (mut focus, _) = focus.split_at(range.end).await;
        let (_, mut focus) = focus.split_at(range.start).await;

        let half_len = focus.len() / 2;
        let (mut left, mut right) = focus.split_at(half_len).await;

        let right_len = right.len();
        for i in 0..half_len {
            let left = left.index(i).await;
            let right = right.index(right_len - 1 - i).await;
            mem::swap(left, right);
        }
    }

    /// Derp
    pub async fn reverse(&mut self) {
        self.reverse_range(..).await
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v: Vector<i32> = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// let f = |x: &i32| {
    ///     if *x < 0 {
    ///         Ordering::Less
    ///     } else if *x > 5 {
    ///         Ordering::Greater
    ///     } else {
    ///         Ordering::Equal
    ///     }
    /// };
    /// assert_eq!(v.equal_range_for_index_in_range_by(1, &f, 1..).await, 1..6);
    /// }
    /// ```
    pub async fn equal_range_for_index_in_range_by<K, F, R>(
        &self,
        index: usize,
        f: &F,
        range: R,
    ) -> Range<usize>
    where
        <Internal::Leaf as LeafTrait>::Item: Borrow<K>,
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

        let mut focus = self.focus().await;
        if f(focus.index(index).await.borrow()) != cmp::Ordering::Equal {
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
                if start + 1 == equals_start {
                    let comparison = f(focus.index(start).await.borrow());
                    if comparison == cmp::Ordering::Equal {
                        equals_start = start;
                    }
                    break;
                } else {
                    let mid = avg(start, equals_start);
                    let comparison = f(focus.index(mid).await.borrow());
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
            if equals_end_inclusive + 1 == end {
                break;
            } else {
                let mid = avg(equals_end_inclusive, end);
                let comparison = f(focus.index(mid).await.borrow());
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v: Vector<i32> = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// let f = |x: &i32| {
    ///     if *x < 0 {
    ///         Ordering::Less
    ///     } else if *x > 5 {
    ///         Ordering::Greater
    ///     } else {
    ///         Ordering::Equal
    ///     }
    /// };
    /// assert_eq!(v.equal_range_for_index_by(1, &f).await, 0..6);
    /// }
    /// ```
    pub async fn equal_range_for_index_by<K, F>(&self, index: usize, f: &F) -> Range<usize>
    where
        <Internal::Leaf as LeafTrait>::Item: Borrow<K>,
        K: ?Sized,
        F: Fn(&K) -> cmp::Ordering,
    {
        self.equal_range_for_index_in_range_by(index, f, ..).await
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v: Vector<i32> = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// let f = |x: &i32| {
    ///     if *x < 0 {
    ///         Ordering::Less
    ///     } else if *x > 5 {
    ///         Ordering::Greater
    ///     } else {
    ///         Ordering::Equal
    ///     }
    /// };
    /// assert_eq!(v.equal_range_in_range_by(&f, 1..).await, Ok(1..6));
    /// }
    /// ```
    pub async fn equal_range_in_range_by<K, F, R>(
        &self,
        f: &F,
        range: R,
    ) -> Result<Range<usize>, usize>
    where
        <Internal::Leaf as LeafTrait>::Item: Borrow<K>,
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
        let mut focus = self.focus().await;

        // Find an equal item, we maintain an invariant that [start, end) should be the only
        // possible equal elements. We binary search down to narrow the range. We stop if we find
        // an equal item for the next step.
        loop {
            if start + 1 == end {
                let comparison = f(focus.index(start).await.borrow());
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
                let comparison = f(focus.index(mid).await.borrow());
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
                        return Ok(self
                            .equal_range_for_index_in_range_by(mid, f, start..end)
                            .await);
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
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
    /// assert_eq!(v.equal_range_by(&f).await, Ok(0..6));
    /// }
    /// ```
    pub async fn equal_range_by<K, F>(&self, f: &F) -> Result<Range<usize>, usize>
    where
        <Internal::Leaf as LeafTrait>::Item: Borrow<K>,
        K: ?Sized,
        F: Fn(&K) -> cmp::Ordering,
    {
        self.equal_range_in_range_by(f, ..).await
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// assert_eq!(v.equal_range_in_range(&1, 0..3).await, Ok(1..3));
    /// }
    /// ```
    pub async fn equal_range_in_range<K, R>(
        &self,
        value: &K,
        range: R,
    ) -> Result<Range<usize>, usize>
    where
        <Internal::Leaf as LeafTrait>::Item: Borrow<K>,
        K: Ord + ?Sized,
        R: RangeBounds<usize>,
    {
        let f = |x: &K| x.cmp(value);
        self.equal_range_in_range_by(&f, range).await
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// assert_eq!(v.equal_range(&1).await, Ok(1..3));
    /// }
    /// ```
    pub async fn equal_range<K>(&self, value: &K) -> Result<Range<usize>, usize>
    where
        <Internal::Leaf as LeafTrait>::Item: Borrow<K>,
        K: Ord + ?Sized,
    {
        let f = |x: &K| x.cmp(value);
        self.equal_range_in_range_by(&f, ..).await
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// assert_eq!(v.between_range_in_range(Bound::Included(&1), Bound::Unbounded, 0..3).await, Ok(1..3));
    /// }
    /// ```
    pub async fn between_range_in_range<K, R>(
        &self,
        start: Bound<&K>,
        end: Bound<&K>,
        range: R,
    ) -> Result<Range<usize>, usize>
    where
        <Internal::Leaf as LeafTrait>::Item: Borrow<K>,
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
        self.equal_range_in_range_by(&f, range).await
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
    /// assert_eq!(v.between_range(Bound::Included(&3), Bound::Unbounded).await, Ok(4..9));
    /// }
    /// ```
    pub async fn between_range<K>(
        &self,
        start: Bound<&K>,
        end: Bound<&K>,
    ) -> Result<Range<usize>, usize>
    where
        <Internal::Leaf as LeafTrait>::Item: Borrow<K>,
        K: Ord + ?Sized,
    {
        self.between_range_in_range(start, end, ..).await
    }

    /// Sorts a range of the sequence by the given comparator.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// v.sort_range_by(&|x: &i32, y: &i32 | (-x).cmp(&(-y)), ..).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0].iter().await.collect::<Vec<i32>>().await);
    /// v.sort_range_by(&Ord::cmp, 2..).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 0, 1, 2, 3, 4, 5, 6, 7].iter().await.collect::<Vec<i32>>().await);
    /// }
    /// ```
    pub async fn sort_range_by<F, R>(&mut self, f: &F, range: R)
    where
        F: Fn(
            &<Internal::Leaf as LeafTrait>::Item,
            &<Internal::Leaf as LeafTrait>::Item,
        ) -> cmp::Ordering,
        R: RangeBounds<usize>,
    {
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
        let mut focus = self.focus_mut().await;
        let (mut focus, _) = focus.split_at(range_end).await;
        let (_, mut focus) = focus.split_at(range_start).await;
        let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
        do_single_sort(&mut focus, &mut rng, f).await;
        // });
    }

    /// Sorts a range of the sequence by the given comparator. Any swap that occurs will be made in
    /// the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort_range_by(&|x: &i32, y: &i32 | (-x).cmp(&(-y)), .., &mut secondary).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0].iter().await.collect::<Vec<i32>>().await);
    /// assert_eq!(secondary.iter().await.collect::<Vec<char>>().await, vector!['j', 'i','h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'].iter().await.collect::<Vec<char>>().await);
    /// v.dual_sort_range_by(&Ord::cmp, 2.., &mut secondary).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 0, 1, 2, 3, 4, 5, 6, 7].iter().await.collect::<Vec<i32>>().await);
    /// assert_eq!(secondary.iter().await.collect::<Vec<char>>().await, vector!['j', 'i', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'].iter().await.collect::<Vec<char>>().await);
    /// }
    /// ```
    pub async fn dual_sort_range_by<F, R, Internal2>(
        &mut self,
        f: &F,
        range: R,
        secondary: &mut InternalVector<Internal2>,
    ) where
        F: Fn(
            &<Internal::Leaf as LeafTrait>::Item,
            &<Internal::Leaf as LeafTrait>::Item,
        ) -> cmp::Ordering,
        R: RangeBounds<usize> + Clone,
        Internal2: InternalTrait,
    {
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
        let mut focus = self.focus_mut().await;
        let (mut focus, _) = focus.split_at(range_end).await;
        let (_, mut focus) = focus.split_at(range_start).await;
        let mut dual = secondary.focus_mut().await;
        let (mut dual, _) = dual.split_at(range_end).await;
        let (_, mut dual) = dual.split_at(range_start).await;
        let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
        do_dual_sort(&mut focus, &mut dual, &mut rng, &f).await;
        // });
        // });
    }

    /// Sorts a range of the sequence by the given comparator.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// v.sort_range_by_key(&|&x| -x, ..).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0].iter().await.collect::<Vec<i32>>().await);
    /// v.sort_range_by_key(&|&x| x, 2..).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 0, 1, 2, 3, 4, 5, 6, 7].iter().await.collect::<Vec<i32>>().await);
    /// }
    /// ```
    pub async fn sort_range_by_key<
        F: Fn(&<Internal::Leaf as LeafTrait>::Item) -> K,
        K: Ord,
        R: RangeBounds<usize>,
    >(
        &mut self,
        f: &F,
        range: R,
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
        let mut focus = self.focus_mut().await;
        let (mut focus, _) = focus.split_at(range_end).await;
        let (_, mut focus) = focus.split_at(range_start).await;
        let comp = |x: &<Internal::Leaf as LeafTrait>::Item,
                    y: &<Internal::Leaf as LeafTrait>::Item| f(x).cmp(&f(y));
        let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
        do_single_sort(&mut focus, &mut rng, &comp).await;
        // });
    }

    /// Sorts a range of the sequence by the given comparator. Any swap that occurs will be made in
    /// the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort_range_by_key(&|&x| -x, .., &mut secondary).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0].iter().await.collect::<Vec<i32>>().await);
    /// assert_eq!(secondary.iter().await.collect::<Vec<char>>().await, vector!['j', 'i','h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'].iter().await.collect::<Vec<char>>().await);
    /// v.dual_sort_range_by_key(&|&x| x, 2.., &mut secondary).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 0, 1, 2, 3, 4, 5, 6, 7].iter().await.collect::<Vec<i32>>().await);
    /// assert_eq!(secondary.iter().await.collect::<Vec<char>>().await, vector!['j', 'i', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'].iter().await.collect::<Vec<char>>().await);
    /// }
    /// ```
    pub async fn dual_sort_range_by_key<F, K, R, Internal2>(
        &mut self,
        f: &F,
        range: R,
        secondary: &mut InternalVector<Internal2>,
    ) where
        F: Fn(&<Internal::Leaf as LeafTrait>::Item) -> K,
        K: Ord,
        R: RangeBounds<usize> + Clone,
        Internal2: InternalTrait,
    {
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
        let comp = |x: &<Internal::Leaf as LeafTrait>::Item,
                    y: &<Internal::Leaf as LeafTrait>::Item| f(x).cmp(&f(y));
        let mut focus = self.focus_mut().await;
        let (mut focus, _) = focus.split_at(range_end).await;
        let (_, mut focus) = focus.split_at(range_start).await;
        let mut dual = secondary.focus_mut().await;
        let (mut dual, _) = dual.split_at(range_end).await;
        let (_, mut dual) = dual.split_at(range_start).await;
        let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(0);
        do_dual_sort(&mut focus, &mut dual, &mut rng, &comp).await;
        // });
        // });
    }

    /// Sorts the entire sequence by the given comparator.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// v.sort_by(&|x: &i32, y: &i32 | (-x).cmp(&(-y))).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0].iter().await.collect::<Vec<i32>>().await);
    /// v.sort_by(&Ord::cmp).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].iter().await.collect::<Vec<i32>>().await);
    /// }
    /// ```
    pub async fn sort_by<
        F: Fn(
            &<Internal::Leaf as LeafTrait>::Item,
            &<Internal::Leaf as LeafTrait>::Item,
        ) -> cmp::Ordering,
    >(
        &mut self,
        f: &F,
    ) {
        self.sort_range_by(f, ..).await;
    }

    /// Sorts the entire sequence by the given comparator. Any swap that occurs will be made in
    /// the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort_by(&|x: &i32, y: &i32 | (-x).cmp(&(-y)), &mut secondary).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0].iter().await.collect::<Vec<i32>>().await);
    /// v.dual_sort_by(&Ord::cmp, &mut secondary).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].iter().await.collect::<Vec<i32>>().await);
    /// }
    /// ```
    pub async fn dual_sort_by<F, Internal2>(
        &mut self,
        f: &F,
        secondary: &mut InternalVector<Internal2>,
    ) where
        F: Fn(
            &<Internal::Leaf as LeafTrait>::Item,
            &<Internal::Leaf as LeafTrait>::Item,
        ) -> cmp::Ordering,
        Internal2: InternalTrait,
    {
        self.dual_sort_range_by(f, .., secondary).await;
    }

    /// Removes item from the vector at the given index.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// v.remove(1).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![1, 3].iter().await.collect::<Vec<i32>>().await);
    /// }
    /// ```
    pub async fn remove(&mut self, index: usize) -> Option<<Internal::Leaf as LeafTrait>::Item> {
        // TODO: This is not really the most efficient way to do this, specialize this function
        if index < self.len {
            let mut last_part = self.split_off(index).await;
            let item = last_part.pop_front().await;
            self.append(last_part).await;
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
    fn spine_ref(&self, side: Side) -> &Vec<NodeRc<Internal>> {
        match side {
            Side::Front => &self.left_spine,
            Side::Back => &self.right_spine,
        }
    }

    /// Returns a reference to the leaf on the requested side of the tree.
    fn leaf_ref(&self, side: Side) -> &Internal::LeafEntry {
        self.spine_ref(side)
            .first()
            .unwrap_or(&self.root)
            .leaf_ref()
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// let mut f = v.focus().await;
    /// assert_eq!(f.get(0).await, Some(&1));
    /// assert_eq!(f.get(1).await, Some(&2));
    /// assert_eq!(f.get(2).await, Some(&3));
    /// }
    /// ```
    pub async fn focus(&self) -> Focus<'_, Internal> {
        Focus::new(self).await
    }

    /// Returns a mutable focus over the vector. A focus tracks the last leaf and positions which
    /// was read. The path down this tree is saved in the focus and is used to accelerate lookups in
    /// nearby locations.
    pub async fn focus_mut(&mut self) -> FocusMut<'_, Internal> {
        let mut nodes = Vec::new();
        for node in self.left_spine.iter_mut() {
            if !node.is_empty(&self.context).await {
                nodes.push(node.borrow_node(&self.context).await);
            }
        }
        if !self.root.is_empty(&self.context).await {
            nodes.push(self.root.borrow_node(&self.context).await);
        }
        for node in self.right_spine.iter_mut().rev() {
            if !node.is_empty(&self.context).await {
                nodes.push(node.borrow_node(&self.context).await);
            }
        }
        FocusMut::from_vector(self, nodes).await
    }

    /// Returns an iterator over the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// let mut iter = v.iter().await;
    /// assert_eq!(iter.next().await, Some(&1));
    /// assert_eq!(iter.next().await, Some(&2));
    /// assert_eq!(iter.next().await, Some(&3));
    /// assert_eq!(iter.next().await, None);
    /// }
    /// ```
    pub async fn iter(&self) -> Iter<'_, Internal> {
        Iter {
            front: 0,
            back: self.len(),
            focus: self.focus().await,
            current_future: None,
        }
    }

    /// Returns a mutable iterator over the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// let mut iter = v.iter_mut().await;
    /// assert_eq!(iter.next().await, Some(&mut 1));
    /// assert_eq!(iter.next().await, Some(&mut 2));
    /// assert_eq!(iter.next().await, Some(&mut 3));
    /// assert_eq!(iter.next().await, None);
    /// }
    /// ```
    pub async fn iter_mut(&mut self) -> IterMut<'_, Internal> {
        IterMut {
            front: 0,
            back: self.len(),
            focus: self.focus_mut().await,
            current_future: None,
        }
    }

    /// Checks the internal invariants that are required by the Vector.
    #[allow(dead_code)]
    pub(crate) async fn assert_invariants(&self) -> bool {
        #[cfg(debug_assertions)]
        {
            // Invariant 1
            // Spines must be of equal length
            assert_eq!(self.left_spine.len(), self.right_spine.len());

            // Invariant 2
            // All nodes in the spine except for the first one (the leaf) must have at least 1 slot free.
            for spine in self.left_spine.iter().skip(1) {
                assert!(spine.free_slots(&self.context).await >= 1);
            }
            for spine in self.right_spine.iter().skip(1) {
                assert!(spine.free_slots(&self.context).await >= 1);
            }

            // Invariant 3
            // The first node(the leaf) in the spine must have at least 1 element, but may be full
            if let Some(leaf) = self.left_spine.first() {
                assert!(leaf.slots(&self.context).await >= 1);
            }
            if let Some(leaf) = self.right_spine.first() {
                assert!(leaf.slots(&self.context).await >= 1);
            }

            // Invariant 4
            // If the root is a non-leaf, it must always have at least 2 slot free, but may be empty
            if !self.root.is_leaf(&self.context).await {
                assert!(self.root.free_slots(&self.context).await >= 2);
            }

            // Invariant 5
            // If the root is an empty non-leaf node then the last two nodes of both spines:
            // 1) Must not be able to be merged into a node of 1 less height
            // 2) Must differ in slots by at most one node
            if self.root.is_empty(&self.context).await && !self.root.is_leaf(&self.context).await {
                let left_children = self.left_spine.last().unwrap().slots(&self.context).await;
                let right_children = self.right_spine.last().unwrap().slots(&self.context).await;

                let difference = if left_children > right_children {
                    left_children - right_children
                } else {
                    right_children - left_children
                };

                assert!(difference <= 1);

                let min_children = if self.root.level(&self.context).await == 1 {
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
                spine
                    .debug_check_invariants(spine.len(&self.context).await, level, &self.context)
                    .await;
            }
            self.root
                .debug_check_invariants(
                    self.root.len(&self.context).await,
                    self.left_spine.len(),
                    &self.context,
                )
                .await;
            for (level, spine) in self.right_spine.iter().enumerate() {
                spine
                    .debug_check_invariants(spine.len(&self.context).await, level, &self.context)
                    .await;
            }

            // Invariant 7
            // The tree's `len` field must match the sum of the spine's lens
            let mut left_spine_len = 0;
            for node in self.left_spine.iter() {
                left_spine_len += node.len(&self.context).await;
            }
            // .map(|x| x.len(&self.context))
            // .sum::<usize>();
            let root_len = self.root.len(&self.context).await;
            let mut right_spine_len = 0;
            for node in self.right_spine.iter() {
                right_spine_len += node.len(&self.context).await;
            }
            // let right_spine_len = self
            //     .right_spine
            //     .iter()
            //     .map(|x| x.len(&self.context))
            //     .sum::<usize>();
            assert_eq!(self.len, left_spine_len + root_len + right_spine_len);
        }
        true
    }
}

impl<Internal> InternalVector<Internal>
where
    Internal: InternalTrait,
    <Internal::Leaf as LeafTrait>::Item: Ord,
{
    /// Sorts the entire sequence by the natural comparator on the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    /// v.sort().await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].iter().await.collect::<Vec<i32>>().await);
    /// }
    /// ```
    pub async fn sort(&mut self) {
        self.sort_by(&Ord::cmp).await
    }

    /// Sorts the entire sequence by the natural comparator on the sequence. Any swap that occurs
    /// will be made in the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort(&mut secondary).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].iter().await.collect::<Vec<i32>>().await);
    /// assert_eq!(secondary.iter().await.collect::<Vec<char>>().await, vector!['j', 'i','h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'].iter().await.collect::<Vec<char>>().await);
    /// }
    /// ```
    pub async fn dual_sort<Internal2>(&mut self, secondary: &mut InternalVector<Internal2>)
    where
        Internal2: InternalTrait,
    {
        self.dual_sort_by(&Ord::cmp, secondary).await
    }

    /// Sorts the range of the sequence by the natural comparator on the sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    /// v.sort_range(5..).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 7, 6, 5, 0, 1, 2, 3, 4].iter().await.collect::<Vec<i32>>().await);
    /// }
    /// ```
    pub async fn sort_range<R>(&mut self, range: R)
    where
        R: RangeBounds<usize>,
    {
        self.sort_range_by(&Ord::cmp, range).await
    }

    /// Sorts the range of the sequence by the natural comparator on the sequence. Any swap that
    /// occurs will be made in the secondary vector provided.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    /// let mut secondary = vector!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'];
    /// v.dual_sort_range(5.., &mut secondary).await;
    /// assert_eq!(v.iter().await.collect::<Vec<i32>>().await, vector![9, 8, 7, 6, 5, 0, 1, 2, 3, 4].iter().await.collect::<Vec<i32>>().await);
    /// assert_eq!(secondary.iter().await.collect::<Vec<char>>().await, vector!['a', 'b', 'c', 'd', 'e', 'j', 'i','h', 'g', 'f'].iter().await.collect::<Vec<char>>().await);
    /// }
    /// ```
    pub async fn dual_sort_range<R, Internal2>(
        &mut self,
        range: R,
        secondary: &mut InternalVector<Internal2>,
    ) where
        R: RangeBounds<usize> + Clone,
        Internal2: InternalTrait,
    {
        self.dual_sort_range_by(&Ord::cmp, range, secondary).await
    }
}

impl<Internal> InternalVector<Internal>
where
    Internal: InternalTrait,
    <Internal::Leaf as LeafTrait>::Item: PartialEq,
{
    /// Tests whether the node is equal to the given vector. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) async fn equal_vec(&self, v: &Vec<<Internal::Leaf as LeafTrait>::Item>) -> bool {
        if self.len() == v.len() {
            let mut iter = v.iter();
            for spine in self.left_spine.iter() {
                if !spine.equal_iter_debug(&mut iter, &self.context).await {
                    println!("Left: {:?} {:?}", self, v);
                    return false;
                }
            }
            if !self.root.equal_iter_debug(&mut iter, &self.context).await {
                println!("Root: {:?} {:?}", self, v);
                return false;
            }
            for spine in self.right_spine.iter().rev() {
                if !spine.equal_iter_debug(&mut iter, &self.context).await {
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

// impl<Internal> Default for InternalVector<Internal>
// where
//     Internal: InternalTrait,
// {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl<Internal> PartialEq for InternalVector<Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: PartialEq,
// {
//     fn eq(&self, other: &Self) -> bool {
//         self.len() == other.len() && self.iter().eq(other.iter())
//     }
// }

// impl<Internal> Eq for InternalVector<Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: Eq,
// {
// }

// impl<Internal> PartialOrd for InternalVector<Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: PartialOrd,
// {
//     fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
//         self.iter().partial_cmp(other.iter())
//     }
// }

// impl<Internal> Ord for InternalVector<Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: Ord,
// {
//     fn cmp(&self, other: &Self) -> cmp::Ordering {
//         self.iter().cmp(other.iter())
//     }
// }

// impl<Internal> Hash for InternalVector<Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: Hash,
// {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         for i in self {
//             i.hash(state)
//         }
//     }
// }

// impl<Internal> FromIterator<<Internal::Leaf as LeafTrait>::Item> for InternalVector<Internal>
// where
//     Internal: InternalTrait,
// {
//     fn from_iter<I: IntoIterator<Item = <Internal::Leaf as LeafTrait>::Item>>(iter: I) -> Self {
//         let mut result = InternalVector::new();
//         for item in iter {
//             result.push_back(item);
//         }
//         result
//     }
// }

/// Derp
#[derive(Debug)]
pub struct MutBoundGuard<'a, Internal>
where
    Internal: InternalTrait,
{
    guard: Internal::ItemMutGuard,
    vector: &'a mut InternalVector<Internal>,
}

impl<'a, Internal> std::ops::Deref for MutBoundGuard<'a, Internal>
where
    Internal: InternalTrait,
{
    type Target = <Internal::Leaf as LeafTrait>::Item;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'a, Internal> std::ops::DerefMut for MutBoundGuard<'a, Internal>
where
    Internal: InternalTrait,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

/// An iterator for a Vector.
// #[derive(Clone, Debug)]
pub struct Iter<'a, Internal>
where
    Internal: InternalTrait,
    <Internal::Leaf as LeafTrait>::Item: 'a,
{
    front: usize,
    back: usize,
    focus: Focus<'a, Internal>,
    current_future: Option<
        std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Option<&'a <Internal::Leaf as LeafTrait>::Item>>
                    + 'a,
            >,
        >,
    >,
}

impl<'a, Internal> Stream for Iter<'a, Internal>
where
    Internal: InternalTrait,
{
    type Item = &'a <Internal::Leaf as LeafTrait>::Item;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut futures::task::Context,
    ) -> futures::task::Poll<Option<Self::Item>> {
        let self_mut = unsafe { std::pin::Pin::into_inner_unchecked(self) };
        if self_mut.current_future.is_none() {
            if self_mut.front != self_mut.back {
                let focus: &'a mut Focus<'a, Internal> =
                    unsafe { &mut *(&mut self_mut.focus as *mut _) };
                let result = focus.get(self_mut.front);
                self_mut.current_future = Some(Box::pin(result));
                self_mut.front += 1;
            }
        }

        if let Some(ref mut future) = self_mut.current_future {
            let state = future.as_mut().poll(cx);
            match state {
                std::task::Poll::Pending => futures::task::Poll::Pending,
                std::task::Poll::Ready(val) => {
                    self_mut.current_future = None;
                    futures::task::Poll::Ready(val)
                }
            }
        } else {
            futures::task::Poll::Ready(None)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

// impl<'a, Internal> IntoIterator for &'a InternalVector<Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: 'a,
// {
//     type Item = &'a <Internal::Leaf as LeafTrait>::Item;
//     type IntoIter = Iter<'a, Internal>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.iter()
//     }
// }

// impl<'a, Internal> DoubleEndedIterator for Iter<'a, Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: 'a,
// {
//     fn next_back(&mut self) -> Option<&'a <Internal::Leaf as LeafTrait>::Item> {
//         if self.front != self.back {
//             self.back -= 1;
//             let focus: &'a mut Focus<Internal> = unsafe { &mut *(&mut self.focus as *mut _) };
//             focus.get(self.back)
//         } else {
//             None
//         }
//     }
// }

// impl<'a, Internal> ExactSizeIterator for Iter<'a, Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: 'a,
// {
// }

// impl<'a, Internal> FusedIterator for Iter<'a, Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: 'a,
// {
// }

/// An iterator for a Vector.
// #[derive(Debug)]
pub struct IterMut<'a, Internal>
where
    Internal: InternalTrait,
{
    front: usize,
    back: usize,
    focus: FocusMut<'a, Internal>,
    current_future: Option<
        std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = Option<&'a mut <Internal::Leaf as LeafTrait>::Item>,
                    > + 'a,
            >,
        >,
    >,
    // dummy: std::marker::PhantomData<&'a ()>,
}

impl<'a, Internal> Stream for IterMut<'a, Internal>
where
    Internal: InternalTrait,
    <Internal::Leaf as LeafTrait>::Item: 'a,
{
    type Item = &'a mut <Internal::Leaf as LeafTrait>::Item;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut futures::task::Context,
    ) -> futures::task::Poll<Option<Self::Item>> {
        let self_mut = unsafe { std::pin::Pin::into_inner_unchecked(self) };
        if self_mut.current_future.is_none() {
            if self_mut.front != self_mut.back {
                let focus: &'a mut FocusMut<'a, Internal> =
                    unsafe { &mut *(&mut self_mut.focus as *mut _) };
                let result = focus.get(self_mut.front);
                self_mut.current_future = Some(Box::pin(result));
                self_mut.front += 1;
            }
        }

        if let Some(ref mut future) = self_mut.current_future {
            let state = future.as_mut().poll(cx);
            match state {
                std::task::Poll::Pending => futures::task::Poll::Pending,
                std::task::Poll::Ready(val) => {
                    self_mut.current_future = None;
                    futures::task::Poll::Ready(val)
                }
            }
        } else {
            futures::task::Poll::Ready(None)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

// impl<'a, Internal> IntoIterator for &'a mut InternalVector<Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: 'a,
// {
//     type Item = &'a mut <Internal::Leaf as LeafTrait>::Item;
//     type IntoIter = IterMut<'a, Internal>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.iter_mut()
//     }
// }

// impl<'a, Internal> DoubleEndedIterator for IterMut<'a, Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: 'a,
// {
//     fn next_back(&mut self) -> Option<&'a mut <Internal::Leaf as LeafTrait>::Item> {
//         if self.front != self.back {
//             self.back -= 1;
//             let focus: &'a mut FocusMut<Internal> = unsafe { &mut *(&mut self.focus as *mut _) };
//             focus.get(self.back)
//         } else {
//             None
//         }
//     }
// }

// impl<'a, Internal> ExactSizeIterator for IterMut<'a, Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: 'a,
// {
// }

// impl<'a, Internal> FusedIterator for IterMut<'a, Internal>
// where
//     Internal: InternalTrait,
//     <Internal::Leaf as LeafTrait>::Item: 'a,
// {
// }

#[allow(clippy::cognitive_complexity)]
#[cfg(test)]
mod test {
    use crate::*;
    use futures::stream::StreamExt;
    use proptest::prelude::*;
    use proptest::proptest;
    use proptest_derive::Arbitrary;
    use tokio::runtime::Runtime;

    const MAX_EXTEND_SIZE: usize = 1000;

    #[derive(Arbitrary)]
    enum Action<A: Clone + std::fmt::Debug + Arbitrary + 'static> {
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

    impl<A: Clone + std::fmt::Debug + Arbitrary + 'static> std::fmt::Debug for Action<A> {
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
    struct ActionList<A: Clone + std::fmt::Debug + Arbitrary + 'static> {
        actions: Vec<Action<A>>,
    }

    impl<A: Clone + std::fmt::Debug + Arbitrary + 'static> std::fmt::Debug for ActionList<A> {
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

            let mut rt = Runtime::new().unwrap();

            // Spawn a future onto the runtime
            rt.block_on(async {
                let mut vec: Vec<u64> = Vec::new();
                let mut vector: Vector<u64> = Vector::new().await;

                for action in &actions.actions {
                    match action {
                        Action::PushFront(item) => {
                            vec.insert(0, item.clone());
                            vector.push_front(item.clone()).await;
                            assert_eq!(vec.len(), vector.len());
                            assert!(vector.equal_vec(&vec).await);
                        },
                        Action::PushBack(item) => {
                            vec.push(item.clone());
                            vector.push_back(item.clone()).await;
                            assert_eq!(vec.len(), vector.len());
                            assert!(vector.equal_vec(&vec).await);
                        },
                        Action::ExtendFront(items) => {
                            for item in items {
                                vec.insert(0, item.clone());
                                vector.push_front(item.clone()).await;
                            }
                            assert_eq!(vec.len(), vector.len());
                            assert!(vector.equal_vec(&vec).await);
                        }
                        Action::ExtendBack(items) => {
                            for item in items {
                                vec.push(item.clone());
                                vector.push_back(item.clone()).await;
                            }
                            assert_eq!(vec.len(), vector.len());
                            assert!(vector.equal_vec(&vec).await);
                        }
                        Action::SplitLeft(index) => {
                            let index = index % (1 + vec.len());
                            vec.truncate(index);
                            vector.slice_from_start(index).await;
                            assert_eq!(vec.len(), vector.len());
                            assert!(vector.equal_vec(&vec).await);
                        }
                        Action::SplitRight(index) => {
                            let index = index % (1 + vec.len());
                            vec = vec.split_off(index);
                            vector.slice_to_end(index).await;
                            assert_eq!(vec.len(), vector.len());
                            assert!(vector.equal_vec(&vec).await);
                        }
                        Action::ConcatFront(items) => {
                            let mut new_vector = Vector::new().await;
                            for item in items {
                                vec.insert(0, item.clone());
                                new_vector.push_front(item.clone()).await;
                            }
                            new_vector.append(vector).await;
                            vector = new_vector;
                            assert_eq!(vec.len(), vector.len());
                            assert!(vector.equal_vec(&vec).await);
                        }
                        Action::ConcatBack(items) => {
                            let mut new_vector = Vector::new().await;
                            for item in items {
                                vec.push(item.clone());
                                new_vector.push_back(item.clone()).await;
                            }
                            vector.append(new_vector).await;
                            assert_eq!(vec.len(), vector.len());
                            assert!(vector.equal_vec(&vec).await);
                        }
                    }
                    assert!(vector.assert_invariants().await);
                }

                assert_eq!(
                    vector.iter().await.collect::<Vec<u64>>().await,
                    vec
                );
                // assert_eq!(
                //     vector.iter().await.rev().collect::<Vec<u64>>().await,
                //     vec.iter().rev().cloned().collect::<Vec<_>>()
                // );
        });
    }
    }

    #[tokio::test]
    pub async fn empty() {
        let empty: Vector<usize> = Vector::new().await;
        // let empty_vec: Vec<usize> = Vec::new();
        // let empty_ref_vec: Vec<&usize> = Vec::new();
        // let empty_ref_mut_vec: Vec<&mut usize> = Vec::new();

        // Len
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        // Back
        assert_eq!(empty.back().await, None);
        assert_eq!(empty.front().await, None);

        // Concat
        let mut empty_concat = empty.clone();
        empty_concat.append(empty.clone()).await;
        assert!(empty_concat.is_empty());
        assert_eq!(empty_concat.len(), 0);

        // Slice
        let mut empty_slice_left = empty.clone();
        empty_slice_left.slice_from_start(10).await;
        assert!(empty_slice_left.is_empty());
        assert_eq!(empty_slice_left.len(), 0);

        let mut empty_slice_right = empty.clone();
        empty_slice_right.slice_to_end(10).await;
        assert!(empty_slice_right.is_empty());
        assert_eq!(empty_slice_right.len(), 0);

        // Iter
        // assert_eq!(empty.iter().collect::<Vec<_>>(), empty_ref_vec);
        // assert_eq!(empty.iter_mut().collect::<Vec<_>>(), empty_ref_mut_vec);
        // assert_eq!(empty.into_iter().collect::<Vec<_>>(), empty_vec);
    }

    #[tokio::test]
    pub async fn single() {
        let mut item = 9;
        let mut single = Vector::new().await;
        single.push_back(item).await;

        // Len
        assert!(!single.is_empty());
        assert_eq!(single.len(), 1);

        // Back
        assert_eq!(single.back().await, Some(&item));
        assert_eq!(single.back_mut().await, Some(&mut item));
        let mut back = single.clone();
        assert_eq!(back.pop_back().await, Some(item));
        assert_eq!(back.pop_back().await, None);
        assert_eq!(back.back().await, None);
        assert_eq!(back.back_mut().await, None);

        // Front
        assert_eq!(single.front().await, Some(&item));
        assert_eq!(single.front_mut().await, Some(&mut item));
        let mut front = single.clone();
        assert_eq!(front.pop_front().await, Some(item));
        assert_eq!(front.pop_front().await, None);

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

    #[tokio::test]
    pub async fn large() {
        const N: usize = 10000;
        let mut vec = Vector::new().await;
        for i in 0..N {
            vec.push_back(i).await;
        }

        // Len
        assert!(!vec.is_empty());
        assert_eq!(vec.len(), N);

        // Back
        assert_eq!(vec.back().await, Some(&(N - 1)));
        assert_eq!(vec.back_mut().await, Some(&mut (N - 1)));
        assert_eq!(vec.clone().pop_back().await, Some(N - 1));

        // Front
        assert_eq!(vec.front().await, Some(&0));
        assert_eq!(vec.front_mut().await, Some(&mut 0));
        assert_eq!(vec.clone().pop_front().await, Some(0));

        // let vec
        // Iter
        let mut derp_vec = Vec::new();
        let mut derp_iter = vec.iter().await;
        while let Some(val) = derp_iter.next().await {
            derp_vec.push(*val);
        }
        // assert_eq!(
        //     vec.iter().await.collect::<Vec<_>>(),
        //     (0..N).collect::<Vec<_>>().iter().collect::<Vec<_>>()
        // );
        // assert_eq!(
        //     vec.iter_mut().collect::<Vec<_>>(),
        //     (0..N).collect::<Vec<_>>()
        // );
        // assert_eq!(
        //     vec.into_iter().collect::<Vec<_>>(),
        //     (0..N).collect::<Vec<_>>()
        // );

        // assert_eq!(
        //     vec.iter().await.rev().collect::<Vec<_>>(),
        //     (0..N).rev().collect::<Vec<_>>().iter().collect::<Vec<_>>()
        // );
    }

    #[tokio::test]
    pub async fn inserts() {
        let mut v = Vector::new().await;
        const N: usize = 1_000;
        for i in 0..N {
            v.insert(v.len() / 2, i).await;
            v.assert_invariants().await;

            // let first_half = (1..i + 1).step_by(2);
            // let second_half = (0..i + 1).step_by(2).rev();

            // let mut vector = Vec::new();
            // vector.extend(first_half);
            // vector.extend(second_half);

            // assert_eq!(v.iter().await.collect::<Vec<usize>>().await, vector);
        }
        let first_half = (1..N).step_by(2);
        let second_half = (0..N).step_by(2).rev();

        let mut vector = Vec::new();
        vector.extend(first_half);
        vector.extend(second_half);

        assert_eq!(v.iter().await.collect::<Vec<usize>>().await, vector);

        println!("{} {}", v.len(), v.height());
    }

    #[tokio::test]
    async fn test_equal_range() {
        let v: Vector<u64> = vector![0, 1, 1, 2, 3, 4, 7, 9, 10];
        assert_eq!(v.equal_range_in_range(&1, 0..3).await, Ok(1..3));
    }
}
