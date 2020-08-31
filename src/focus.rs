//! A focus for a vector.
//!
//! A focus tracks the last leaf and positions which was read. The path down this tree is saved in
//! the focus and is used to accelerate lookups in nearby locations.
use crate::node_traits::{
    BorrowedInternalTrait, BorrowedLeafTrait, BorrowedNode, Entry, InternalTrait, LeafTrait,
    NodeMut, NodeRc, NodeRef,
};
use crate::vector::InternalVector;
use crate::Side;
use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::ops::{Bound, Range, RangeBounds};

/// A focus for a particular node in the spine.
///
/// This tracks the path down to a particular leaf in the tree.
#[derive(Clone, Debug)]
struct PartialFocus<Internal>
where
    Internal: InternalTrait,
{
    path: Vec<(Internal::InternalEntry, Range<usize>)>,
    leaf: Internal::LeafEntry,
    leaf_range: Range<usize>,
}

impl<'a, Internal> PartialFocus<Internal>
where
    Internal: InternalTrait,
{
    /// A helper method to compute the remainder of a path down a tree to a particular index.
    async fn tree_path(
        nodes: &mut Vec<(Internal::InternalEntry, Range<usize>)>,
        mut idx: usize,
        context: &Internal::Context,
    ) -> (Range<usize>, Internal::LeafEntry) {
        while let Some((previous_root, range)) = nodes.last() {
            let (next_node, next_range) = if let Some((subchild, subchild_range)) = previous_root
                .load(context)
                .await
                .get_child_ref_for_position(idx)
            {
                let subrange_len = subchild_range.end - subchild_range.start;
                let absolute_subrange_start = range.start + subchild_range.start;
                let absolute_subrange =
                    absolute_subrange_start..absolute_subrange_start + subrange_len;
                if let NodeRef::Internal(internal) = subchild {
                    idx -= subchild_range.start;
                    (internal.clone(), absolute_subrange)
                } else {
                    return (absolute_subrange, subchild.leaf().clone());
                }
            } else {
                panic!("Attempt to move a focus to an out of bounds location.")
            };
            nodes.push((next_node, next_range));
        }
        unreachable!()
    }

    /// Constructs the focus from a tree node. This will focus on the first element in the node.
    async fn from_tree(tree: &'a NodeRc<Internal>, context: &Internal::Context) -> Self {
        match tree {
            NodeRc::Internal(internal) => {
                let mut path = vec![(internal.clone(), 0..tree.len(context).await)];
                let (leaf_range, leaf) =
                    PartialFocus::<Internal>::tree_path(&mut path, 0, context).await;
                PartialFocus {
                    path,
                    leaf,
                    leaf_range,
                }
            }
            NodeRc::Leaf(leaf) => PartialFocus {
                path: Vec::new(),
                leaf: leaf.clone(),
                leaf_range: 0..leaf.load(context).await.len(),
            },
        }
    }

    /// Moves the focus to a new index in the tree.
    pub async fn move_focus(&mut self, idx: usize, context: &Internal::Context) {
        if !self.leaf_range.contains(&idx) {
            while !self.path.last().unwrap().1.contains(&idx) {
                self.path.pop();
            }
            let new_idx = idx - self.path.last().unwrap().1.start;
            let (leaf_range, leaf) =
                PartialFocus::<Internal>::tree_path(&mut self.path, new_idx, context).await;
            self.leaf_range = leaf_range;
            self.leaf = leaf;
        }
    }

    /// Gets an element from the tree. If the element does not exist this will return `None`. This
    /// will move the focus along if necessary.
    async fn get(
        &mut self,
        idx: usize,
        context: &Internal::Context,
    ) -> Option<&<Internal::Leaf as LeafTrait>::Item> {
        if self.path.is_empty() {
            unsafe { Some(&*self.leaf.load(context).await.get(idx)?) }
        } else if idx >= self.path[0].0.load(context).await.len() {
            None
        } else {
            self.move_focus(idx, context).await;
            unsafe {
                Some(
                    &*self
                        .leaf
                        .load(context)
                        .await
                        .get(idx - self.leaf_range.start)?,
                )
            }
        }
    }
}

/// A focus for the entire the tree. Like a `PartialFocus`, but this also takes the position in the
/// spine into account.
#[derive(Debug)]
pub struct Focus<'a, Internal>
where
    Internal: InternalTrait,
{
    tree: &'a InternalVector<Internal>,
    spine_position: Option<(Side, usize)>,
    spine_node_focus: PartialFocus<Internal>,
    focus_range: Range<usize>,
    range: Range<usize>,
}

impl<'a, Internal> Clone for Focus<'a, Internal>
where
    Internal: InternalTrait,
{
    fn clone(&self) -> Self {
        Self {
            tree: self.tree,
            spine_position: self.spine_position.clone(),
            spine_node_focus: self.spine_node_focus.clone(),
            focus_range: self.focus_range.clone(),
            range: self.range.clone(),
        }
    }
}

impl<'a, Internal> Focus<'a, Internal>
where
    Internal: InternalTrait + 'a,
{
    /// Constructs a new focus for a Vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// let mut focus = Focus::new(&v).await;
    /// assert_eq!(focus.get(0).await, Some(&1));
    /// }
    /// ```
    pub async fn new(tree: &'a InternalVector<Internal>) -> Focus<'a, Internal> {
        Focus::narrowed_tree(tree, 0..tree.len()).await
    }

    /// Constructs a new focus for a Vector. The focus is narrowed by the given range, only
    /// elements within this range are accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// let mut focus = Focus::narrowed_tree(&v, 1..3).await;
    /// assert_eq!(focus.get(0).await, Some(&2));
    /// }
    /// ```
    pub async fn narrowed_tree(
        tree: &'a InternalVector<Internal>,
        mut range: Range<usize>,
    ) -> Focus<'a, Internal> {
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
            range_end = range_start + node.len(&tree.context).await;

            if range.start < range_end && range.end > range_start {
                focus_range = range_start..range_end;
                spine_position = node_position;
                focus_node = node;
            }
        }
        Focus {
            tree,
            spine_position,
            spine_node_focus: PartialFocus::from_tree(focus_node, &tree.context).await,
            focus_range,
            range,
        }
    }

    /// Refocuses to a different index within the tree. This will move to a different spine node if
    /// necessary.
    async fn refocus(&mut self, idx: usize) {
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
                let range_start = range_end - node.len(&self.tree.context).await;
                let range = range_start..range_end;
                if range.contains(&idx) {
                    self.spine_position = position;
                    self.focus_range = range;
                    self.spine_node_focus = PartialFocus::from_tree(node, &self.tree.context).await;
                    self.spine_node_focus
                        .move_focus(idx - range_start, &self.tree.context)
                        .await;
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
                let range_end = range_start + node.len(&self.tree.context).await;
                let range = range_start..range_end;
                if range.contains(&idx) {
                    self.spine_position = position;
                    self.focus_range = range;
                    self.spine_node_focus = PartialFocus::from_tree(node, &self.tree.context).await;
                    self.spine_node_focus
                        .move_focus(idx - range_start, &self.tree.context)
                        .await;
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// let mut focus = Focus::new(&v).await;
    /// assert_eq!(focus.get(0).await, Some(&1));
    /// assert_eq!(focus.get(1).await, Some(&2));
    /// assert_eq!(focus.get(2).await, Some(&3));
    /// assert_eq!(focus.get(3).await, None);
    /// }
    /// ```
    pub async fn get(&mut self, idx: usize) -> Option<&<Internal::Leaf as LeafTrait>::Item> {
        let new_idx = idx + self.range.start;
        if self.range.contains(&new_idx) {
            if !self.focus_range.contains(&new_idx) {
                self.refocus(new_idx).await;
            }
            self.spine_node_focus
                .get(new_idx - self.focus_range.start, &self.tree.context)
                .await
        } else {
            None
        }
    }

    /// Derp
    pub async fn index(&mut self, idx: usize) -> &<Internal::Leaf as LeafTrait>::Item {
        self.get(idx).await.unwrap()
    }

    /// Returns the length of the range that is accessible through the focus.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::{Focus, Vector};
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// assert_eq!(Focus::new(&v).await.len(), 3);
    /// assert_eq!(Focus::narrowed_tree(&v, 3..3).await.len(), 0);
    /// }
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// assert!(!Focus::new(&v).await.is_empty());
    /// assert!(Focus::narrowed_tree(&v, 3..3).await.is_empty());
    /// }
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// let mut focus = Focus::new(&v).await;
    /// assert_eq!(focus.get(0).await, Some(&1));
    /// focus.narrow(1..3).await;
    /// assert_eq!(focus.get(0).await, Some(&2));
    /// focus.narrow(1..2).await;
    /// assert_eq!(focus.get(0).await, Some(&3));
    /// focus.narrow(1..1).await;
    /// assert_eq!(focus.get(0).await, None);
    /// }
    /// ```
    pub async fn narrow<R: RangeBounds<usize>>(&mut self, range: R) {
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
        let new_focus = Focus::narrowed_tree(&self.tree, new_start..new_end).await;
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let v = vector![1, 2, 3];
    /// let focus = Focus::new(&v).await;
    /// let (mut first_focus, mut second_focus) = focus.split_at(1).await;
    /// assert_eq!(first_focus.get(0).await, Some(&1));
    /// assert_eq!(first_focus.get(1).await, None);
    /// assert_eq!(second_focus.get(0).await, Some(&2));
    /// assert_eq!(second_focus.get(1).await, Some(&3));
    /// }
    /// ```
    pub async fn split_at(self, idx: usize) -> (Focus<'a, Internal>, Focus<'a, Internal>) {
        if idx >= self.len() {
            panic!("Split is out of range")
        }
        let first_range = self.range.start..idx;
        let second_range = idx..self.range.end;
        let first = Focus::narrowed_tree(self.tree, first_range).await;
        let second = Focus::narrowed_tree(self.tree, second_range).await;
        (first, second)
    }
}

/// derp
pub enum FocusMut<'a, Internal>
where
    Internal: InternalTrait,
{
    /// derp
    Rooted {
        /// derp
        origin: &'a mut InternalVector<Internal>,
        /// derp
        focus: InnerFocusMut<Internal>,
        /// derp
        borrowed_roots: Vec<BorrowedNode<Internal>>,
        /// derp
        _marker: std::marker::PhantomData<&'a mut <Internal::Leaf as LeafTrait>::Item>,
    },
    /// derp
    Nonrooted {
        /// derp
        parent: &'a FocusMut<'a, Internal>,
        /// derp
        focus: InnerFocusMut<Internal>,
        /// derp
        _marker: std::marker::PhantomData<&'a mut <Internal::Leaf as LeafTrait>::Item>,
    },
}

unsafe impl<'a, Internal> Send for FocusMut<'a, Internal>
where
    Internal: InternalTrait,
    <Internal::Leaf as LeafTrait>::Item: Send,
{
}

unsafe impl<'a, Internal> Sync for FocusMut<'a, Internal>
where
    Internal: InternalTrait,
    <Internal::Leaf as LeafTrait>::Item: Sync,
{
}

impl<'a, Internal> Drop for FocusMut<'a, Internal>
where
    Internal: InternalTrait,
{
    fn drop(&mut self) {
        if let FocusMut::Rooted { borrowed_roots, .. } = self {
            while let Some(_) = borrowed_roots.pop() {
                // println!("Dropping borrowed root");
            }
        // println!("Lel dropping FocusMutRoot here");
        } else {
            // println!("Lel dropping FocusMutSub here");
        }
    }
}

// The above needs to be refactored such that the borrowed nodes are their own Drop type.

impl<'a, Internal> FocusMut<'a, Internal>
where
    Internal: InternalTrait,
{
    pub(crate) async fn from_vector(
        origin: &'a mut InternalVector<Internal>,
        nodes: Vec<BorrowedNode<Internal>>,
    ) -> FocusMut<'a, Internal> {
        FocusMut::Rooted {
            origin,
            focus: InnerFocusMut::from_vector(nodes.clone()).await,
            borrowed_roots: nodes,
            _marker: std::marker::PhantomData,
        }
    }

    /// Gets the length of the focus.
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
    /// let mut focus_1 = v.focus_mut().await;
    /// let (focus_1, focus_2) = focus_1.split_at(1).await;
    /// assert_eq!(focus_1.len(), 1);
    /// assert_eq!(focus_2.len(), 2);
    /// }
    /// ```
    pub fn len(&self) -> usize {
        match self {
            FocusMut::Rooted { focus, .. } | FocusMut::Nonrooted { focus, .. } => focus.len(),
        }
    }

    /// Tests whether the focus represents no elements.
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
    /// let mut focus_1 = v.focus_mut().await;
    /// let (focus_1, focus_2) = focus_1.split_at(0).await;
    /// assert!(focus_1.is_empty());
    /// assert!(!focus_2.is_empty());
    /// }
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// let mut focus_1 = v.focus_mut().await;
    /// let (mut focus_1, mut focus_2) = focus_1.split_at(1).await;
    /// assert_eq!(focus_1.get(0).await, Some(&mut 1));
    /// assert_eq!(focus_1.get(1).await, None);
    /// assert_eq!(focus_2.get(0).await, Some(&mut 2));
    /// assert_eq!(focus_2.get(1).await, Some(&mut 3));
    /// }
    /// ```
    /// # Panics
    ///
    /// Panics if the given index is greater than the focus' length.
    ///
    pub async fn split_at(
        &mut self,
        index: usize,
    ) -> (FocusMut<'_, Internal>, FocusMut<'_, Internal>) {
        match self {
            FocusMut::Rooted { focus, origin, .. } => {
                let (left, right) = focus.split_at(index, &origin.context).await;
                (
                    FocusMut::Nonrooted {
                        focus: left,
                        parent: self,
                        _marker: std::marker::PhantomData,
                    },
                    FocusMut::Nonrooted {
                        focus: right,
                        parent: self,
                        _marker: std::marker::PhantomData,
                    },
                )
            }
            FocusMut::Nonrooted { focus, parent, .. } => {
                let (left, right) = focus.split_at(index, parent.context()).await;
                (
                    FocusMut::Nonrooted {
                        focus: left,
                        parent: self,
                        _marker: std::marker::PhantomData,
                    },
                    FocusMut::Nonrooted {
                        focus: right,
                        parent: self,
                        _marker: std::marker::PhantomData,
                    },
                )
            }
        }
    }

    // /// Derp
    // pub fn append(&mut self, other: Self) {
    //     // The focus part remains the same, but update other bits.
    //     self.origins.extend(other.origins);
    //     self.nodes.extend(other.nodes);
    //     self.len += other.len;
    // }

    // /// Derp
    // pub fn prepend(&mut self, mut other: Self) {
    //     // The focus part must be updated to point to the new place before the other bits
    //     if let Some((ref mut root_id, ref mut root_range)) = self.root {
    //         *root_id += other.nodes.len();
    //         root_range.end += other.len;
    //         root_range.start += other.len;
    //         for (_, path_range) in self.path.iter_mut() {
    //             path_range.end += other.len;
    //             path_range.start += other.len;
    //         }
    //         self.leaf_range.end += other.len();
    //         self.leaf_range.start += other.len();
    //     }
    //     self.origins.reverse();
    //     other.origins.reverse();
    //     self.origins.extend(other.origins);
    //     self.origins.reverse();
    //     mem::swap(&mut self.nodes, &mut other.nodes);
    //     self.nodes.extend(other.nodes);
    //     self.len += other.len;
    // }

    /// Gets a mutable reference to the element at the given index of the focus. The index is
    /// relative to the start of the focus. If the index does not exist this will return `None`.
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
    /// let mut focus = v.focus_mut().await;
    /// assert_eq!(focus.get(0).await, Some(&mut 1));
    /// assert_eq!(focus.get(1).await, Some(&mut 2));
    /// assert_eq!(focus.get(2).await, Some(&mut 3));
    /// assert_eq!(focus.get(3).await, None);
    /// }
    /// ```
    pub async fn get(&mut self, idx: usize) -> Option<&mut <Internal::Leaf as LeafTrait>::Item> {
        match self {
            FocusMut::Rooted { focus, origin, .. } => focus.get(idx, &origin.context).await,
            FocusMut::Nonrooted { focus, parent, .. } => focus.get(idx, parent.context()).await,
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// let mut focus = v.focus_mut().await;
    /// assert_eq!(focus.index(0).await, &mut 1);
    /// assert_eq!(focus.index(1).await, &mut 2);
    /// assert_eq!(focus.index(2).await, &mut 3);
    /// }
    /// ```
    pub async fn index(&mut self, idx: usize) -> &mut <Internal::Leaf as LeafTrait>::Item {
        self.get(idx).await.expect("Index out of range.")
    }

    /// derp
    pub fn assert_invariants(&self) -> bool {
        true
    }

    fn context(&self) -> &Internal::Context {
        match self {
            FocusMut::Rooted { origin, .. } => &origin.context,
            FocusMut::Nonrooted { parent, .. } => parent.context(),
        }
    }
}

/// A focus of the elements of a vector. The focus allows mutation of the elements in the vector.
#[derive(Clone)]
pub struct InnerFocusMut<Internal>
where
    Internal: InternalTrait,
{
    // origin: Rc<&'a mut InternalVector<Internal, Leaf, BorrowedInternal>>,
    pub(crate) nodes: ManuallyDrop<Vec<BorrowedNode<Internal>>>,
    len: usize,
    // Focus part
    // This indicates the index of the root in the node list and the range of that is covered by it
    root: Option<(usize, Range<usize>)>,
    // The listing of internal nodes below the borrowed root node along with their associated ranges
    path: Vec<(Internal::Borrowed, Range<usize>)>,
    // The leaf of the focus part, might not exist if the borrowed root is a leaf node
    leaf: Option<<Internal::Leaf as LeafTrait>::Borrowed>,
    // The range that is covered by the lowest part of the focus
    leaf_range: Range<usize>,
    // Information for when the focus is split, the full borrowed nodes can be destroyed correctly.
    borrowed_nodes: Vec<BorrowedNode<Internal>>,
}

impl<Internal> Drop for InnerFocusMut<Internal>
where
    Internal: InternalTrait,
{
    fn drop(&mut self) {
        self.drop_path_nodes();
        self.drop_borrowed_nodes();
        while let Some(node) = self.nodes.pop() {
            std::mem::forget(node);
        }
        unsafe {
            ManuallyDrop::drop(&mut self.nodes);
        }
    }
}

impl<Internal> InnerFocusMut<Internal>
where
    Internal: InternalTrait,
{
    fn empty(&mut self) -> Self {
        InnerFocusMut {
            // origin: self.origin.clone(),
            nodes: ManuallyDrop::new(vec![]),
            len: 0,
            root: None,
            path: vec![],
            leaf: None,
            leaf_range: 0..0,
            borrowed_nodes: vec![],
        }
    }

    pub(crate) async fn from_vector(nodes: Vec<BorrowedNode<Internal>>) -> Self {
        let mut len = 0;
        for node in nodes.iter() {
            len += node.len().await;
        }
        InnerFocusMut {
            borrowed_nodes: vec![],
            nodes: ManuallyDrop::new(nodes),
            len,
            root: None,
            path: vec![],
            leaf: None,
            leaf_range: 0..0,
        }
    }

    /// Drops the borrowed nodes for the focus. The borrowed nodes get populated when the focus is
    /// split. As the split foci don't clean up the parent when both splits are dropped, this is
    /// necessary to be called when the parent is re-used or dropped. The borrowed nodes cannot
    /// exist at the same time as the path nodes.
    fn drop_borrowed_nodes(&mut self) {
        while let Some(_) = self.borrowed_nodes.pop() {
            // println!("dropping borrowed");
        }
    }

    /// Drops the path nodes for the focus. The path nodes get populated when the focus is used for
    /// get. As the split foci don't clean up the parent when both splits are dropped, this is
    /// necessary to be called when the parent is re-used or dropped. The path nodes cannot exist at
    /// the same time as the borrowed nodes.
    fn drop_path_nodes(&mut self) {
        self.root = None;
        // panic!("rawr {}", self.path.len());
        if let Some(_) = self.leaf.take() {
            // println!("dropping leaf {:?} ", self.leaf_range);
            self.leaf_range = 0..0;
        }
        while let Some(_) = self.path.pop() {
            // println!("dropping internal {:?} ", p.1);
        }
    }

    /// Gets the length of the focus.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Tests whether the focus represents no elements.
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
    async fn split_at(&mut self, index: usize, context: &Internal::Context) -> (Self, Self) {
        // We split the vector in two at the position, we need to find the two positions that denote
        // the last of this vector and the first of the next vector.
        self.drop_borrowed_nodes(); // Might be going from one split to another so we may need drop.

        if index == 0 {
            // This vector becomes empty and the returned one is self.
            let first_part = self.empty();
            let second_part = self.clone();
            debug_assert!(first_part.assert_invariants().await);
            debug_assert!(second_part.assert_invariants().await);
            return (first_part, second_part);
        } else if index == self.len {
            // This vector is unchanged and the returned one is empty.
            let first_part = self.clone();
            let second_part = self.empty();
            debug_assert!(first_part.assert_invariants().await);
            debug_assert!(second_part.assert_invariants().await);
            return (first_part, second_part);
        }
        let original_len = self.len;
        // index is now 1..self.len()
        let (self_child_position, mut subindex) =
            self.find_node_info_for_index(index).await.unwrap();
        // self.len = index;

        let mut left_nodes = ManuallyDrop::into_inner(self.nodes.clone());
        let mut right_nodes = left_nodes.split_off(self_child_position);
        right_nodes.reverse();

        loop {
            if subindex == 0 {
                break;
            }
            let node = right_nodes.pop().unwrap();
            match node {
                BorrowedNode::Internal(mut internal) => {
                    let (new_subindex, left, mut right) = internal.split_at_position(subindex);
                    subindex = new_subindex;
                    let child = right.pop_child(Side::Front, context).await.unwrap();
                    if !left.is_empty() {
                        left_nodes.push(BorrowedNode::Internal(left));
                    }
                    if !right.is_empty() {
                        right_nodes.push(BorrowedNode::Internal(right));
                    }
                    right_nodes.push(child);
                }
                BorrowedNode::Leaf(mut leaf) => {
                    let (left, right) = leaf.split_at(subindex);
                    if !left.is_empty() {
                        left_nodes.push(BorrowedNode::Leaf(left));
                    }
                    if !right.is_empty() {
                        right_nodes.push(BorrowedNode::Leaf(right));
                    }
                    break;
                }
            }
        }
        right_nodes.reverse();

        // self.root.take();
        // self.path.clear();
        // self.leaf = None;
        // self.leaf_range = 0..0;

        let first_part = InnerFocusMut::from_vector(left_nodes).await;
        let second_part = InnerFocusMut::from_vector(right_nodes).await;
        assert_eq!(first_part.len + second_part.len, original_len);
        debug_assert!(first_part.assert_invariants().await);
        debug_assert!(second_part.assert_invariants().await);
        (first_part, second_part)
    }

    // fn combine(&mut self, mut other: Self) {
    //     // Recombine both side FocusMuts into back into the original FocusMut
    //     if self.is_empty() {
    //         *self = other;
    //         return;
    //     } else if other.is_empty() {
    //         return;
    //     }

    //     other.nodes.reverse();

    //     while !self.nodes.is_empty() && !other.nodes.is_empty() {
    //         let mut left_node = self.nodes.pop().unwrap();
    //         let right_node = other.nodes.pop().unwrap();
    //         let right_level = right_node.level();

    //         if !left_node.from_same_source(&right_node) {
    //             self.nodes.push(left_node);
    //             other.nodes.push(right_node);
    //             break;
    //         }

    //         left_node.combine(right_node);
    //         if let Some(right_parent) = other.nodes.last_mut() {
    //             // If we have a borrowed Leaf as the root then we could have exhausted the list.
    //             if right_level + 1 == right_parent.level() {
    //                 // The parent range must be updated to include the whole borrowed node now.
    //                 right_parent.internal_mut().unpop_child(Side::Front);
    //             } else {
    //                 other.nodes.push(left_node);
    //             }
    //         } else {
    //             other.nodes.push(left_node);
    //         }
    //     }
    //     other.nodes.reverse();
    //     self.nodes.append(&mut other.nodes);
    //     self.len += other.len;
    // }

    // /// Derp
    // pub fn append(&mut self, other: Self) {
    //     // The focus part remains the same, but update other bits.
    //     self.origins.extend(other.origins);
    //     self.nodes.extend(other.nodes);
    //     self.len += other.len;
    // }

    // /// Derp
    // pub fn prepend(&mut self, mut other: Self) {
    //     // The focus part must be updated to point to the new place before the other bits
    //     if let Some((ref mut root_id, ref mut root_range)) = self.root {
    //         *root_id += other.nodes.len();
    //         root_range.end += other.len;
    //         root_range.start += other.len;
    //         for (_, path_range) in self.path.iter_mut() {
    //             path_range.end += other.len;
    //             path_range.start += other.len;
    //         }
    //         self.leaf_range.end += other.len();
    //         self.leaf_range.start += other.len();
    //     }
    //     self.origins.reverse();
    //     other.origins.reverse();
    //     self.origins.extend(other.origins);
    //     self.origins.reverse();
    //     mem::swap(&mut self.nodes, &mut other.nodes);
    //     self.nodes.extend(other.nodes);
    //     self.len += other.len;
    // }

    // /// Narrows the focus so it only represents the given subrange of the focus.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// # #[macro_use] extern crate librrb;
    // /// # use librrb::Vector;
    // /// let mut v = vector![1, 2, 3];
    // /// let mut focus = v.focus_mut();
    // /// focus.narrow(1..3, |focus| {
    // ///     assert_eq!(focus.get(0), Some(&mut 2));
    // ///     assert_eq!(focus.get(1), Some(&mut 3));
    // ///     assert_eq!(focus.get(2), None);
    // /// });
    // /// ```
    // pub fn narrow<
    //     R: RangeBounds<usize>,
    //     F: FnMut(&mut FocusMut<Internal, Leaf, BorrowedInternal>),
    // >(
    //     &mut self,
    //     range: R,
    //     mut f: F,
    //     context: &Internal::Context,
    // ) {
    //     let range_start = match range.start_bound() {
    //         Bound::Unbounded => 0,
    //         Bound::Included(x) => *x,
    //         Bound::Excluded(x) => x + 1,
    //     };
    //     let range_end = match range.end_bound() {
    //         Bound::Unbounded => self.len(),
    //         Bound::Included(x) => x + 1,
    //         Bound::Excluded(x) => *x,
    //     };
    //     self.split_at_fn(range_end, context, |left, _right| {
    //         left.split_at_fn(range_start, context, |_left, right| f(right))
    //     });
    // }

    async fn move_focus(&mut self, mut idx: usize, context: &Internal::Context) {
        if !self.borrowed_nodes.is_empty() {
            self.drop_borrowed_nodes();
        }
        if self.leaf_range.contains(&idx) {
            // Nothing needs to move here
            return;
        }
        if let Some(_) = self.leaf.take() {
            // print!("Dropping leaf {:?} ", self.leaf_range);
            self.leaf_range = 0..0;
        }
        if let Some((_, ref mut range)) = self.root {
            if !range.contains(&idx) {
                self.drop_path_nodes();
            }
        }
        if self.root.is_none() {
            // If the root is unassigned we can potentially find a new one.
            if idx < self.len {
                let (node_position, new_idx) = self.find_node_info_for_index(idx).await.unwrap();
                let node_start = idx - new_idx;
                let node_len = self.nodes[node_position].len().await;
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
            // print!("Popping {:?} ", range);
            self.path.pop();
        }
        if self.path.is_empty() {
            if let Some((ref root, ref range)) = self.root {
                // Over here we are guaranteed only the root and the elements in path are correct
                // Our job now is to refresh the remainder of the path and the leave.
                let root = &mut self.nodes[*root];
                match root {
                    BorrowedNode::Internal(internal) => {
                        let (subchild, subchild_range) = internal
                            .get_child_mut_for_position(idx - range.start)
                            .unwrap();
                        let absolute_subchild_range = (subchild_range.start + range.start)
                            ..(subchild_range.end + range.start);

                        match subchild {
                            NodeMut::Internal(subchild) => {
                                self.path.push((
                                    subchild.load_mut(context).await.borrow_node(),
                                    absolute_subchild_range,
                                ));
                            }
                            NodeMut::Leaf(subchild) => {
                                self.leaf = Some(subchild.load_mut(context).await.borrow_node());
                                self.leaf_range = absolute_subchild_range;
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
        loop {
            let (parent, parent_subrange) = self.path.last_mut().unwrap();
            let parent = &mut *parent;
            let (child_node, child_subrange) = parent.get_child_mut_for_position(idx).unwrap();
            idx -= child_subrange.start;
            let child_subrange = (parent_subrange.start + child_subrange.start)
                ..(parent_subrange.start + child_subrange.end);
            // let (child_idx, new_idx) = parent.position_info_for(idx).unwrap();
            // let this_skipped_items = idx - new_idx;
            // idx = new_idx;
            match child_node {
                NodeMut::Internal(internal) => {
                    let mut new_root = internal.load_mut(context).await;
                    self.path.push((new_root.borrow_node(), child_subrange));
                }
                NodeMut::Leaf(leaf) => {
                    // skipped_items.start += this_skipped_items;
                    // skipped_items.end = skipped_items.start + leaf_len;
                    self.leaf = Some(leaf.load_mut(context).await.borrow_node());
                    self.leaf_range = child_subrange;
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
    /// # use futures::stream::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() {
    /// let mut v = vector![1, 2, 3];
    /// let mut focus = v.focus_mut().await;
    /// assert_eq!(focus.get(0).await, Some(&mut 1));
    /// assert_eq!(focus.get(1).await, Some(&mut 2));
    /// assert_eq!(focus.get(2).await, Some(&mut 3));
    /// assert_eq!(focus.get(3).await, None);
    /// }
    /// ```
    pub async fn get(
        &mut self,
        idx: usize,
        context: &Internal::Context,
    ) -> Option<&mut <Internal::Leaf as LeafTrait>::Item> {
        self.move_focus(idx, context).await;
        if self.leaf_range.contains(&idx) {
            // println!("In branch A {} with idx for in {:?}", idx, self.leaf_range);
            let position = idx - self.leaf_range.start;
            if let Some(ref mut leaf) = self.leaf {
                let ptr = leaf.get_mut(position)?;
                // println!("In branch B with idx {:?}", unsafe { &mut *ptr });
                unsafe { Some(&mut *ptr) }
            } else {
                // println!(
                //     "In branch C with idx, {:?} {:?}",
                //     self.root.as_ref(),
                //     self.nodes.get(1)
                // );
                let root_index = self.root.as_ref().unwrap().0;
                let ptr = self.nodes[root_index].leaf_mut().get_mut(position)?;
                // println!("doodah focus {:p}", ptr);
                unsafe { Some(&mut *ptr) }
            }
        } else {
            // println!("In branch D");
            None
        }
    }

    // /// Gets a mutable reference to the element at the given index of the focus. The index is
    // /// relative to the start of the focus. If the index does not exist this will panic.
    // ///
    // /// # Examples
    // ///
    // /// ```
    // /// # #[macro_use] extern crate librrb;
    // /// # use librrb::Vector;
    // /// let mut v = vector![1, 2, 3];
    // /// let mut focus = v.focus_mut();
    // /// assert_eq!(focus.index(0), &mut 1);
    // /// assert_eq!(focus.index(1), &mut 2);
    // /// assert_eq!(focus.index(2), &mut 3);
    // /// ```
    // pub fn index(&mut self, idx: usize, context: &Internal::Context) -> &mut Leaf::Item {
    //     self.get(idx).expect("Index out of range.")
    // }

    /// Returns the spine position and subindex corresponding the given index.
    async fn find_node_info_for_index(&self, index: usize) -> Option<(usize, usize)> {
        if index >= self.len {
            None
        } else {
            let mut forward_end = 0;

            for (idx, node) in self.nodes.iter().enumerate() {
                if index < forward_end + node.len().await {
                    return Some((idx, index - forward_end));
                }
                forward_end += node.len().await;
            }
            unreachable!();
        }
    }

    async fn assert_invariants(&self) -> bool {
        let mut cumulative = 0;
        for node in self.nodes.iter() {
            cumulative += node.len().await;
        }
        cumulative == self.len
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    use crossbeam;
    use futures::stream::StreamExt;

    #[tokio::test]
    pub async fn single_focus_mut() {
        let mut v = Vector::new().await;
        const N: isize = 1_000_000;
        for i in 0..N {
            v.push_back(i).await;
        }

        {
            let mut focus = v.focus_mut().await;
            for i in 0..N {
                let thing = focus.index(i as usize).await;
                *thing = -i;
            }
        }
        let mut iter = v.iter().await.enumerate();
        while let Some((i, v)) = iter.next().await {
            let r = -(i as isize);
            assert_eq!(v, &r);
        }
    }

    #[tokio::test]
    pub async fn split_focus_mut() {
        let mut v = Vector::new().await;
        const N: usize = 1_000_000;
        for i in 0..N {
            v.push_back(i).await;
        }

        const S: usize = N / 2;
        println!("Prebuild");
        {
            let mut focus = v.focus_mut().await;
            println!("Presplit");
            {
                let (mut left, mut right) = focus.split_at(S).await;
                println!("Postsplit");
                for i in 0..S {
                    let thing = left.get(i).await;
                    if let Some(thing) = thing {
                        *thing = 0;
                    }
                }
                for i in 0..N - S {
                    let thing = right.get(i).await;
                    if let Some(thing) = thing {
                        *thing = 1;
                    }
                }
                println!("End of split");
            }
            for i in 0..N {
                if i < S {
                    assert_eq!(focus.get(i).await, Some(&mut 0));
                } else {
                    assert_eq!(focus.get(i).await, Some(&mut 1));
                }
            }
            println!("Predrop");
        }
        println!("Postdrop");
        let mut iter = v.iter().await.enumerate();
        while let Some((i, v)) = iter.next().await {
            if i < S {
                assert_eq!(v, &0);
            } else {
                assert_eq!(v, &1);
            }
        }
        println!("Post test");
    }
}
