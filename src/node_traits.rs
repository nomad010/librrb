use crate::Side;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut, Range};

use async_trait::async_trait;

#[async_trait(?Send)]
pub trait Entry: Clone + std::fmt::Debug {
    type Item: Clone + std::fmt::Debug;
    type LoadGuard: Deref<Target = Self::Item>;
    type LoadMutGuard: DerefMut<Target = Self::Item>;
    type Context: Clone + std::fmt::Debug + Default;

    async fn new(item: Self::Item) -> Self;

    async fn load(&self, context: &Self::Context) -> Self::LoadGuard;

    async fn load_mut(&mut self, context: &Self::Context) -> Self::LoadMutGuard;
}

#[async_trait(?Send)]
pub trait BorrowedLeafTrait: Clone + std::fmt::Debug {
    type Concrete: LeafTrait<Borrowed = Self>;
    type ItemMutGuard: DerefMut<Target = <Self::Concrete as LeafTrait>::Item>;

    async fn split_at(&mut self, idx: usize) -> (Self, Self);

    async fn len(&self) -> usize;

    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    async fn get_mut_guarded(&mut self, idx: usize) -> Option<Self::ItemMutGuard>;

    async fn get_mut(&mut self, idx: usize) -> Option<*mut <Self::Concrete as LeafTrait>::Item>;

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    async fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        debug_assert_eq!(reported_size, self.len().await);
        debug_assert_eq!(reported_level, 0);
    }
}

pub trait LeafTrait: Clone + std::fmt::Debug {
    type Item: Clone + std::fmt::Debug;
    type Context: Clone + std::fmt::Debug + Default;
    type Borrowed: BorrowedLeafTrait<Concrete = Self> + std::fmt::Debug;
    type ItemMutGuard: DerefMut<Target = Self::Item>;

    /// Constructs a new empty leaf.
    fn empty() -> Self;

    /// Constructs a new leaf with a single item.
    fn with_item(item: Self::Item, context: &Self::Context) -> Self;

    /// Returns the number of items stored in the leaf.
    fn len(&self) -> usize;

    /// Returns whether the leaf is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the amount of space left in the leaf.
    fn free_space(&self) -> usize;

    /// Returns whether the leaf is full.
    fn is_full(&self) -> bool {
        self.free_space() == 0
    }

    /// Gets a reference to the item at requested position if it exists.
    fn get(&self, position: usize) -> Option<*const Self::Item>;

    /// Gets a reference to the item at the front of the leaf.
    fn front(&self) -> Option<*const Self::Item> {
        self.get(0)
    }

    /// Gets a reference to the item at the back of the leaf.
    fn back(&self) -> Option<*const Self::Item> {
        self.get(self.len().saturating_sub(1))
    }

    /// Gets a mutable reference to the item at requested position if it exists.
    fn get_mut_guarded(
        &mut self,
        position: usize,
        context: &Self::Context,
    ) -> Option<Self::ItemMutGuard>;

    /// Gets a mutable reference to the item at requested position if it exists.
    fn get_mut(&mut self, position: usize, context: &Self::Context) -> Option<*mut Self::Item>;

    /// Gets a mutable reference to the item at the front of the leaf.
    fn front_mut(&mut self, context: &Self::Context) -> Option<*mut Self::Item> {
        self.get_mut(0, context)
    }

    /// Gets a reference to the item at the back of the leaf.
    fn back_mut(&mut self, context: &Self::Context) -> Option<*mut Self::Item> {
        self.get_mut(self.len().saturating_sub(1), context)
    }

    /// Attempts to push an element to a side of the buffer.
    ///
    /// Panics if the buffer is full.
    fn push(&mut self, side: Side, item: Self::Item, context: &Self::Context);

    /// Attempts to pop an element from a side of the buffer.
    ///
    /// Panics if the buffer is empty.
    fn pop(&mut self, side: Side, context: &Self::Context) -> Self::Item;

    /// Removes elements from `self` and inserts these elements into `destination`. At most `len`
    /// elements will be removed. The actual number of elements decanted is returned. Elements are
    /// popped from `share_side` but pushed into the destination via `share_side.negate()`
    fn share_children_with(
        &mut self,
        destination: &mut Self,
        share_side: Side,
        len: usize,
        context: &Self::Context,
    ) -> usize;

    fn borrow_node(&mut self) -> Self::Borrowed;

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        _context: &Self::Context,
    ) {
        debug_assert_eq!(reported_size, self.len());
        debug_assert_eq!(reported_level, 0);
    }

    fn split(&mut self, idx: usize, context: &Self::Context) -> Self;

    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    fn equal_iter_debug<'a>(
        &self,
        iter: &mut std::slice::Iter<'a, Self::Item>,
        context: &Self::Context,
    ) -> bool
    where
        Self::Item: PartialEq;
}

#[async_trait(?Send)]
pub trait BorrowedInternalTrait: Clone + std::fmt::Debug {
    type Concrete: InternalTrait<Borrowed = Self>;
    type ItemMutGuard: DerefMut<Target = Self::Concrete>;

    async fn len(&self) -> usize;

    async fn slots(&self) -> usize;

    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    async fn range(&self) -> Range<usize>;

    async fn level(&self) -> usize;

    async fn split_at_child(&mut self, index: usize) -> (Self, Self);

    async fn split_at_position(&mut self, position: usize) -> (usize, Self, Self);

    /// Returns a mutable reference to the Rc of the child node at the given slot in this node.
    async fn get_child_mut_at_slot(
        &mut self,
        idx: usize,
    ) -> Option<(NodeMut<'_, Self::Concrete>, Range<usize>)>;

    async fn get_child_mut_at_side(
        &mut self,
        side: Side,
    ) -> Option<(NodeMut<'_, Self::Concrete>, Range<usize>)> {
        match side {
            Side::Front => self.get_child_mut_at_slot(0).await,
            Side::Back => {
                self.get_child_mut_at_slot(self.slots().await.saturating_sub(1))
                    .await
            }
        }
    }

    /// Returns a mutable reference to the Rc of the child node that covers the leaf position in this node.
    async fn get_child_mut_for_position(
        &mut self,
        position: usize,
    ) -> Option<(NodeMut<Self::Concrete>, Range<usize>)>;

    /// Logically pops a child from the node. The child is then returned.
    async fn pop_child(
        &mut self,
        side: Side,
        context: &<<Self::Concrete as InternalTrait>::Leaf as LeafTrait>::Context,
    ) -> Option<BorrowedNode<Self::Concrete>>;

    /// Undoes the popping that has occurred via a call to `pop_child`.
    async fn unpop_child(&mut self, side: Side);

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    async fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        debug_assert_eq!(reported_size, self.len().await);
        debug_assert_eq!(reported_level, self.level().await);
    }
}

#[async_trait(?Send)]
pub trait InternalTrait: Clone + std::fmt::Debug {
    type Context: Clone + std::fmt::Debug + Default;
    type Borrowed: BorrowedInternalTrait<Concrete = Self> + std::fmt::Debug;

    type Leaf: LeafTrait<Context = Self::Context, ItemMutGuard = Self::ItemMutGuard>;
    type LeafEntry: Entry<Item = Self::Leaf, Context = Self::Context>;
    type InternalEntry: Entry<Item = Self, Context = Self::Context>;

    type ItemMutGuard: DerefMut<Target = <Self::Leaf as LeafTrait>::Item>;

    /// Constructs a new empty internal node that is at the given level in the tree.
    async fn empty_internal(level: usize) -> Self;

    /// Constructs a new internal node of the same level, but with no children.
    async fn new_empty(&self) -> Self;

    /// Removes elements from `self` and inserts these elements into `destination`. At most `len`
    /// elements will be removed. The actual number of elements decanted is returned. Elements are
    /// popped from `share_side` but pushed into the destination via `share_side.negate()`. Both
    /// self and destination should be at the same level in the tree.
    async fn share_children_with(
        &mut self,
        destination: &mut Self,
        share_side: Side,
        len: usize,
        context: &Self::Context,
    ) -> usize;

    /// Packs the children of the to the left of the node so that all but the last is dense.
    async fn pack_children(&mut self, context: &Self::Context);

    /// Returns a reference to the element at the given index in the tree.
    async fn get(
        &self,
        idx: usize,
        context: &Self::Context,
    ) -> Option<*const <Self::Leaf as LeafTrait>::Item>;

    /// Returns a mutable reference to the element at the given index in the tree.
    async fn get_mut_guarded(
        &mut self,
        idx: usize,
        context: &Self::Context,
    ) -> Option<Self::ItemMutGuard>;

    /// Returns a mutable reference to the element at the given index in the tree.
    async fn get_mut(
        &mut self,
        idx: usize,
        context: &Self::Context,
    ) -> Option<*mut <Self::Leaf as LeafTrait>::Item>;

    /// Removes and returns the node at the given side of this node.
    /// # Panics
    // This should panic if the node is empty.
    async fn pop_child(&mut self, side: Side, context: &Self::Context) -> NodeRc<Self>;

    /// Adds a node to the given side of this node.
    /// # Panics
    // This should panic if the node does not have a slot free.
    async fn push_child(&mut self, side: Side, node: NodeRc<Self>, context: &Self::Context);

    async fn borrow_node(&mut self) -> Self::Borrowed;

    /// Returns the level the node is at in the tree.
    async fn level(&self) -> usize;

    /// Returns the size(number of elements hanging off) of the node.
    async fn len(&self) -> usize;

    /// Returns whether the node is empty. This should almost never be the case.
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Returns the number of direct children of the node.
    async fn slots(&self) -> usize;

    /// Returns the number of direct children that could be inserted into the node.
    async fn free_slots(&self) -> usize;

    /// Returns whether no more children can be inserted into the node.
    async fn is_full(&self) -> bool {
        self.free_slots().await == 0
    }

    /// Returns a reference to the Rc of the child node at the given slot in this node.
    async fn get_child_ref_at_slot(&self, idx: usize) -> Option<(NodeRef<Self>, Range<usize>)>;

    async fn get_child_ref_at_side(&self, side: Side) -> Option<(NodeRef<'_, Self>, Range<usize>)> {
        match side {
            Side::Front => self.get_child_ref_at_slot(0).await,
            Side::Back => {
                self.get_child_ref_at_slot(self.slots().await.saturating_sub(1))
                    .await
            }
        }
    }

    /// Returns a reference to the Rc of the child node that covers the leaf position in this node.
    async fn get_child_ref_for_position(
        &self,
        position: usize,
    ) -> Option<(NodeRef<Self>, Range<usize>)>;

    /// Returns a mutable reference to the Rc of the child node at the given slot in this node.
    async fn get_child_mut_at_slot(&mut self, idx: usize) -> Option<(NodeMut<Self>, Range<usize>)>;

    async fn get_child_mut_at_side(
        &mut self,
        side: Side,
    ) -> Option<(NodeMut<'_, Self>, Range<usize>)> {
        match side {
            Side::Front => self.get_child_mut_at_slot(0).await,
            Side::Back => {
                self.get_child_mut_at_slot(self.slots().await.saturating_sub(1))
                    .await
            }
        }
    }

    /// Returns a mutable reference of the Rc of the child node that covers the leaf position in this node.
    async fn get_child_mut_for_position(
        &mut self,
        position: usize,
    ) -> Option<(NodeMut<Self>, Range<usize>)>;

    /// Splits the node into two at the given slot index.
    async fn split_at_child(&mut self, idx: usize, context: &Self::Context) -> Self;

    /// Splits the node into two at the given position.
    async fn split_at_position(&mut self, position: usize, context: &Self::Context) -> Self;

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    async fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        _context: &Self::Context,
    ) {
        debug_assert_eq!(reported_size, self.len().await);
        debug_assert_ne!(reported_level, self.level().await);
    }

    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    async fn equal_iter_debug<'a>(
        &self,
        iter: &mut std::slice::Iter<'a, <Self::Leaf as LeafTrait>::Item>,
        context: &Self::Context,
    ) -> bool
    where
        <Self::Leaf as LeafTrait>::Item: PartialEq;
}

/// Represents an arbitrary node in the tree.
#[derive(Debug)]
pub enum NodeRc<Internal>
where
    Internal: InternalTrait,
{
    Leaf(Internal::LeafEntry),
    Internal(Internal::InternalEntry),
}

impl<Internal> Clone for NodeRc<Internal>
where
    Internal: InternalTrait,
{
    fn clone(&self) -> Self {
        match self {
            NodeRc::Leaf(t) => NodeRc::Leaf(t.clone()),
            NodeRc::Internal(t) => NodeRc::Internal(t.clone()),
        }
    }
}

impl<Internal> NodeRc<Internal>
where
    Internal: InternalTrait,
{
    /// Constructs a new empty of the same level.
    pub async fn new_empty(&self, context: &Internal::Context) -> Self {
        //TODO: This shouldn't have context
        match self {
            NodeRc::Internal(x) => NodeRc::Internal(
                Internal::InternalEntry::new(x.load(context).await.new_empty().await).await,
            ),
            NodeRc::Leaf(_) => {
                NodeRc::Leaf(Internal::LeafEntry::new(Internal::Leaf::empty()).await)
            }
        }
    }

    /// Removes elements from `self` and inserts these elements into `destination`. At most `len`
    /// elements will be removed. The actual number of elements decanted is returned. Elements are
    /// popped from `share_side` but pushed into the destination via `share_side.negate()`. Both
    /// self and destination should be at the same level in the tree.
    pub async fn share_children_with(
        &mut self,
        destination: &mut Self,
        share_side: Side,
        len: usize,
        context: &Internal::Context,
    ) -> usize {
        match self {
            NodeRc::Internal(ref mut origin_internal) => {
                let mut destination_internal = destination.internal_mut().load_mut(context).await;
                origin_internal
                    .load_mut(context)
                    .await
                    .share_children_with(&mut *destination_internal, share_side, len, context)
                    .await
            }
            NodeRc::Leaf(ref mut origin_leaf) => {
                let mut destination_leaf = destination.leaf_mut().load_mut(context).await;
                origin_leaf.load_mut(context).await.share_children_with(
                    &mut *destination_leaf,
                    share_side,
                    len,
                    context,
                )
            }
        }
    }

    /// Returns the size of the of node.
    pub async fn len(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRc::Leaf(x) => x.load(context).await.len(),
            NodeRc::Internal(x) => x.load(context).await.len().await,
        }
    }

    /// Returns the number of direct children of the node.
    pub async fn slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRc::Leaf(x) => x.load(context).await.len(),
            NodeRc::Internal(x) => x.load(context).await.slots().await,
        }
    }

    /// Returns whether the node is completely empty.
    pub async fn is_empty(&self, context: &Internal::Context) -> bool {
        match self {
            NodeRc::Leaf(x) => x.load(context).await.is_empty(),
            NodeRc::Internal(x) => x.load(context).await.is_empty().await,
        }
    }

    /// Returns whether the node is completely full.
    pub async fn is_full(&self, context: &Internal::Context) -> bool {
        match self {
            NodeRc::Leaf(x) => x.load(context).await.is_full(),
            NodeRc::Internal(x) => x.load(context).await.is_full().await,
        }
    }

    /// Returns the level of the node.
    pub async fn level(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRc::Leaf(_) => 0,
            NodeRc::Internal(x) => x.load(context).await.level().await,
        }
    }

    /// Tests whether the node is a leaf.
    pub async fn is_leaf(&self, context: &Internal::Context) -> bool {
        self.level(context).await == 0
    }

    /// Returns the element at the given position in the node.
    pub async fn get(
        &self,
        idx: usize,
        context: &Internal::Context,
    ) -> Option<*const <Internal::Leaf as LeafTrait>::Item> {
        match self {
            NodeRc::Leaf(x) => x.load(context).await.get(idx),
            NodeRc::Internal(x) => x.load(context).await.get(idx, context).await,
        }
    }

    /// Returns the element at the given position in the node.
    pub async fn get_mut(
        &mut self,
        idx: usize,
        context: &Internal::Context,
    ) -> Option<*mut <Internal::Leaf as LeafTrait>::Item> {
        match self {
            NodeRc::Leaf(ref mut x) => x.load_mut(context).await.get_mut(idx, context),
            NodeRc::Internal(ref mut x) => x.load_mut(context).await.get_mut(idx, context).await,
        }
    }

    /// Returns the element at the given position in the node.
    pub async fn get_mut_guarded(
        &mut self,
        idx: usize,
        context: &Internal::Context,
    ) -> Option<<Internal::Leaf as LeafTrait>::ItemMutGuard> {
        match self {
            NodeRc::Leaf(ref mut x) => x.load_mut(context).await.get_mut_guarded(idx, context),
            NodeRc::Internal(ref mut x) => {
                x.load_mut(context)
                    .await
                    .get_mut_guarded(idx, context)
                    .await
            }
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub async fn free_slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRc::Leaf(x) => x.load(context).await.free_space(),
            NodeRc::Internal(x) => x.load(context).await.free_slots().await,
        }
    }

    /// Consumes `self` and returns the node as a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a leaf node.
    pub fn leaf(self) -> Internal::LeafEntry {
        if let NodeRc::Leaf(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as a leaf node")
        }
    }

    /// Consumes `self` and returns the node as an internal node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not an internal node.
    pub fn internal(self) -> Internal::InternalEntry {
        if let NodeRc::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    /// Returns a reference to the node as a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a leaf node.
    pub fn leaf_ref(&self) -> &Internal::LeafEntry {
        if let NodeRc::Leaf(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as a leaf node")
        }
    }

    /// Returns a mutable reference to the node as a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a leaf node.
    pub fn leaf_mut(&mut self) -> &mut Internal::LeafEntry {
        if let NodeRc::Leaf(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as a leaf node")
        }
    }

    /// Returns a mutable reference to the node as an internal node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a internal node.
    pub fn internal_mut(&mut self) -> &mut Internal::InternalEntry {
        if let NodeRc::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    pub async fn borrow_node(&mut self, context: &Internal::Context) -> BorrowedNode<Internal> {
        match self {
            NodeRc::Internal(internal) => {
                BorrowedNode::Internal(internal.load_mut(context).await.borrow_node().await)
            }
            NodeRc::Leaf(leaf) => BorrowedNode::Leaf(leaf.load_mut(context).await.borrow_node()),
        }
    }

    pub async fn split_at_child(&mut self, idx: usize, context: &Internal::Context) -> Self {
        match self {
            NodeRc::Internal(internal) => NodeRc::Internal(
                Internal::InternalEntry::new(
                    internal
                        .load_mut(context)
                        .await
                        .split_at_child(idx, context)
                        .await,
                )
                .await,
            ),
            NodeRc::Leaf(leaf) => NodeRc::Leaf(
                Internal::LeafEntry::new(leaf.load_mut(context).await.split(idx, context)).await,
            ),
        }
    }

    pub async fn split_at_position(
        &mut self,
        position: usize,
        context: &Internal::Context,
    ) -> Self {
        match self {
            NodeRc::Internal(internal) => NodeRc::Internal(
                Internal::InternalEntry::new(
                    internal
                        .load_mut(context)
                        .await
                        .split_at_position(position, context)
                        .await,
                )
                .await,
            ),
            NodeRc::Leaf(leaf) => NodeRc::Leaf(
                Internal::LeafEntry::new(leaf.load_mut(context).await.split(position, context))
                    .await,
            ),
        }
    }

    /// Checks internal invariants of the node.
    #[allow(dead_code)]
    pub async fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        context: &Internal::Context,
    ) {
        match self {
            NodeRc::Leaf(x) => {
                x.load(context)
                    .await
                    .debug_check_invariants(reported_size, reported_level, context)
            }
            NodeRc::Internal(x) => {
                x.load(context)
                    .await
                    .debug_check_invariants(reported_size, reported_level, context)
                    .await
            }
        }
    }
}

impl<Internal> NodeRc<Internal>
where
    Internal: InternalTrait,
    <Internal::Leaf as LeafTrait>::Item: PartialEq,
{
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) async fn equal_iter_debug<'a>(
        &self,
        iter: &mut std::slice::Iter<'a, <Internal::Leaf as LeafTrait>::Item>,
        context: &Internal::Context,
    ) -> bool {
        match self {
            NodeRc::Internal(ref internal) => {
                internal
                    .load(context)
                    .await
                    .equal_iter_debug(iter, context)
                    .await
            }
            NodeRc::Leaf(ref leaf) => leaf.load(context).await.equal_iter_debug(iter, context),
        }
    }
}

#[derive(Clone, Debug)]
pub enum BorrowedNode<Internal>
where
    Internal: InternalTrait,
{
    Internal(Internal::Borrowed),
    Leaf(<Internal::Leaf as LeafTrait>::Borrowed),
}

impl<Internal> BorrowedNode<Internal>
where
    Internal: InternalTrait,
{
    pub async fn len(&self) -> usize {
        match self {
            BorrowedNode::Leaf(leaf) => leaf.len().await,
            BorrowedNode::Internal(internal) => internal.len().await,
        }
    }

    pub async fn level(&self) -> usize {
        match self {
            BorrowedNode::Leaf(_) => 0,
            BorrowedNode::Internal(internal) => internal.level().await,
        }
    }

    pub fn leaf(self) -> <Internal::Leaf as LeafTrait>::Borrowed {
        if let BorrowedNode::Leaf(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as a leaf node")
        }
    }

    /// Consumes `self` and returns the node as an internal node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not an internal node.
    pub fn internal(self) -> Internal::Borrowed {
        if let BorrowedNode::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    /// Returns a mutable reference to the node as a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a leaf node.
    pub fn leaf_mut(&mut self) -> &mut <Internal::Leaf as LeafTrait>::Borrowed {
        if let BorrowedNode::Leaf(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as a leaf node")
        }
    }

    /// Returns a mutable reference to the node as an internal node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a internal node.
    pub fn internal_mut(&mut self) -> &mut Internal::Borrowed {
        if let BorrowedNode::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    pub async fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        match self {
            BorrowedNode::Leaf(leaf) => {
                leaf.debug_check_invariants(reported_size, reported_level)
                    .await
            }
            BorrowedNode::Internal(internal) => {
                internal
                    .debug_check_invariants(reported_size, reported_level)
                    .await
            }
        }
    }
}

/// Represents an immutable reference to an arbitrary node in the tree.
#[derive(Debug)]
pub enum NodeRef<'a, Internal>
where
    Internal: InternalTrait,
{
    Leaf(&'a Internal::LeafEntry),
    Internal(&'a Internal::InternalEntry),
}

impl<'a, Internal> NodeRef<'a, Internal>
where
    Internal: InternalTrait,
{
    /// Returns the size of the of node.
    pub async fn len(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRef::Leaf(x) => x.load(context).await.len(),
            NodeRef::Internal(x) => x.load(context).await.len().await,
        }
    }

    /// Returns the number of direct children of the node.
    pub async fn slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRef::Leaf(x) => x.load(context).await.len(),
            NodeRef::Internal(x) => x.load(context).await.slots().await,
        }
    }

    /// Returns whether the node is completely empty.
    pub async fn is_empty(&self, context: &Internal::Context) -> bool {
        match self {
            NodeRef::Leaf(x) => x.load(context).await.is_empty(),
            NodeRef::Internal(x) => x.load(context).await.is_empty().await,
        }
    }

    /// Returns the level of the node.
    pub async fn level(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRef::Leaf(_) => 0,
            NodeRef::Internal(x) => x.load(context).await.level().await,
        }
    }

    /// Tests whether the node is a leaf.
    pub async fn is_leaf(&self, context: &Internal::Context) -> bool {
        self.level(context).await == 0
    }

    /// Returns the element at the given position in the node.
    pub async fn get(
        &self,
        idx: usize,
        context: &Internal::Context,
    ) -> Option<*const <Internal::Leaf as LeafTrait>::Item> {
        match self {
            NodeRef::Leaf(x) => x.load(context).await.get(idx),
            NodeRef::Internal(x) => x.load(context).await.get(idx, context).await,
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub async fn free_slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRef::Leaf(x) => x.load(context).await.free_space(),
            NodeRef::Internal(x) => x.load(context).await.free_slots().await,
        }
    }

    /// Consumes `self` and returns the node as a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a leaf node.
    pub fn leaf(self) -> &'a Internal::LeafEntry {
        if let NodeRef::Leaf(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as a leaf node")
        }
    }

    /// Consumes `self` and returns the node as an internal node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not an internal node.
    pub fn internal(self) -> &'a Internal::InternalEntry {
        if let NodeRef::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    /// Checks internal invariants of the node.
    #[allow(dead_code)]
    pub async fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        context: &Internal::Context,
    ) {
        match self {
            NodeRef::Leaf(x) => {
                x.load(context)
                    .await
                    .debug_check_invariants(reported_size, reported_level, context)
            }
            NodeRef::Internal(x) => {
                x.load(context)
                    .await
                    .debug_check_invariants(reported_size, reported_level, context)
                    .await
            }
        }
    }
}

impl<'a, Internal> NodeRef<'a, Internal>
where
    Internal: InternalTrait,
    <Internal::Leaf as LeafTrait>::Item: PartialEq,
{
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) async fn equal_iter_debug<'b>(
        &self,
        iter: &mut std::slice::Iter<'b, <Internal::Leaf as LeafTrait>::Item>,
        context: &Internal::Context,
    ) -> bool {
        match self {
            NodeRef::Internal(ref internal) => {
                internal
                    .load(context)
                    .await
                    .equal_iter_debug(iter, context)
                    .await
            }
            NodeRef::Leaf(ref leaf) => leaf.load(context).await.equal_iter_debug(iter, context),
        }
    }
}

/// Represents an immutable reference to an arbitrary node in the tree.
#[derive(Debug)]
pub enum NodeMut<'a, Internal>
where
    Internal: InternalTrait,
{
    Leaf(&'a mut Internal::LeafEntry),
    Internal(&'a mut Internal::InternalEntry),
}

impl<'a, Internal> NodeMut<'a, Internal>
where
    Internal: InternalTrait,
{
    /// Returns the size of the of node.
    pub async fn len(&self, context: &Internal::Context) -> usize {
        match self {
            NodeMut::Leaf(x) => x.load(context).await.len(),
            NodeMut::Internal(x) => x.load(context).await.len().await,
        }
    }

    /// Returns the number of direct children of the node.
    pub async fn slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeMut::Leaf(x) => x.load(context).await.len(),
            NodeMut::Internal(x) => x.load(context).await.slots().await,
        }
    }

    /// Returns whether the node is completely empty.
    pub async fn is_empty(&self, context: &Internal::Context) -> bool {
        match self {
            NodeMut::Leaf(x) => x.load(context).await.is_empty(),
            NodeMut::Internal(x) => x.load(context).await.is_empty().await,
        }
    }

    /// Returns the level of the node.
    pub async fn level(&self, context: &Internal::Context) -> usize {
        match self {
            NodeMut::Leaf(_) => 0,
            NodeMut::Internal(x) => x.load(context).await.level().await,
        }
    }

    /// Tests whether the node is a leaf.
    pub async fn is_leaf(&self, context: &Internal::Context) -> bool {
        self.level(context).await == 0
    }

    /// Returns the element at the given position in the node.
    pub async fn get(
        &self,
        idx: usize,
        context: &Internal::Context,
    ) -> Option<*const <Internal::Leaf as LeafTrait>::Item> {
        match self {
            NodeMut::Leaf(x) => x.load(context).await.get(idx),
            NodeMut::Internal(x) => x.load(context).await.get(idx, context).await,
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub async fn free_slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeMut::Leaf(x) => x.load(context).await.free_space(),
            NodeMut::Internal(x) => x.load(context).await.free_slots().await,
        }
    }

    /// Consumes `self` and returns the node as a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a leaf node.
    pub fn leaf(self) -> &'a Internal::LeafEntry {
        if let NodeMut::Leaf(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as a leaf node")
        }
    }

    /// Consumes `self` and returns the node as an internal node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not an internal node.
    pub fn internal(self) -> &'a Internal::InternalEntry {
        if let NodeMut::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    /// derp
    pub async fn borrow_node(self, context: &Internal::Context) -> BorrowedNode<Internal> {
        match self {
            NodeMut::Internal(internal) => {
                BorrowedNode::Internal(internal.load_mut(context).await.borrow_node().await)
            }
            NodeMut::Leaf(leaf) => BorrowedNode::Leaf(leaf.load_mut(context).await.borrow_node()),
        }
    }

    /// Checks internal invariants of the node.
    #[allow(dead_code)]
    pub async fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        context: &Internal::Context,
    ) {
        match self {
            NodeMut::Leaf(x) => {
                x.load(context)
                    .await
                    .debug_check_invariants(reported_size, reported_level, context)
            }
            NodeMut::Internal(x) => {
                x.load(context)
                    .await
                    .debug_check_invariants(reported_size, reported_level, context)
                    .await
            }
        }
    }
}

impl<'a, Internal> NodeMut<'a, Internal>
where
    Internal: InternalTrait,
    <Internal::Leaf as LeafTrait>::Item: Clone + std::fmt::Debug + PartialEq,
{
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) async fn equal_iter_debug<'b>(
        &self,
        iter: &mut std::slice::Iter<'b, <Internal::Leaf as LeafTrait>::Item>,
        context: &Internal::Context,
    ) -> bool {
        match self {
            NodeMut::Internal(ref internal) => {
                internal
                    .load(context)
                    .await
                    .equal_iter_debug(iter, context)
                    .await
            }
            NodeMut::Leaf(ref leaf) => leaf.load(context).await.equal_iter_debug(iter, context),
        }
    }
}
