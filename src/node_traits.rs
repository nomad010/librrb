use crate::Side;
use archery::{SharedPointer, SharedPointerKind};
use std::fmt::Debug;
use std::ops::Range;

pub trait Entry: Clone + std::fmt::Debug {
    type Item: Clone + std::fmt::Debug;

    fn new(item: Self::Item) -> Self;

    fn load(&self) -> &Self::Item;

    fn load_mut(&mut self) -> &mut Self::Item;
}

impl<I: Clone + std::fmt::Debug, P: SharedPointerKind> Entry for SharedPointer<I, P> {
    type Item = I;

    fn new(item: Self::Item) -> Self {
        SharedPointer::new(item)
    }

    fn load(&self) -> &Self::Item {
        &self
    }

    fn load_mut(&mut self) -> &mut Self::Item {
        SharedPointer::make_mut(self)
    }
}

pub trait BorrowedLeafTrait {
    type Item;

    fn split_at(&mut self, idx: usize) -> Self;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn from_same_source(&self, other: &Self) -> bool;

    fn combine(&mut self, other: Self);

    fn get_mut(&mut self, idx: usize) -> Option<&mut Self::Item>;

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        debug_assert_eq!(reported_size, self.len());
        debug_assert_eq!(reported_level, 0);
    }
}

pub trait LeafTrait: Clone + std::fmt::Debug {
    type Item: Clone + std::fmt::Debug;
    type Borrowed: BorrowedLeafTrait<Item = Self::Item> + std::fmt::Debug;

    /// Constructs a new empty leaf.
    fn empty() -> Self;

    /// Constructs a new leaf with a single item.
    fn with_item(item: Self::Item) -> Self;

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
    fn get(&self, position: usize) -> Option<&Self::Item>;

    /// Gets a reference to the item at the front of the leaf.
    fn front(&self) -> Option<&Self::Item> {
        self.get(0)
    }

    /// Gets a reference to the item at the back of the leaf.
    fn back(&self) -> Option<&Self::Item> {
        self.get(self.len().saturating_sub(1))
    }

    /// Gets a mutable reference to the item at requested position if it exists.
    fn get_mut(&mut self, position: usize) -> Option<&mut Self::Item>;

    /// Gets a mutable reference to the item at the front of the leaf.
    fn front_mut(&mut self) -> Option<&mut Self::Item> {
        self.get_mut(0)
    }

    /// Gets a reference to the item at the back of the leaf.
    fn back_mut(&mut self) -> Option<&mut Self::Item> {
        self.get_mut(self.len().saturating_sub(1))
    }

    /// Attempts to push an element to a side of the buffer.
    ///
    /// Panics if the buffer is full.
    fn push(&mut self, side: Side, item: Self::Item);

    /// Attempts to pop an element from a side of the buffer.
    ///
    /// Panics if the buffer is empty.
    fn pop(&mut self, side: Side) -> Self::Item;

    /// Removes elements from `self` and inserts these elements into `destination`. At most `len`
    /// elements will be removed. The actual number of elements decanted is returned. Elements are
    /// popped from `share_side` but pushed into the destination via `share_side.negate()`
    fn share_children_with(
        &mut self,
        destination: &mut Self,
        share_side: Side,
        len: usize,
    ) -> usize;

    fn borrow_node(&mut self) -> Self::Borrowed;

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        debug_assert_eq!(reported_size, self.len());
        debug_assert_eq!(reported_level, 0);
    }

    fn split(&mut self, idx: usize) -> Self;

    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    fn equal_iter_debug<'a>(&self, iter: &mut std::slice::Iter<'a, Self::Item>) -> bool
    where
        Self::Item: PartialEq;
}

pub trait BorrowedInternalTrait<P: SharedPointerKind, Leaf: LeafTrait> {
    type InternalChild: InternalTrait<P, Leaf>;

    fn len(&self) -> usize;

    fn slots(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn range(&self) -> Range<usize>;

    fn level(&self) -> usize;

    fn from_same_source(&self, other: &Self) -> bool;

    fn combine(&mut self, other: Self);

    fn split_at_child(&mut self, index: usize) -> Self;

    fn split_at_position(&mut self, position: usize) -> (usize, Self);

    /// Returns a mutable reference to the Rc of the child node at the given slot in this node.
    fn get_child_mut_at_slot(
        &mut self,
        idx: usize,
    ) -> Option<(NodeMut<P, Self::InternalChild, Leaf>, Range<usize>)>;

    fn get_child_mut_at_side(
        &mut self,
        side: Side,
    ) -> Option<(NodeMut<P, Self::InternalChild, Leaf>, Range<usize>)> {
        match side {
            Side::Front => self.get_child_mut_at_slot(0),
            Side::Back => self.get_child_mut_at_slot(self.slots().saturating_sub(1)),
        }
    }

    /// Returns a mutable reference to the Rc of the child node that covers the leaf position in this node.
    fn get_child_mut_for_position(
        &mut self,
        position: usize,
    ) -> Option<(NodeMut<P, Self::InternalChild, Leaf>, Range<usize>)>;

    /// Logically pops a child from the node. The child is then returned.
    fn pop_child(&mut self, side: Side) -> Option<BorrowedNode<P, Self::InternalChild, Leaf>>;

    /// Undoes the popping that has occurred via a call to `pop_child`.
    fn unpop_child(&mut self, side: Side);

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        debug_assert_eq!(reported_size, self.len());
        debug_assert_eq!(reported_level, self.level());
    }
}

pub trait InternalTrait<P: SharedPointerKind, Leaf: LeafTrait>: Clone + std::fmt::Debug {
    type Borrowed: BorrowedInternalTrait<P, Leaf> + std::fmt::Debug;

    type LeafEntry: Entry<Item = Leaf>;
    type InternalEntry: Entry<Item = Self>;

    /// Constructs a new empty internal node that is at the given level in the tree.
    fn empty_internal(level: usize) -> Self;

    /// Constructs a new internal node of the same level, but with no children.
    fn new_empty(&self) -> Self;

    /// Removes elements from `self` and inserts these elements into `destination`. At most `len`
    /// elements will be removed. The actual number of elements decanted is returned. Elements are
    /// popped from `share_side` but pushed into the destination via `share_side.negate()`. Both
    /// self and destination should be at the same level in the tree.
    fn share_children_with(
        &mut self,
        destination: &mut Self,
        share_side: Side,
        len: usize,
    ) -> usize;

    /// Packs the children of the to the left of the node so that all but the last is dense.
    fn pack_children(&mut self);

    /// Returns a reference to the element at the given index in the tree.
    fn get(&self, idx: usize) -> Option<&Leaf::Item>;

    /// Returns a mutable reference to the element at the given index in the tree.
    fn get_mut(&mut self, idx: usize) -> Option<&mut Leaf::Item>;

    /// Removes and returns the node at the given side of this node.
    /// # Panics
    // This should panic if the node is empty.
    fn pop_child(&mut self, side: Side) -> NodeRc<P, Self, Leaf>;

    /// Adds a node to the given side of this node.
    /// # Panics
    // This should panic if the node does not have a slot free.
    fn push_child(&mut self, side: Side, node: NodeRc<P, Self, Leaf>);

    fn borrow_node(&mut self) -> Self::Borrowed;

    /// Returns the level the node is at in the tree.
    fn level(&self) -> usize;

    /// Returns the size(number of elements hanging off) of the node.
    fn len(&self) -> usize;

    /// Returns whether the node is empty. This should almost never be the case.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of direct children of the node.
    fn slots(&self) -> usize;

    /// Returns the number of direct children that could be inserted into the node.
    fn free_slots(&self) -> usize;

    /// Returns whether no more children can be inserted into the node.
    fn is_full(&self) -> bool {
        self.free_slots() == 0
    }

    /// Returns a reference to the Rc of the child node at the given slot in this node.
    fn get_child_ref_at_slot(&self, idx: usize) -> Option<(NodeRef<P, Self, Leaf>, Range<usize>)>;

    fn get_child_ref_at_side(&self, side: Side) -> Option<(NodeRef<P, Self, Leaf>, Range<usize>)> {
        match side {
            Side::Front => self.get_child_ref_at_slot(0),
            Side::Back => self.get_child_ref_at_slot(self.slots().saturating_sub(1)),
        }
    }

    /// Returns a reference to the Rc of the child node that covers the leaf position in this node.
    fn get_child_ref_for_position(
        &self,
        position: usize,
    ) -> Option<(NodeRef<P, Self, Leaf>, Range<usize>)>;

    /// Returns a mutable reference to the Rc of the child node at the given slot in this node.
    fn get_child_mut_at_slot(
        &mut self,
        idx: usize,
    ) -> Option<(NodeMut<P, Self, Leaf>, Range<usize>)>;

    fn get_child_mut_at_side(
        &mut self,
        side: Side,
    ) -> Option<(NodeMut<P, Self, Leaf>, Range<usize>)> {
        match side {
            Side::Front => self.get_child_mut_at_slot(0),
            Side::Back => self.get_child_mut_at_slot(self.slots().saturating_sub(1)),
        }
    }

    /// Returns a mutable reference of the Rc of the child node that covers the leaf position in this node.
    fn get_child_mut_for_position(
        &mut self,
        position: usize,
    ) -> Option<(NodeMut<P, Self, Leaf>, Range<usize>)>;

    /// Splits the node into two at the given slot index.
    fn split_at_child(&mut self, idx: usize) -> Self;

    /// Splits the node into two at the given position.
    fn split_at_position(&mut self, position: usize) -> Self;

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        debug_assert_eq!(reported_size, self.len());
        debug_assert_ne!(reported_level, self.level());
    }

    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    fn equal_iter_debug<'a>(&self, iter: &mut std::slice::Iter<'a, Leaf::Item>) -> bool
    where
        Leaf::Item: PartialEq;
}

/// Represents an arbitrary node in the tree.
#[derive(Debug)]
pub enum NodeRc<P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    Leaf(Internal::LeafEntry),
    Internal(Internal::InternalEntry),
}

impl<P, Internal, Leaf> Clone for NodeRc<P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    fn clone(&self) -> Self {
        match self {
            NodeRc::Leaf(t) => NodeRc::Leaf(t.clone()),
            NodeRc::Internal(t) => NodeRc::Internal(t.clone()),
        }
    }
}

impl<P, Internal, Leaf> NodeRc<P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    /// Constructs a new empty of the same level.
    pub fn new_empty(&self) -> Self {
        match self {
            NodeRc::Internal(x) => {
                NodeRc::Internal(Internal::InternalEntry::new(x.load().new_empty()))
            }
            NodeRc::Leaf(_) => NodeRc::Leaf(Internal::LeafEntry::new(Leaf::empty())),
        }
    }

    /// Removes elements from `self` and inserts these elements into `destination`. At most `len`
    /// elements will be removed. The actual number of elements decanted is returned. Elements are
    /// popped from `share_side` but pushed into the destination via `share_side.negate()`. Both
    /// self and destination should be at the same level in the tree.
    pub fn share_children_with(
        &mut self,
        destination: &mut Self,
        share_side: Side,
        len: usize,
    ) -> usize {
        match self {
            NodeRc::Internal(ref mut origin_internal) => {
                let destination_internal = destination.internal_mut().load_mut();
                origin_internal.load_mut().share_children_with(
                    destination_internal,
                    share_side,
                    len,
                )
            }
            NodeRc::Leaf(ref mut origin_leaf) => {
                let destination_leaf = destination.leaf_mut().load_mut();
                origin_leaf
                    .load_mut()
                    .share_children_with(destination_leaf, share_side, len)
            }
        }
    }

    /// Returns the size of the of node.
    pub fn len(&self) -> usize {
        match self {
            NodeRc::Leaf(x) => x.load().len(),
            NodeRc::Internal(x) => x.load().len(),
        }
    }

    /// Returns the number of direct children of the node.
    pub fn slots(&self) -> usize {
        match self {
            NodeRc::Leaf(x) => x.load().len(),
            NodeRc::Internal(x) => x.load().slots(),
        }
    }

    /// Returns whether the node is completely empty.
    pub fn is_empty(&self) -> bool {
        match self {
            NodeRc::Leaf(x) => x.load().is_empty(),
            NodeRc::Internal(x) => x.load().is_empty(),
        }
    }

    /// Returns whether the node is completely full.
    pub fn is_full(&self) -> bool {
        match self {
            NodeRc::Leaf(x) => x.load().is_full(),
            NodeRc::Internal(x) => x.load().is_full(),
        }
    }

    /// Returns the level of the node.
    pub fn level(&self) -> usize {
        match self {
            NodeRc::Leaf(_) => 0,
            NodeRc::Internal(x) => x.load().level(),
        }
    }

    /// Tests whether the node is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.level() == 0
    }

    /// Returns the element at the given position in the node.
    pub fn get(&self, idx: usize) -> Option<&Leaf::Item> {
        match self {
            NodeRc::Leaf(x) => x.load().get(idx),
            NodeRc::Internal(x) => x.load().get(idx),
        }
    }

    /// Returns the element at the given position in the node.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Leaf::Item> {
        match self {
            NodeRc::Leaf(ref mut x) => x.load_mut().get_mut(idx),
            NodeRc::Internal(ref mut x) => x.load_mut().get_mut(idx),
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub fn free_slots(&self) -> usize {
        match self {
            NodeRc::Leaf(x) => x.load().free_space(),
            NodeRc::Internal(x) => x.load().free_slots(),
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

    pub fn borrow_node(&mut self) -> BorrowedNode<P, Internal, Leaf> {
        match self {
            NodeRc::Internal(internal) => BorrowedNode::Internal(internal.load_mut().borrow_node()),
            NodeRc::Leaf(leaf) => BorrowedNode::Leaf(leaf.load_mut().borrow_node()),
        }
    }

    pub fn split_at_child(&mut self, idx: usize) -> Self {
        match self {
            NodeRc::Internal(internal) => NodeRc::Internal(Internal::InternalEntry::new(
                internal.load_mut().split_at_child(idx),
            )),
            NodeRc::Leaf(leaf) => {
                NodeRc::Leaf(Internal::LeafEntry::new(leaf.load_mut().split(idx)))
            }
        }
    }

    pub fn split_at_position(&mut self, position: usize) -> Self {
        match self {
            NodeRc::Internal(internal) => NodeRc::Internal(Internal::InternalEntry::new(
                internal.load_mut().split_at_position(position),
            )),
            NodeRc::Leaf(leaf) => {
                NodeRc::Leaf(Internal::LeafEntry::new(leaf.load_mut().split(position)))
            }
        }
    }

    /// Checks internal invariants of the node.
    #[allow(dead_code)]
    pub fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        match self {
            NodeRc::Leaf(x) => x
                .load()
                .debug_check_invariants(reported_size, reported_level),
            NodeRc::Internal(x) => x
                .load()
                .debug_check_invariants(reported_size, reported_level),
        }
    }
}

impl<P, Internal, Leaf> NodeRc<P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
    Leaf::Item: PartialEq,
{
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) fn equal_iter_debug<'a>(&self, iter: &mut std::slice::Iter<'a, Leaf::Item>) -> bool {
        match self {
            NodeRc::Internal(ref internal) => internal.load().equal_iter_debug(iter),
            NodeRc::Leaf(ref leaf) => leaf.load().equal_iter_debug(iter),
        }
    }
}

#[derive(Debug)]
pub enum BorrowedNode<P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    Internal(Internal::Borrowed),
    Leaf(Leaf::Borrowed),
}

impl<P, Internal, Leaf> BorrowedNode<P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    pub fn len(&self) -> usize {
        match self {
            BorrowedNode::Leaf(leaf) => leaf.len(),
            BorrowedNode::Internal(internal) => internal.len(),
        }
    }

    pub fn level(&self) -> usize {
        match self {
            BorrowedNode::Leaf(_) => 0,
            BorrowedNode::Internal(internal) => internal.level(),
        }
    }

    pub fn leaf(self) -> Leaf::Borrowed {
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
    pub fn leaf_mut(&mut self) -> &mut Leaf::Borrowed {
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

    pub fn from_same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (BorrowedNode::Internal(node), BorrowedNode::Internal(other)) => {
                node.from_same_source(other)
            }
            (BorrowedNode::Leaf(node), BorrowedNode::Leaf(other)) => node.from_same_source(other),
            _ => false,
        }
    }

    pub fn combine(&mut self, other: Self) {
        match self {
            BorrowedNode::Leaf(leaf) => leaf.combine(other.leaf()),
            BorrowedNode::Internal(internal) => internal.combine(other.internal()),
        }
    }

    pub fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        match self {
            BorrowedNode::Leaf(leaf) => leaf.debug_check_invariants(reported_size, reported_level),
            BorrowedNode::Internal(internal) => {
                internal.debug_check_invariants(reported_size, reported_level)
            }
        }
    }
}

/// Represents an immutable reference to an arbitrary node in the tree.
#[derive(Debug)]
pub enum NodeRef<'a, P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    Leaf(&'a Internal::LeafEntry),
    Internal(&'a Internal::InternalEntry),
}

impl<'a, P, Internal, Leaf> NodeRef<'a, P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    /// Returns the size of the of node.
    pub fn len(&self) -> usize {
        match self {
            NodeRef::Leaf(x) => x.load().len(),
            NodeRef::Internal(x) => x.load().len(),
        }
    }

    /// Returns the number of direct children of the node.
    pub fn slots(&self) -> usize {
        match self {
            NodeRef::Leaf(x) => x.load().len(),
            NodeRef::Internal(x) => x.load().slots(),
        }
    }

    /// Returns whether the node is completely empty.
    pub fn is_empty(&self) -> bool {
        match self {
            NodeRef::Leaf(x) => x.load().is_empty(),
            NodeRef::Internal(x) => x.load().is_empty(),
        }
    }

    /// Returns the level of the node.
    pub fn level(&self) -> usize {
        match self {
            NodeRef::Leaf(_) => 0,
            NodeRef::Internal(x) => x.load().level(),
        }
    }

    /// Tests whether the node is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.level() == 0
    }

    /// Returns the element at the given position in the node.
    pub fn get(&self, idx: usize) -> Option<&Leaf::Item> {
        match self {
            NodeRef::Leaf(x) => x.load().get(idx),
            NodeRef::Internal(x) => x.load().get(idx),
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub fn free_slots(&self) -> usize {
        match self {
            NodeRef::Leaf(x) => x.load().free_space(),
            NodeRef::Internal(x) => x.load().free_slots(),
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
    pub fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        match self {
            NodeRef::Leaf(x) => x
                .load()
                .debug_check_invariants(reported_size, reported_level),
            NodeRef::Internal(x) => x
                .load()
                .debug_check_invariants(reported_size, reported_level),
        }
    }
}

impl<'a, P, Internal, Leaf> NodeRef<'a, P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
    Leaf::Item: PartialEq,
{
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) fn equal_iter_debug<'b>(&self, iter: &mut std::slice::Iter<'b, Leaf::Item>) -> bool {
        match self {
            NodeRef::Internal(ref internal) => internal.load().equal_iter_debug(iter),
            NodeRef::Leaf(ref leaf) => leaf.load().equal_iter_debug(iter),
        }
    }
}

/// Represents an immutable reference to an arbitrary node in the tree.
#[derive(Debug)]
pub enum NodeMut<'a, P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    Leaf(&'a mut SharedPointer<Leaf, P>),
    Internal(&'a mut SharedPointer<Internal, P>),
}

impl<'a, P, Internal, Leaf> NodeMut<'a, P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
{
    /// Returns the size of the of node.
    pub fn len(&self) -> usize {
        match self {
            NodeMut::Leaf(x) => x.len(),
            NodeMut::Internal(x) => x.len(),
        }
    }

    /// Returns the number of direct children of the node.
    pub fn slots(&self) -> usize {
        match self {
            NodeMut::Leaf(x) => x.len(),
            NodeMut::Internal(x) => x.slots(),
        }
    }

    /// Returns whether the node is completely empty.
    pub fn is_empty(&self) -> bool {
        match self {
            NodeMut::Leaf(x) => x.is_empty(),
            NodeMut::Internal(x) => x.is_empty(),
        }
    }

    /// Returns the level of the node.
    pub fn level(&self) -> usize {
        match self {
            NodeMut::Leaf(_) => 0,
            NodeMut::Internal(x) => x.level(),
        }
    }

    /// Tests whether the node is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.level() == 0
    }

    /// Returns the element at the given position in the node.
    pub fn get(&self, idx: usize) -> Option<&<Leaf as LeafTrait>::Item> {
        match self {
            NodeMut::Leaf(x) => x.get(idx),
            NodeMut::Internal(x) => x.get(idx),
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub fn free_slots(&self) -> usize {
        match self {
            NodeMut::Leaf(x) => x.free_space(),
            NodeMut::Internal(x) => x.free_slots(),
        }
    }

    /// Consumes `self` and returns the node as a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a leaf node.
    pub fn leaf(self) -> &'a SharedPointer<Leaf, P> {
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
    pub fn internal(self) -> &'a SharedPointer<Internal, P> {
        if let NodeMut::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    /// derp
    pub fn borrow_node(self) -> BorrowedNode<P, Internal, Leaf> {
        match self {
            NodeMut::Internal(internal) => {
                BorrowedNode::Internal(SharedPointer::make_mut(internal).borrow_node())
            }
            NodeMut::Leaf(leaf) => BorrowedNode::Leaf(SharedPointer::make_mut(leaf).borrow_node()),
        }
    }

    /// Checks internal invariants of the node.
    #[allow(dead_code)]
    pub fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        match self {
            NodeMut::Leaf(x) => x.debug_check_invariants(reported_size, reported_level),
            NodeMut::Internal(x) => x.debug_check_invariants(reported_size, reported_level),
        }
    }
}

impl<'a, P, Internal, Leaf> NodeMut<'a, P, Internal, Leaf>
where
    P: SharedPointerKind,
    Internal: InternalTrait<P, Leaf>,
    Leaf: LeafTrait,
    Leaf::Item: Clone + std::fmt::Debug + PartialEq,
{
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) fn equal_iter_debug<'b>(&self, iter: &mut std::slice::Iter<'b, Leaf::Item>) -> bool {
        match self {
            NodeMut::Internal(ref internal) => internal.equal_iter_debug(iter),
            NodeMut::Leaf(ref leaf) => leaf.equal_iter_debug(iter),
        }
    }
}
