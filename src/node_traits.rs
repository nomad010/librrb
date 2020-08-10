use crate::Side;
use std::fmt::Debug;
use std::ops::Range;

pub trait Entry: Clone + std::fmt::Debug {
    type Item: Clone + std::fmt::Debug;
    type Context: Clone + std::fmt::Debug + Default;

    fn new(item: Self::Item) -> Self;

    fn load(&self, context: &Self::Context) -> &Self::Item;

    fn load_mut(&mut self, context: &Self::Context) -> &mut Self::Item;
}

pub trait BorrowedLeafTrait {
    type Context: Clone + std::fmt::Debug + Default;
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
    type Context: Clone + std::fmt::Debug + Default;
    type Borrowed: BorrowedLeafTrait<Item = Self::Item, Context = Self::Context> + std::fmt::Debug;

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
    fn get_mut(&mut self, position: usize, context: &Self::Context) -> Option<&mut Self::Item>;

    /// Gets a mutable reference to the item at the front of the leaf.
    fn front_mut(&mut self, context: &Self::Context) -> Option<&mut Self::Item> {
        self.get_mut(0, context)
    }

    /// Gets a reference to the item at the back of the leaf.
    fn back_mut(&mut self, context: &Self::Context) -> Option<&mut Self::Item> {
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

pub trait BorrowedInternalTrait<
    Leaf: LeafTrait<Context = <Self::InternalChild as InternalTrait<Leaf>>::Context>,
>
{
    type InternalChild: InternalTrait<Leaf>;

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
    ) -> Option<(NodeMut<Self::InternalChild, Leaf>, Range<usize>)>;

    fn get_child_mut_at_side(
        &mut self,
        side: Side,
    ) -> Option<(NodeMut<Self::InternalChild, Leaf>, Range<usize>)> {
        match side {
            Side::Front => self.get_child_mut_at_slot(0),
            Side::Back => self.get_child_mut_at_slot(self.slots().saturating_sub(1)),
        }
    }

    /// Returns a mutable reference to the Rc of the child node that covers the leaf position in this node.
    fn get_child_mut_for_position(
        &mut self,
        position: usize,
    ) -> Option<(NodeMut<Self::InternalChild, Leaf>, Range<usize>)>;

    /// Logically pops a child from the node. The child is then returned.
    fn pop_child(
        &mut self,
        side: Side,
        context: &Leaf::Context,
    ) -> Option<BorrowedNode<Self::InternalChild, Leaf>>;

    /// Undoes the popping that has occurred via a call to `pop_child`.
    fn unpop_child(&mut self, side: Side);

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    fn debug_check_invariants(&self, reported_size: usize, reported_level: usize) {
        debug_assert_eq!(reported_size, self.len());
        debug_assert_eq!(reported_level, self.level());
    }
}

pub trait InternalTrait<Leaf: LeafTrait<Context = <Self as InternalTrait<Leaf>>::Context>>:
    Clone + std::fmt::Debug
{
    type Context: Clone + std::fmt::Debug + Default;
    type Borrowed: BorrowedInternalTrait<Leaf, InternalChild = Self> + std::fmt::Debug;

    type LeafEntry: Entry<Item = Leaf, Context = Self::Context>;
    type InternalEntry: Entry<Item = Self, Context = Self::Context>;

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
        context: &Self::Context,
    ) -> usize;

    /// Packs the children of the to the left of the node so that all but the last is dense.
    fn pack_children(&mut self, context: &Self::Context);

    /// Returns a reference to the element at the given index in the tree.
    fn get(&self, idx: usize) -> Option<&Leaf::Item>;

    /// Returns a mutable reference to the element at the given index in the tree.
    fn get_mut(&mut self, idx: usize, context: &Self::Context) -> Option<&mut Leaf::Item>;

    /// Removes and returns the node at the given side of this node.
    /// # Panics
    // This should panic if the node is empty.
    fn pop_child(&mut self, side: Side, context: &Self::Context) -> NodeRc<Self, Leaf>;

    /// Adds a node to the given side of this node.
    /// # Panics
    // This should panic if the node does not have a slot free.
    fn push_child(&mut self, side: Side, node: NodeRc<Self, Leaf>, context: &Self::Context);

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
    fn get_child_ref_at_slot(&self, idx: usize) -> Option<(NodeRef<Self, Leaf>, Range<usize>)>;

    fn get_child_ref_at_side(&self, side: Side) -> Option<(NodeRef<Self, Leaf>, Range<usize>)> {
        match side {
            Side::Front => self.get_child_ref_at_slot(0),
            Side::Back => self.get_child_ref_at_slot(self.slots().saturating_sub(1)),
        }
    }

    /// Returns a reference to the Rc of the child node that covers the leaf position in this node.
    fn get_child_ref_for_position(
        &self,
        position: usize,
    ) -> Option<(NodeRef<Self, Leaf>, Range<usize>)>;

    /// Returns a mutable reference to the Rc of the child node at the given slot in this node.
    fn get_child_mut_at_slot(&mut self, idx: usize) -> Option<(NodeMut<Self, Leaf>, Range<usize>)>;

    fn get_child_mut_at_side(&mut self, side: Side) -> Option<(NodeMut<Self, Leaf>, Range<usize>)> {
        match side {
            Side::Front => self.get_child_mut_at_slot(0),
            Side::Back => self.get_child_mut_at_slot(self.slots().saturating_sub(1)),
        }
    }

    /// Returns a mutable reference of the Rc of the child node that covers the leaf position in this node.
    fn get_child_mut_for_position(
        &mut self,
        position: usize,
    ) -> Option<(NodeMut<Self, Leaf>, Range<usize>)>;

    /// Splits the node into two at the given slot index.
    fn split_at_child(&mut self, idx: usize, context: &Self::Context) -> Self;

    /// Splits the node into two at the given position.
    fn split_at_position(&mut self, position: usize, context: &Self::Context) -> Self;

    /// Checks the invariants that this node may hold if any.
    #[allow(dead_code)]
    fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        _context: &Self::Context,
    ) {
        debug_assert_eq!(reported_size, self.len());
        debug_assert_ne!(reported_level, self.level());
    }

    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    fn equal_iter_debug<'a>(
        &self,
        iter: &mut std::slice::Iter<'a, Leaf::Item>,
        context: &Self::Context,
    ) -> bool
    where
        Leaf::Item: PartialEq;
}

/// Represents an arbitrary node in the tree.
#[derive(Debug)]
pub enum NodeRc<Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
{
    Leaf(Internal::LeafEntry),
    Internal(Internal::InternalEntry),
}

impl<Internal, Leaf> Clone for NodeRc<Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
{
    fn clone(&self) -> Self {
        match self {
            NodeRc::Leaf(t) => NodeRc::Leaf(t.clone()),
            NodeRc::Internal(t) => NodeRc::Internal(t.clone()),
        }
    }
}

impl<Internal, Leaf> NodeRc<Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
{
    /// Constructs a new empty of the same level.
    pub fn new_empty(&self, context: &Internal::Context) -> Self {
        //TODO: This shouldn't have context
        match self {
            NodeRc::Internal(x) => {
                NodeRc::Internal(Internal::InternalEntry::new(x.load(context).new_empty()))
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
        context: &Internal::Context,
    ) -> usize {
        match self {
            NodeRc::Internal(ref mut origin_internal) => {
                let destination_internal = destination.internal_mut().load_mut(context);
                origin_internal.load_mut(context).share_children_with(
                    destination_internal,
                    share_side,
                    len,
                    context,
                )
            }
            NodeRc::Leaf(ref mut origin_leaf) => {
                let destination_leaf = destination.leaf_mut().load_mut(context);
                origin_leaf.load_mut(context).share_children_with(
                    destination_leaf,
                    share_side,
                    len,
                    context,
                )
            }
        }
    }

    /// Returns the size of the of node.
    pub fn len(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRc::Leaf(x) => x.load(context).len(),
            NodeRc::Internal(x) => x.load(context).len(),
        }
    }

    /// Returns the number of direct children of the node.
    pub fn slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRc::Leaf(x) => x.load(context).len(),
            NodeRc::Internal(x) => x.load(context).slots(),
        }
    }

    /// Returns whether the node is completely empty.
    pub fn is_empty(&self, context: &Internal::Context) -> bool {
        match self {
            NodeRc::Leaf(x) => x.load(context).is_empty(),
            NodeRc::Internal(x) => x.load(context).is_empty(),
        }
    }

    /// Returns whether the node is completely full.
    pub fn is_full(&self, context: &Internal::Context) -> bool {
        match self {
            NodeRc::Leaf(x) => x.load(context).is_full(),
            NodeRc::Internal(x) => x.load(context).is_full(),
        }
    }

    /// Returns the level of the node.
    pub fn level(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRc::Leaf(_) => 0,
            NodeRc::Internal(x) => x.load(context).level(),
        }
    }

    /// Tests whether the node is a leaf.
    pub fn is_leaf(&self, context: &Internal::Context) -> bool {
        self.level(context) == 0
    }

    /// Returns the element at the given position in the node.
    pub fn get(&self, idx: usize, context: &Internal::Context) -> Option<&Leaf::Item> {
        match self {
            NodeRc::Leaf(x) => x.load(context).get(idx),
            NodeRc::Internal(x) => x.load(context).get(idx),
        }
    }

    /// Returns the element at the given position in the node.
    pub fn get_mut(&mut self, idx: usize, context: &Internal::Context) -> Option<&mut Leaf::Item> {
        match self {
            NodeRc::Leaf(ref mut x) => x.load_mut(context).get_mut(idx, context),
            NodeRc::Internal(ref mut x) => x.load_mut(context).get_mut(idx, context),
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub fn free_slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRc::Leaf(x) => x.load(context).free_space(),
            NodeRc::Internal(x) => x.load(context).free_slots(),
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

    pub fn borrow_node(&mut self, context: &Internal::Context) -> BorrowedNode<Internal, Leaf> {
        match self {
            NodeRc::Internal(internal) => {
                BorrowedNode::Internal(internal.load_mut(context).borrow_node())
            }
            NodeRc::Leaf(leaf) => BorrowedNode::Leaf(leaf.load_mut(context).borrow_node()),
        }
    }

    pub fn split_at_child(&mut self, idx: usize, context: &Internal::Context) -> Self {
        match self {
            NodeRc::Internal(internal) => NodeRc::Internal(Internal::InternalEntry::new(
                internal.load_mut(context).split_at_child(idx, context),
            )),
            NodeRc::Leaf(leaf) => NodeRc::Leaf(Internal::LeafEntry::new(
                leaf.load_mut(context).split(idx, context),
            )),
        }
    }

    pub fn split_at_position(&mut self, position: usize, context: &Internal::Context) -> Self {
        match self {
            NodeRc::Internal(internal) => NodeRc::Internal(Internal::InternalEntry::new(
                internal
                    .load_mut(context)
                    .split_at_position(position, context),
            )),
            NodeRc::Leaf(leaf) => NodeRc::Leaf(Internal::LeafEntry::new(
                leaf.load_mut(context).split(position, context),
            )),
        }
    }

    /// Checks internal invariants of the node.
    #[allow(dead_code)]
    pub fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        context: &Internal::Context,
    ) {
        match self {
            NodeRc::Leaf(x) => {
                x.load(context)
                    .debug_check_invariants(reported_size, reported_level, context)
            }
            NodeRc::Internal(x) => {
                x.load(context)
                    .debug_check_invariants(reported_size, reported_level, context)
            }
        }
    }
}

impl<Internal, Leaf> NodeRc<Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
    Leaf::Item: PartialEq,
{
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) fn equal_iter_debug<'a>(
        &self,
        iter: &mut std::slice::Iter<'a, Leaf::Item>,
        context: &Internal::Context,
    ) -> bool {
        match self {
            NodeRc::Internal(ref internal) => {
                internal.load(context).equal_iter_debug(iter, context)
            }
            NodeRc::Leaf(ref leaf) => leaf.load(context).equal_iter_debug(iter, context),
        }
    }
}

#[derive(Debug)]
pub enum BorrowedNode<Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
{
    Internal(Internal::Borrowed),
    Leaf(Leaf::Borrowed),
}

impl<Internal, Leaf> BorrowedNode<Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
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
pub enum NodeRef<'a, Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
{
    Leaf(&'a Internal::LeafEntry),
    Internal(&'a Internal::InternalEntry),
}

impl<'a, Internal, Leaf> NodeRef<'a, Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
{
    /// Returns the size of the of node.
    pub fn len(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRef::Leaf(x) => x.load(context).len(),
            NodeRef::Internal(x) => x.load(context).len(),
        }
    }

    /// Returns the number of direct children of the node.
    pub fn slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRef::Leaf(x) => x.load(context).len(),
            NodeRef::Internal(x) => x.load(context).slots(),
        }
    }

    /// Returns whether the node is completely empty.
    pub fn is_empty(&self, context: &Internal::Context) -> bool {
        match self {
            NodeRef::Leaf(x) => x.load(context).is_empty(),
            NodeRef::Internal(x) => x.load(context).is_empty(),
        }
    }

    /// Returns the level of the node.
    pub fn level(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRef::Leaf(_) => 0,
            NodeRef::Internal(x) => x.load(context).level(),
        }
    }

    /// Tests whether the node is a leaf.
    pub fn is_leaf(&self, context: &Internal::Context) -> bool {
        self.level(context) == 0
    }

    /// Returns the element at the given position in the node.
    pub fn get(&self, idx: usize, context: &Internal::Context) -> Option<&Leaf::Item> {
        match self {
            NodeRef::Leaf(x) => x.load(context).get(idx),
            NodeRef::Internal(x) => x.load(context).get(idx),
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub fn free_slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeRef::Leaf(x) => x.load(context).free_space(),
            NodeRef::Internal(x) => x.load(context).free_slots(),
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
    pub fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        context: &Internal::Context,
    ) {
        match self {
            NodeRef::Leaf(x) => {
                x.load(context)
                    .debug_check_invariants(reported_size, reported_level, context)
            }
            NodeRef::Internal(x) => {
                x.load(context)
                    .debug_check_invariants(reported_size, reported_level, context)
            }
        }
    }
}

impl<'a, Internal, Leaf> NodeRef<'a, Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
    Leaf::Item: PartialEq,
{
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) fn equal_iter_debug<'b>(
        &self,
        iter: &mut std::slice::Iter<'b, Leaf::Item>,
        context: &Internal::Context,
    ) -> bool {
        match self {
            NodeRef::Internal(ref internal) => {
                internal.load(context).equal_iter_debug(iter, context)
            }
            NodeRef::Leaf(ref leaf) => leaf.load(context).equal_iter_debug(iter, context),
        }
    }
}

/// Represents an immutable reference to an arbitrary node in the tree.
#[derive(Debug)]
pub enum NodeMut<'a, Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
{
    Leaf(&'a mut Internal::LeafEntry),
    Internal(&'a mut Internal::InternalEntry),
}

impl<'a, Internal, Leaf> NodeMut<'a, Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
{
    /// Returns the size of the of node.
    pub fn len(&self, context: &Internal::Context) -> usize {
        match self {
            NodeMut::Leaf(x) => x.load(context).len(),
            NodeMut::Internal(x) => x.load(context).len(),
        }
    }

    /// Returns the number of direct children of the node.
    pub fn slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeMut::Leaf(x) => x.load(context).len(),
            NodeMut::Internal(x) => x.load(context).slots(),
        }
    }

    /// Returns whether the node is completely empty.
    pub fn is_empty(&self, context: &Internal::Context) -> bool {
        match self {
            NodeMut::Leaf(x) => x.load(context).is_empty(),
            NodeMut::Internal(x) => x.load(context).is_empty(),
        }
    }

    /// Returns the level of the node.
    pub fn level(&self, context: &Internal::Context) -> usize {
        match self {
            NodeMut::Leaf(_) => 0,
            NodeMut::Internal(x) => x.load(context).level(),
        }
    }

    /// Tests whether the node is a leaf.
    pub fn is_leaf(&self, context: &Internal::Context) -> bool {
        self.level(context) == 0
    }

    /// Returns the element at the given position in the node.
    pub fn get(
        &self,
        idx: usize,
        context: &Internal::Context,
    ) -> Option<&<Leaf as LeafTrait>::Item> {
        match self {
            NodeMut::Leaf(x) => x.load(context).get(idx),
            NodeMut::Internal(x) => x.load(context).get(idx),
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub fn free_slots(&self, context: &Internal::Context) -> usize {
        match self {
            NodeMut::Leaf(x) => x.load(context).free_space(),
            NodeMut::Internal(x) => x.load(context).free_slots(),
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
    pub fn borrow_node(self, context: &Internal::Context) -> BorrowedNode<Internal, Leaf> {
        match self {
            NodeMut::Internal(internal) => {
                BorrowedNode::Internal(internal.load_mut(context).borrow_node())
            }
            NodeMut::Leaf(leaf) => BorrowedNode::Leaf(leaf.load_mut(context).borrow_node()),
        }
    }

    /// Checks internal invariants of the node.
    #[allow(dead_code)]
    pub fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        context: &Internal::Context,
    ) {
        match self {
            NodeMut::Leaf(x) => {
                x.load(context)
                    .debug_check_invariants(reported_size, reported_level, context)
            }
            NodeMut::Internal(x) => {
                x.load(context)
                    .debug_check_invariants(reported_size, reported_level, context)
            }
        }
    }
}

impl<'a, Internal, Leaf> NodeMut<'a, Internal, Leaf>
where
    Internal: InternalTrait<Leaf>,
    Leaf: LeafTrait<Context = Internal::Context>,
    Leaf::Item: Clone + std::fmt::Debug + PartialEq,
{
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    #[allow(dead_code)]
    pub(crate) fn equal_iter_debug<'b>(
        &self,
        iter: &mut std::slice::Iter<'b, Leaf::Item>,
        context: &Internal::Context,
    ) -> bool {
        match self {
            NodeMut::Internal(ref internal) => {
                internal.load(context).equal_iter_debug(iter, context)
            }
            NodeMut::Leaf(ref leaf) => leaf.load(context).equal_iter_debug(iter, context),
        }
    }
}
