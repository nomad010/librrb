use crate::circular::{BorrowBufferMut, CircularBuffer};
use crate::node_traits::*;
use crate::size_table::SizeTable;
use crate::{Side, RRB_WIDTH};
use archery::{SharedPointer, SharedPointerKind};
use std::ops::Range;

#[derive(Debug)]
pub struct SharedPointerEntry<
    I: Clone + std::fmt::Debug,
    P: SharedPointerKind,
    C: Clone + std::fmt::Debug + Default,
>(SharedPointer<I, P>, std::marker::PhantomData<C>);

impl<I, P, C> Clone for SharedPointerEntry<I, P, C>
where
    I: Clone + std::fmt::Debug,
    P: SharedPointerKind,
    C: Clone + std::fmt::Debug + Default,
{
    fn clone(&self) -> Self {
        SharedPointerEntry(self.0.clone(), std::marker::PhantomData)
    }
}

impl<I, P, C> Entry for SharedPointerEntry<I, P, C>
where
    I: Clone + std::fmt::Debug,
    P: SharedPointerKind,
    C: Clone + std::fmt::Debug + Default,
{
    type Item = I;
    type Context = C;

    fn new(item: Self::Item) -> Self {
        SharedPointerEntry(SharedPointer::new(item), std::marker::PhantomData)
    }

    fn load(&self, _context: &Self::Context) -> &Self::Item {
        &self.0
    }

    fn load_mut(&mut self, _context: &Self::Context) -> &mut Self::Item {
        SharedPointer::make_mut(&mut self.0)
    }
}

#[derive(Debug)]
pub struct BorrowedLeaf<A: Clone + std::fmt::Debug> {
    buffer: BorrowBufferMut<A>,
}

impl<A: Clone + std::fmt::Debug> BorrowedLeafTrait for BorrowedLeaf<A> {
    type Item = A;
    type Context = ();

    fn split_at(&mut self, idx: usize) -> Self {
        BorrowedLeaf {
            buffer: self.buffer.split_at(idx),
        }
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn from_same_source(&self, other: &Self) -> bool {
        self.buffer.from_same_source(&other.buffer)
    }

    fn get_mut(&mut self, idx: usize) -> Option<&mut Self::Item> {
        self.buffer.get_mut(idx)
    }

    fn combine(&mut self, other: Self) {
        self.buffer.combine(other.buffer)
    }
}

#[derive(Clone, Debug)]
pub struct Leaf<A: Clone + std::fmt::Debug> {
    buffer: CircularBuffer<A>,
}

impl<A: Clone + std::fmt::Debug> LeafTrait for Leaf<A> {
    type Item = A;
    type Context = ();
    type Borrowed = BorrowedLeaf<A>;

    fn empty() -> Self {
        Leaf {
            buffer: CircularBuffer::new(),
        }
    }

    fn with_item(item: Self::Item, _context: &Self::Context) -> Self {
        Leaf {
            buffer: CircularBuffer::with_item(item),
        }
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn free_space(&self) -> usize {
        self.buffer.free_space()
    }

    fn get(&self, position: usize) -> Option<&Self::Item> {
        self.buffer.get(position)
    }

    fn get_mut(&mut self, position: usize, _context: &Self::Context) -> Option<&mut Self::Item> {
        self.buffer.get_mut(position)
    }

    fn push(&mut self, side: Side, item: Self::Item, _context: &Self::Context) {
        self.buffer.push(side, item)
    }

    fn pop(&mut self, side: Side, _context: &Self::Context) -> Self::Item {
        self.buffer.pop(side)
    }

    fn split(&mut self, idx: usize, context: &Self::Context) -> Self {
        if idx <= self.len() {
            let mut other = Self::empty();
            for _ in 0..idx {
                let node = self.pop(Side::Front, context);
                other.push(Side::Back, node, context);
            }
            std::mem::replace(self, other)
        } else {
            panic!("Trying to split at a position out of bounds of the tree");
        }
    }

    fn share_children_with(
        &mut self,
        destination: &mut Self,
        share_side: Side,
        len: usize,
        _context: &Self::Context,
    ) -> usize {
        self.buffer
            .decant_into(&mut destination.buffer, share_side, len)
    }

    fn borrow_node(&mut self) -> Self::Borrowed {
        BorrowedLeaf {
            buffer: self.buffer.mutable_view(),
        }
    }

    fn equal_iter_debug<'a>(
        &self,
        iter: &mut std::slice::Iter<'a, Self::Item>,
        _context: &Self::Context,
    ) -> bool
    where
        Self::Item: PartialEq,
    {
        let mut result = true;
        for value in &self.buffer {
            let iter_value = iter.next().unwrap();
            let cmp = value == iter_value;
            result &= cmp;
            println!("{:?} == {:?} = {} ", value, iter_value, cmp);
        }
        result
    }
}

/*
/// Represents a homogenous list of nodes.
#[derive(Debug)]
pub(crate) enum ChildList<A: Clone + std::fmt::Debug, P: SharedPointerKind> {
    Leaves(CircularBuffer<SharedPointer<Leaf<A>, P>>),
    Internals(CircularBuffer<SharedPointer<Internal<A, P>, P>>),
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind> Clone for ChildList<A, P> {
    fn clone(&self) -> Self {
        match self {
            ChildList::Leaves(buf) => ChildList::Leaves(buf.clone()),
            ChildList::Internals(buf) => ChildList::Internals(buf.clone()),
        }
    }
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind> ChildList<A, P> {
    /// Constructs a new empty list of nodes.
    pub fn new_empty(&self) -> Self {
        match self {
            ChildList::Internals(_) => ChildList::Internals(CircularBuffer::new()),
            ChildList::Leaves(_) => ChildList::Leaves(CircularBuffer::new()),
        }
    }

    /// Returns a reference to the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves_ref(&self) -> &CircularBuffer<SharedPointer<Leaf<A>, P>> {
        if let ChildList::Leaves(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as a leaves list")
        }
    }

    /// Returns a reference to the list as a list of internal nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of internal nodes.
    pub fn internals_ref(&self) -> &CircularBuffer<SharedPointer<Internal<A, P>, P>> {
        if let ChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
        }
    }

    /// Returns a mutable reference to the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves_mut(&mut self) -> &mut CircularBuffer<SharedPointer<Leaf<A>, P>> {
        if let ChildList::Leaves(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as a leaves list")
        }
    }

    /// Returns a mutable reference to the list as a list of internal nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of internal nodes.
    pub fn internals_mut(&mut self) -> &mut CircularBuffer<SharedPointer<Internal<A, P>, P>> {
        if let ChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
        }
    }

    pub fn borrow(&mut self) -> BorrowedChildList<A, P> {
        match self {
            ChildList::Internals(children) => BorrowedChildList::Internals(children.mutable_view()),
            ChildList::Leaves(children) => BorrowedChildList::Leaves(children.mutable_view()),
        }
    }

    /// Returns a mutable reference to the list as a list of internal nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of internal nodes.
    pub fn get(&self, child_idx: usize, idx: usize) -> Option<&A> {
        match self {
            ChildList::Leaves(children) => children.get(child_idx).unwrap().get(idx),
            ChildList::Internals(children) => children.get(child_idx).unwrap().get(idx),
        }
    }

    pub fn get_mut(&mut self, child_idx: usize, idx: usize) -> Option<&mut A> {
        match self {
            ChildList::Leaves(children) => {
                SharedPointer::make_mut(children.get_mut(child_idx).unwrap()).get_mut(idx)
            }
            ChildList::Internals(children) => {
                SharedPointer::make_mut(children.get_mut(child_idx).unwrap()).get_mut(idx)
            }
        }
    }

    /// Returns a copy of the Rc of the node at the given position in the node.
    pub fn get_child_node(&self, idx: usize) -> Option<NodeRc<A, P, Internal<A, P>, Leaf<A>>> {
        // TODO: Get rid of this function
        match self {
            ChildList::Leaves(children) => {
                children.get(idx).map(|x| SharedPointer::clone(x).into())
            }
            ChildList::Internals(children) => {
                children.get(idx).map(|x| SharedPointer::clone(x).into())
            }
        }
    }

    /// Returns the length of the child list.
    pub fn slots(&self) -> usize {
        match self {
            ChildList::Leaves(children) => children.len(),
            ChildList::Internals(children) => children.len(),
        }
    }

    /// Returns the number of nodes that could still be inserted into the child list.
    pub fn free_slots(&self) -> usize {
        RRB_WIDTH - self.slots()
    }
}

#[derive(Debug)]
pub(crate) enum BorrowedChildList<A: Clone + std::fmt::Debug, P: SharedPointerKind> {
    Internals(BorrowBufferMut<SharedPointer<Internal<A, P>, P>>),
    Leaves(BorrowBufferMut<SharedPointer<Leaf<A>, P>>),
}

impl<'a, A: Clone + std::fmt::Debug, P: SharedPointerKind> BorrowedChildList<A, P> {
    /// Consumes `self` and returns the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves(self) -> BorrowBufferMut<SharedPointer<Leaf<A>, P>> {
        if let BorrowedChildList::Leaves(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as a leaves list")
        }
    }

    /// Consumes `self` and returns the list as a list of internal nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of internal nodes.
    pub fn internals(self) -> BorrowBufferMut<SharedPointer<Internal<A, P>, P>> {
        if let BorrowedChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
        }
    }

    fn split_at(&mut self, index: usize) -> Self {
        match self {
            BorrowedChildList::Internals(ref mut children) => {
                let new_children = children.split_at(index);
                BorrowedChildList::Internals(new_children)
            }
            BorrowedChildList::Leaves(ref mut children) => {
                let new_children = children.split_at(index);
                BorrowedChildList::Leaves(new_children)
            }
        }
    }

    pub fn from_same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (
                BorrowedChildList::Internals(children),
                BorrowedChildList::Internals(other_children),
            ) => children.from_same_source(other_children),
            (BorrowedChildList::Leaves(children), BorrowedChildList::Leaves(other_children)) => {
                children.from_same_source(other_children)
            }
            _ => false,
        }
    }

    pub fn combine(&mut self, other: Self) {
        match self {
            BorrowedChildList::Internals(children) => children.combine(other.internals()),
            BorrowedChildList::Leaves(children) => children.combine(other.leaves()),
        }
    }

    pub(crate) fn range(&self) -> &Range<usize> {
        match self {
            BorrowedChildList::Internals(children) => &children.range,
            BorrowedChildList::Leaves(children) => &children.range,
        }
    }

    pub(crate) fn range_mut(&mut self) -> &mut Range<usize> {
        match self {
            BorrowedChildList::Internals(children) => &mut children.range,
            BorrowedChildList::Leaves(children) => &mut children.range,
        }
    }
}
*/

pub struct BorrowedInternal<
    A: Clone + std::fmt::Debug,
    P: SharedPointerKind,
    Leaf: LeafTrait<Item = A, Context = ()>,
> {
    pub(crate) sizes: SharedPointer<SizeTable, P>,
    pub(crate) children: BorrowedChildList<A, P, Leaf>,
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind, Leaf: LeafTrait<Item = A, Context = ()>>
    BorrowedInternal<A, P, Leaf>
{
    fn left_size(&self) -> usize {
        let range = self.children.range();
        if range.start == 0 {
            0
        } else {
            *self
                .sizes
                .get_cumulative_child_size(range.start - 1)
                .unwrap()
        }
    }

    pub fn position_info_for(&self, idx: usize) -> Option<(usize, usize)> {
        let position = self.sizes.position_info_for(self.left_size() + idx);
        if let Some((child_number, subidx)) = position {
            if child_number >= self.children.range().end {
                None
            } else {
                Some((child_number - self.children.range().start, subidx))
            }
        } else {
            None
        }
    }
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind, Leaf: LeafTrait<Item = A, Context = ()>>
    std::fmt::Debug for BorrowedInternal<A, P, Leaf>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BorrowedInternal").finish()
    }
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind, Leaf: LeafTrait<Item = A, Context = ()>>
    BorrowedInternalTrait<Leaf> for BorrowedInternal<A, P, Leaf>
{
    type InternalChild = Internal<A, P, Leaf>;

    fn len(&self) -> usize {
        let range = self.children.range();
        if range.start == range.end {
            0
        } else {
            self.sizes.get_cumulative_child_size(range.end - 1).unwrap() - self.left_size()
        }
    }

    fn slots(&self) -> usize {
        self.children.range().len()
    }

    fn range(&self) -> Range<usize> {
        self.children.range().clone()
    }

    fn level(&self) -> usize {
        self.sizes.level()
    }

    fn from_same_source(&self, other: &Self) -> bool {
        self.children.from_same_source(&other.children)
    }

    fn combine(&mut self, other: Self) {
        self.children.combine(other.children)
    }

    fn get_child_mut_at_slot(
        &mut self,
        idx: usize,
    ) -> Option<(NodeMut<Self::InternalChild, Leaf>, Range<usize>)> {
        let left_skipped = self.children.range().start;
        let mut subrange = self.sizes.get_child_range(idx + left_skipped)?;
        subrange.start -= self.left_size();
        subrange.end -= self.left_size();
        let node = match self.children {
            BorrowedChildList::Internals(ref mut children) => {
                NodeMut::Internal(children.get_mut(idx)?)
            }
            BorrowedChildList::Leaves(ref mut children) => NodeMut::Leaf(children.get_mut(idx)?),
        };
        Some((node, subrange))
    }

    fn get_child_mut_for_position(
        &mut self,
        position: usize,
    ) -> Option<(NodeMut<Self::InternalChild, Leaf>, Range<usize>)> {
        let index = self.position_info_for(position)?.0;
        self.get_child_mut_at_slot(index)
    }

    fn pop_child(
        &mut self,
        side: Side,
        context: &(),
    ) -> Option<BorrowedNode<Self::InternalChild, Leaf>> {
        let child = self.get_child_mut_at_side(side)?.0.borrow_node(context);
        if side == Side::Front {
            self.children.range_mut().start += 1;
        } else {
            self.children.range_mut().end -= 1;
        }
        Some(child)
    }

    fn unpop_child(&mut self, side: Side) {
        if side == Side::Front {
            self.children.range_mut().start -= 1;
        } else {
            self.children.range_mut().end += 1;
        }
    }

    fn split_at_child(&mut self, index: usize) -> Self {
        BorrowedInternal {
            children: self.children.split_at(index),
            sizes: SharedPointer::clone(&self.sizes),
        }
    }

    fn split_at_position(&mut self, position: usize) -> (usize, Self) {
        let index = self.position_info_for(position).unwrap();
        println!(
            "rawr split says {:?} for {} {} {:?}",
            index,
            self.len(),
            self.slots(),
            self.range()
        );
        let result = self.split_at_child(index.0);
        println!("nogtma {}", result.len());
        (index.1, result)
    }
}

#[derive(Debug)]
pub(crate) enum BorrowedChildList<
    A: Clone + std::fmt::Debug,
    P: SharedPointerKind,
    Leaf: LeafTrait<Item = A, Context = ()>,
> {
    Internals(BorrowBufferMut<SharedPointerEntry<Internal<A, P, Leaf>, P, ()>>),
    Leaves(BorrowBufferMut<SharedPointerEntry<Leaf, P, ()>>),
}

impl<
        'a,
        A: Clone + std::fmt::Debug,
        P: SharedPointerKind,
        Leaf: LeafTrait<Item = A, Context = ()>,
    > BorrowedChildList<A, P, Leaf>
{
    /// Consumes `self` and returns the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves(self) -> BorrowBufferMut<SharedPointerEntry<Leaf, P, ()>> {
        if let BorrowedChildList::Leaves(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as a leaves list")
        }
    }

    /// Consumes `self` and returns the list as a list of internal nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of internal nodes.
    pub fn internals(self) -> BorrowBufferMut<SharedPointerEntry<Internal<A, P, Leaf>, P, ()>> {
        if let BorrowedChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
        }
    }

    fn split_at(&mut self, index: usize) -> Self {
        println!(
            "Splitting RAwr borrowchildlist at {} out of {:?}",
            index,
            self.range()
        );
        match self {
            BorrowedChildList::Internals(ref mut children) => {
                let new_children = children.split_at(index);
                BorrowedChildList::Internals(new_children)
            }
            BorrowedChildList::Leaves(ref mut children) => {
                let new_children = children.split_at(index);
                BorrowedChildList::Leaves(new_children)
            }
        }
    }

    pub fn from_same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (
                BorrowedChildList::Internals(children),
                BorrowedChildList::Internals(other_children),
            ) => children.from_same_source(other_children),
            (BorrowedChildList::Leaves(children), BorrowedChildList::Leaves(other_children)) => {
                children.from_same_source(other_children)
            }
            _ => false,
        }
    }

    pub fn combine(&mut self, other: Self) {
        match self {
            BorrowedChildList::Internals(children) => children.combine(other.internals()),
            BorrowedChildList::Leaves(children) => children.combine(other.leaves()),
        }
    }

    pub(crate) fn range(&self) -> &Range<usize> {
        match self {
            BorrowedChildList::Internals(children) => &children.range,
            BorrowedChildList::Leaves(children) => &children.range,
        }
    }

    pub(crate) fn range_mut(&mut self) -> &mut Range<usize> {
        match self {
            BorrowedChildList::Internals(children) => &mut children.range,
            BorrowedChildList::Leaves(children) => &mut children.range,
        }
    }
}

/// Represents a homogenous list of nodes.
#[derive(Debug)]
pub(crate) enum ChildList<
    A: Clone + std::fmt::Debug,
    P: SharedPointerKind,
    Leaf: LeafTrait<Item = A, Context = ()>,
> {
    Leaves(CircularBuffer<SharedPointerEntry<Leaf, P, ()>>),
    Internals(CircularBuffer<SharedPointerEntry<Internal<A, P, Leaf>, P, ()>>),
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind, Leaf: LeafTrait<Item = A, Context = ()>>
    Clone for ChildList<A, P, Leaf>
{
    fn clone(&self) -> Self {
        match self {
            ChildList::Leaves(buf) => ChildList::Leaves(buf.clone()),
            ChildList::Internals(buf) => ChildList::Internals(buf.clone()),
        }
    }
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind, Leaf: LeafTrait<Item = A, Context = ()>>
    ChildList<A, P, Leaf>
{
    /// Constructs a new empty list of nodes.
    pub fn new_empty(&self) -> Self {
        match self {
            ChildList::Internals(_) => ChildList::Internals(CircularBuffer::new()),
            ChildList::Leaves(_) => ChildList::Leaves(CircularBuffer::new()),
        }
    }

    /// Returns a reference to the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves_ref(&self) -> &CircularBuffer<SharedPointerEntry<Leaf, P, ()>> {
        if let ChildList::Leaves(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as a leaves list")
        }
    }

    /// Returns a reference to the list as a list of internal nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of internal nodes.
    pub fn internals_ref(
        &self,
    ) -> &CircularBuffer<SharedPointerEntry<Internal<A, P, Leaf>, P, ()>> {
        if let ChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
        }
    }

    /// Returns a mutable reference to the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves_mut(&mut self) -> &mut CircularBuffer<SharedPointerEntry<Leaf, P, ()>> {
        if let ChildList::Leaves(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as a leaves list")
        }
    }

    /// Returns a mutable reference to the list as a list of internal nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of internal nodes.
    pub fn internals_mut(
        &mut self,
    ) -> &mut CircularBuffer<SharedPointerEntry<Internal<A, P, Leaf>, P, ()>> {
        if let ChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
        }
    }

    pub fn borrow(&mut self) -> BorrowedChildList<A, P, Leaf> {
        match self {
            ChildList::Internals(children) => BorrowedChildList::Internals(children.mutable_view()),
            ChildList::Leaves(children) => BorrowedChildList::Leaves(children.mutable_view()),
        }
    }

    /// Returns a mutable reference to the list as a list of internal nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of internal nodes.
    pub fn get(&self, child_idx: usize, idx: usize) -> Option<&A> {
        match self {
            ChildList::Leaves(children) => children.get(child_idx).unwrap().load(&()).get(idx),
            ChildList::Internals(children) => children.get(child_idx).unwrap().load(&()).get(idx),
        }
    }

    pub fn get_mut(
        &mut self,
        child_idx: usize,
        idx: usize,
        context: &Leaf::Context,
    ) -> Option<&mut A> {
        match self {
            ChildList::Leaves(children) => children
                .get_mut(child_idx)
                .unwrap()
                .load_mut(&())
                .get_mut(idx, context),
            ChildList::Internals(children) => children
                .get_mut(child_idx)
                .unwrap()
                .load_mut(&())
                .get_mut(idx, context),
        }
    }

    /// Returns the length of the child list.
    pub fn slots(&self) -> usize {
        match self {
            ChildList::Leaves(children) => children.len(),
            ChildList::Internals(children) => children.len(),
        }
    }

    /// Returns the number of nodes that could still be inserted into the child list.
    pub fn free_slots(&self) -> usize {
        RRB_WIDTH - self.slots()
    }
}

#[derive(Debug)]
pub struct Internal<
    A: Clone + std::fmt::Debug,
    P: SharedPointerKind,
    Leaf: LeafTrait<Item = A, Context = ()>,
> {
    sizes: SharedPointer<SizeTable, P>,
    children: ChildList<A, P, Leaf>,
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind, Leaf: LeafTrait<Item = A, Context = ()>>
    Clone for Internal<A, P, Leaf>
{
    fn clone(&self) -> Self {
        Internal {
            sizes: SharedPointer::clone(&self.sizes),
            children: self.children.clone(),
        }
    }
}

// impl<A: Clone + std::fmt::Debug, P: SharedPointerKind, Leaf: LeafTrait<Item = A>> std::fmt::Debug
//     for Internal<A, P, Leaf>
// {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         f.debug_struct("Internal").finish()
//     }
// }

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind, Leaf: LeafTrait<Item = A, Context = ()>>
    InternalTrait<Leaf> for Internal<A, P, Leaf>
{
    // type Item = A;
    type Borrowed = BorrowedInternal<A, P, Leaf>;
    type Context = ();

    type LeafEntry = SharedPointerEntry<Leaf, P, ()>;
    type InternalEntry = SharedPointerEntry<Self, P, ()>;

    fn empty_internal(level: usize) -> Self {
        debug_assert_ne!(level, 0); // Should be a Leaf
        if level == 1 {
            Internal {
                sizes: SharedPointer::new(SizeTable::new(1)),
                children: ChildList::Leaves(CircularBuffer::new()),
            }
        } else {
            Internal {
                sizes: SharedPointer::new(SizeTable::new(level)),
                children: ChildList::Internals(CircularBuffer::new()),
            }
        }
    }

    fn new_empty(&self) -> Self {
        Internal {
            sizes: SharedPointer::new(SizeTable::new(self.sizes.level())),
            children: self.children.new_empty(),
        }
    }

    fn share_children_with(
        &mut self,
        destination: &mut Self,
        share_side: Side,
        len: usize,
        _context: &Self::Context,
    ) -> usize {
        debug_assert_eq!(self.level(), destination.level());
        let shared = match self.children {
            ChildList::Internals(ref mut children) => {
                debug_assert_eq!(children.len(), self.sizes.len());
                children.decant_into(destination.children.internals_mut(), share_side, len)
            }
            ChildList::Leaves(ref mut children) => {
                debug_assert_eq!(children.len(), self.sizes.len());
                children.decant_into(destination.children.leaves_mut(), share_side, len)
            }
        };
        if shared != 0 {
            let origin_sizes = SharedPointer::make_mut(&mut self.sizes);
            let destination_sizes = SharedPointer::make_mut(&mut destination.sizes);

            for _ in 0..shared {
                destination_sizes
                    .push_child(share_side.negate(), origin_sizes.pop_child(share_side));
            }
        }

        shared
    }

    fn pack_children(&mut self, context: &Self::Context) {
        if self.is_empty() {
            return;
        }
        if self.level() == 1 {
            assert_eq!(self.children.leaves_ref().len(), self.sizes.len());
        } else {
            assert_eq!(self.children.internals_ref().len(), self.sizes.len());
        }
        // TODO: Genericize on a side
        let mut write_position = 0;
        let mut read_position = write_position + 1;
        match self.children {
            ChildList::Internals(ref mut children) => {
                while read_position < children.len() {
                    if read_position == write_position {
                        read_position += 1;
                        continue;
                    } else {
                        let (write, read) = children.pair_mut(write_position, read_position);
                        read.load_mut(context).share_children_with(
                            write.load_mut(context),
                            Side::Front,
                            RRB_WIDTH,
                            context,
                        );

                        if write.load_mut(context).is_full() {
                            write_position += 1;
                        }
                        if read.load_mut(context).is_empty() {
                            read_position += 1;
                        }
                    }
                }
                while children.back().unwrap().load(context).is_empty() {
                    children.pop_back();
                }
                let sizes = SharedPointer::make_mut(&mut self.sizes);
                *sizes = SizeTable::new(sizes.level());
                for child in children {
                    sizes.push_child(Side::Back, child.load(context).len());
                }
            }
            ChildList::Leaves(ref mut children) => {
                while read_position < children.len() {
                    if read_position == write_position {
                        read_position += 1;
                        continue;
                    } else {
                        let (write, read) = children.pair_mut(write_position, read_position);
                        read.load_mut(context).share_children_with(
                            write.load_mut(context),
                            Side::Front,
                            RRB_WIDTH,
                            context,
                        );

                        if write.load_mut(context).is_full() {
                            write_position += 1;
                        }
                        if read.load_mut(context).is_empty() {
                            read_position += 1;
                        }
                    }
                }

                while children.back().unwrap().load(context).is_empty() {
                    children.pop_back();
                }
                let sizes = SharedPointer::make_mut(&mut self.sizes);
                *sizes = SizeTable::new(sizes.level());
                for child in children {
                    sizes.push_child(Side::Back, child.load(context).len());
                }
            }
        }

        if self.level() == 1 {
            assert_eq!(self.children.leaves_ref().len(), self.sizes.len());
        } else {
            assert_eq!(self.children.internals_ref().len(), self.sizes.len());
        }
    }

    fn get(&self, idx: usize) -> Option<&Leaf::Item> {
        if let Some((array_idx, new_idx)) = self.sizes.position_info_for(idx) {
            self.children.get(array_idx, new_idx)
        } else {
            None
        }
    }

    fn get_mut(&mut self, idx: usize, context: &Self::Context) -> Option<&mut Leaf::Item> {
        if let Some((array_idx, new_idx)) = self.sizes.position_info_for(idx) {
            self.children.get_mut(array_idx, new_idx, context)
        } else {
            None
        }
    }

    fn pop_child(&mut self, side: Side, _context: &Self::Context) -> NodeRc<Self, Leaf> {
        SharedPointer::make_mut(&mut self.sizes).pop_child(side);
        match self.children {
            ChildList::Internals(ref mut children) => NodeRc::Internal(children.pop(side)),
            ChildList::Leaves(ref mut children) => NodeRc::Leaf(children.pop(side)),
        }
    }

    fn push_child(&mut self, side: Side, node: NodeRc<Self, Leaf>, context: &Self::Context) {
        SharedPointer::make_mut(&mut self.sizes).push_child(side, node.len(context));
        match self.children {
            ChildList::Internals(ref mut children) => children.push(side, node.internal()),
            ChildList::Leaves(ref mut children) => children.push(side, node.leaf()),
        }
    }

    fn borrow_node(&mut self) -> Self::Borrowed {
        BorrowedInternal {
            children: self.children.borrow(),
            sizes: SharedPointer::clone(&self.sizes),
        }
    }

    fn level(&self) -> usize {
        self.sizes.level()
    }

    fn len(&self) -> usize {
        self.sizes.cumulative_size()
    }

    fn slots(&self) -> usize {
        self.children.slots()
    }

    fn free_slots(&self) -> usize {
        self.children.free_slots()
    }

    fn get_child_ref_at_slot(&self, idx: usize) -> Option<(NodeRef<Self, Leaf>, Range<usize>)> {
        if let Some(range) = self.sizes.get_child_range(idx) {
            match self.children {
                ChildList::Internals(ref internals) => {
                    Some((NodeRef::Internal(internals.get(idx).unwrap()), range))
                }
                ChildList::Leaves(ref leaves) => {
                    Some((NodeRef::Leaf(leaves.get(idx).unwrap()), range))
                }
            }
        } else {
            None
        }
    }

    fn get_child_ref_for_position(
        &self,
        position: usize,
    ) -> Option<(NodeRef<Self, Leaf>, Range<usize>)> {
        if let Some((child_idx, _)) = self.sizes.position_info_for(position) {
            self.get_child_ref_at_slot(child_idx)
        } else {
            None
        }
    }

    fn get_child_mut_at_slot(&mut self, idx: usize) -> Option<(NodeMut<Self, Leaf>, Range<usize>)> {
        if let Some(range) = self.sizes.get_child_range(idx) {
            match self.children {
                ChildList::Internals(ref mut internals) => {
                    Some((NodeMut::Internal(internals.get_mut(idx).unwrap()), range))
                }
                ChildList::Leaves(ref mut leaves) => {
                    Some((NodeMut::Leaf(leaves.get_mut(idx).unwrap()), range))
                }
            }
        } else {
            None
        }
    }

    fn get_child_mut_for_position(
        &mut self,
        position: usize,
    ) -> Option<(NodeMut<Self, Leaf>, Range<usize>)> {
        if let Some((child_idx, _)) = self.sizes.position_info_for(position) {
            self.get_child_mut_at_slot(child_idx)
        } else {
            None
        }
    }

    fn split_at_child(&mut self, idx: usize, context: &Self::Context) -> Self {
        if idx <= self.slots() {
            let mut result = self.new_empty();
            self.share_children_with(&mut result, Side::Front, idx, context);
            std::mem::swap(self, &mut result);
            result
        } else {
            panic!("Trying to split at a position out of bounds of the tree");
        }
    }

    fn split_at_position(&mut self, position: usize, context: &Self::Context) -> Self {
        if position > self.len() {
            panic!("Trying to split at a position out of bounds of the tree");
        }
        let original_position = position;
        let original_len = self.len();
        let (child_position, position) = self.sizes.position_info_for(position).unwrap();
        let mut result = self.split_at_child(child_position, context);
        let mut next_child = result.pop_child(Side::Front, context);

        if position == 0 {
            result.push_child(Side::Front, next_child, context);
        } else if position == next_child.len(context) {
            self.push_child(Side::Back, next_child, context);
        } else {
            let subresult = next_child.split_at_position(position, context);
            self.push_child(Side::Back, next_child, context);
            result.push_child(Side::Front, subresult, context);
        }

        self.debug_check_invariants(original_position, self.level(), context);
        result.debug_check_invariants(original_len - original_position, result.level(), context);
        result
    }

    fn equal_iter_debug<'a>(
        &self,
        iter: &mut std::slice::Iter<'a, Leaf::Item>,
        context: &Self::Context,
    ) -> bool
    where
        Leaf::Item: PartialEq,
    {
        let mut result = true;
        match &self.children {
            ChildList::Internals(internals) => {
                for internal in internals {
                    result &= internal.load(context).equal_iter_debug(iter, context);
                }
            }
            ChildList::Leaves(leaves) => {
                for leaf in leaves {
                    result &= leaf.load(context).equal_iter_debug(iter, context);
                }
            }
        }
        result
    }

    fn debug_check_invariants(
        &self,
        reported_size: usize,
        reported_level: usize,
        context: &Self::Context,
    ) {
        debug_assert_eq!(reported_level, self.level());
        match &self.children {
            ChildList::Internals(internals) => {
                debug_assert_eq!(internals.len(), self.sizes.len());
                let mut sum = 0;
                for (idx, internal) in internals.iter().enumerate() {
                    let child_size = self.sizes.get_child_size(idx).unwrap();
                    internal.load(context).debug_check_invariants(
                        child_size,
                        reported_level - 1,
                        context,
                    );
                    sum += child_size;
                }
                debug_assert_eq!(sum, reported_size);
            }
            ChildList::Leaves(leaves) => {
                debug_assert_eq!(leaves.len(), self.sizes.len());
                let mut sum = 0;
                for (idx, leaf) in leaves.iter().enumerate() {
                    let child_size = self.sizes.get_child_size(idx).unwrap();
                    leaf.load(context).debug_check_invariants(
                        child_size,
                        reported_level - 1,
                        context,
                    );
                    sum += child_size;
                }
                debug_assert_eq!(sum, reported_size);
            }
        }
    }
}
