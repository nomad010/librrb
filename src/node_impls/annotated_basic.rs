use crate::circular::{BorrowBufferMut, CircularBuffer};
use crate::node_traits::*;
use crate::size_table::SizeTable;
use crate::{Side, RRB_WIDTH};
use archery::{SharedPointer, SharedPointerKind};
use num::Zero;
use std::ops::{Bound, Deref, DerefMut, Range, RangeBounds};

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

pub struct DerefPtr<A: Clone + std::fmt::Debug>(*const A);

impl<A: Clone + std::fmt::Debug> Deref for DerefPtr<A> {
    type Target = A;

    fn deref(&self) -> &A {
        unsafe { &*self.0 }
    }
}

pub struct DerefMutPtr<A: Clone + std::fmt::Debug>(*mut A);

impl<A: Clone + std::fmt::Debug> Deref for DerefMutPtr<A> {
    type Target = A;

    fn deref(&self) -> &A {
        unsafe { &*self.0 }
    }
}

impl<A: Clone + std::fmt::Debug> DerefMut for DerefMutPtr<A> {
    fn deref_mut(&mut self) -> &mut A {
        unsafe { &mut *self.0 }
    }
}

impl<A: Clone + std::fmt::Debug> Drop for DerefPtr<A> {
    fn drop(&mut self) {
        // println!("Ok dropping derefptr");
    }
}

impl<A: Clone + std::fmt::Debug> Drop for DerefMutPtr<A> {
    fn drop(&mut self) {
        // println!("Ok dropping derefptrmut");
    }
}

impl<I, P, C> Entry for SharedPointerEntry<I, P, C>
where
    I: Clone + std::fmt::Debug,
    P: SharedPointerKind,
    C: Clone + std::fmt::Debug + Default,
{
    type Item = I;
    type LoadGuard = DerefPtr<I>;
    type LoadMutGuard = DerefMutPtr<I>;
    type Context = C;

    fn new(item: Self::Item) -> Self {
        SharedPointerEntry(SharedPointer::new(item), std::marker::PhantomData)
    }

    fn load<'a>(&'a self, _context: &Self::Context) -> Self::LoadGuard {
        DerefPtr(self.0.deref() as *const I)
    }

    fn load_mut<'a>(&'a mut self, _context: &Self::Context) -> Self::LoadMutGuard {
        DerefMutPtr(SharedPointer::make_mut(&mut self.0))
    }
}

pub struct ItemMutGuard<
    A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
> {
    item: *mut A,
    previous: A,
    ptrs: Vec<*mut A>,
}

impl<A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero>
    Deref for ItemMutGuard<A>
{
    type Target = A;
    fn deref(&self) -> &A {
        unsafe { &*self.item }
    }
}

impl<A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero>
    DerefMut for ItemMutGuard<A>
{
    fn deref_mut(&mut self) -> &mut A {
        unsafe { &mut *self.item }
    }
}

impl<A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero> Drop
    for ItemMutGuard<A>
{
    fn drop(&mut self) {
        println!("Ok dropping itemmutguard");
        for sum_ptr in &self.ptrs {
            unsafe {
                let sum = &mut **sum_ptr;
                *sum = (*sum).clone() - self.previous.clone() + (*self.item).clone();
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct BorrowedLeaf<
    A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
> {
    buffer: BorrowBufferMut<A>,
    sum: *mut A,
}

impl<A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero> Drop
    for BorrowedLeaf<A>
{
    fn drop(&mut self) {
        // println!("Dropping leaf")
        let mut new_sum = A::zero();
        for child in self.buffer.iter() {
            new_sum = child.clone() + new_sum;
        }
        unsafe {
            *self.sum = new_sum;
        }
    }
}

impl<A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero>
    BorrowedLeafTrait for BorrowedLeaf<A>
{
    type Item = A;
    type Context = ();
    type ItemMutGuard = DerefMutPtr<A>;

    fn split_at(&mut self, idx: usize) -> (Self, Self) {
        let (left, right) = self.buffer.split_at(idx);
        (
            BorrowedLeaf {
                buffer: left,
                sum: self.sum,
            },
            BorrowedLeaf {
                buffer: right,
                sum: self.sum,
            },
        )
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn get_mut_guarded(&mut self, idx: usize) -> Option<Self::ItemMutGuard> {
        Some(DerefMutPtr(self.buffer.get_mut_ptr(idx)?))
    }

    fn get_mut(&mut self, idx: usize) -> Option<*mut Self::Item> {
        self.buffer.get_mut_ptr(idx)
    }
}

#[derive(Clone, Debug)]
pub struct Leaf<
    A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
> {
    buffer: CircularBuffer<A>,
    sum: A,
}

impl<A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero>
    LeafTrait for Leaf<A>
{
    type Item = A;
    type Context = ();
    type Borrowed = BorrowedLeaf<A>;
    type ItemMutGuard = ItemMutGuard<A>;

    fn empty() -> Self {
        Leaf {
            buffer: CircularBuffer::new(),
            sum: A::zero(),
        }
    }

    fn with_item(item: Self::Item, _context: &Self::Context) -> Self {
        Leaf {
            sum: item.clone(),
            buffer: CircularBuffer::with_item(item),
        }
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn free_space(&self) -> usize {
        self.buffer.free_space()
    }

    fn get(&self, position: usize) -> Option<*const Self::Item> {
        self.buffer.get_ptr(position)
    }

    /// Gets a mutable reference to the item at requested position if it exists.
    fn get_mut_guarded(
        &mut self,
        position: usize,
        context: &Self::Context,
    ) -> Option<Self::ItemMutGuard> {
        let item = self.get_mut(position, context)?;
        Some(ItemMutGuard {
            item,
            previous: unsafe { (*item).clone() },
            ptrs: vec![&mut self.sum],
        })
    }

    fn get_mut(&mut self, position: usize, _context: &Self::Context) -> Option<*mut Self::Item> {
        self.buffer.get_mut_ptr(position)
    }

    fn push(&mut self, side: Side, item: Self::Item, _context: &Self::Context) {
        self.sum = self.sum.clone() + item.clone();
        self.buffer.push(side, item)
    }

    fn pop(&mut self, side: Side, _context: &Self::Context) -> Self::Item {
        let item = self.buffer.pop(side);
        self.sum = self.sum.clone() - item.clone();
        item
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
        let shared = self
            .buffer
            .decant_into(&mut destination.buffer, share_side, len);
        if shared != 0 {
            let mut decanted_sum = A::zero();
            for i in 0..shared {
                let idx = self.buffer.len() - i - 1;
                let sum = self.buffer.get(idx).unwrap().clone();
                decanted_sum = decanted_sum.clone() + sum;
            }
            self.sum = self.sum.clone() - decanted_sum.clone();
            destination.sum = destination.sum.clone() + decanted_sum;
        }
        shared
    }

    fn borrow_node(&mut self) -> Self::Borrowed {
        BorrowedLeaf {
            buffer: self.buffer.mutable_view(),
            sum: &mut self.sum as *mut _,
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

impl<A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero>
    Leaf<A>
{
    fn sum<R: RangeBounds<usize>>(&self, range: R) -> A {
        let start = match range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => x + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(x) => x + 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.len(),
        };
        let end = end.min(self.len());
        let mut result = A::zero();
        for idx in start..end {
            result = result.clone() + self.buffer.get(idx).unwrap().clone();
        }
        result
    }
}

pub struct BorrowedInternal<
    A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
    P: SharedPointerKind,
> {
    pub(crate) sizes: SharedPointer<SizeTable, P>,
    pub(crate) children: BorrowedChildList<A, P>,
    sum: *mut A,
}

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > Drop for BorrowedInternal<A, P>
{
    fn drop(&mut self) {
        // let f = match &self.children {
        //     BorrowedChildList::Internals(c) => c.represents_full_range(),
        //     BorrowedChildList::Leaves(c) => c.represents_full_range(),
        // };
        // println!("Dropping borrowedinternal")
        // println!("Dropping leaf")
        let mut new_sum = A::zero();
        match self.children {
            BorrowedChildList::Internals(ref buffer) => {
                for child in buffer.iter() {
                    new_sum = child.0.sum.clone() + new_sum;
                }
            }
            BorrowedChildList::Leaves(ref buffer) => {
                for child in buffer.iter() {
                    new_sum = child.0.as_ref().sum.clone() + new_sum;
                }
            }
        }
        unsafe {
            *self.sum = new_sum;
        }
    }
}

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > Clone for BorrowedInternal<A, P>
{
    fn clone(&self) -> Self {
        BorrowedInternal {
            sizes: SharedPointer::clone(&self.sizes),
            children: self.children.clone(),
            sum: self.sum,
        }
    }
}

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > BorrowedInternal<A, P>
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

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > std::fmt::Debug for BorrowedInternal<A, P>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BorrowedInternal").finish()
    }
}

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > BorrowedInternalTrait<Leaf<A>> for BorrowedInternal<A, P>
{
    type InternalChild = Internal<A, P>;
    type ItemMutGuard = DerefMutPtr<Self::InternalChild>;

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

    fn get_child_mut_at_slot(
        &mut self,
        idx: usize,
    ) -> Option<(NodeMut<Self::InternalChild, Leaf<A>>, Range<usize>)> {
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
    ) -> Option<(NodeMut<Self::InternalChild, Leaf<A>>, Range<usize>)> {
        let index = self.position_info_for(position)?.0;
        self.get_child_mut_at_slot(index)
    }

    fn pop_child(
        &mut self,
        side: Side,
        context: &(),
    ) -> Option<BorrowedNode<Self::InternalChild, Leaf<A>>> {
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

    fn split_at_child(&mut self, index: usize) -> (Self, Self) {
        let (left, right) = self.children.split_at(index);
        (
            BorrowedInternal {
                children: left,
                sizes: SharedPointer::clone(&self.sizes),
                sum: self.sum,
            },
            BorrowedInternal {
                children: right,
                sizes: SharedPointer::clone(&self.sizes),
                sum: self.sum,
            },
        )
    }

    fn split_at_position(&mut self, position: usize) -> (usize, Self, Self) {
        let index = self.position_info_for(position).unwrap();
        let result = self.split_at_child(index.0);
        (index.1, result.0, result.1)
    }
}

#[derive(Debug)]
pub(crate) enum BorrowedChildList<
    A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
    P: SharedPointerKind,
> {
    Internals(BorrowBufferMut<SharedPointerEntry<Internal<A, P>, P, ()>>),
    Leaves(BorrowBufferMut<SharedPointerEntry<Leaf<A>, P, ()>>),
}

impl<
        'a,
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > Clone for BorrowedChildList<A, P>
{
    fn clone(&self) -> Self {
        match self {
            BorrowedChildList::Internals(children) => {
                BorrowedChildList::Internals(children.clone())
            }
            BorrowedChildList::Leaves(children) => BorrowedChildList::Leaves(children.clone()),
        }
    }
}

impl<
        'a,
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > BorrowedChildList<A, P>
{
    fn split_at(&mut self, index: usize) -> (Self, Self) {
        match self {
            BorrowedChildList::Internals(ref mut children) => {
                let (left, right) = children.split_at(index);
                (
                    BorrowedChildList::Internals(left),
                    BorrowedChildList::Internals(right),
                )
            }
            BorrowedChildList::Leaves(ref mut children) => {
                let (left, right) = children.split_at(index);
                (
                    BorrowedChildList::Leaves(left),
                    BorrowedChildList::Leaves(right),
                )
            }
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
    A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
    P: SharedPointerKind,
> {
    Leaves(CircularBuffer<SharedPointerEntry<Leaf<A>, P, ()>>),
    Internals(CircularBuffer<SharedPointerEntry<Internal<A, P>, P, ()>>),
}

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > Clone for ChildList<A, P>
{
    fn clone(&self) -> Self {
        match self {
            ChildList::Leaves(buf) => ChildList::Leaves(buf.clone()),
            ChildList::Internals(buf) => ChildList::Internals(buf.clone()),
        }
    }
}

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > ChildList<A, P>
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
    pub fn leaves_ref(&self) -> &CircularBuffer<SharedPointerEntry<Leaf<A>, P, ()>> {
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
    pub fn internals_ref(&self) -> &CircularBuffer<SharedPointerEntry<Internal<A, P>, P, ()>> {
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
    pub fn leaves_mut(&mut self) -> &mut CircularBuffer<SharedPointerEntry<Leaf<A>, P, ()>> {
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
    ) -> &mut CircularBuffer<SharedPointerEntry<Internal<A, P>, P, ()>> {
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
    pub fn get(&self, child_idx: usize, idx: usize) -> Option<*const A> {
        match self {
            ChildList::Leaves(children) => children.get(child_idx).unwrap().load(&()).get(idx),
            ChildList::Internals(children) => children.get(child_idx).unwrap().load(&()).get(idx),
        }
    }

    pub fn get_mut(
        &mut self,
        child_idx: usize,
        idx: usize,
        context: &<Leaf<A> as LeafTrait>::Context,
    ) -> Option<*mut A> {
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

    pub fn get_mut_guarded(
        &mut self,
        child_idx: usize,
        idx: usize,
        context: &<Leaf<A> as LeafTrait>::Context,
    ) -> Option<ItemMutGuard<A>> {
        match self {
            ChildList::Leaves(children) => children
                .get_mut(child_idx)
                .unwrap()
                .load_mut(&())
                .get_mut_guarded(idx, context),
            ChildList::Internals(children) => children
                .get_mut(child_idx)
                .unwrap()
                .load_mut(&())
                .get_mut_guarded(idx, context),
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
    A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
    P: SharedPointerKind,
> {
    sizes: SharedPointer<SizeTable, P>,
    children: ChildList<A, P>,
    sum: A,
}

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > Clone for Internal<A, P>
{
    fn clone(&self) -> Self {
        Internal {
            sizes: SharedPointer::clone(&self.sizes),
            children: self.children.clone(),
            sum: self.sum.clone(),
        }
    }
}

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > Internal<A, P>
{
    fn sum<R: RangeBounds<usize>>(&self, range: R) -> A {
        let start = match range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => x + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(x) => x + 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.len(),
        };
        let end = end.min(self.len());

        if start == 0 && end == self.len() {
            self.sum.clone()
        } else {
            let mut result = A::zero();
            match self.children {
                ChildList::Internals(ref children) => {
                    let mut start_pos = 0;
                    for idx in 0..children.len() {
                        let child = children.get(idx).unwrap();
                        // result = result.clone() + children.get(idx).unwrap().clone();
                        let end_pos = start_pos + child.0.len();
                        if start_pos < end && end_pos > start {
                            let subrange = (start - start_pos)..(end - start_pos);
                            result = result.clone() + child.0.sum(subrange);
                        }
                        start_pos = end_pos;
                    }
                }
                ChildList::Leaves(ref children) => {
                    let mut start_pos = 0;
                    for idx in 0..children.len() {
                        let child = children.get(idx).unwrap();
                        // result = result.clone() + children.get(idx).unwrap().clone();
                        let end_pos = start_pos + child.0.len();
                        if start_pos < end && end_pos > start {
                            let subrange = (start.saturating_sub(start_pos))..(end - start_pos);
                            result = result.clone() + child.0.sum(subrange);
                        }
                        start_pos = end_pos;
                    }
                }
            }
            result
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

impl<
        A: Clone + std::fmt::Debug + std::ops::Add<Output = A> + std::ops::Sub<Output = A> + Zero,
        P: SharedPointerKind,
    > InternalTrait<Leaf<A>> for Internal<A, P>
{
    // type Item = A;
    type Borrowed = BorrowedInternal<A, P>;
    type Context = ();

    type LeafEntry = SharedPointerEntry<Leaf<A>, P, ()>;
    type InternalEntry = SharedPointerEntry<Self, P, ()>;
    type ItemMutGuard = ItemMutGuard<A>;

    fn empty_internal(level: usize) -> Self {
        debug_assert_ne!(level, 0); // Should be a Leaf
        if level == 1 {
            Internal {
                sizes: SharedPointer::new(SizeTable::new(1)),
                children: ChildList::Leaves(CircularBuffer::new()),
                sum: A::zero(),
            }
        } else {
            Internal {
                sizes: SharedPointer::new(SizeTable::new(level)),
                children: ChildList::Internals(CircularBuffer::new()),
                sum: A::zero(),
            }
        }
    }

    fn new_empty(&self) -> Self {
        Internal {
            sizes: SharedPointer::new(SizeTable::new(self.sizes.level())),
            children: self.children.new_empty(),
            sum: A::zero(),
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
            let mut decanted_sum = A::zero();
            for i in 0..shared {
                let idx = self.children.slots() - i - 1;
                let sum = match self.children {
                    ChildList::Internals(ref children) => children.get(idx).unwrap().0.sum.clone(),
                    ChildList::Leaves(ref children) => children.get(idx).unwrap().0.sum.clone(),
                };
                decanted_sum = decanted_sum.clone() + sum;
            }
            self.sum = self.sum.clone() - decanted_sum.clone();
            destination.sum = destination.sum.clone() + decanted_sum;
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
                            &mut *write.load_mut(context),
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
                            &mut *write.load_mut(context),
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

    fn get(&self, idx: usize) -> Option<*const A> {
        if let Some((array_idx, new_idx)) = self.sizes.position_info_for(idx) {
            self.children.get(array_idx, new_idx)
        } else {
            None
        }
    }

    fn get_mut_guarded(
        &mut self,
        idx: usize,
        context: &Self::Context,
    ) -> Option<Self::ItemMutGuard> {
        let (array_idx, new_idx) = self.sizes.position_info_for(idx)?;
        let mut guard = self.children.get_mut_guarded(array_idx, new_idx, context)?;
        guard.ptrs.push(&mut self.sum);
        Some(guard)
    }

    fn get_mut(&mut self, idx: usize, context: &Self::Context) -> Option<*mut A> {
        if let Some((array_idx, new_idx)) = self.sizes.position_info_for(idx) {
            self.children.get_mut(array_idx, new_idx, context)
        } else {
            None
        }
    }

    fn pop_child(&mut self, side: Side, _context: &Self::Context) -> NodeRc<Self, Leaf<A>> {
        SharedPointer::make_mut(&mut self.sizes).pop_child(side);
        let (sum, node) = match self.children {
            ChildList::Internals(ref mut children) => {
                let child = children.pop(side);
                (child.0.sum.clone(), NodeRc::Internal(child))
            }
            ChildList::Leaves(ref mut children) => {
                let child = children.pop(side);
                (child.0.sum.clone(), NodeRc::Leaf(child))
            }
        };
        self.sum = self.sum.clone() - sum;
        node
    }

    fn push_child(&mut self, side: Side, node: NodeRc<Self, Leaf<A>>, context: &Self::Context) {
        SharedPointer::make_mut(&mut self.sizes).push_child(side, node.len(context));
        let sum = match &node {
            NodeRc::Internal(internal) => internal.0.sum.clone(),
            NodeRc::Leaf(leaf) => leaf.0.sum.clone(),
        };
        self.sum = self.sum.clone() + sum;
        match self.children {
            ChildList::Internals(ref mut children) => children.push(side, node.internal()),
            ChildList::Leaves(ref mut children) => children.push(side, node.leaf()),
        }
    }

    fn borrow_node(&mut self) -> Self::Borrowed {
        BorrowedInternal {
            children: self.children.borrow(),
            sizes: SharedPointer::clone(&self.sizes),
            sum: &mut self.sum as *mut _,
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

    fn get_child_ref_at_slot(&self, idx: usize) -> Option<(NodeRef<Self, Leaf<A>>, Range<usize>)> {
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
    ) -> Option<(NodeRef<Self, Leaf<A>>, Range<usize>)> {
        if let Some((child_idx, _)) = self.sizes.position_info_for(position) {
            self.get_child_ref_at_slot(child_idx)
        } else {
            None
        }
    }

    fn get_child_mut_at_slot(
        &mut self,
        idx: usize,
    ) -> Option<(NodeMut<Self, Leaf<A>>, Range<usize>)> {
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
    ) -> Option<(NodeMut<Self, Leaf<A>>, Range<usize>)> {
        if let Some((child_idx, _)) = self.sizes.position_info_for(position) {
            self.get_child_mut_at_slot(child_idx)
        } else {
            None
        }
    }

    fn split_at_child(&mut self, idx: usize, context: &Self::Context) -> Self {
        // TODO: Need to set the sum values correctly
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
        // TODO: Need to set the sum values correctly
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
        iter: &mut std::slice::Iter<'a, A>,
        context: &Self::Context,
    ) -> bool
    where
        A: PartialEq,
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
pub trait SumVector<A> {
    fn sum<R: RangeBounds<usize>>(&self, range: R) -> A;

    fn full_sum(&self) -> A {
        self.sum(..)
    }
}

impl SumVector<usize>
    for crate::InternalVector<
        crate::node_impls::annotated_basic::Internal<usize, archery::RcK>,
        crate::node_impls::annotated_basic::Leaf<usize>,
        crate::node_impls::annotated_basic::BorrowedInternal<usize, archery::RcK>,
    >
{
    fn sum<R: RangeBounds<usize>>(&self, range: R) -> usize {
        let start = match range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => x + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(x) => x + 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.len(),
        };
        let mut result = 0;
        let mut start_pos = 0;
        for node_idx in self.spine_iter() {
            let node = node_idx.1;
            let end_pos = start_pos + node.len(&self.context);
            println!("Range derp {:?}", start_pos..end_pos);
            if start_pos < end && end_pos > start {
                let subrange = start.saturating_sub(start_pos)..(end - start_pos);
                println!(
                    "Hitting sum subrange {:?} at {:?}",
                    subrange,
                    start_pos..end_pos
                );
                result += match node {
                    NodeRc::Internal(ref node) => {
                        let internal: &Internal<usize, archery::RcK> = &*node.0;
                        internal.sum(subrange)
                    }
                    NodeRc::Leaf(ref node) => {
                        let leaf: &Leaf<usize> = &*node.0;
                        leaf.sum(subrange)
                    }
                };
            }
            start_pos = end_pos;
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::SumVector;
    use crate::*;
    use crossbeam;

    #[test]
    pub fn split_focus_mut_annotations() {
        let mut v: InternalVector<
            node_impls::annotated_basic::Internal<usize, archery::RcK>,
            node_impls::annotated_basic::Leaf<usize>,
            node_impls::annotated_basic::BorrowedInternal<usize, archery::RcK>,
        > = InternalVector::new();
        const N: usize = 1_000;
        for i in 0..N {
            v.push_back(1);
        }
        // for i in 0..N {
        //     *v.get_mut_guarded(i).unwrap() = 1;
        // }

        println!("Kappa {:#?}", v);
        println!("SumKappa {:#?}", v.sum(6..506));

        // const S: usize = N / 2;
        // println!("Prebuild");
        // {
        //     let mut focus = v.focus_mut();
        //     println!("Presplit");
        //     {
        //         let (mut left, mut right) = focus.split_at(S);
        //         println!("Postsplit");
        //         for i in 0..S {
        //             let thing = left.get(i);
        //             if let Some(thing) = thing {
        //                 *thing = 0;
        //             }
        //         }
        //         for i in 0..N - S {
        //             let thing = right.get(i);
        //             if let Some(thing) = thing {
        //                 *thing = 1;
        //             }
        //         }
        //     }
        //     for i in 0..N {
        //         if i < S {
        //             assert_eq!(focus.get(i), Some(&mut 0));
        //         } else {
        //             assert_eq!(focus.get(i), Some(&mut 1));
        //         }
        //     }
        //     println!("Predrop");
        // }
        // println!("Postdrop");
        // for (i, v) in v.iter().enumerate() {
        //     if i < S {
        //         assert_eq!(v, &0);
        //     } else {
        //         assert_eq!(v, &1);
        //     }
        // }
        // println!("Post test");
        // println!("Kappa {:#?}", v);
    }
}
