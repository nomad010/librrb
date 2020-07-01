//! Collection of nodes used for the RRB tree.

use crate::circular::{BorrowBufferMut, CircularBuffer};
use crate::size_table::SizeTable;
use crate::{Side, RRB_WIDTH};
use archery::{ArcK, RcK, SharedPointer, SharedPointerKind};
use std::cmp;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut, Range};
use std::rc::Rc;

/// A leaf indicates a terminal node in the tree.
#[derive(Debug)]
pub(crate) struct Leaf<A: Debug> {
    pub buffer: CircularBuffer<A>,
}

impl<A: Debug> Leaf<A> {
    /// Sorts the leaf by the given comparator.
    pub fn sort_by<F: FnMut(&A, &A) -> cmp::Ordering>(&mut self, range: Range<usize>, f: &mut F) {
        self.buffer.sort_by(range, f)
    }

    /// Constructs a new empty leaf.
    pub fn empty() -> Self {
        Leaf {
            buffer: CircularBuffer::new(),
        }
    }

    /// Constructs a new leaf with a single item.
    pub fn with_item(item: A) -> Self {
        Leaf {
            buffer: CircularBuffer::with_item(item),
        }
    }

    /// Constructs a new leaf from the given items.
    ///
    /// # Panics
    ///
    /// This will panic if the length of items is greater than `RRB_WIDTH`.
    pub fn with_items(items: Vec<A>) -> Self {
        Leaf {
            buffer: CircularBuffer::with_items(items),
        }
    }

    /// Removes elements from `self` and inserts these elements into `destination`. At most `len`
    /// elements will be removed. The actual number of elements decanted is returned. Elements are
    /// popped from `share_side` but pushed into the destination via `share_side.negate()`
    pub fn share_children_with(
        &mut self,
        destination: &mut Self,
        share_side: Side,
        len: usize,
    ) -> usize {
        self.buffer.decant_into(destination, share_side, len)
    }

    /// Returns the level of the node. Always 0 in the case of leaves.
    pub fn level(&self) -> usize {
        0
    }

    /// Returns whether the node is a leaf. Always true in the case of leaves.
    pub fn is_leaf(&self) -> bool {
        true
    }

    /// Updates an element in the leaf.
    pub fn update(&mut self, idx: usize, item: A) {
        *self.buffer.get_mut(idx).unwrap() = item;
    }

    /// Checks whether the node corresponds to its size table, Leaves do not have these tables so
    /// this never fails.
    pub fn debug_check_lens_size_table(&self) -> (usize, bool) {
        (self.len(), true)
    }

    /// Returns the number of used positions in the node. This is equivalent to len in leaves.
    pub fn slots(&self) -> usize {
        self.len()
    }

    /// Returns the number of unused positions in the node. This is equivalent to the amount of
    /// space left in the leaf.
    pub fn free_slots(&self) -> usize {
        self.free_space()
    }

    pub fn borrow_node(&mut self) -> BorrowedLeaf<A> {
        BorrowedLeaf {
            buffer: self.buffer.mutable_view(),
        }
    }

    /// Checks whether the node's heights and heights of its children make sense. This never fails
    /// for leaves.
    pub(crate) fn debug_check_child_heights(&self) -> usize {
        debug_assert_eq!(self.level(), 0);
        debug_assert!(self.is_leaf());
        self.level()
    }
}

impl<A: Debug + PartialEq> Leaf<A> {
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_iter<'a>(&self, iter: &mut std::slice::Iter<'a, A>) -> bool {
        for value in &self.buffer {
            if value != iter.next().unwrap() {
                return false;
            }
        }
        true
    }

    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_iter_debug<'a>(&self, iter: &mut std::slice::Iter<'a, A>) {
        for value in &self.buffer {
            let iter_value = iter.next().unwrap();
            println!("{:?} == {:?} = {} ", value, iter_value, value == iter_value);
        }
    }
}

impl<A: Clone + Debug> Clone for Leaf<A> {
    fn clone(&self) -> Self {
        Leaf {
            buffer: self.buffer.clone(),
        }
    }
}

impl<A: Debug> Deref for Leaf<A> {
    type Target = CircularBuffer<A>;

    fn deref(&self) -> &CircularBuffer<A> {
        &self.buffer
    }
}

impl<A: Debug> DerefMut for Leaf<A> {
    fn deref_mut(&mut self) -> &mut CircularBuffer<A> {
        &mut self.buffer
    }
}

#[derive(Debug)]
pub(crate) struct BorrowedLeaf<A: Debug> {
    buffer: BorrowBufferMut<A>,
}

impl<'a, A: Debug> BorrowedLeaf<A> {
    pub fn empty(&mut self) -> Self {
        BorrowedLeaf {
            buffer: self.buffer.empty(),
        }
    }

    pub fn update(&mut self, idx: usize, item: A) {
        *self.buffer.get_mut(idx).unwrap() = item;
    }
    pub fn split_at(&mut self, idx: usize) -> Self {
        let buffer = self.buffer.split_at(idx);
        BorrowedLeaf { buffer }
    }
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn level(&self) -> usize {
        0
    }

    pub fn from_same_source(&self, other: &Self) -> bool {
        self.buffer.from_same_source(&other.buffer)
    }

    pub fn combine(&mut self, other: Self) {
        self.buffer.combine(other.buffer)
    }

    pub fn get(&self, idx: usize) -> Option<&A> {
        self.buffer.get(idx)
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut A> {
        self.buffer.get_mut(idx)
    }
}

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

impl<A: Clone + Debug, P: SharedPointerKind> ChildList<A, P> {
    /// Constructs a new empty list of nodes.
    pub fn new_empty(&self) -> Self {
        match self {
            ChildList::Internals(_) => ChildList::Internals(CircularBuffer::new()),
            ChildList::Leaves(_) => ChildList::Leaves(CircularBuffer::new()),
        }
    }

    /// Consumes `self` and returns the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves(self) -> CircularBuffer<SharedPointer<Leaf<A>, P>> {
        if let ChildList::Leaves(x) = self {
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
    pub fn internals(self) -> CircularBuffer<SharedPointer<Internal<A, P>, P>> {
        if let ChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
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
    pub fn get_child_node(&self, idx: usize) -> Option<NodeRc<A, P>> {
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

// impl<A: Clone + std::fmt::Debug, P: SharedPointerKind> Internal<A, P> {
//     /// Constructs a new empty internal node that is at level 1 in the tree.
//     pub fn empty_leaves() -> Self {
//         Internal {
//             sizes: SharedPointer::new(SizeTable::new(1)),
//             children: ChildList::Leaves(CircularBuffer::new()),
//         }
//     }

//     /// Constructs a new empty internal node that is at the given level in the tree.
//     pub fn empty_internal(level: usize) -> Self {
//         debug_assert_ne!(level, 0); // Should be a Leaf
//         if level == 1 {
//             Self::empty_leaves()
//         } else {
//             Internal {
//                 sizes: SharedPointer::new(SizeTable::new(level)),
//                 children: ChildList::Internals(CircularBuffer::new()),
//             }
//         }
//     }

//     /// Constructs a new internal node of the same level, but with no children.
//     pub fn new_empty(&self) -> Self {
//         Internal {
//             sizes: SharedPointer::new(SizeTable::new(self.sizes.level())),
//             children: self.children.new_empty(),
//         }
//     }

//     /// Removes elements from `self` and inserts these elements into `destination`. At most `len`
//     /// elements will be removed. The actual number of elements decanted is returned. Elements are
//     /// popped from `share_side` but pushed into the destination via `share_side.negate()`. Both
//     /// self and destination should be at the same level in the tree.
//     pub fn share_children_with(
//         &mut self,
//         destination: &mut Self,
//         share_side: Side,
//         len: usize,
//     ) -> usize {
//         debug_assert_eq!(self.level(), destination.level());
//         let shared = match self.children {
//             ChildList::Internals(ref mut children) => {
//                 debug_assert_eq!(children.len(), self.sizes.len());
//                 children.decant_into(destination.children.internals_mut(), share_side, len)
//             }
//             ChildList::Leaves(ref mut children) => {
//                 debug_assert_eq!(children.len(), self.sizes.len());
//                 children.decant_into(destination.children.leaves_mut(), share_side, len)
//             }
//         };
//         if shared != 0 {
//             let origin_sizes = SharedPointer::make_mut(&mut self.sizes);
//             let destination_sizes = SharedPointer::make_mut(&mut destination.sizes);

//             for _ in 0..shared {
//                 destination_sizes
//                     .push_child(share_side.negate(), origin_sizes.pop_child(share_side));
//             }
//         }

//         shared
//     }

//     /// Packs the children of the to the left of the node so that all but the last is dense.
//     pub fn pack_children(&mut self) {
//         if self.is_empty() {
//             return;
//         }
//         if self.level() == 1 {
//             assert_eq!(self.children.leaves_ref().len(), self.sizes.len());
//         } else {
//             assert_eq!(self.children.internals_ref().len(), self.sizes.len());
//         }
//         // TODO: Genericize on a side
//         let mut write_position = 0;
//         let mut read_position = write_position + 1;
//         match self.children {
//             ChildList::Internals(ref mut children) => {
//                 while read_position < children.len() {
//                     if read_position == write_position {
//                         read_position += 1;
//                         continue;
//                     } else {
//                         let (write, read) = children.pair_mut(write_position, read_position);
//                         SharedPointer::make_mut(read).share_children_with(
//                             SharedPointer::make_mut(write),
//                             Side::Front,
//                             RRB_WIDTH,
//                         );

//                         if write.is_full() {
//                             write_position += 1;
//                         }
//                         if read.is_empty() {
//                             read_position += 1;
//                         }
//                     }
//                 }
//                 while children.back().unwrap().is_empty() {
//                     children.pop_back();
//                 }
//                 let sizes = SharedPointer::make_mut(&mut self.sizes);
//                 *sizes = SizeTable::new(sizes.level());
//                 for child in children {
//                     sizes.push_child(Side::Back, child.len());
//                 }
//             }
//             ChildList::Leaves(ref mut children) => {
//                 while read_position < children.len() {
//                     if read_position == write_position {
//                         read_position += 1;
//                         continue;
//                     } else {
//                         let (write, read) = children.pair_mut(write_position, read_position);
//                         SharedPointer::make_mut(read).share_children_with(
//                             SharedPointer::make_mut(write),
//                             Side::Front,
//                             RRB_WIDTH,
//                         );

//                         if write.is_full() {
//                             write_position += 1;
//                         }
//                         if read.is_empty() {
//                             read_position += 1;
//                         }
//                     }
//                 }

//                 while children.back().unwrap().is_empty() {
//                     children.pop_back();
//                 }
//                 let sizes = SharedPointer::make_mut(&mut self.sizes);
//                 *sizes = SizeTable::new(sizes.level());
//                 for child in children {
//                     sizes.push_child(Side::Back, child.len());
//                 }
//             }
//         }

//         if self.level() == 1 {
//             assert_eq!(self.children.leaves_ref().len(), self.sizes.len());
//         } else {
//             assert_eq!(self.children.internals_ref().len(), self.sizes.len());
//         }
//     }

//     /// Returns a reference to the element at the given index in the tree.
//     pub fn get(&self, idx: usize) -> Option<&A> {
//         if let Some((array_idx, new_idx)) = self.sizes.position_info_for(idx) {
//             self.children.get(array_idx, new_idx)
//         } else {
//             None
//         }
//     }

//     pub fn get_mut(&mut self, idx: usize) -> Option<&mut A> {
//         if let Some((array_idx, new_idx)) = self.sizes.position_info_for(idx) {
//             self.children.get_mut(array_idx, new_idx)
//         } else {
//             None
//         }
//     }

//     /// Overwrites the element at the given index in the tree with a new value.
//     pub fn update(&mut self, idx: usize, item: A) {
//         let (array_idx, new_idx) = self.sizes.position_info_for(idx).unwrap();

//         match &mut self.children {
//             ChildList::Internals(children) => {
//                 SharedPointer::make_mut(children.get_mut(array_idx).unwrap()).update(new_idx, item)
//             }
//             ChildList::Leaves(children) => {
//                 SharedPointer::make_mut(children.get_mut(array_idx).unwrap()).update(new_idx, item)
//             }
//         }
//     }

//     pub fn borrow_node(&mut self) -> BorrowedInternal<A, P> {
//         BorrowedInternal {
//             children: self.children.borrow(),
//             sizes: Rc::clone(&self.sizes),
//         }
//     }

//     /// Returns the level the node is at in the tree.
//     pub fn level(&self) -> usize {
//         self.sizes.level()
//     }

//     /// Returns the size(number of elements hanging off) of the node.
//     pub fn len(&self) -> usize {
//         debug_assert_eq!(
//             self.debug_check_lens_size_table(),
//             self.sizes.cumulative_size()
//         );
//         self.sizes.cumulative_size()
//     }

//     /// Returns whether the node is empty. This should almost never be the case.
//     pub fn is_empty(&self) -> bool {
//         self.len() == 0
//     }

//     /// Returns the number of elements that could be inserted into the node.
//     pub fn free_space(&self) -> usize {
//         debug_assert_eq!(
//             RRB_WIDTH.pow(1 + self.level() as u32) - self.len(),
//             self.sizes.free_space()
//         );
//         self.sizes.free_space()
//     }

//     /// Returns the number of direct children of the node.
//     pub fn slots(&self) -> usize {
//         debug_assert_eq!(self.sizes.len(), self.children.slots());
//         self.children.slots()
//     }

//     /// Returns the number of direct children that could be inserted into the node.
//     pub fn free_slots(&self) -> usize {
//         debug_assert_eq!(RRB_WIDTH - self.sizes.len(), self.children.free_slots());
//         self.children.free_slots()
//     }

//     /// Returns whether no more children can be inserted into the node.
//     pub fn is_full(&self) -> bool {
//         self.free_slots() == 0
//     }

//     /// Returns whether the node is a leaf. This is always false for an internal node.
//     pub fn is_leaf(&self) -> bool {
//         false
//     }

//     /// Returns the position of the child that corresponds to the given index along with a new
//     /// index to query in that child.
//     pub fn position_info_for(&self, idx: usize) -> Option<(usize, usize)> {
//         self.sizes.position_info_for(idx)
//     }

//     /// Checks whether the node corresponds to its size table.
//     pub fn debug_check_lens_size_table(&self) -> usize {
//         let mut cumulative_size = 0;
//         match &self.children {
//             ChildList::Internals(children) => {
//                 debug_assert_eq!(self.sizes.len(), children.len());
//                 for i in 0..children.len() {
//                     debug_assert_eq!(
//                         children.get(i).unwrap().len(),
//                         self.sizes.get_child_size(i).unwrap()
//                     );
//                     cumulative_size += children.get(i).unwrap().len();
//                 }
//                 cumulative_size
//             }
//             ChildList::Leaves(children) => {
//                 debug_assert_eq!(self.sizes.len(), children.len());
//                 for i in 0..children.len() {
//                     debug_assert_eq!(
//                         children.get(i).unwrap().len(),
//                         self.sizes.get_child_size(i).unwrap()
//                     );
//                     cumulative_size += children.get(i).unwrap().len();
//                 }
//                 cumulative_size
//             }
//         }
//     }

//     /// Checks whether the node's heights and heights of its children make sense.
//     pub fn debug_check_child_heights(&self) -> usize {
//         match &self.children {
//             ChildList::Internals(children) => {
//                 for i in children {
//                     debug_assert_eq!(i.debug_check_child_heights() + 1, self.level());
//                 }
//             }
//             ChildList::Leaves(children) => {
//                 for i in children {
//                     debug_assert_eq!(i.debug_check_child_heights() + 1, self.level());
//                 }
//             }
//         }
//         self.level()
//     }

//     /// Check the invariants of the node.
//     pub fn debug_check_invariants(&self) -> bool {
//         self.debug_check_child_heights();
//         self.len();
//         true
//     }
// }

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

    /// Returns a reference to the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves_ref(&self) -> &BorrowBufferMut<SharedPointer<Leaf<A>, P>> {
        if let BorrowedChildList::Leaves(x) = self {
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
    pub fn internals_ref(&self) -> &BorrowBufferMut<SharedPointer<Internal<A, P>, P>> {
        if let BorrowedChildList::Internals(x) = self {
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
    pub fn leaves_mut(&mut self) -> &mut BorrowBufferMut<SharedPointer<Leaf<A>, P>> {
        if let BorrowedChildList::Leaves(x) = self {
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
    pub fn internals_mut(&mut self) -> &mut BorrowBufferMut<SharedPointer<Internal<A, P>, P>> {
        if let BorrowedChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
        }
    }

    fn empty(&mut self) -> Self {
        match self {
            BorrowedChildList::Internals(children) => {
                BorrowedChildList::Internals(children.empty())
            }
            BorrowedChildList::Leaves(children) => BorrowedChildList::Leaves(children.empty()),
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

    fn len(&self) -> usize {
        match self {
            BorrowedChildList::Internals(children) => children.len(),
            BorrowedChildList::Leaves(children) => children.len(),
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

/// An internal node indicates a non-terminal node in the tree.
#[derive(Debug)]
pub(crate) struct Internal<A: Clone + std::fmt::Debug, P: SharedPointerKind> {
    pub sizes: SharedPointer<SizeTable, P>,
    pub children: ChildList<A, P>,
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind> Clone for Internal<A, P> {
    fn clone(&self) -> Self {
        Internal {
            sizes: self.sizes.clone(),
            children: self.children.clone(),
        }
    }
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind> Internal<A, P> {
    /// Constructs a new empty internal node that is at level 1 in the tree.
    pub fn empty_leaves() -> Self {
        Internal {
            sizes: SharedPointer::new(SizeTable::new(1)),
            children: ChildList::Leaves(CircularBuffer::new()),
        }
    }

    /// Constructs a new empty internal node that is at the given level in the tree.
    pub fn empty_internal(level: usize) -> Self {
        debug_assert_ne!(level, 0); // Should be a Leaf
        if level == 1 {
            Self::empty_leaves()
        } else {
            Internal {
                sizes: SharedPointer::new(SizeTable::new(level)),
                children: ChildList::Internals(CircularBuffer::new()),
            }
        }
    }

    /// Constructs a new internal node of the same level, but with no children.
    pub fn new_empty(&self) -> Self {
        Internal {
            sizes: SharedPointer::new(SizeTable::new(self.sizes.level())),
            children: self.children.new_empty(),
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

    /// Packs the children of the to the left of the node so that all but the last is dense.
    pub fn pack_children(&mut self) {
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
                        SharedPointer::make_mut(read).share_children_with(
                            SharedPointer::make_mut(write),
                            Side::Front,
                            RRB_WIDTH,
                        );

                        if write.is_full() {
                            write_position += 1;
                        }
                        if read.is_empty() {
                            read_position += 1;
                        }
                    }
                }
                while children.back().unwrap().is_empty() {
                    children.pop_back();
                }
                let sizes = SharedPointer::make_mut(&mut self.sizes);
                *sizes = SizeTable::new(sizes.level());
                for child in children {
                    sizes.push_child(Side::Back, child.len());
                }
            }
            ChildList::Leaves(ref mut children) => {
                while read_position < children.len() {
                    if read_position == write_position {
                        read_position += 1;
                        continue;
                    } else {
                        let (write, read) = children.pair_mut(write_position, read_position);
                        SharedPointer::make_mut(read).share_children_with(
                            SharedPointer::make_mut(write),
                            Side::Front,
                            RRB_WIDTH,
                        );

                        if write.is_full() {
                            write_position += 1;
                        }
                        if read.is_empty() {
                            read_position += 1;
                        }
                    }
                }

                while children.back().unwrap().is_empty() {
                    children.pop_back();
                }
                let sizes = SharedPointer::make_mut(&mut self.sizes);
                *sizes = SizeTable::new(sizes.level());
                for child in children {
                    sizes.push_child(Side::Back, child.len());
                }
            }
        }

        if self.level() == 1 {
            assert_eq!(self.children.leaves_ref().len(), self.sizes.len());
        } else {
            assert_eq!(self.children.internals_ref().len(), self.sizes.len());
        }
    }

    /// Returns a reference to the element at the given index in the tree.
    pub fn get(&self, idx: usize) -> Option<&A> {
        if let Some((array_idx, new_idx)) = self.sizes.position_info_for(idx) {
            self.children.get(array_idx, new_idx)
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut A> {
        if let Some((array_idx, new_idx)) = self.sizes.position_info_for(idx) {
            self.children.get_mut(array_idx, new_idx)
        } else {
            None
        }
    }

    /// Overwrites the element at the given index in the tree with a new value.
    pub fn update(&mut self, idx: usize, item: A) {
        let (array_idx, new_idx) = self.sizes.position_info_for(idx).unwrap();

        match &mut self.children {
            ChildList::Internals(children) => {
                SharedPointer::make_mut(children.get_mut(array_idx).unwrap()).update(new_idx, item)
            }
            ChildList::Leaves(children) => {
                SharedPointer::make_mut(children.get_mut(array_idx).unwrap()).update(new_idx, item)
            }
        }
    }

    pub fn borrow_node(&mut self) -> BorrowedInternal<A, P> {
        BorrowedInternal {
            children: self.children.borrow(),
            sizes: SharedPointer::clone(&self.sizes),
        }
    }

    /// Returns the level the node is at in the tree.
    pub fn level(&self) -> usize {
        self.sizes.level()
    }

    /// Returns the size(number of elements hanging off) of the node.
    pub fn len(&self) -> usize {
        debug_assert_eq!(
            self.debug_check_lens_size_table(),
            self.sizes.cumulative_size()
        );
        self.sizes.cumulative_size()
    }

    /// Returns whether the node is empty. This should almost never be the case.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements that could be inserted into the node.
    pub fn free_space(&self) -> usize {
        debug_assert_eq!(
            RRB_WIDTH.pow(1 + self.level() as u32) - self.len(),
            self.sizes.free_space()
        );
        self.sizes.free_space()
    }

    /// Returns the number of direct children of the node.
    pub fn slots(&self) -> usize {
        debug_assert_eq!(self.sizes.len(), self.children.slots());
        self.children.slots()
    }

    /// Returns the number of direct children that could be inserted into the node.
    pub fn free_slots(&self) -> usize {
        debug_assert_eq!(RRB_WIDTH - self.sizes.len(), self.children.free_slots());
        self.children.free_slots()
    }

    /// Returns whether no more children can be inserted into the node.
    pub fn is_full(&self) -> bool {
        self.free_slots() == 0
    }

    /// Returns whether the node is a leaf. This is always false for an internal node.
    pub fn is_leaf(&self) -> bool {
        false
    }

    /// Returns the position of the child that corresponds to the given index along with a new
    /// index to query in that child.
    pub fn position_info_for(&self, idx: usize) -> Option<(usize, usize)> {
        self.sizes.position_info_for(idx)
    }

    /// Checks whether the node corresponds to its size table.
    pub fn debug_check_lens_size_table(&self) -> usize {
        let mut cumulative_size = 0;
        match &self.children {
            ChildList::Internals(children) => {
                debug_assert_eq!(self.sizes.len(), children.len());
                for i in 0..children.len() {
                    debug_assert_eq!(
                        children.get(i).unwrap().len(),
                        self.sizes.get_child_size(i).unwrap()
                    );
                    cumulative_size += children.get(i).unwrap().len();
                }
                cumulative_size
            }
            ChildList::Leaves(children) => {
                debug_assert_eq!(self.sizes.len(), children.len());
                for i in 0..children.len() {
                    debug_assert_eq!(
                        children.get(i).unwrap().len(),
                        self.sizes.get_child_size(i).unwrap()
                    );
                    cumulative_size += children.get(i).unwrap().len();
                }
                cumulative_size
            }
        }
    }

    /// Checks whether the node's heights and heights of its children make sense.
    pub fn debug_check_child_heights(&self) -> usize {
        match &self.children {
            ChildList::Internals(children) => {
                for i in children {
                    debug_assert_eq!(i.debug_check_child_heights() + 1, self.level());
                }
            }
            ChildList::Leaves(children) => {
                for i in children {
                    debug_assert_eq!(i.debug_check_child_heights() + 1, self.level());
                }
            }
        }
        self.level()
    }

    /// Check the invariants of the node.
    pub fn debug_check_invariants(&self) -> bool {
        self.debug_check_child_heights();
        self.len();
        true
    }
}

impl<A: Clone + std::fmt::Debug + PartialEq, P: SharedPointerKind> Internal<A, P> {
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_iter<'a>(&self, iter: &mut std::slice::Iter<'a, A>) -> bool {
        match self.children {
            ChildList::Internals(ref children) => {
                for child in children {
                    if !child.equal_iter(iter) {
                        return false;
                    }
                }
            }
            ChildList::Leaves(ref children) => {
                for child in children {
                    if !child.equal_iter(iter) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_iter_debug<'a>(&self, iter: &mut std::slice::Iter<'a, A>) {
        match self.children {
            ChildList::Internals(ref children) => {
                for child in children {
                    child.equal_iter_debug(iter);
                }
            }
            ChildList::Leaves(ref children) => {
                for child in children {
                    child.equal_iter_debug(iter);
                }
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct BorrowedInternal<A: Clone + Debug, P: SharedPointerKind> {
    pub(crate) sizes: SharedPointer<SizeTable, P>,
    pub(crate) children: BorrowedChildList<A, P>,
}

impl<'a, A: Clone + Debug, P: SharedPointerKind> BorrowedInternal<A, P> {
    fn empty(&mut self) -> Self {
        BorrowedInternal {
            sizes: SharedPointer::clone(&self.sizes),
            children: self.children.empty(),
        }
    }

    pub fn left_size(&self) -> usize {
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

    pub fn len(&self) -> usize {
        let range = self.children.range();
        if range.start == range.end {
            0
        } else {
            self.sizes.get_cumulative_child_size(range.end - 1).unwrap() - self.left_size()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn level(&self) -> usize {
        self.sizes.level()
    }

    pub fn from_same_source(&self, other: &Self) -> bool {
        self.children.from_same_source(&other.children)
    }

    pub fn combine(&mut self, other: Self) {
        self.children.combine(other.children)
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

    pub fn split_at_child(&mut self, index: usize) -> Self {
        let new_children = self.children.split_at(index);
        BorrowedInternal {
            children: new_children,
            sizes: SharedPointer::clone(&self.sizes),
        }
    }

    pub fn debug_check_lens_size_table(&self) -> usize {
        let mut cumulative_size = 0;
        match &self.children {
            BorrowedChildList::Internals(children) => {
                // debug_assert_eq!(self.sizes.len(), children.len());
                for i in 0..children.len() {
                    // println!(
                    //     "{} vs {}",
                    //     children.get(i).unwrap().len(),
                    //     children.get(i).unwrap().debug_check_lens_size_table()
                    // );
                    debug_assert_eq!(
                        children.get(i).unwrap().len(),
                        children.get(i).unwrap().debug_check_lens_size_table()
                    );
                    cumulative_size += children.get(i).unwrap().len();
                }
                cumulative_size
            }
            BorrowedChildList::Leaves(children) => {
                // debug_assert_eq!(self.sizes.len(), children.len());
                for i in 0..children.len() {
                    // println!(
                    //     "{} vs {}",
                    //     children.get(i).unwrap().len(),
                    //     self.sizes.get_child_size(i).unwrap()
                    // );
                    debug_assert_eq!(
                        children.get(i).unwrap().len(),
                        self.sizes.get_child_size(i).unwrap()
                    );
                    cumulative_size += children.get(i).unwrap().len();
                }
                cumulative_size
            }
        }
    }
}

/// Represents an arbitrary node in the tree.
#[derive(Debug)]
pub(crate) enum NodeRc<A: Clone + std::fmt::Debug, P: SharedPointerKind> {
    Leaf(SharedPointer<Leaf<A>, P>),
    Internal(SharedPointer<Internal<A, P>, P>),
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind> Clone for NodeRc<A, P> {
    fn clone(&self) -> Self {
        match self {
            NodeRc::Leaf(buf) => NodeRc::Leaf(buf.clone()),
            NodeRc::Internal(buf) => NodeRc::Internal(buf.clone()),
        }
    }
}

impl<A: Clone + std::fmt::Debug, P: SharedPointerKind> NodeRc<A, P> {
    /// Constructs a new empty of the same level.
    pub fn new_empty(&self) -> Self {
        match self {
            NodeRc::Internal(x) => NodeRc::Internal(SharedPointer::new(x.new_empty())),
            NodeRc::Leaf(_) => NodeRc::Leaf(SharedPointer::new(Leaf::empty())),
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
                let destination_internal = SharedPointer::make_mut(destination.internal_mut());
                SharedPointer::make_mut(origin_internal).share_children_with(
                    destination_internal,
                    share_side,
                    len,
                )
            }
            NodeRc::Leaf(ref mut origin_leaf) => {
                let destination_leaf = SharedPointer::make_mut(destination.leaf_mut());
                SharedPointer::make_mut(origin_leaf).share_children_with(
                    destination_leaf,
                    share_side,
                    len,
                )
            }
        }
    }

    /// Returns the size of the of node.
    pub fn len(&self) -> usize {
        match self {
            NodeRc::Leaf(x) => x.len(),
            NodeRc::Internal(x) => x.len(),
        }
    }

    /// Returns a reference to the size table if it has one.
    pub fn size_table_ref(&self) -> Option<&SizeTable> {
        match self {
            NodeRc::Leaf(_) => None,
            NodeRc::Internal(x) => Some(&x.sizes),
        }
    }

    /// Returns the number of direct children of the node.
    pub fn slots(&self) -> usize {
        match self {
            NodeRc::Leaf(x) => x.len(),
            _ => self.size_table_ref().unwrap().len(),
        }
    }

    /// Returns whether the node is completely empty.
    pub fn is_empty(&self) -> bool {
        match self {
            NodeRc::Leaf(x) => x.is_empty(),
            NodeRc::Internal(x) => x.is_empty(),
        }
    }

    /// Returns the level of the node.
    pub fn level(&self) -> usize {
        match self {
            NodeRc::Leaf(x) => x.level(),
            NodeRc::Internal(x) => x.level(),
        }
    }

    /// Tests whether the node is a leaf.
    pub fn is_leaf(&self) -> bool {
        match self {
            NodeRc::Leaf(x) => x.is_leaf(),
            NodeRc::Internal(x) => x.is_leaf(),
        }
    }

    /// Returns the element at the given position in the node.
    pub fn get(&self, idx: usize) -> Option<&A> {
        match self {
            NodeRc::Leaf(x) => x.get(idx),
            NodeRc::Internal(x) => x.get(idx),
        }
    }

    /// Returns the element at the given position in the node.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut A> {
        match self {
            NodeRc::Leaf(ref mut x) => SharedPointer::make_mut(x).get_mut(idx),
            NodeRc::Internal(ref mut x) => SharedPointer::make_mut(x).get_mut(idx),
        }
    }

    /// Returns the number of elements that can be inserted into this node.
    pub fn free_space(&self) -> usize {
        match self {
            NodeRc::Leaf(x) => x.free_space(),
            NodeRc::Internal(x) => x.free_space(),
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub fn free_slots(&self) -> usize {
        match self {
            NodeRc::Leaf(x) => x.free_slots(),
            NodeRc::Internal(x) => x.free_slots(),
        }
    }

    /// Consumes `self` and returns the node as a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a leaf node.
    pub fn leaf(self) -> SharedPointer<Leaf<A>, P> {
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
    pub fn internal(self) -> SharedPointer<Internal<A, P>, P> {
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
    pub fn leaf_ref(&self) -> &SharedPointer<Leaf<A>, P> {
        if let NodeRc::Leaf(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as a leaf node")
        }
    }

    /// Returns a reference to the node as an internal node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not an internal node.
    pub fn internal_ref(&self) -> &SharedPointer<Internal<A, P>, P> {
        if let NodeRc::Internal(x) = self {
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
    pub fn leaf_mut(&mut self) -> &mut SharedPointer<Leaf<A>, P> {
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
    pub fn internal_mut(&mut self) -> &mut SharedPointer<Internal<A, P>, P> {
        if let NodeRc::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    pub fn borrow_node(&mut self) -> BorrowedNode<A, P> {
        match self {
            NodeRc::Internal(internal) => {
                BorrowedNode::Internal(SharedPointer::make_mut(internal).borrow_node())
            }
            NodeRc::Leaf(leaf) => BorrowedNode::Leaf(SharedPointer::make_mut(leaf).borrow_node()),
        }
    }

    /// Checks internal invariants of the node.
    pub fn debug_check_invariants(&self) -> bool {
        if let NodeRc::Internal(internal) = self {
            internal.debug_check_invariants()
        } else {
            true
        }
    }
}

impl<A: Clone + std::fmt::Debug + PartialEq, P: SharedPointerKind> NodeRc<A, P> {
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_iter<'a>(&self, iter: &mut std::slice::Iter<'a, A>) -> bool {
        match self {
            NodeRc::Internal(ref internal) => internal.equal_iter(iter),
            NodeRc::Leaf(ref leaf) => leaf.equal_iter(iter),
        }
    }

    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_iter_debug<'a>(&self, iter: &mut std::slice::Iter<'a, A>) {
        match self {
            NodeRc::Internal(ref internal) => internal.equal_iter_debug(iter),
            NodeRc::Leaf(ref leaf) => leaf.equal_iter_debug(iter),
        }
    }
}

impl<A: Clone + Debug, P: SharedPointerKind> From<SharedPointer<Leaf<A>, P>> for NodeRc<A, P> {
    fn from(t: SharedPointer<Leaf<A>, P>) -> NodeRc<A, P> {
        NodeRc::Leaf(t)
    }
}

impl<A: Clone + Debug, P: SharedPointerKind> From<SharedPointer<Internal<A, P>, P>>
    for NodeRc<A, P>
{
    fn from(t: SharedPointer<Internal<A, P>, P>) -> NodeRc<A, P> {
        NodeRc::Internal(t)
    }
}

#[derive(Debug)]
pub(crate) enum BorrowedNode<A: Clone + Debug, P: SharedPointerKind> {
    Internal(BorrowedInternal<A, P>),
    Leaf(BorrowedLeaf<A>),
}

impl<A: Clone + Debug, P: SharedPointerKind> BorrowedNode<A, P> {
    pub fn empty(&mut self) -> Self {
        match self {
            BorrowedNode::Internal(internal) => BorrowedNode::Internal(internal.empty()),
            BorrowedNode::Leaf(leaf) => BorrowedNode::Leaf(leaf.empty()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            BorrowedNode::Internal(internal) => internal.len(),
            BorrowedNode::Leaf(leaf) => leaf.len(),
        }
    }

    pub fn level(&self) -> usize {
        match self {
            BorrowedNode::Internal(internal) => internal.level(),
            BorrowedNode::Leaf(leaf) => leaf.level(),
        }
    }

    pub fn make_spine(&mut self, side: Side) -> Vec<NodeRc<A, P>> {
        let mut spine = Vec::new();
        if let BorrowedNode::Internal(ref mut internal) = self {
            let idx = if side == Side::Front {
                0
            } else {
                internal.children.len() - 1
            };
            unsafe {
                let child = match &mut internal.children {
                    BorrowedChildList::Internals(children) => {
                        NodeRc::Internal(children.take_item(idx))
                    }
                    BorrowedChildList::Leaves(children) => NodeRc::Leaf(children.take_item(idx)),
                };
                spine.push(child);
            }
        }
        while let NodeRc::Internal(internal) = spine.last_mut().unwrap() {
            let idx = if side == Side::Front {
                0
            } else {
                internal.children.slots() - 1
            };
            unsafe {
                let child = match SharedPointer::make_mut(internal).children {
                    ChildList::Internals(ref mut children) => {
                        NodeRc::Internal(children.take_item(idx))
                    }
                    ChildList::Leaves(ref mut children) => NodeRc::Leaf(children.take_item(idx)),
                };
                spine.push(child);
            }
        }
        spine
    }

    pub fn split_info(&mut self, index: usize) -> (Vec<Self>, Vec<Self>) {
        if index == 0 {
            let empty = self.empty();
            *self = empty;
        } else if index == self.len() {
            let empty = self.empty();
        }
        let shared: Vec<Self> = Vec::new();
        // let result
        if let BorrowedNode::Internal(internal) = self {}
        unimplemented!();
    }

    pub fn leaf(self) -> BorrowedLeaf<A> {
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
    pub fn internal(self) -> BorrowedInternal<A, P> {
        if let BorrowedNode::Internal(x) = self {
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
    pub fn leaf_ref(&self) -> &BorrowedLeaf<A> {
        if let BorrowedNode::Leaf(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as a leaf node")
        }
    }

    /// Returns a reference to the node as an internal node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not an internal node.
    pub fn internal_ref(&self) -> &BorrowedInternal<A, P> {
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
    pub fn leaf_mut(&mut self) -> &mut BorrowedLeaf<A> {
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
    pub fn internal_mut(&mut self) -> &mut BorrowedInternal<A, P> {
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
            BorrowedNode::Internal(internal) => internal.combine(other.internal()),
            BorrowedNode::Leaf(internal) => internal.combine(other.leaf()),
        }
    }

    pub fn assert_size_invariants(&self) -> usize {
        match self {
            BorrowedNode::Internal(internal) => internal.debug_check_lens_size_table(),
            BorrowedNode::Leaf(leaf) => leaf.len(),
        }
    }
}
