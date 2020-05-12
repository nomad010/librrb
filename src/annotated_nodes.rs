//! Collection of nodes used for the RRB tree.

use crate::circular::{BorrowBufferMut, CircularBuffer};
use crate::nodes::{BorrowedLeaf, Leaf};
use crate::size_table::SizeTable;
use crate::{Side, RRB_WIDTH};
use std::cmp;
use std::fmt::Debug;
use std::ops::{Deref, DerefMut, Range};
use std::rc::Rc;

/// Represents a homogenous list of nodes.
#[derive(Clone, Debug)]
pub(crate) enum AnnotatedChildList<A: Clone + Debug> {
    Leaves(CircularBuffer<Rc<Leaf<A>>>),
    Internals(CircularBuffer<Rc<AnnotatedInternal<A>>>),
}

impl<A: Clone + Debug> AnnotatedChildList<A> {
    /// Constructs a new empty list of nodes.
    pub fn new_empty(&self) -> Self {
        match self {
            AnnotatedChildList::Internals(_) => {
                AnnotatedChildList::Internals(CircularBuffer::new())
            }
            AnnotatedChildList::Leaves(_) => AnnotatedChildList::Leaves(CircularBuffer::new()),
        }
    }

    /// Consumes `self` and returns the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves(self) -> CircularBuffer<Rc<Leaf<A>>> {
        if let AnnotatedChildList::Leaves(x) = self {
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
    pub fn internals(self) -> CircularBuffer<Rc<AnnotatedInternal<A>>> {
        if let AnnotatedChildList::Internals(x) = self {
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
    pub fn leaves_ref(&self) -> &CircularBuffer<Rc<Leaf<A>>> {
        if let AnnotatedChildList::Leaves(x) = self {
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
    pub fn internals_ref(&self) -> &CircularBuffer<Rc<AnnotatedInternal<A>>> {
        if let AnnotatedChildList::Internals(x) = self {
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
    pub fn leaves_mut(&mut self) -> &mut CircularBuffer<Rc<Leaf<A>>> {
        if let AnnotatedChildList::Leaves(x) = self {
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
    pub fn internals_mut(&mut self) -> &mut CircularBuffer<Rc<AnnotatedInternal<A>>> {
        if let AnnotatedChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
        }
    }

    pub fn borrow(&mut self) -> BorrowedAnnotatedChildList<A> {
        match self {
            AnnotatedChildList::Internals(children) => {
                BorrowedAnnotatedChildList::Internals(children.mutable_view())
            }
            AnnotatedChildList::Leaves(children) => {
                BorrowedAnnotatedChildList::Leaves(children.mutable_view())
            }
        }
    }

    /// Returns a mutable reference to the list as a list of internal nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of internal nodes.
    pub fn get(&self, child_idx: usize, idx: usize) -> Option<&A> {
        match self {
            AnnotatedChildList::Leaves(children) => children.get(child_idx).unwrap().get(idx),
            AnnotatedChildList::Internals(children) => children.get(child_idx).unwrap().get(idx),
        }
    }

    pub fn get_mut(&mut self, child_idx: usize, idx: usize) -> Option<&mut A> {
        match self {
            AnnotatedChildList::Leaves(children) => {
                Rc::make_mut(children.get_mut(child_idx).unwrap()).get_mut(idx)
            }
            AnnotatedChildList::Internals(children) => {
                Rc::make_mut(children.get_mut(child_idx).unwrap()).get_mut(idx)
            }
        }
    }

    /// Returns a copy of the Rc of the node at the given position in the node.
    pub fn get_child_node(&self, idx: usize) -> Option<AnnotatedNodeRc<A>> {
        // TODO: Get rid of this function
        match self {
            AnnotatedChildList::Leaves(children) => children.get(idx).map(|x| Rc::clone(x).into()),
            AnnotatedChildList::Internals(children) => {
                children.get(idx).map(|x| Rc::clone(x).into())
            }
        }
    }

    /// Returns the length of the child list.
    pub fn slots(&self) -> usize {
        match self {
            AnnotatedChildList::Leaves(children) => children.len(),
            AnnotatedChildList::Internals(children) => children.len(),
        }
    }

    /// Returns the number of nodes that could still be inserted into the child list.
    pub fn free_slots(&self) -> usize {
        RRB_WIDTH - self.slots()
    }
}

#[derive(Debug)]
pub(crate) enum BorrowedAnnotatedChildList<A: Clone + Debug> {
    Internals(BorrowBufferMut<Rc<AnnotatedInternal<A>>>),
    Leaves(BorrowBufferMut<Rc<Leaf<A>>>),
}

impl<'a, A: Clone + Debug> BorrowedAnnotatedChildList<A> {
    /// Consumes `self` and returns the list as a list of leaf nodes.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a list of leaf nodes.
    pub fn leaves(self) -> BorrowBufferMut<Rc<Leaf<A>>> {
        if let BorrowedAnnotatedChildList::Leaves(x) = self {
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
    pub fn internals(self) -> BorrowBufferMut<Rc<AnnotatedInternal<A>>> {
        if let BorrowedAnnotatedChildList::Internals(x) = self {
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
    pub fn leaves_ref(&self) -> &BorrowBufferMut<Rc<Leaf<A>>> {
        if let BorrowedAnnotatedChildList::Leaves(x) = self {
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
    pub fn internals_ref(&self) -> &BorrowBufferMut<Rc<AnnotatedInternal<A>>> {
        if let BorrowedAnnotatedChildList::Internals(x) = self {
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
    pub fn leaves_mut(&mut self) -> &mut BorrowBufferMut<Rc<Leaf<A>>> {
        if let BorrowedAnnotatedChildList::Leaves(x) = self {
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
    pub fn internals_mut(&mut self) -> &mut BorrowBufferMut<Rc<AnnotatedInternal<A>>> {
        if let BorrowedAnnotatedChildList::Internals(x) = self {
            x
        } else {
            panic!("Failed to unwrap a child list as an internals list")
        }
    }

    fn empty(&mut self) -> Self {
        match self {
            BorrowedAnnotatedChildList::Internals(children) => {
                BorrowedAnnotatedChildList::Internals(children.empty())
            }
            BorrowedAnnotatedChildList::Leaves(children) => {
                BorrowedAnnotatedChildList::Leaves(children.empty())
            }
        }
    }

    fn split_at(&mut self, index: usize) -> Self {
        match self {
            BorrowedAnnotatedChildList::Internals(ref mut children) => {
                let new_children = children.split_at(index);
                BorrowedAnnotatedChildList::Internals(new_children)
            }
            BorrowedAnnotatedChildList::Leaves(ref mut children) => {
                let new_children = children.split_at(index);
                BorrowedAnnotatedChildList::Leaves(new_children)
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            BorrowedAnnotatedChildList::Internals(children) => children.len(),
            BorrowedAnnotatedChildList::Leaves(children) => children.len(),
        }
    }

    pub fn from_same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (
                BorrowedAnnotatedChildList::Internals(children),
                BorrowedAnnotatedChildList::Internals(other_children),
            ) => children.from_same_source(other_children),
            (
                BorrowedAnnotatedChildList::Leaves(children),
                BorrowedAnnotatedChildList::Leaves(other_children),
            ) => children.from_same_source(other_children),
            _ => false,
        }
    }

    pub fn combine(&mut self, other: Self) {
        match self {
            BorrowedAnnotatedChildList::Internals(children) => children.combine(other.internals()),
            BorrowedAnnotatedChildList::Leaves(children) => children.combine(other.leaves()),
        }
    }

    pub(crate) fn range(&self) -> &Range<usize> {
        match self {
            BorrowedAnnotatedChildList::Internals(children) => &children.range,
            BorrowedAnnotatedChildList::Leaves(children) => &children.range,
        }
    }

    pub(crate) fn range_mut(&mut self) -> &mut Range<usize> {
        match self {
            BorrowedAnnotatedChildList::Internals(children) => &mut children.range,
            BorrowedAnnotatedChildList::Leaves(children) => &mut children.range,
        }
    }
}

/// An internal node indicates a non-terminal node in the tree.
#[derive(Clone, Debug)]
pub(crate) struct AnnotatedInternal<A: Clone + Debug> {
    pub sizes: Rc<SizeTable>,
    pub children: AnnotatedChildList<A>,
}

impl<A: Clone + Debug> AnnotatedInternal<A> {
    /// Constructs a new empty internal node that is at level 1 in the tree.
    pub fn empty_leaves() -> Self {
        AnnotatedInternal {
            sizes: Rc::new(SizeTable::new(1)),
            children: AnnotatedChildList::Leaves(CircularBuffer::new()),
        }
    }

    /// Constructs a new empty internal node that is at the given level in the tree.
    pub fn empty_internal(level: usize) -> Self {
        debug_assert_ne!(level, 0); // Should be a Leaf
        if level == 1 {
            Self::empty_leaves()
        } else {
            AnnotatedInternal {
                sizes: Rc::new(SizeTable::new(level)),
                children: AnnotatedChildList::Internals(CircularBuffer::new()),
            }
        }
    }

    /// Constructs a new internal node of the same level, but with no children.
    pub fn new_empty(&self) -> Self {
        AnnotatedInternal {
            sizes: Rc::new(SizeTable::new(self.sizes.level())),
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
            AnnotatedChildList::Internals(ref mut children) => {
                debug_assert_eq!(children.len(), self.sizes.len());
                children.decant_into(destination.children.internals_mut(), share_side, len)
            }
            AnnotatedChildList::Leaves(ref mut children) => {
                debug_assert_eq!(children.len(), self.sizes.len());
                children.decant_into(destination.children.leaves_mut(), share_side, len)
            }
        };
        if shared != 0 {
            let origin_sizes = Rc::make_mut(&mut self.sizes);
            let destination_sizes = Rc::make_mut(&mut destination.sizes);

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
            AnnotatedChildList::Internals(ref mut children) => {
                while read_position < children.len() {
                    if read_position == write_position {
                        read_position += 1;
                        continue;
                    } else {
                        let (write, read) = children.pair_mut(write_position, read_position);
                        Rc::make_mut(read).share_children_with(
                            Rc::make_mut(write),
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
                let sizes = Rc::make_mut(&mut self.sizes);
                *sizes = SizeTable::new(sizes.level());
                for child in children {
                    sizes.push_child(Side::Back, child.len());
                }
            }
            AnnotatedChildList::Leaves(ref mut children) => {
                while read_position < children.len() {
                    if read_position == write_position {
                        read_position += 1;
                        continue;
                    } else {
                        let (write, read) = children.pair_mut(write_position, read_position);
                        Rc::make_mut(read).share_children_with(
                            Rc::make_mut(write),
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
                let sizes = Rc::make_mut(&mut self.sizes);
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
            AnnotatedChildList::Internals(children) => {
                Rc::make_mut(children.get_mut(array_idx).unwrap()).update(new_idx, item)
            }
            AnnotatedChildList::Leaves(children) => {
                Rc::make_mut(children.get_mut(array_idx).unwrap()).update(new_idx, item)
            }
        }
    }

    /// Sorts the elements in the tree by the given comparator.
    ///
    /// # NOTE
    ///
    /// This is a work in progress. And the functionality doesn't work at all at the moment.
    pub fn sort_by<F: FnMut(&A, &A) -> cmp::Ordering>(&mut self, range: Range<usize>, f: &mut F) {
        let mut start = 0;
        match self.children {
            AnnotatedChildList::Internals(ref mut children) => {
                for child in children {
                    let end = start + child.len();
                    if end > range.start && start < range.end {
                        // Range intersects
                        let new_range = start.max(range.start)..end.min(range.end);
                        Rc::make_mut(child).sort_by(new_range, f);
                    }
                    start = end;
                }
            }
            AnnotatedChildList::Leaves(ref mut children) => {
                for child in children {
                    let end = start + child.len();
                    if end > range.start && start < range.end {
                        let new_range = start.max(range.start)..end.min(range.end);
                        Rc::make_mut(child).sort_by(new_range, f);
                    }
                    start = end;
                }
            }
        }
    }

    pub fn borrow(&mut self) -> BorrowedAnnotatedInternal<A> {
        BorrowedAnnotatedInternal {
            children: self.children.borrow(),
            sizes: Rc::clone(&self.sizes),
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
            AnnotatedChildList::Internals(children) => {
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
            AnnotatedChildList::Leaves(children) => {
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
            AnnotatedChildList::Internals(children) => {
                for i in children {
                    debug_assert_eq!(i.debug_check_child_heights() + 1, self.level());
                }
            }
            AnnotatedChildList::Leaves(children) => {
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

impl<A: Clone + Debug + PartialEq> AnnotatedInternal<A> {
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_iter<'a>(&self, iter: &mut std::slice::Iter<'a, A>) -> bool {
        match self.children {
            AnnotatedChildList::Internals(ref children) => {
                for child in children {
                    if !child.equal_iter(iter) {
                        return false;
                    }
                }
            }
            AnnotatedChildList::Leaves(ref children) => {
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
            AnnotatedChildList::Internals(ref children) => {
                for child in children {
                    child.equal_iter_debug(iter);
                }
            }
            AnnotatedChildList::Leaves(ref children) => {
                for child in children {
                    child.equal_iter_debug(iter);
                }
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct BorrowedAnnotatedInternal<A: Clone + Debug> {
    pub(crate) sizes: Rc<SizeTable>,
    pub(crate) children: BorrowedAnnotatedChildList<A>,
}

impl<'a, A: Clone + Debug> BorrowedAnnotatedInternal<A> {
    fn empty(&mut self) -> Self {
        BorrowedAnnotatedInternal {
            sizes: Rc::clone(&self.sizes),
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
        BorrowedAnnotatedInternal {
            children: new_children,
            sizes: Rc::clone(&self.sizes),
        }
    }

    pub fn debug_check_lens_size_table(&self) -> usize {
        let mut cumulative_size = 0;
        match &self.children {
            BorrowedAnnotatedChildList::Internals(children) => {
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
            BorrowedAnnotatedChildList::Leaves(children) => {
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
#[derive(Clone, Debug)]
pub(crate) enum AnnotatedNodeRc<A: Clone + Debug> {
    Leaf(Rc<Leaf<A>>),
    Internal(Rc<AnnotatedInternal<A>>),
}

impl<A: Clone + Debug> AnnotatedNodeRc<A> {
    /// Constructs a new empty of the same level.
    pub fn new_empty(&self) -> Self {
        match self {
            AnnotatedNodeRc::Internal(x) => Rc::new(x.new_empty()).into(),
            AnnotatedNodeRc::Leaf(_) => Rc::new(Leaf::empty()).into(),
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
            AnnotatedNodeRc::Internal(ref mut origin_internal) => {
                let destination_internal = Rc::make_mut(destination.internal_mut());
                Rc::make_mut(origin_internal).share_children_with(
                    destination_internal,
                    share_side,
                    len,
                )
            }
            AnnotatedNodeRc::Leaf(ref mut origin_leaf) => {
                let destination_leaf = Rc::make_mut(destination.leaf_mut());
                Rc::make_mut(origin_leaf).share_children_with(destination_leaf, share_side, len)
            }
        }
    }

    /// Returns the size of the of node.
    pub fn len(&self) -> usize {
        match self {
            AnnotatedNodeRc::Leaf(x) => x.len(),
            AnnotatedNodeRc::Internal(x) => x.len(),
        }
    }

    /// Returns a reference to the size table if it has one.
    pub fn size_table_ref(&self) -> Option<&SizeTable> {
        match self {
            AnnotatedNodeRc::Leaf(_) => None,
            AnnotatedNodeRc::Internal(x) => Some(&x.sizes),
        }
    }

    /// Returns the number of direct children of the node.
    pub fn slots(&self) -> usize {
        match self {
            AnnotatedNodeRc::Leaf(x) => x.len(),
            _ => self.size_table_ref().unwrap().len(),
        }
    }

    /// Returns whether the node is completely empty.
    pub fn is_empty(&self) -> bool {
        match self {
            AnnotatedNodeRc::Leaf(x) => x.is_empty(),
            AnnotatedNodeRc::Internal(x) => x.is_empty(),
        }
    }

    /// Returns the level of the node.
    pub fn level(&self) -> usize {
        match self {
            AnnotatedNodeRc::Leaf(x) => x.level(),
            AnnotatedNodeRc::Internal(x) => x.level(),
        }
    }

    /// Tests whether the node is a leaf.
    pub fn is_leaf(&self) -> bool {
        match self {
            AnnotatedNodeRc::Leaf(x) => x.is_leaf(),
            AnnotatedNodeRc::Internal(x) => x.is_leaf(),
        }
    }

    /// Returns the element at the given position in the node.
    pub fn get(&self, idx: usize) -> Option<&A> {
        match self {
            AnnotatedNodeRc::Leaf(x) => x.get(idx),
            AnnotatedNodeRc::Internal(x) => x.get(idx),
        }
    }

    /// Returns the element at the given position in the node.
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut A> {
        match self {
            AnnotatedNodeRc::Leaf(ref mut x) => Rc::make_mut(x).get_mut(idx),
            AnnotatedNodeRc::Internal(ref mut x) => Rc::make_mut(x).get_mut(idx),
        }
    }

    /// Returns the number of elements that can be inserted into this node.
    pub fn free_space(&self) -> usize {
        match self {
            AnnotatedNodeRc::Leaf(x) => x.free_space(),
            AnnotatedNodeRc::Internal(x) => x.free_space(),
        }
    }

    /// Returns the number of children that can be inserted into this node.
    pub fn free_slots(&self) -> usize {
        match self {
            AnnotatedNodeRc::Leaf(x) => x.free_slots(),
            AnnotatedNodeRc::Internal(x) => x.free_slots(),
        }
    }

    /// Consumes `self` and returns the node as a leaf node.
    ///
    /// # Panics
    ///
    /// Panics if `self` is not a leaf node.
    pub fn leaf(self) -> Rc<Leaf<A>> {
        if let AnnotatedNodeRc::Leaf(x) = self {
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
    pub fn internal(self) -> Rc<AnnotatedInternal<A>> {
        if let AnnotatedNodeRc::Internal(x) = self {
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
    pub fn leaf_ref(&self) -> &Rc<Leaf<A>> {
        if let AnnotatedNodeRc::Leaf(x) = self {
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
    pub fn internal_ref(&self) -> &Rc<AnnotatedInternal<A>> {
        if let AnnotatedNodeRc::Internal(x) = self {
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
    pub fn leaf_mut(&mut self) -> &mut Rc<Leaf<A>> {
        if let AnnotatedNodeRc::Leaf(x) = self {
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
    pub fn internal_mut(&mut self) -> &mut Rc<AnnotatedInternal<A>> {
        if let AnnotatedNodeRc::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    pub fn borrow(&mut self) -> BorrowedAnnotatedNode<A> {
        match self {
            AnnotatedNodeRc::Internal(internal) => {
                BorrowedAnnotatedNode::Internal(Rc::make_mut(internal).borrow())
            }
            AnnotatedNodeRc::Leaf(leaf) => {
                BorrowedAnnotatedNode::Leaf(Rc::make_mut(leaf).borrow_node())
            }
        }
    }

    /// Checks internal invariants of the node.
    pub fn debug_check_invariants(&self) -> bool {
        if let AnnotatedNodeRc::Internal(internal) = self {
            internal.debug_check_invariants()
        } else {
            true
        }
    }
}

impl<A: Clone + Debug + PartialEq> AnnotatedNodeRc<A> {
    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_iter<'a>(&self, iter: &mut std::slice::Iter<'a, A>) -> bool {
        match self {
            AnnotatedNodeRc::Internal(ref internal) => internal.equal_iter(iter),
            AnnotatedNodeRc::Leaf(ref leaf) => leaf.equal_iter(iter),
        }
    }

    /// Tests whether the node is compatible with the given iterator. This is mainly used for
    /// debugging purposes.
    pub(crate) fn equal_iter_debug<'a>(&self, iter: &mut std::slice::Iter<'a, A>) {
        match self {
            AnnotatedNodeRc::Internal(ref internal) => internal.equal_iter_debug(iter),
            AnnotatedNodeRc::Leaf(ref leaf) => leaf.equal_iter_debug(iter),
        }
    }
}

impl<A: Clone + Debug> From<Rc<Leaf<A>>> for AnnotatedNodeRc<A> {
    fn from(t: Rc<Leaf<A>>) -> AnnotatedNodeRc<A> {
        AnnotatedNodeRc::Leaf(t)
    }
}

impl<A: Clone + Debug> From<Rc<AnnotatedInternal<A>>> for AnnotatedNodeRc<A> {
    fn from(t: Rc<AnnotatedInternal<A>>) -> AnnotatedNodeRc<A> {
        AnnotatedNodeRc::Internal(t)
    }
}

#[derive(Debug)]
pub(crate) enum BorrowedAnnotatedNode<A: Clone + Debug> {
    Internal(BorrowedAnnotatedInternal<A>),
    Leaf(BorrowedLeaf<A>),
}

impl<A: Clone + Debug> BorrowedAnnotatedNode<A> {
    pub fn empty(&mut self) -> Self {
        match self {
            BorrowedAnnotatedNode::Internal(internal) => {
                BorrowedAnnotatedNode::Internal(internal.empty())
            }
            BorrowedAnnotatedNode::Leaf(leaf) => BorrowedAnnotatedNode::Leaf(leaf.empty()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            BorrowedAnnotatedNode::Internal(internal) => internal.len(),
            BorrowedAnnotatedNode::Leaf(leaf) => leaf.len(),
        }
    }

    pub fn level(&self) -> usize {
        match self {
            BorrowedAnnotatedNode::Internal(internal) => internal.level(),
            BorrowedAnnotatedNode::Leaf(leaf) => leaf.level(),
        }
    }

    pub fn make_spine(&mut self, side: Side) -> Vec<AnnotatedNodeRc<A>> {
        let mut spine = Vec::new();
        if let BorrowedAnnotatedNode::Internal(ref mut internal) = self {
            let idx = if side == Side::Front {
                0
            } else {
                internal.children.len() - 1
            };
            unsafe {
                let child = match &mut internal.children {
                    BorrowedAnnotatedChildList::Internals(children) => {
                        AnnotatedNodeRc::Internal(children.take_item(idx))
                    }
                    BorrowedAnnotatedChildList::Leaves(children) => {
                        AnnotatedNodeRc::Leaf(children.take_item(idx))
                    }
                };
                spine.push(child);
            }
        }
        while let AnnotatedNodeRc::Internal(internal) = spine.last_mut().unwrap() {
            let idx = if side == Side::Front {
                0
            } else {
                internal.children.slots() - 1
            };
            unsafe {
                let child = match Rc::make_mut(internal).children {
                    AnnotatedChildList::Internals(ref mut children) => {
                        AnnotatedNodeRc::Internal(children.take_item(idx))
                    }
                    AnnotatedChildList::Leaves(ref mut children) => {
                        AnnotatedNodeRc::Leaf(children.take_item(idx))
                    }
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
        if let BorrowedAnnotatedNode::Internal(internal) = self {}
        unimplemented!();
    }

    pub fn leaf(self) -> BorrowedLeaf<A> {
        if let BorrowedAnnotatedNode::Leaf(x) = self {
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
    pub fn internal(self) -> BorrowedAnnotatedInternal<A> {
        if let BorrowedAnnotatedNode::Internal(x) = self {
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
        if let BorrowedAnnotatedNode::Leaf(x) = self {
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
    pub fn internal_ref(&self) -> &BorrowedAnnotatedInternal<A> {
        if let BorrowedAnnotatedNode::Internal(x) = self {
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
        if let BorrowedAnnotatedNode::Leaf(x) = self {
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
    pub fn internal_mut(&mut self) -> &mut BorrowedAnnotatedInternal<A> {
        if let BorrowedAnnotatedNode::Internal(x) = self {
            x
        } else {
            panic!("Failed to unwrap a node as an internal node")
        }
    }

    pub fn from_same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (BorrowedAnnotatedNode::Internal(node), BorrowedAnnotatedNode::Internal(other)) => {
                node.from_same_source(other)
            }
            (BorrowedAnnotatedNode::Leaf(node), BorrowedAnnotatedNode::Leaf(other)) => {
                node.from_same_source(other)
            }
            _ => false,
        }
    }

    pub fn combine(&mut self, other: Self) {
        match self {
            BorrowedAnnotatedNode::Internal(internal) => internal.combine(other.internal()),
            BorrowedAnnotatedNode::Leaf(internal) => internal.combine(other.leaf()),
        }
    }

    pub fn assert_size_invariants(&self) -> usize {
        match self {
            BorrowedAnnotatedNode::Internal(internal) => internal.debug_check_lens_size_table(),
            BorrowedAnnotatedNode::Leaf(leaf) => leaf.len(),
        }
    }
}
