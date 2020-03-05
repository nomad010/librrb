//! A container for representing sequence of elements.

use crate::focus::Focus;
use crate::nodes::{ChildList, Internal, Leaf, NodeRc};
use crate::{Side, RRB_WIDTH};
use std::fmt::Debug;
use std::iter::{self, FusedIterator};
use std::mem;
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
        let (spine, other_spine) = match side {
            Side::Back => (&mut self.right_spine, &mut self.left_spine),
            Side::Front => (&mut self.left_spine, &mut self.right_spine),
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
            // println!("Gar {}/{} {} ", level, spine_len, node.slots());
            // assert_ne!(node.slots(), 0);
            let child: NodeRc<A> = match Rc::make_mut(node).children {
                ChildList::Internals(ref mut children) => children.pop(side).into(),
                ChildList::Leaves(ref mut children) => children.pop(side).into(),
            };
            // println!("Garfield {:?}", child);
            Rc::make_mut(&mut Rc::make_mut(node).sizes).pop_child(side);
            mem::replace(&mut spine[level], child);
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
                // println!("Shrinking {:?} to {:?}", spine_last, other_spine_last);
                left_spine_top.share_children_with(&mut right_spine_top, Side::Back, RRB_WIDTH);
                mem::replace(&mut self.root, right_spine_top);
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
        // println!("DERP\n{:#?}\n", self);
        if self.spine_ref(side).is_empty() {
            // println!("roflderpedoo {:#?}", self);
            if let Ok(item) = Rc::make_mut(self.root.leaf_mut()).try_pop(side) {
                self.len -= 1;
                Some(item)
            } else {
                None
            }
        // println!("derpedoo {:#?}", self);
        } else {
            // Can never be none as the is of height at least 1
            let leaf = self.leaf_mut(side);
            let item = Rc::make_mut(leaf).pop(side);

            if leaf.is_empty() {
                // println!("Derp");
                self.empty_leaf(side);
            // println!("roflderpedoofromher {:#?}", self);
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

    /// Appends the given vector onto the back of this vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate librrb;
    /// # use librrb::Vector;
    /// let mut v = vector![1, 2, 3];
    /// v.concatenate(vector![4, 5, 6]);
    /// assert_eq!(v, vector![1, 2, 3, 4, 5, 6]);
    /// ```
    pub fn concatenate(&mut self, mut other: Self) {
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

        println!("roy derpison {:#?}", self);

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
            let new_left = mem::replace(&mut self.root, new_root);
            let new_right = new_left.new_empty();
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
            let new_right = mem::replace(&mut other.root, new_root);
            let new_left = new_right.new_empty();
            other.left_spine.push(new_left);
            other.right_spine.push(new_right);
        }
        // if self.left_spine.len() < other.left_spine.len() {
        // self's root spine gets enlarged with
        // let mut new_root = match self.root Rc::new().into()
        // }

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
            // let total_slots = left_node.slots() + right_node.slots();
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
        // self.right_spine.reverse();
        // other.left_spine.reverse();
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

        // let total_slots = self.root.slots() + other.root.slots();

        // println!("RAWR {}", total_slots);

        if !other.root.is_empty() {
            println!("neither root empty");
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
        } else {
            println!("other root empty");
        }
        self.len += other.len;
        self.fixup_spine_tops();

        println!("roy arbison {:?}", self);
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
        self.assert_invariants();
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

    /// Slices the vector from the given index inclusive to the end..
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
        println!("slice_to_end {} {}", start, self.len);
        if start >= self.len {
            self.left_spine.clear();
            self.right_spine.clear();
            self.root = Rc::new(Leaf::empty()).into();
            self.len = 0;
            return;
        }
        self.assert_invariants();
        let index = start;
        self.make_index_side(index, Side::Front);
        self.fixup_spine_tops();
    }

    /// Returns the spine position and subindex corresponding the given index.
    fn find_node_info_for_index(&mut self, index: usize) -> Option<(Option<(Side, usize)>, usize)> {
        if index >= self.len {
            None
        } else {
            let mut forward_end = 0;
            let mut backward_start = self.len;

            for spine in &self.right_spine {
                println!("Gar {} {}", spine.len(), spine.level());
            }

            for (idx, (left, right)) in self
                .left_spine
                .iter_mut()
                .zip(self.right_spine.iter_mut())
                .enumerate()
            {
                if index < forward_end + left.len() {
                    return Some((Some((Side::Front, idx)), index - forward_end));
                }
                forward_end += left.len();
                backward_start -= right.len();
                println!(
                    "saltwater wells {} {} {} {} {}",
                    index,
                    backward_start,
                    right.len(),
                    self.len,
                    right.level()
                );
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
            println!(
                "DerpLOlololololo {:?} {:?} {:?}",
                side, node_index, node_position
            );
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
                    mem::replace(&mut self.root, new_root);
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
                    // let range = match side {
                    //     BufferSide::Back => spine_position + 1..self.spine_ref(side).len(),
                    //     BufferSide::Front => ,
                    // };
                    // println!("WTTTF {:?}", range);
                    for _ in 0..spine_position {
                        println!("Popping");
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
                println!(
                    "derpenstein {} {} {} {} {}",
                    node_index,
                    child_position,
                    new_index,
                    internal.len(),
                    internal.level()
                );
                let internal_mut = Rc::make_mut(internal);
                let children = &mut internal_mut.children;
                let sizes = Rc::make_mut(&mut internal_mut.sizes);
                let range = match side {
                    Side::Back => child_position + 1..num_slots,
                    Side::Front => 0..child_position,
                };
                for _ in range {
                    println!("pop");
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
                println!("final pop");
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
            println!(
                "derlol {:?} {:?} {:?} {} {}",
                leaf.len(),
                range,
                side,
                node_index,
                self.len
            );
            assert!(node_index < leaf.len());
            for _ in range {
                leaf.buffer.pop(side);
                self.len -= 1;
            }
            println!("derlol2 {:?} {}", leaf.len(), self.len);

            // Now we are done, we can reverse the spine here to get it back to normal
            spine.reverse();
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

            println!(
                "derpydoo {} {} {}",
                left_children, right_children, difference
            );
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

        println!(
            "Derp {} {} {} {:?}",
            left_spine_len, root_len, right_spine_len, self
        );

        assert_eq!(self.len, left_spine_len + root_len + right_spine_len);
        true
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
                    println!("OK Gar Left: {:?} {:?}", self, v);
                    return false;
                }
            }
            if !self.root.equal_iter(&mut iter) {
                println!("OK Gar Root: {:?} {:?}", self, v);
                return false;
            }
            for spine in self.right_spine.iter().rev() {
                if !spine.equal_iter(&mut iter) {
                    println!("OK Gar Right: {:?} {:?}", self, v);
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
            if focus.get(self.front).is_none() {
                println!("RAWR {:#?}", self);
            }
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
                        vec.split_off(index);
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
                        new_vector.concatenate(vector);
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
                        vector.concatenate(new_vector);
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

            // println!("Rawr {:?} {:?}", vec, vector);
        }
    }

    #[test]
    pub fn empty() {
        let mut empty: Vector<usize> = Vector::new();
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
        empty_concat.concatenate(empty.clone());
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
        let mut vec: Vec<usize> = vec![item];

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

    // #[test]
    // pub fn sort() {
    //     let items = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    //     let mut buffer = CircularBuffer::new();
    //     for item in &items {
    //         buffer.push_back(*item);
    //     }

    //     // Len
    //     assert!(!buffer.is_empty());
    //     assert_eq!(buffer.len(), 10);
    //     assert!(!buffer.is_full() || RRB_WIDTH == 10);

    //     // Back
    //     assert_eq!(buffer.back(), Some(&0));
    //     assert_eq!(buffer.back_mut(), Some(&mut 0));
    //     assert_eq!(buffer.end(BufferSide::Back), Some(&0));
    //     assert_eq!(buffer.end_mut(BufferSide::Back), Some(&mut 0));
    //     let mut back = buffer.clone();
    //     assert_eq!(back.try_pop_back(), Ok(0));
    //     assert_eq!(back.try_pop_back(), Ok(1));
    //     assert_eq!(back.try_pop_back(), Ok(2));
    //     assert_eq!(back.try_pop_back(), Ok(3));
    //     assert_eq!(back.try_pop_back(), Ok(4));
    //     assert_eq!(back.try_pop_back(), Ok(5));
    //     assert_eq!(back.try_pop_back(), Ok(6));
    //     assert_eq!(back.try_pop_back(), Ok(7));
    //     assert_eq!(back.try_pop_back(), Ok(8));
    //     assert_eq!(back.try_pop_back(), Ok(9));
    //     assert_eq!(back.try_pop_back(), Err(()));
    //     assert_eq!(back.back(), None);
    //     assert_eq!(back.back_mut(), None);
    //     assert_eq!(back.end(BufferSide::Back), None);
    //     assert_eq!(back.end_mut(BufferSide::Back), None);

    //     // Front
    //     assert_eq!(buffer.front(), Some(&9));
    //     assert_eq!(buffer.front_mut(), Some(&mut 9));
    //     assert_eq!(buffer.end(BufferSide::Front), Some(&9));
    //     assert_eq!(buffer.end_mut(BufferSide::Front), Some(&mut 9));
    //     let mut front = buffer.clone();
    //     assert_eq!(front.try_pop_front(), Ok(9));
    //     assert_eq!(front.try_pop_front(), Ok(8));
    //     assert_eq!(front.try_pop_front(), Ok(7));
    //     assert_eq!(front.try_pop_front(), Ok(6));
    //     assert_eq!(front.try_pop_front(), Ok(5));
    //     assert_eq!(front.try_pop_front(), Ok(4));
    //     assert_eq!(front.try_pop_front(), Ok(3));
    //     assert_eq!(front.try_pop_front(), Ok(2));
    //     assert_eq!(front.try_pop_front(), Ok(1));
    //     assert_eq!(front.try_pop_front(), Ok(0));
    //     assert_eq!(front.try_pop_front(), Err(()));
    //     assert_eq!(front.end(BufferSide::Front), None);
    //     assert_eq!(front.end_mut(BufferSide::Front), None);

    //     // Sort
    //     buffer.sort();
    //     assert_eq!(buffer.try_pop_front(), Ok(0));
    //     assert_eq!(buffer.try_pop_front(), Ok(1));
    //     assert_eq!(buffer.try_pop_front(), Ok(2));
    //     assert_eq!(buffer.try_pop_front(), Ok(3));
    //     assert_eq!(buffer.try_pop_front(), Ok(4));
    //     assert_eq!(buffer.try_pop_front(), Ok(5));
    //     assert_eq!(buffer.try_pop_front(), Ok(6));
    //     assert_eq!(buffer.try_pop_front(), Ok(7));
    //     assert_eq!(buffer.try_pop_front(), Ok(8));
    //     assert_eq!(buffer.try_pop_front(), Ok(9));
    // }
}
