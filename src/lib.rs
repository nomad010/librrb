//! A library implementing an RRB tree.
//!
//! # What are RRB trees?
//!
//! An RRB tree is a data structure for representing a sequence of items, similar to a [`Vec<T>`].
//! RRB trees support most of the operation that [`Vec<T>`]'s do, albeit with better algorithmic
//! complexities. Second, the data structure supports very fast copying with [`Vector<T>`]'s clone
//! method by a technique called structural sharing. A node in the tree may be shared among multiple
//! instances and even used in multiple places in the same instance.
//!
//! The downside of all of these benefits is hefty constant factors that most operations have. This
//! means that for small collections, [`Vec<T>`] will be faster than [`Vector<T>`]. The focus of
//! this library is to provide an implementation with low constant factors. Many of the techniques
//! in this are experimental or are incomplete. For a more reliable and complete library, see the
//! [`im`] library.
//!
//! [`Vec<T>`]: https://doc.rust-lang.org/std/vec/struct.Vec.html
//! [`im`]: https://docs.rs/im/
//!
//! # What are some differences between librrb's Vector vs im's Vector?
//! * At the moment, `librrb` uses a branching factor of 32, while `im` uses a branching factor
//! of 64. This leads to deeper trees in `librrb`.
//! * `librrb` does not yet support mutating iterators.
//! * `librrb` is not thread-safe, but may be in the future. `im` has a thread-safe Vector and a
//! non-thread-safe Vector. It is likely that `librrb` may use a different scheme to `im`.
//! * `librrb` keeps the tree in a format that allows quick access to the spines of the RRB tree.
//! `im` uses additional buffers on the sides that can be accessed quickly. Both of these techniques
//! facilitate quick mutation/access at the edges of the tree. `librrb`'s approach requires extra
//! invariants to be maintained in addition to RRB tree's invariants. On the whole, however, certain
//! operations are easier to be implement. The aim is that this translates to a faster data
//! structure.
//! * Both `librrb`'s and `im`'s concatenation method balance the resulting tree, but do so
//! differently. `librrb` only loosely balances the resulting tree, the spines are zipped together.
//! On the other hand, `im` zips the children of spine nodes as well. This better balances the tree.
//! All trees of less than 64^2 elements are fully balanced, vs only 32 elements for `librrb`. This
//! results in a slightly slower operation for `im`. Another potential downside is that more
//! internal clones may be required, reducing opportunities for structural sharing among trees. For
//! more information on the properties of the concatenation method, see the [`Vector::concat`]
//! method.
#![deny(missing_docs)]

mod circular;
mod nodes;
mod size_table;

#[macro_use]
pub mod vector;
pub mod focus;

#[doc(inline)]
pub use vector::Vector;

#[doc(inline)]
pub use focus::Focus;

/// The width of the RRB tree nodes. The maximum number of elements in a leaf or internal node.
const RRB_WIDTH: usize = 32;

/// Represents a side of a container.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Side {
    Front,
    Back,
}

impl Side {
    /// Returns the opposite side to `self`.
    pub fn negate(self) -> Side {
        match self {
            Side::Front => Side::Back,
            Side::Back => Side::Front,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn side() {
        assert_eq!(Side::Front.negate(), Side::Back);
        assert_eq!(Side::Back.negate(), Side::Front);
    }

    #[test]
    fn width_is_power_of_two() {
        assert!(RRB_WIDTH.is_power_of_two());
    }
}
