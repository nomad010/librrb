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
//! * Both `librrb` and `im` are thread-safe but they have slightly different implementations.
//! `librrb` uses archery to abstract over either Rc or Arc internally, while `im` exposes two
//! seperate crates. In the future, it will be possible to convert between these types.
//! * `librrb` keeps the tree in a format that allows quick access to the spines of the RRB tree.
//! `im` uses additional buffers on the sides that can be accessed quickly. Both of these techniques
//! facilitate quick mutation/access at the edges of the tree. `librrb`'s approach requires extra
//! invariants to be maintained in addition to RRB tree's invariants. On the whole, however, certain
//! operations are easier to be implement. The aim is that this translates to a faster data
//! structure.
#![deny(missing_docs)]
#![cfg_attr(feature = "may_dangle", feature(dropck_eyepatch))]

mod circular;
mod node_impls;
mod node_traits;
mod size_table;
mod sort;

#[macro_use]
pub mod vector;
pub mod focus;

#[doc(inline)]
pub use vector::{Iter, IterMut, ThreadSafeVector, Vector};

#[doc(inline)]
pub use focus::{Focus, FocusMut};

/// The width of the RRB tree nodes. The maximum number of elements in a leaf or internal node.
pub const RRB_WIDTH: usize = 64;

/// Represents a side of a container.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Side {
    /// derp
    Front,
    /// derp
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
