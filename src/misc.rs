//! Collection of utility structs used for the RRB tree and its algorithms.

// Sorting algorithm works as follows:
// 1) Choose a pivot and swap it to the end (potentially `take` it)
// 2) Move from left to right to find element(a) that is >= pivot
// 3) Move from right to left to find element(b) that is <= pivot
// 4) Swap both elements
// 5) If a == pivot then we swap with left equal range end
// 6) If b == pivot then we swap with right equal range start

// Four positions must be tracked 2, 3, 5, 6.
// 2: Left iter(proceeds forward)
// 3: Right iter(proceeds backwards)
// 5: Left equal iter(proceeds forwards)
// 6: Right equal iter(proceed backwards)

// Invariants
// 5 is not after 2, but can be equal
// 6 is not before or equal to 5

// When 2 and 6 cross, the algorithm moves equal areas and recurses

// pub(crate) struct SortFocusMut<A: Clone> {

// }
