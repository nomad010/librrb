use crate::focus::FocusMut;
use crate::node_traits::{BorrowedInternalTrait, InternalTrait, LeafTrait};
use rand_core::RngCore;
use std::cmp;
use std::fmt::Debug;
use std::mem;

pub(crate) fn do_single_sort<R, F, Internal, Leaf, BorrowedInternal>(
    focus: &mut FocusMut<Internal, Leaf, BorrowedInternal>,
    rng: &mut R,
    comparator: &F,
) where
    R: RngCore,
    F: Fn(&Leaf::Item, &Leaf::Item) -> cmp::Ordering,
    Internal: InternalTrait<Leaf, Borrowed = BorrowedInternal>,
    BorrowedInternal: BorrowedInternalTrait<Leaf, InternalChild = Internal> + Debug,
    Leaf: LeafTrait<Context = Internal::Context>,
{
    if focus.len() <= 1 {
        return;
    }

    // We know there are at least 2 elements here
    let pivot_index = rng.next_u64() as usize % focus.len();
    focus.split_at_fn(1, |first, rest| {
        if pivot_index > 0 {
            mem::swap(rest.index(pivot_index - 1), first.index(0));
        }
        // Pivot is now always in the first slice

        // Find the exact place to put the pivot or pivot-equal items
        let mut less_count = 0;
        let mut equal_count = 0;
        for index in 0..rest.len() {
            let rest_val = rest.index(index);
            let first_val = first.index(0);
            let comp = comparator(rest_val, first_val);
            match comp {
                cmp::Ordering::Less => less_count += 1,
                cmp::Ordering::Equal => equal_count += 1,
                cmp::Ordering::Greater => {}
            }
        }
        // If by accident we picked the minimum element as a pivot, we just call sort again with the
        // rest of the vector.
        if less_count == 0 {
            do_single_sort(rest, rng, comparator);
            return;
        }

        // We know here that there is at least one item before the pivot, so we move the minimum to the
        // beginning part of the vector. First, however we swap the pivot to the start of the equal
        // zone.
        less_count -= 1;
        equal_count += 1;
        mem::swap(first.index(0), rest.index(less_count));
        for index in 0..rest.len() {
            if index == less_count {
                // This is the position we swapped the pivot to. We can't move it from its position, and
                // we know its not the minimum.
                continue;
            }
            // let rest_item = rest.index(index);
            if comparator(rest.index(index), first.index(0)) == cmp::Ordering::Less {
                mem::swap(first.index(0), rest.index(index));
            }
        }
        // Split the vector up into less_than, equal to and greater than parts.
        rest.split_at_fn(less_count + equal_count, |rest, greater_focus| {
            rest.split_at_fn(less_count, |less_focus, equal_focus| {
                let mut less_position = 0;
                let mut equal_position = 0;
                let mut greater_position = 0;

                while less_position != less_focus.len() || greater_position != greater_focus.len() {
                    // At start of this loop, equal_position always points to an equal item
                    let mut equal_swap_side = None;

                    // Advance the less_position until we find an out of place item
                    while less_position != less_focus.len() {
                        match comparator(
                            less_focus.index(less_position),
                            equal_focus.index(equal_position),
                        ) {
                            cmp::Ordering::Equal => {
                                equal_swap_side = Some(cmp::Ordering::Less);
                                break;
                            }
                            cmp::Ordering::Greater => {
                                break;
                            }
                            _ => {}
                        }
                        less_position += 1;
                    }

                    // Advance the greater until we find an out of place item
                    while greater_position != greater_focus.len() {
                        match comparator(
                            greater_focus.index(greater_position),
                            equal_focus.index(equal_position),
                        ) {
                            cmp::Ordering::Less => break,
                            cmp::Ordering::Equal => {
                                equal_swap_side = Some(cmp::Ordering::Greater);
                                break;
                            }
                            _ => {}
                        }
                        greater_position += 1;
                    }

                    if let Some(swap_side) = equal_swap_side {
                        // One of the sides is equal to the pivot, advance the pivot
                        if swap_side == cmp::Ordering::Less {
                            while comparator(
                                less_focus.index(less_position),
                                equal_focus.index(equal_position),
                            ) == cmp::Ordering::Equal
                            {
                                equal_position += 1;
                            }

                            // Swap the equal position and the desired side, it's important to note that only the
                            // equals focus is guaranteed to have made progress so we don't advance the side's index
                            mem::swap(
                                less_focus.index(less_position),
                                equal_focus.index(equal_position),
                            );
                        } else {
                            while comparator(
                                greater_focus.index(greater_position),
                                equal_focus.index(equal_position),
                            ) == cmp::Ordering::Equal
                            {
                                equal_position += 1;
                            }

                            // Swap the equal position and the desired side, it's important to note that only the
                            // equals focus is guaranteed to have made progress so we don't advance the side's index
                            mem::swap(
                                greater_focus.index(greater_position),
                                equal_focus.index(equal_position),
                            );
                        }
                    } else if less_position != less_focus.len()
                        && greater_position != greater_focus.len()
                    {
                        // Both sides are out of place and not equal to the pivot, this can only happen if there
                        // is a greater item in the lesser zone and a lesser item in the greater zone. The
                        // solution is to swap both sides and advance both side's indices.
                        debug_assert_ne!(
                            comparator(
                                less_focus.index(less_position),
                                equal_focus.index(equal_position),
                            ),
                            cmp::Ordering::Equal
                        );
                        debug_assert_ne!(
                            comparator(
                                greater_focus.index(greater_position),
                                equal_focus.index(equal_position)
                            ),
                            cmp::Ordering::Equal
                        );
                        mem::swap(
                            less_focus.index(less_position),
                            greater_focus.index(greater_position),
                        );
                        less_position += 1;
                        greater_position += 1;
                    }
                }

                // Now we have partitioned both sides correctly, we just have to recurse now
                do_single_sort(less_focus, rng, comparator);
                if !greater_focus.is_empty() {
                    do_single_sort(greater_focus, rng, comparator);
                }
            });
        });
    });
}

pub(crate) fn do_dual_sort<
    R,
    F,
    Internal1,
    Leaf1,
    Internal2,
    Leaf2,
    BorrowedInternal1,
    BorrowedInternal2,
>(
    focus: &mut FocusMut<Internal1, Leaf1, BorrowedInternal1>,
    dual: &mut FocusMut<Internal2, Leaf2, BorrowedInternal2>,
    rng: &mut R,
    comparator: &F,
) where
    R: RngCore,
    F: Fn(&Leaf1::Item, &Leaf1::Item) -> cmp::Ordering,
    Internal1: InternalTrait<Leaf1, Borrowed = BorrowedInternal1>,
    BorrowedInternal1: BorrowedInternalTrait<Leaf1, InternalChild = Internal1> + Debug,
    Leaf1: LeafTrait<Context = Internal1::Context>,
    Internal2: InternalTrait<Leaf2, Borrowed = BorrowedInternal2>,
    BorrowedInternal2: BorrowedInternalTrait<Leaf2, InternalChild = Internal2> + Debug,
    Leaf2: LeafTrait<Context = Internal2::Context>,
{
    if focus.len() <= 1 {
        return;
    }

    // We know there are at least 2 elements here
    let pivot_index = rng.next_u64() as usize % focus.len();
    focus.split_at_fn(1, |first_focus, rest_focus| {
        dual.split_at_fn(1, |first_dual, rest_dual| {
            debug_assert_eq!(first_focus.len(), first_dual.len());
            debug_assert_eq!(rest_focus.len(), rest_dual.len());
            if pivot_index > 0 {
                mem::swap(rest_focus.index(pivot_index - 1), first_focus.index(0));
                mem::swap(rest_dual.index(pivot_index - 1), first_dual.index(0));
            }
            // Pivot is now always in the first slice

            // Find the exact place to put the pivot or pivot-equal items
            let mut less_count = 0;
            let mut equal_count = 0;
            for index in 0..rest_focus.len() {
                let comp = comparator(rest_focus.index(index), first_focus.index(0));
                match comp {
                    cmp::Ordering::Less => less_count += 1,
                    cmp::Ordering::Equal => equal_count += 1,
                    cmp::Ordering::Greater => {}
                }
            }
            // If by accident we picked the minimum element as a pivot, we just call sort again with the
            // rest of the vector.
            if less_count == 0 {
                do_dual_sort(rest_focus, rest_dual, rng, comparator);
                return;
            }

            // We know here that there is at least one item before the pivot, so we move the minimum to the
            // beginning part of the vector. First, however we swap the pivot to the start of the equal
            // zone.
            less_count -= 1;
            equal_count += 1;
            mem::swap(first_focus.index(0), rest_focus.index(less_count));
            mem::swap(first_dual.index(0), rest_dual.index(less_count));
            for index in 0..rest_focus.len() {
                if index == less_count {
                    // This is the position we swapped the pivot to. We can't move it from its position, and
                    // we know its not the minimum.
                    continue;
                }
                // let rest_item = rest.index(index);
                if comparator(rest_focus.index(index), first_focus.index(0)) == cmp::Ordering::Less
                {
                    mem::swap(first_focus.index(0), rest_focus.index(index));
                    mem::swap(first_dual.index(0), rest_dual.index(index));
                }
            }
            // Split the vector up into less_than, equal to and greater than parts.
            rest_focus.split_at_fn(less_count + equal_count, |rest_focus, greater_focus| {
                rest_dual.split_at_fn(less_count + equal_count, |rest_dual, greater_dual| {
                    assert_eq!(greater_focus.len(), greater_dual.len());

                    rest_focus.split_at_fn(less_count, |less_focus, equal_focus| {
                        rest_dual.split_at_fn(less_count, |less_dual, equal_dual| {
                            assert_eq!(equal_focus.len(), equal_dual.len());
                            assert_eq!(less_focus.len(), less_dual.len());

                            let mut less_position = 0;
                            let mut equal_position = 0;
                            let mut greater_position = 0;

                            while less_position != less_focus.len()
                                || greater_position != greater_focus.len()
                            {
                                // At start of this loop, equal_position always points to an equal item
                                let mut equal_swap_side = None;

                                // Advance the less_position until we find an out of place item
                                while less_position != less_focus.len() {
                                    match comparator(
                                        less_focus.index(less_position),
                                        equal_focus.index(equal_position),
                                    ) {
                                        cmp::Ordering::Equal => {
                                            equal_swap_side = Some(cmp::Ordering::Less);
                                            break;
                                        }
                                        cmp::Ordering::Greater => {
                                            break;
                                        }
                                        _ => {}
                                    }
                                    less_position += 1;
                                }

                                // Advance the greater until we find an out of place item
                                while greater_position != greater_focus.len() {
                                    match comparator(
                                        greater_focus.index(greater_position),
                                        equal_focus.index(equal_position),
                                    ) {
                                        cmp::Ordering::Less => break,
                                        cmp::Ordering::Equal => {
                                            equal_swap_side = Some(cmp::Ordering::Greater);
                                            break;
                                        }
                                        _ => {}
                                    }
                                    greater_position += 1;
                                }

                                if let Some(swap_side) = equal_swap_side {
                                    // One of the sides is equal to the pivot, advance the pivot
                                    if swap_side == cmp::Ordering::Less {
                                        while comparator(
                                            less_focus.index(less_position),
                                            equal_focus.index(equal_position),
                                        ) == cmp::Ordering::Equal
                                        {
                                            equal_position += 1;
                                        }

                                        // Swap the equal position and the desired side, it's important to note that only the
                                        // equals focus is guaranteed to have made progress so we don't advance the side's index
                                        mem::swap(
                                            less_focus.index(less_position),
                                            equal_focus.index(equal_position),
                                        );
                                        mem::swap(
                                            less_dual.index(less_position),
                                            equal_dual.index(equal_position),
                                        );
                                    } else {
                                        while comparator(
                                            greater_focus.index(greater_position),
                                            equal_focus.index(equal_position),
                                        ) == cmp::Ordering::Equal
                                        {
                                            equal_position += 1;
                                        }

                                        // Swap the equal position and the desired side, it's important to note that only the
                                        // equals focus is guaranteed to have made progress so we don't advance the side's index
                                        mem::swap(
                                            greater_focus.index(greater_position),
                                            equal_focus.index(equal_position),
                                        );
                                        mem::swap(
                                            greater_dual.index(greater_position),
                                            equal_dual.index(equal_position),
                                        );
                                    }
                                } else if less_position != less_focus.len()
                                    && greater_position != greater_focus.len()
                                {
                                    // Both sides are out of place and not equal to the pivot, this can only happen if there
                                    // is a greater item in the lesser zone and a lesser item in the greater zone. The
                                    // solution is to swap both sides and advance both side's indices.
                                    debug_assert_ne!(
                                        comparator(
                                            less_focus.index(less_position),
                                            equal_focus.index(equal_position),
                                        ),
                                        cmp::Ordering::Equal
                                    );
                                    debug_assert_ne!(
                                        comparator(
                                            greater_focus.index(greater_position),
                                            equal_focus.index(equal_position)
                                        ),
                                        cmp::Ordering::Equal
                                    );
                                    mem::swap(
                                        less_focus.index(less_position),
                                        greater_focus.index(greater_position),
                                    );
                                    mem::swap(
                                        less_dual.index(less_position),
                                        greater_dual.index(greater_position),
                                    );
                                    less_position += 1;
                                    greater_position += 1;
                                }
                            }

                            // Now we have partitioned both sides correctly, we just have to recurse now
                            do_dual_sort(less_focus, less_dual, rng, comparator);
                            if !greater_focus.is_empty() {
                                do_dual_sort(greater_focus, greater_dual, rng, comparator);
                            }
                        });
                    });
                });
            });
        });
    });
}

#[allow(clippy::cognitive_complexity)]
#[cfg(test)]
mod test {
    use crate::*;
    use ::proptest::num::i32;
    use proptest::proptest;

    proptest! {
        #[test]
        fn test_quicksort(ref input in proptest::collection::vec(i32::ANY, 0..10_000)) {
            let mut vec = input.clone();
            let mut vector = Vector::new();
            for i in vec.iter() {
                vector.push_back(*i);
            }
            assert!(vec.iter().eq(vector.iter()));
            vector.sort();
            vec.sort();
            assert!(vec.iter().eq(vector.iter()));
        }
    }
}
