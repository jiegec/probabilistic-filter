use crate::traits::{Constructible, ProbabilisticFilter};
use num_traits::Zero;
use std::collections::VecDeque;
use std::ops::BitXor;

/// An XOR filter is a probabilistic data structure that supports membership testing.
/// It is constructed in one pass from a set of items and cannot be modified afterward.
/// It has better space efficiency than Bloom filters but does not support insertion or deletion.
///
/// # Type Parameters
/// * `T` - The type of items stored in the filter
/// * `F` - The fingerprint type (e.g., u8, u16)
/// * `H0` - First hash function that computes index 0
/// * `H1` - Second hash function that computes index 1
/// * `H2` - Third hash function that computes index 2
/// * `FP` - Function that computes fingerprint from an item
///
/// # References
/// * Thomas Mueller Graf and Daniel Lemire. 2020. Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters. ACM J. Exp. Algorithmics 25, Article 1.5 (2020), 16 pages. https://doi.org/10.1145/3376122
pub struct XorFilter<T, F, H0, H1, H2, FP>
where
    F: Copy + PartialEq + Zero + BitXor<Output = F>,
    H0: Fn(&T) -> usize,
    H1: Fn(&T) -> usize,
    H2: Fn(&T) -> usize,
    FP: Fn(&T) -> F,
{
    array_size: usize,
    fingerprints: Vec<F>,
    hash0: H0,
    hash1: H1,
    hash2: H2,
    fingerprint_func: FP,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F, H0, H1, H2, FP> XorFilter<T, F, H0, H1, H2, FP>
where
    F: Copy + PartialEq + Zero + BitXor<Output = F>,
    H0: Fn(&T) -> usize,
    H1: Fn(&T) -> usize,
    H2: Fn(&T) -> usize,
    FP: Fn(&T) -> F,
{
    /// Create a new XOR filter from an iterator of items with provided hash functions.
    ///
    /// # Arguments
    /// * `items` - Iterator of items to include in the filter
    /// * `hash0` - First hash function to compute index 0
    /// * `hash1` - Second hash function to compute index 1
    /// * `hash2` - Third hash function to compute index 2
    /// * `fingerprint_func` - Function to compute fingerprint from an item
    pub fn new<U>(items: U, hash0: H0, hash1: H1, hash2: H2, fingerprint_func: FP) -> Self
    where
        U: IntoIterator<Item = T>,
    {
        let items: Vec<T> = items.into_iter().collect();
        let num_items = items.len();

        if num_items == 0 {
            return XorFilter {
                array_size: 32,
                fingerprints: vec![F::zero(); 32],
                hash0,
                hash1,
                hash2,
                fingerprint_func,
                _phantom: std::marker::PhantomData,
            };
        }

        // Calculate array size: 1.23 * num_items + 32, rounded up to next power of 2
        let mut array_size = ((num_items as f64 * 1.23 + 32.0).ceil()) as usize;
        array_size = array_size.max(32).next_power_of_two();

        let mut fingerprints = vec![F::zero(); array_size];

        // Build a map from position to list of items that use this position
        // Each entry is (item_index, other_position_1, other_position_2)
        let mut pos_to_items: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new(); array_size];

        for (item_idx, item) in items.iter().enumerate() {
            let idx0 = (hash0)(item) % array_size;
            let idx1 = (hash1)(item) % array_size;
            let idx2 = (hash2)(item) % array_size;

            pos_to_items[idx0].push((item_idx, idx1, idx2));
            pos_to_items[idx1].push((item_idx, idx0, idx2));
            pos_to_items[idx2].push((item_idx, idx0, idx1));
        }

        // Graph peeling: process items that have at least one available position
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut in_queue: Vec<bool> = vec![false; array_size];

        // Stack to store (item_idx, pos) for reconstruction
        let mut stack: Vec<(usize, usize)> = Vec::new();

        // Find initial positions with exactly one item
        for (pos, items) in pos_to_items.iter().enumerate().take(array_size) {
            if items.len() == 1 && !in_queue[pos] {
                in_queue[pos] = true;
                queue.push_back(pos);
            }
        }

        // Process positions from queue
        while let Some(pos) = queue.pop_front() {
            in_queue[pos] = false;

            // Skip if position no longer has exactly one item
            if pos_to_items[pos].len() != 1 {
                continue;
            }

            // Get the only item at this position
            let (item_idx, _, _) = pos_to_items[pos][0];
            let item = &items[item_idx];

            // Compute indices for this item
            let idx0 = (hash0)(item) % array_size;
            let idx1 = (hash1)(item) % array_size;
            let idx2 = (hash2)(item) % array_size;

            // Push to stack for later reconstruction
            // Store (item_idx, pos) where pos is where we'll store fingerprint
            stack.push((item_idx, pos));

            // Remove this item from all its positions
            remove_item_from_position(&mut pos_to_items, idx0, item_idx);
            remove_item_from_position(&mut pos_to_items, idx1, item_idx);
            remove_item_from_position(&mut pos_to_items, idx2, item_idx);

            // Check if any positions now have exactly one item
            check_positions_for_queue(&pos_to_items, &mut queue, &mut in_queue, idx0, idx1, idx2);
        }

        // Reconstruct fingerprints in reverse peeling order
        while let Some((item_idx, pos)) = stack.pop() {
            let item = &items[item_idx];
            let fingerprint = (fingerprint_func)(item);

            // Compute indices for this item
            let idx0 = (hash0)(item) % array_size;
            let idx1 = (hash1)(item) % array_size;
            let idx2 = (hash2)(item) % array_size;

            // Since fingerprints[pos] starts at 0, we get f XOR fp[idx0] XOR fp[idx1] XOR fp[idx2]
            // XOR with zero is identity, so this correctly computes the needed value
            fingerprints[pos] =
                fingerprint ^ fingerprints[idx0] ^ fingerprints[idx1] ^ fingerprints[idx2];
        }

        XorFilter {
            array_size,
            fingerprints,
            hash0,
            hash1,
            hash2,
            fingerprint_func,
            _phantom: std::marker::PhantomData,
        }
    }
}

fn remove_item_from_position(
    pos_to_items: &mut [Vec<(usize, usize, usize)>],
    pos: usize,
    item_idx: usize,
) {
    pos_to_items[pos].retain(|&(idx, _, _)| idx != item_idx);
}

fn check_positions_for_queue(
    pos_to_items: &[Vec<(usize, usize, usize)>],
    queue: &mut VecDeque<usize>,
    in_queue: &mut [bool],
    idx0: usize,
    idx1: usize,
    idx2: usize,
) {
    for pos in [idx0, idx1, idx2] {
        if pos_to_items[pos].len() == 1 && !in_queue[pos] {
            in_queue[pos] = true;
            queue.push_back(pos);
        }
    }
}

impl<T, F, H0, H1, H2, FP> ProbabilisticFilter<T> for XorFilter<T, F, H0, H1, H2, FP>
where
    F: Copy + PartialEq + Zero + BitXor<Output = F>,
    H0: Fn(&T) -> usize,
    H1: Fn(&T) -> usize,
    H2: Fn(&T) -> usize,
    FP: Fn(&T) -> F,
{
    fn contains(&self, item: &T) -> bool {
        let fingerprint = (self.fingerprint_func)(item);

        let idx0 = (self.hash0)(item) % self.array_size;
        let idx1 = (self.hash1)(item) % self.array_size;
        let idx2 = (self.hash2)(item) % self.array_size;

        let xored = self.fingerprints[idx0] ^ self.fingerprints[idx1] ^ self.fingerprints[idx2];
        xored == fingerprint
    }

    fn size_in_bits(&self) -> usize {
        self.fingerprints.len() * std::mem::size_of::<F>() * 8
    }
}

impl<T, F, H0, H1, H2, FP> Constructible<T> for XorFilter<T, F, H0, H1, H2, FP>
where
    F: Copy + PartialEq + Zero + BitXor<Output = F>,
    H0: Fn(&T) -> usize,
    H1: Fn(&T) -> usize,
    H2: Fn(&T) -> usize,
    FP: Fn(&T) -> F,
{
    type Config = (H0, H1, H2, FP);

    fn from_iter_with<U>(items: U, config: Self::Config) -> Self
    where
        Self: Sized,
        U: IntoIterator<Item = T>,
    {
        Self::new(items, config.0, config.1, config.2, config.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_filter_basic() {
        use crate::traits::Constructible;

        let hash0 = |item: &u64| -> usize { item.wrapping_mul(0x5bd1e9955bd1e995) as usize };
        let hash1 = |item: &u64| -> usize { item.wrapping_mul(0x9e3779b97f4a7c1) as usize };
        let hash2 = |item: &u64| -> usize { item.wrapping_mul(0xc13fa9a902a6328f) as usize };
        let fp_func =
            |item: &u64| -> u8 { (item.wrapping_mul(0x5bd1e9955bd1e995) & 0xFF) as u8 | 1 };
        let items: Vec<u64> = vec![1, 2, 3];

        let xor = XorFilter::from_iter_with(items, (hash0, hash1, hash2, fp_func));

        assert!(xor.contains(&1));
        assert!(xor.contains(&2));
        assert!(xor.contains(&3));
        assert!(!xor.contains(&4));
    }

    #[test]
    fn test_xor_filter_empty() {
        use crate::traits::Constructible;

        let hash0 = |item: &u64| -> usize { item.wrapping_mul(0x5bd1e9955bd1e995) as usize };
        let hash1 = |item: &u64| -> usize { item.wrapping_mul(0x9e3779b97f4a7c1) as usize };
        let hash2 = |item: &u64| -> usize { item.wrapping_mul(0xc13fa9a902a6328f) as usize };
        let fp_func =
            |item: &u64| -> u8 { (item.wrapping_mul(0x5bd1e9955bd1e995) & 0xFF) as u8 | 1 };
        let items: Vec<u64> = vec![];

        let xor = XorFilter::from_iter_with(items, (hash0, hash1, hash2, fp_func));

        assert!(!xor.contains(&1));
    }
}
