use crate::traits::{Constructible, ProbabilisticFilter};
use crate::utils::{SetQuery, mask_fingerprint, reduce};
use num_traits::{One, Zero};
use rand::Rng;
use std::collections::VecDeque;
use std::ops::{BitAnd, BitXor, Not, Shl, Sub};

/// Configuration for an XOR filter.
///
/// # Type Parameters
/// * `T` - The type of items stored in the filter
/// * `F` - The fingerprint type (e.g., u8, u16, u32)
/// * `H` - The hash function that computes indices from data and seed
/// * `FP` - Function that computes fingerprint from an item
///
/// # Fields
/// * `hash` - The hash function to compute bucket indices
/// * `fingerprint_func` - Function to compute fingerprint from an item
/// * `effective_bits` - Number of bits used from the fingerprint type.
#[derive(Clone, Copy)]
pub struct XorFilterConfig<T, F, H, FP>
where
    F: Copy,
    H: Fn(&T, u64) -> u64,
    FP: Fn(&T) -> F,
{
    pub hash: H,
    pub fingerprint_func: FP,
    pub effective_bits: usize,
    _phantom: std::marker::PhantomData<(T, F)>,
}

impl<T, F, H, FP> XorFilterConfig<T, F, H, FP>
where
    F: Copy,
    H: Fn(&T, u64) -> u64,
    FP: Fn(&T) -> F,
{
    /// Create a new configuration with the given hash function, fingerprint function, and effective bits.
    ///
    /// # Example
    /// ```
    /// use probabilistic_filter::xor::XorFilterConfig;
    /// use probabilistic_filter::utils::hash_with_seed;
    ///
    /// let hash = |item: &u64, seed: u64| hash_with_seed(item, seed);
    /// let fp_func = |item: &u64| (item.wrapping_mul(0x5bd1e9955bd1e995) & 0xFF) as u8;
    /// let config = XorFilterConfig::new(hash, fp_func, 8);
    /// ```
    pub fn new(hash: H, fingerprint_func: FP, effective_bits: usize) -> Self {
        XorFilterConfig {
            hash,
            fingerprint_func,
            effective_bits,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// An XOR filter is a probabilistic data structure that supports membership testing.
/// It is constructed in one pass from a set of items and cannot be modified afterward.
/// It has better space efficiency than Bloom filters but does not support insertion or deletion.
///
/// # Type Parameters
/// * `T` - The type of items stored in the filter
/// * `F` - The fingerprint type (e.g., u8, u16)
/// * `H` - The hash function that computes indices from data and seed
/// * `FP` - Function that computes fingerprint from an item
///
/// # References
/// * Thomas Mueller Graf and Daniel Lemire. 2020. Xor Filters: Faster and Smaller Than Bloom and Cuckoo Filters. ACM J. Exp. Algorithmics 25, Article 1.5 (2020), 16 pages. https://doi.org/10.1145/3376122
pub struct XorFilter<T, F, H, FP>
where
    F: Copy
        + PartialEq
        + Zero
        + One
        + BitAnd<Output = F>
        + BitXor<Output = F>
        + Not<Output = F>
        + Shl<usize, Output = F>
        + Sub<Output = F>,
    H: Fn(&T, u64) -> u64,
    FP: Fn(&T) -> F,
{
    array_size: usize,
    fingerprints: Vec<F>,
    hash: H,
    seed0: u64,
    seed1: u64,
    seed2: u64,
    fingerprint_func: FP,
    effective_bits: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F, H, FP> XorFilter<T, F, H, FP>
where
    F: Copy
        + PartialEq
        + Zero
        + One
        + BitAnd<Output = F>
        + BitXor<Output = F>
        + Not<Output = F>
        + Shl<usize, Output = F>
        + Sub<Output = F>,
    H: Fn(&T, u64) -> u64,
    FP: Fn(&T) -> F,
{
    /// Create a new XOR filter from an iterator of items with provided hash functions.
    ///
    /// # Arguments
    /// * `items` - Iterator of items to include in the filter
    /// * `hash` - The hash function to compute indices
    /// * `fingerprint_func` - Function to compute fingerprint from an item
    /// * `effective_bits` - Number of bits actually used from the fingerprint type
    pub fn new<U>(items: U, hash: H, fingerprint_func: FP, effective_bits: usize) -> Self
    where
        U: IntoIterator<Item = T>,
    {
        let items: Vec<T> = items.into_iter().collect();
        let num_items = items.len();

        if num_items == 0 {
            return XorFilter {
                array_size: 32,
                fingerprints: vec![F::zero(); 32],
                hash,
                seed0: 0,
                seed1: 0,
                seed2: 0,
                fingerprint_func,
                effective_bits,
                _phantom: std::marker::PhantomData,
            };
        }

        // Calculate array size: 1.23 * num_items + 32
        let array_size = ((num_items as f64 * 1.23 + 32.0).ceil()) as usize;
        let mut fingerprints = vec![F::zero(); array_size];
        let mut rng = rand::rng();

        loop {
            // Build a map from position to list of items that use this position
            // Each entry is item_index
            // Use SetQuery to optimize allocation away
            let mut pos_to_items: Vec<SetQuery> = vec![SetQuery::new(); array_size];

            let seed0 = rng.random::<u64>();
            let seed1 = rng.random::<u64>();
            let seed2 = rng.random::<u64>();

            // wrap three hash functions
            let third_size = array_size / 3;
            let two_thirds_size = 2 * third_size;

            let hash0 = |item: &T| -> usize {
                let hash = hash(item, seed0);
                reduce(hash as u32, third_size as u32) as usize
            };

            let hash1 = |item: &T| -> usize {
                let hash = hash(item, seed1);
                third_size + reduce(hash as u32, third_size as u32) as usize
            };

            let hash2 = |item: &T| -> usize {
                let hash = hash(item, seed2);
                two_thirds_size
                    + reduce(hash as u32, (array_size - two_thirds_size) as u32) as usize
            };

            for (item_idx, item) in items.iter().enumerate() {
                let idx0 = (hash0)(item);
                let idx1 = (hash1)(item);
                let idx2 = (hash2)(item);

                pos_to_items[idx0].insert(item_idx);
                pos_to_items[idx1].insert(item_idx);
                pos_to_items[idx2].insert(item_idx);
            }

            // Graph peeling: process items that have at least one available position
            let mut queue: VecDeque<usize> = VecDeque::new();
            let mut in_queue: Vec<bool> = vec![false; array_size];

            // Stack to store (item_idx, pos, idx0, idx1, idx2) for reconstruction
            let mut stack: Vec<(usize, usize, usize, usize, usize)> = Vec::new();

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
                    debug_assert!(pos_to_items[pos].is_empty());
                    continue;
                }

                // Get the only item at this position
                let (item_idx,) = pos_to_items[pos].query();

                // Compute indices again
                let item = &items[item_idx];
                let idx0 = (hash0)(item);
                let idx1 = (hash1)(item);
                let idx2 = (hash2)(item);

                // Push to stack for later reconstruction
                // Store (item_idx, pos, idx1, idx2, idx3) where pos is where we'll store fingerprint
                stack.push((item_idx, pos, idx0, idx1, idx2));

                // Remove this item from all its positions
                pos_to_items[idx0].remove(item_idx);
                pos_to_items[idx1].remove(item_idx);
                pos_to_items[idx2].remove(item_idx);

                // Check if any positions now have exactly one item
                for pos in [idx0, idx1, idx2] {
                    if pos_to_items[pos].len() == 1 && !in_queue[pos] {
                        in_queue[pos] = true;
                        queue.push_back(pos);
                    }
                }
            }

            if stack.len() != num_items {
                // try again
                continue;
            }

            // Reconstruct fingerprints in reverse peeling order
            while let Some((item_idx, pos, idx0, idx1, idx2)) = stack.pop() {
                let item = &items[item_idx];
                let fingerprint = Self::get_fingerprint(&fingerprint_func, effective_bits, item);

                // Since fingerprints[pos] starts at 0, we get f XOR fp[idx0] XOR fp[idx1] XOR fp[idx2]
                // XOR with zero is identity, so this correctly computes the needed value
                fingerprints[pos] =
                    fingerprint ^ fingerprints[idx0] ^ fingerprints[idx1] ^ fingerprints[idx2];
            }
            return XorFilter {
                array_size,
                fingerprints,
                hash,
                seed0,
                seed1,
                seed2,
                fingerprint_func,
                effective_bits,
                _phantom: std::marker::PhantomData,
            };
        }
    }

    /// Create a new XOR filter from an iterator of items with configuration.
    pub fn new_with_config<U>(items: U, config: XorFilterConfig<T, F, H, FP>) -> Self
    where
        U: IntoIterator<Item = T>,
    {
        Self::new(
            items,
            config.hash,
            config.fingerprint_func,
            config.effective_bits,
        )
    }

    fn get_fingerprint(fingerprint_func: &FP, effective_bits: usize, item: &T) -> F {
        let fp = fingerprint_func(item);
        let masked = mask_fingerprint(effective_bits, fp);
        // Ensure non-zero fingerprint (0 does not change xor result)
        // Only adjust if masked result is actually zero
        if masked.is_zero() { F::one() } else { masked }
    }

    /// Returns the number of fingerprints stored in the filter.
    pub fn fingerprint_count(&self) -> usize {
        self.fingerprints.len()
    }
}

impl<T, F, H, FP> ProbabilisticFilter<T> for XorFilter<T, F, H, FP>
where
    F: Copy
        + PartialEq
        + Zero
        + One
        + BitAnd<Output = F>
        + BitXor<Output = F>
        + Not<Output = F>
        + Shl<usize, Output = F>
        + Sub<Output = F>,
    H: Fn(&T, u64) -> u64,
    FP: Fn(&T) -> F,
{
    fn contains(&self, item: &T) -> bool {
        let fingerprint = Self::get_fingerprint(&self.fingerprint_func, self.effective_bits, item);

        // wrap three hash functions
        let array_size = self.array_size;
        let hash = &self.hash;
        let seed0 = self.seed0;
        let seed1 = self.seed1;
        let seed2 = self.seed2;
        let third_size = array_size / 3;
        let two_thirds_size = 2 * third_size;

        let hash0 = hash(item, seed0);
        let idx0 = reduce(hash0 as u32, third_size as u32) as usize;

        let hash1 = hash(item, seed1);
        let idx1 = third_size + reduce(hash1 as u32, third_size as u32) as usize;

        let hash2 = hash(item, seed2);
        let idx2 =
            two_thirds_size + reduce(hash2 as u32, (array_size - two_thirds_size) as u32) as usize;

        let xored = self.fingerprints[idx0] ^ self.fingerprints[idx1] ^ self.fingerprints[idx2];
        xored == fingerprint
    }

    fn size_in_bits(&self) -> usize {
        self.fingerprints.len() * self.effective_bits
    }
}

impl<T, F, H, FP> Constructible<T> for XorFilter<T, F, H, FP>
where
    F: Copy
        + PartialEq
        + Zero
        + One
        + BitAnd<Output = F>
        + BitXor<Output = F>
        + Not<Output = F>
        + Shl<usize, Output = F>
        + Sub<Output = F>,
    H: Fn(&T, u64) -> u64,
    FP: Fn(&T) -> F,
{
    type Config = XorFilterConfig<T, F, H, FP>;

    fn from_iter_with<U>(items: U, config: Self::Config) -> Self
    where
        Self: Sized,
        U: IntoIterator<Item = T>,
    {
        Self::new_with_config(items, config)
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::hash_with_seed;

    use super::*;

    #[test]
    fn test_xor_filter_basic() {
        use crate::traits::Constructible;

        let hash = |item: &u64, seed: u64| -> u64 { hash_with_seed(item, seed) };
        let fp_func = |item: &u64| -> u8 { item.wrapping_mul(0x5bd1e9955bd1e995) as u8 };
        let items: Vec<u64> = vec![1, 2, 3];
        let config = XorFilterConfig::new(hash, fp_func, 8);

        let xor = XorFilter::from_iter_with(items, config);

        assert!(xor.contains(&1));
        assert!(xor.contains(&2));
        assert!(xor.contains(&3));
        assert!(!xor.contains(&4));
    }

    #[test]
    fn test_xor_filter_empty() {
        use crate::traits::Constructible;

        let hash = |item: &u64, seed: u64| -> u64 { hash_with_seed(item, seed) };
        let fp_func = |item: &u64| -> u8 { item.wrapping_mul(0x5bd1e9955bd1e995) as u8 };
        let items: Vec<u64> = vec![];
        let config = XorFilterConfig::new(hash, fp_func, 8);

        let xor = XorFilter::from_iter_with(items, config);

        assert!(!xor.contains(&1));
    }
}
