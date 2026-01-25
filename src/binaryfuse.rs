use crate::traits::{Constructible, ProbabilisticFilter};
use crate::utils::mask_fingerprint;
use num_traits::{One, Zero};
use rand::Rng;
use std::{collections::VecDeque, ops::Not};

/// Configuration for a binary fuse filter.
///
/// # Fields
/// * `arity` - The arity of the filter (3 or 4). Each key is hashed to `arity` candidate buckets.
/// * `effective_bits` - Number of bits used from the fingerprint type.
#[derive(Clone, Copy)]
pub struct BinaryFuseConfig {
    pub arity: usize,
    pub effective_bits: usize,
}

impl BinaryFuseConfig {
    /// Create a new configuration with the given arity and effective bits.
    pub fn new(arity: usize, effective_bits: usize) -> Self {
        assert!(arity == 3 || arity == 4);
        BinaryFuseConfig {
            arity,
            effective_bits,
        }
    }
}

/// A binary fuse filter is a space-efficient probabilistic data structure for membership testing.
/// It is constructed in one pass from a set of items and cannot be modified afterward.
///
/// The filter uses a table of segments.
///
/// # Type Parameters
/// * `T` - The type of items stored in the filter
/// * `F` - The fingerprint type (e.g., u8, u16, u32)
/// * `H` - The hash function that computes indices from data and seed
/// * `FP` - Function that computes fingerprint from an item
///
/// # References
/// * Thomas Mueller Graf and Daniel Lemire. 2022. Binary Fuse Filters: Fast and Smaller Than Xor Filters. ACM J. Exp. Algorithmics 27, Article 1.5 (December 2022), 15 pages. https://doi.org/10.1145/3510449
pub struct BinaryFuseFilter<T, F, H, FP>
where
    F: Copy
        + PartialEq
        + std::ops::BitXor<Output = F>
        + std::ops::BitAnd<Output = F>
        + std::ops::Shl<usize, Output = F>
        + std::ops::Sub<Output = F>,
    H: Fn(&T, u64) -> usize,
    FP: Fn(&T) -> F,
{
    num_segments: usize,
    segment_size: usize,
    // flattened to contain fingerprints from all segments
    fingerprints: Vec<F>,
    // seed for computing the first segment index from hash
    seed0: u64,
    // seeds for the 3/4 hash functions
    seeds: Vec<u64>,
    hash: H,
    fingerprint_func: FP,
    effective_bits: usize,
    arity: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, F, H, FP> BinaryFuseFilter<T, F, H, FP>
where
    F: Copy
        + PartialEq
        + std::ops::BitXor<Output = F>
        + std::ops::BitAnd<Output = F>
        + std::ops::Shl<usize, Output = F>
        + std::ops::Sub<Output = F>
        + Zero
        + One
        + Not<Output = F>,
    H: Fn(&T, u64) -> usize,
    FP: Fn(&T) -> F,
{
    /// Create a new binary fuse filter from an iterator of items.
    ///
    /// # Arguments
    /// * `items` - Iterator of items to include in the filter
    /// * `hash` - The hash function to compute bucket indices
    /// * `fingerprint_func` - Function to compute fingerprint from an item
    /// * `effective_bits` - Number of bits actually used from the fingerprint type
    /// * `arity` - The arity of the filter (3 or 4)
    pub fn new<U>(
        items: U,
        hash: H,
        fingerprint_func: FP,
        effective_bits: usize,
        arity: usize,
    ) -> Self
    where
        U: IntoIterator<Item = T>,
    {
        let config = BinaryFuseConfig::new(arity, effective_bits);
        Self::new_with_config(items, hash, fingerprint_func, config)
    }

    /// Create a new binary fuse filter from an iterator of items with configuration.
    pub fn new_with_config<U>(
        items: U,
        hash: H,
        fingerprint_func: FP,
        config: BinaryFuseConfig,
    ) -> Self
    where
        U: IntoIterator<Item = T>,
    {
        let items: Vec<T> = items.into_iter().collect();
        let num_items = items.len();

        // from paper Table 1
        let segment_size = match config.arity {
            3 => {
                let pow = ((num_items as f64).log(3.33) + 2.25).floor() as usize;
                1 << pow
            }
            4 => {
                let pow = ((num_items as f64).log(2.91) - 0.5).floor() as usize;
                1 << pow
            }
            _ => unreachable!("Invalid arity"),
        };

        let mut array_size = match config.arity {
            3 => {
                let log = (1000000.0f64).log(num_items as f64);
                ((0.875 + 0.25 * (if log > 1.0 { log } else { 1.0 })) * (num_items as f64)).floor()
                    as usize
            }
            4 => {
                let log = (600000.0f64).log(num_items as f64);
                ((0.77 + 0.305 * (if log > 1.0 { log } else { 1.0 })) * (num_items as f64)).floor()
                    as usize
            }
            _ => unreachable!("Invalid arity"),
        };
        // round up to multiples of segment size
        let mut num_segments = array_size.div_ceil(segment_size);
        if num_segments < config.arity {
            num_segments = config.arity;
        }

        // add extra segments to avoid modulo operation
        // and make it easy to find solution
        array_size = (num_segments + config.arity - 1) * segment_size;
        let mut fingerprints = vec![F::zero(); array_size];

        let mut rng = rand::rng();

        loop {
            // Build a map from position to list of items that use this position
            // Each entry is (item_index, other_positions)
            let mut pos_to_items: Vec<Vec<(usize, Vec<usize>)>> = vec![Vec::new(); array_size];

            // Generate random seeds for each hash function and segment index
            let seed0 = rng.random::<u64>();
            let seeds: Vec<u64> = (0..config.arity).map(|_| rng.random::<u64>()).collect();

            for (item_idx, item) in items.iter().enumerate() {
                // starting segment index
                let segment_idx = hash(item, seed0) % num_segments;
                let mut indices = vec![];
                for (i, seed) in seeds.iter().enumerate() {
                    // consecutive segment indices
                    let idx = (segment_idx + i) * segment_size + hash(item, *seed) % segment_size;
                    indices.push(idx);
                }

                for i in 0..config.arity {
                    let mut other_indices = indices.clone();
                    other_indices.remove(i);
                    pos_to_items[indices[i]].push((item_idx, other_indices));
                }
            }

            // Graph peeling: process items that have at least one available position
            let mut queue: VecDeque<usize> = VecDeque::new();
            let mut in_queue: Vec<bool> = vec![false; array_size];

            // Stack to store (item_idx, pos, other_pos) for reconstruction
            let mut stack: Vec<(usize, usize, Vec<usize>)> = Vec::new();

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
                let (item_idx, other_indices) = pos_to_items[pos][0].clone();

                // Push to stack for later reconstruction
                // Store (item_idx, pos, other_indices) where pos is where we'll store fingerprint
                stack.push((item_idx, pos, other_indices.clone()));

                // Remove this item from all its positions
                remove_item_from_position(&mut pos_to_items, pos, item_idx);
                for idx in &other_indices {
                    remove_item_from_position(&mut pos_to_items, *idx, item_idx);
                }

                // Check if any positions now have exactly one item
                check_positions_for_queue(&pos_to_items, &mut queue, &mut in_queue, &other_indices);
            }

            if stack.len() != num_items {
                // try again
                continue;
            }

            // Reconstruct fingerprints in reverse peeling order
            while let Some((item_idx, pos, other_pos)) = stack.pop() {
                let item = &items[item_idx];
                let fingerprint =
                    Self::get_fingerprint(&fingerprint_func, config.effective_bits, item);

                // Since fingerprints[pos] starts at 0, we get f XOR fp[idx0] XOR fp[idx1] XOR fp[idx2]
                // XOR with zero is identity, so this correctly computes the needed value
                let mut fp = fingerprint;
                fp = fp ^ fingerprints[pos];
                for idx in other_pos {
                    fp = fp ^ fingerprints[idx];
                }
                fingerprints[pos] = fp;
            }
            return BinaryFuseFilter {
                num_segments,
                segment_size,
                fingerprints,
                seed0,
                seeds,
                hash,
                fingerprint_func,
                effective_bits: config.effective_bits,
                arity: config.arity,
                _phantom: std::marker::PhantomData,
            };
        }
    }

    fn get_fingerprint(fingerprint_func: &FP, effective_bits: usize, item: &T) -> F {
        let fp = fingerprint_func(item);
        let masked = mask_fingerprint(effective_bits, fp);
        // Ensure non-zero fingerprint (0 represents empty slot)
        if masked.is_zero() { F::one() } else { masked }
    }

    /// Returns the number of segments (including padding segments) in the filter.
    pub fn num_segments(&self) -> usize {
        self.num_segments
    }

    /// Returns the size (number of fingerprints) in each segment.
    pub fn segment_size(&self) -> usize {
        self.segment_size
    }

    /// Returns the total number of segments (including padding segments).
    pub fn total_segments(&self) -> usize {
        self.num_segments + self.arity - 1
    }
}

fn remove_item_from_position(
    pos_to_items: &mut [Vec<(usize, Vec<usize>)>],
    pos: usize,
    item_idx: usize,
) {
    pos_to_items[pos].retain(|&(idx, _)| idx != item_idx);
}

fn check_positions_for_queue(
    pos_to_items: &[Vec<(usize, Vec<usize>)>],
    queue: &mut VecDeque<usize>,
    in_queue: &mut [bool],
    other_pos: &Vec<usize>,
) {
    for pos in other_pos {
        if pos_to_items[*pos].len() == 1 && !in_queue[*pos] {
            in_queue[*pos] = true;
            queue.push_back(*pos);
        }
    }
}

impl<T, F, H, FP> ProbabilisticFilter<T> for BinaryFuseFilter<T, F, H, FP>
where
    F: Copy
        + PartialEq
        + std::ops::BitXor<Output = F>
        + std::ops::BitAnd<Output = F>
        + std::ops::Shl<usize, Output = F>
        + std::ops::Sub<Output = F>
        + Zero
        + One
        + Not<Output = F>,
    H: Fn(&T, u64) -> usize,
    FP: Fn(&T) -> F,
{
    fn contains(&self, item: &T) -> bool {
        let mut fingerprint =
            Self::get_fingerprint(&self.fingerprint_func, self.effective_bits, item);

        // starting segment index
        let segment_idx = (self.hash)(item, self.seed0) % self.num_segments;
        for i in 0..self.arity {
            // consecutive segment indices
            let idx = (segment_idx + i) * self.segment_size
                + (self.hash)(item, self.seeds[i]) % self.segment_size;
            fingerprint = fingerprint ^ self.fingerprints[idx];
        }

        fingerprint == F::zero()
    }

    fn size_in_bits(&self) -> usize {
        self.fingerprints.len() * self.effective_bits
    }
}

impl<T, F, H, FP> Constructible<T> for BinaryFuseFilter<T, F, H, FP>
where
    F: Copy
        + PartialEq
        + std::ops::BitXor<Output = F>
        + std::ops::BitAnd<Output = F>
        + std::ops::Shl<usize, Output = F>
        + std::ops::Sub<Output = F>
        + Zero
        + One
        + Not<Output = F>,
    H: Fn(&T, u64) -> usize,
    FP: Fn(&T) -> F,
{
    type Config = (H, FP, BinaryFuseConfig);

    fn from_iter_with<U>(items: U, config: Self::Config) -> Self
    where
        Self: Sized,
        U: IntoIterator<Item = T>,
    {
        Self::new_with_config(items, config.0, config.1, config.2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_fuse_3way_basic() {
        use crate::traits::Constructible;

        let hash = |item: &u64, seed: u64| -> usize { (*item ^ seed) as usize };
        let fp_func = |item: &u64| -> u8 { (item.wrapping_mul(0x5bd1e9955bd1e995) & 0xFF) as u8 };
        let items: Vec<u64> = vec![1, 2, 3];
        let config = BinaryFuseConfig::new(3, 8);

        let filter = BinaryFuseFilter::from_iter_with(items, (hash, fp_func, config));

        assert!(filter.contains(&1));
        assert!(filter.contains(&2));
        assert!(filter.contains(&3));
        assert!(!filter.contains(&4));
    }

    #[test]
    fn test_binary_fuse_4way_basic() {
        use crate::traits::Constructible;

        let hash = |item: &u64, seed: u64| -> usize { (*item ^ seed) as usize };
        let fp_func = |item: &u64| -> u8 { (item.wrapping_mul(0x5bd1e9955bd1e995) & 0xFF) as u8 };
        let items: Vec<u64> = vec![1, 2, 3];
        let config = BinaryFuseConfig::new(4, 8);

        let filter = BinaryFuseFilter::from_iter_with(items, (hash, fp_func, config));

        assert!(filter.contains(&1));
        assert!(filter.contains(&2));
        assert!(filter.contains(&3));
        assert!(!filter.contains(&4));
    }

    #[test]
    fn test_binary_fuse_empty() {
        use crate::traits::Constructible;

        let hash = |item: &u64, seed: u64| -> usize { (*item ^ seed) as usize };
        let fp_func = |item: &u64| -> u8 { (item.wrapping_mul(0x5bd1e9955bd1e995) & 0xFF) as u8 };
        let items: Vec<u64> = vec![];
        let config = BinaryFuseConfig::new(3, 8);

        let filter = BinaryFuseFilter::from_iter_with(items, (hash, fp_func, config));

        assert!(!filter.contains(&1));
    }

    #[test]
    fn test_binary_fuse_large_set() {
        use crate::traits::Constructible;

        // Use better hash functions for the large set test
        let hash = |item: &u64, seed: u64| -> usize {
            let mut h = item.wrapping_mul(0x5bd1e9955bd1e995);
            h = h.wrapping_add(seed);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
            h ^= h >> 33;
            h as usize
        };
        let fp_func = |item: &u64| -> u8 {
            let mut h = item.wrapping_mul(0x9e3779b97f4a7c15);
            h ^= h >> 16;
            h = h.wrapping_mul(0x85ebca6b);
            h ^= h >> 13;
            h = h.wrapping_mul(0xc2b2ae35);
            h ^= h >> 16;
            (h & 0xFF) as u8
        };
        let items: Vec<u64> = (0..1000).collect();
        let config = BinaryFuseConfig::new(4, 8);

        let filter = BinaryFuseFilter::from_iter_with(items, (hash, fp_func, config));

        // All inserted items should be found
        for i in 0..1000 {
            assert!(filter.contains(&i), "Should contain item {}", i);
        }

        // Items not inserted should mostly not be found (some false positives possible)
        let mut found_count = 0;
        for i in 1000..1100 {
            if filter.contains(&i) {
                found_count += 1;
            }
        }
        // With ~8 bits per element and 1000 items, false positive rate should be low
        // Allow some false positives but they should be reasonable
        assert!(
            found_count < 20,
            "Too many false positives: {}",
            found_count
        );
    }
}
