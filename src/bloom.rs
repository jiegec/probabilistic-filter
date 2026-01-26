use crate::{
    traits::{Insertable, ProbabilisticFilter},
    utils::reduce,
};
use fixedbitset::FixedBitSet;
use std::hash::Hash;

/// A Bloom filter is a space-efficient probabilistic data structure that supports insertion and membership testing.
/// It may produce false positives but never false negatives.
///
/// # Type Parameters
/// * `T` - The type of items stored in the filter
/// * `H` - A hash function that takes a reference to T and returns a hash value
///
/// # References
/// * Burton H. Bloom. 1970. Space/time trade-offs in hash coding with allowable errors. Commun. ACM 13, 7 (July 1970), 422–426. https://doi.org/10.1145/362686.362692
pub struct BloomFilter<T, H>
where
    H: Fn(&T) -> u64,
{
    bits: FixedBitSet,
    hash_funcs: Vec<H>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, H> BloomFilter<T, H>
where
    T: Hash,
    H: Fn(&T) -> u64,
{
    /// Create a new Bloom filter with the given number of bits and hash functions.
    ///
    /// # Arguments
    /// * `num_bits` - The number of bits in the filter
    /// * `hash_funcs` - A vector of hash functions
    ///
    /// # Example
    /// ```
    /// use probabilistic_filter::{bloom::BloomFilter, Insertable, ProbabilisticFilter};
    ///
    /// let hash_fn1 = |item: &u64| item.wrapping_mul(0x5bd1e9955bd1e995);
    /// let hash_fn2 = |item: &u64| item.wrapping_mul(0x9e3779b97f4a7c1);
    ///
    /// let mut bloom = BloomFilter::new(1024, vec![hash_fn1, hash_fn2]);
    /// bloom.insert(&1);
    /// assert!(bloom.contains(&1));
    /// ```
    pub fn new<IT>(num_bits: usize, hash_funcs: IT) -> Self
    where
        IT: IntoIterator<Item = H>,
    {
        BloomFilter {
            bits: FixedBitSet::with_capacity(num_bits),
            hash_funcs: hash_funcs.into_iter().collect(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Calculate optimal number of bits for a given expected number of items and false positive rate.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of items to be inserted
    /// * `false_positive_rate` - Desired false positive rate (between 0 and 1)
    pub fn optimal_num_bits(expected_items: usize, false_positive_rate: f64) -> usize {
        if expected_items == 0 || false_positive_rate <= 0.0 || false_positive_rate >= 1.0 {
            return 1;
        }
        let ln_2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let bits =
            (-(expected_items as f64) * false_positive_rate.ln() / ln_2_squared).ceil() as usize;
        bits.max(1)
    }

    /// Calculate optimal number of hash functions for a given number of bits and expected items.
    ///
    /// # Arguments
    /// * `num_bits` - Number of bits in the filter
    /// * `expected_items` - Expected number of items to be inserted
    pub fn optimal_num_hashes(num_bits: usize, expected_items: usize) -> usize {
        if expected_items == 0 {
            return 1;
        }
        let hashes =
            (num_bits as f64 / expected_items as f64 * std::f64::consts::LN_2).ceil() as usize;
        hashes.max(1)
    }
}

impl<T, H> ProbabilisticFilter<T> for BloomFilter<T, H>
where
    T: Hash,
    H: Fn(&T) -> u64,
{
    fn contains(&self, item: &T) -> bool {
        for hash_func in &self.hash_funcs {
            let hash = hash_func(item);
            let idx = reduce(hash as u32, self.bits.len() as u32);
            if !self.bits.contains(idx as usize) {
                return false;
            }
        }
        true
    }

    fn size_in_bits(&self) -> usize {
        self.bits.len()
    }
}

impl<T, H> Insertable<T> for BloomFilter<T, H>
where
    T: Hash,
    H: Fn(&T) -> u64,
{
    fn insert(&mut self, item: &T) -> bool {
        for hash_func in &self.hash_funcs {
            let hash = hash_func(item);
            let idx = reduce(hash as u32, self.bits.len() as u32);
            self.bits.set(idx as usize, true);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_insert_and_contains() {
        // Use a simple hash function
        let hash_fn1 = |item: &u64| -> u64 { item.wrapping_mul(0x5bd1e9955bd1e995) };
        let hash_fn2 = |item: &u64| -> u64 { item.wrapping_mul(0x9e3779b97f4a7c1) };
        let mut bloom = BloomFilter::new(1024, vec![hash_fn1, hash_fn2]);

        assert!(!bloom.contains(&1));
        bloom.insert(&1);
        assert!(bloom.contains(&1));
        assert!(!bloom.contains(&2));
    }

    #[test]
    fn test_bloom_no_false_negatives() {
        let hash_fn1 = |item: &u64| -> u64 { item.wrapping_mul(0x5bd1e9955bd1e995) };
        let hash_fn2 = |item: &u64| -> u64 { item.wrapping_mul(0x9e3779b97f4a7c1) };
        let mut bloom = BloomFilter::new(1024, vec![hash_fn1, hash_fn2]);

        for i in 0..100 {
            bloom.insert(&i);
        }

        for i in 0..100 {
            assert!(bloom.contains(&i), "Should contain item {}", i);
        }
    }

    #[test]
    fn test_bloom_optimal_parameters() {
        let bits = BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_bits(1000, 0.01);
        assert!(bits > 0);

        let hashes = BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_hashes(bits, 1000);
        assert!(hashes > 0);
    }

    #[test]
    fn test_bloom_false_positive_rate() {
        let hash_fn1 = |item: &u64| -> u64 { item.wrapping_mul(0x5bd1e9955bd1e995) };
        let hash_fn2 = |item: &u64| -> u64 { item.wrapping_mul(0x9e3779b97f4a7c1) };
        let false_positive_rate = 0.1;
        let num_items = 1000;

        let bits =
            BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_bits(num_items, false_positive_rate);
        let _hashes = BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_hashes(bits, num_items);

        let mut bloom = BloomFilter::new(bits, vec![hash_fn1, hash_fn2]);

        // Insert 1000 items
        for i in 0..num_items {
            bloom.insert(&(i as u64));
        }

        // Check false positive rate for items not in the filter
        let mut false_positives = 0;
        for i in num_items..(num_items + 1000) {
            if bloom.contains(&(i as u64)) {
                false_positives += 1;
            }
        }

        let actual_rate = false_positives as f64 / 1000.0;
        // Allow some margin of error
        assert!(
            actual_rate < false_positive_rate * 2.0,
            "Actual rate {} exceeds expected rate {}",
            actual_rate,
            false_positive_rate
        );
    }
}
