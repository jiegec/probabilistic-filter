use crate::{
    traits::{Insertable, ProbabilisticFilter},
    utils::{optimal_num_bits, optimal_num_hashes},
};
use fixedbitset::FixedBitSet;
use std::hash::Hash;

/// A One-Hashing Bloom Filter (OHBF).
///
/// Instead of computing `k` independent hash values, OHBF splits the bit vector
/// into `k` partitions and derives one index per partition from a **single**
/// hash value via `h(x) mod m_i`. The partition lengths `m_i` are consecutive
/// primes, which makes them pairwise coprime. By a number-theoretic argument
/// (see the reference), the resulting `k` indices are pairwise independent, so
/// OHBF achieves almost the same false positive rate as an ideal Bloom filter
/// while computing only one base hash per operation.
///
/// # Type Parameters
/// * `T` - The type of items stored in the filter
/// * `H` - A hash function that takes a reference to T and returns a 64-bit hash value
///
/// # References
/// * Jianyuan Lu, Tong Yang, Yi Wang, Huichen Dai, Linxiao Jin, Haoyu Song, and Bin Liu. 2015. One-Hashing Bloom Filter. In 2015 IEEE 23rd International Symposium on Quality of Service (IWQoS). 289–298. https://doi.org/10.1109/IWQoS.2015.7404739
///
/// # Example
/// ```
/// use probabilistic_filter::{onehashing::OneHashingBloomFilter, Insertable, ProbabilisticFilter};
///
/// let hash_fn = |item: &u64| item.wrapping_mul(0x5bd1e9955bd1e995);
/// let mut ohbf = OneHashingBloomFilter::new(1024, 3, hash_fn);
///
/// ohbf.insert(&1);
/// assert!(ohbf.contains(&1));
/// assert!(!ohbf.contains(&2));
/// // The actual size may differ slightly from the planned size.
/// assert!((ohbf.size_in_bits() as i64 - 1024).abs() < 100);
/// ```
pub struct OneHashingBloomFilter<T, H>
where
    H: Fn(&T) -> u64,
{
    bits: FixedBitSet,
    partition_lengths: Vec<usize>,
    /// Start offset of each partition, with one extra sentinel equal to the total length.
    offsets: Vec<usize>,
    hash_func: H,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, H> OneHashingBloomFilter<T, H>
where
    T: Hash,
    H: Fn(&T) -> u64,
{
    /// Create a new One-Hashing Bloom Filter.
    ///
    /// The bit vector is split into `num_hashes` partitions whose lengths are
    /// consecutive primes chosen so that their sum is as close as possible to
    /// `num_bits` (Algorithm 1 of the paper). The actual number of bits may
    /// therefore differ slightly from `num_bits`.
    ///
    /// # Arguments
    /// * `num_bits` - Planned number of bits in the filter
    /// * `num_hashes` - Number of hash values (i.e. number of partitions) per element
    /// * `hash_func` - The single base hash function
    ///
    /// # Panics
    /// Panics if `num_hashes` is zero.
    ///
    /// # Example
    /// ```
    /// use probabilistic_filter::onehashing::OneHashingBloomFilter;
    ///
    /// let hash_fn = |item: &u64| *item;
    /// let ohbf = OneHashingBloomFilter::new(10_000, 4, hash_fn);
    /// assert_eq!(ohbf.partition_lengths().len(), 4);
    /// ```
    pub fn new(num_bits: usize, num_hashes: usize, hash_func: H) -> Self {
        Self::with_partition_lengths(&partition_lengths(num_bits, num_hashes), hash_func)
    }

    /// Create a filter from explicit partition lengths.
    ///
    /// The lengths are expected to be pairwise coprime (consecutive primes in
    /// the paper) in order to preserve the independence of the derived indices.
    /// This constructor is mostly useful to reproduce a specific layout, e.g.
    /// the partition examples given in Table II of the paper.
    ///
    /// # Arguments
    /// * `partition_lengths` - Length of every partition, pairwise coprime
    /// * `hash_func` - The single base hash function
    ///
    /// # Panics
    /// Panics if `partition_lengths` is empty.
    pub fn with_partition_lengths(partition_lengths: &[usize], hash_func: H) -> Self {
        assert!(
            !partition_lengths.is_empty(),
            "a One-Hashing Bloom Filter needs at least one partition"
        );

        let mut offsets = Vec::with_capacity(partition_lengths.len() + 1);
        let mut total = 0;
        offsets.push(0);
        for &len in partition_lengths {
            assert!(len > 0, "partition lengths must be positive");
            total += len;
            offsets.push(total);
        }

        debug_assert!(are_pairwise_coprime(partition_lengths));

        OneHashingBloomFilter {
            bits: FixedBitSet::with_capacity(total),
            partition_lengths: partition_lengths.to_vec(),
            offsets,
            hash_func,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Calculate optimal number of bits for a given expected number of items and false positive rate.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of items to be inserted
    /// * `false_positive_rate` - Desired false positive rate (between 0 and 1)
    pub fn optimal_num_bits(expected_items: usize, false_positive_rate: f64) -> usize {
        optimal_num_bits(expected_items, false_positive_rate)
    }

    /// Calculate optimal number of hash functions for a given number of bits and expected items.
    ///
    /// # Arguments
    /// * `num_bits` - Number of bits in the filter
    /// * `expected_items` - Expected number of items to be inserted
    pub fn optimal_num_hashes(num_bits: usize, expected_items: usize) -> usize {
        optimal_num_hashes(num_bits, expected_items)
    }

    /// The length of every partition, in order.
    pub fn partition_lengths(&self) -> &[usize] {
        &self.partition_lengths
    }

    /// Estimate the false positive rate after `num_items` insertions.
    ///
    /// This is Equation 6 of the paper, which ignores the (practically
    /// negligible) machine-word collision in the hash stage:
    ///
    /// `f = prod_i (1 - (1 - 1/m_i)^n)`
    pub fn false_positive_rate(&self, num_items: usize) -> f64 {
        let n = num_items as f64;
        self.partition_lengths
            .iter()
            .map(|&m| 1.0 - (1.0 - 1.0 / m as f64).powf(n))
            .product()
    }
}

impl<T, H> ProbabilisticFilter<T> for OneHashingBloomFilter<T, H>
where
    T: Hash,
    H: Fn(&T) -> u64,
{
    fn contains(&self, item: &T) -> bool {
        let hash = (self.hash_func)(item);
        // One index per partition, all derived from the same base hash.
        for (&len, &offset) in self.partition_lengths.iter().zip(&self.offsets) {
            if !self.bits.contains(offset + (hash % len as u64) as usize) {
                return false;
            }
        }
        true
    }

    fn size_in_bits(&self) -> usize {
        self.bits.len()
    }
}

impl<T, H> Insertable<T> for OneHashingBloomFilter<T, H>
where
    T: Hash,
    H: Fn(&T) -> u64,
{
    fn insert(&mut self, item: &T) -> bool {
        let hash = (self.hash_func)(item);
        for (&len, &offset) in self.partition_lengths.iter().zip(&self.offsets) {
            self.bits.set(offset + (hash % len as u64) as usize, true);
        }
        true
    }
}

/// Determine partition lengths for a One-Hashing Bloom Filter.
///
/// Implements Algorithm 1 of the paper: `k = num_hashes` consecutive primes are
/// chosen such that their sum is as close as possible to `num_bits`. Because
/// the primes are consecutive, they are pairwise coprime, which is the
/// condition guaranteeing the independence of `h(x) mod m_i` (Theorem 1).
///
/// The returned lengths are ascending and sum to the *actual* filter length,
/// which can differ slightly from `num_bits`.
///
/// # Panics
/// Panics if `num_hashes` is zero.
///
/// # Example
/// ```
/// use probabilistic_filter::onehashing::partition_lengths;
///
/// // The k = 10 example from Table II of the paper.
/// let parts = partition_lengths(10_000, 10);
/// assert_eq!(
///     parts,
///     vec![971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031]
/// );
/// ```
pub fn partition_lengths(num_bits: usize, num_hashes: usize) -> Vec<usize> {
    assert!(num_hashes > 0, "num_hashes must be positive");

    if num_hashes == 1 {
        // A single partition does not need to be prime.
        return vec![num_bits.max(1)];
    }

    // With k partitions the paper suggests primes around m/k + delta, delta < 300.
    let target = (num_bits / num_hashes).max(2) as u64;

    // Step 1: find the prime closest to the target average partition length.
    let base = nearest_prime(target);

    // Build the window of k consecutive primes ending at `base`.
    let mut window = consecutive_primes_ending_at(base, num_hashes);
    let mut sum: u64 = window.iter().sum();
    let mut best_window = window.clone();
    let planned = num_bits as u64;
    let mut best_diff = sum.abs_diff(planned);

    // Steps 8-16: slide the window upwards while the sum gets closer to the plan.
    // Since every window sum is strictly increasing, the difference is unimodal
    // and we can stop as soon as it stops improving.
    loop {
        let next = next_prime(window[num_hashes - 1] + 1);
        let first = window.remove(0);
        window.push(next);
        sum = sum + next - first;

        let diff = sum.abs_diff(planned);
        if diff >= best_diff {
            break;
        }
        best_diff = diff;
        best_window = window.clone();
    }

    best_window.into_iter().map(|p| p as usize).collect()
}

/// Check whether all the given integers are pairwise coprime.
fn are_pairwise_coprime(values: &[usize]) -> bool {
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            if gcd(values[i], values[j]) != 1 {
                return false;
            }
        }
    }
    true
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Deterministic Miller-Rabin primality test, valid for all `u64` inputs.
fn is_prime(n: u64) -> bool {
    const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    if n < 2 {
        return false;
    }
    for p in SMALL_PRIMES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }

    // n - 1 = d * 2^s with d odd.
    let mut d = n - 1;
    let mut s = 0;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }

    // These bases are sufficient to make the test deterministic for n < 2^64.
    'witness: for a in SMALL_PRIMES {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..s - 1 {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

#[inline]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, m);
        }
        base = mul_mod(base, base, m);
        exp >>= 1;
    }
    result
}

/// Smallest prime greater than or equal to `n`.
fn next_prime(n: u64) -> u64 {
    let mut candidate = n.max(2);
    if candidate == 2 {
        return 2;
    }
    if candidate.is_multiple_of(2) {
        candidate += 1;
    }
    while !is_prime(candidate) {
        candidate += 2;
    }
    candidate
}

/// Largest prime less than or equal to `n` (or 2 when `n < 2`).
fn prev_prime(n: u64) -> u64 {
    if n <= 2 {
        return 2;
    }
    let mut candidate = n;
    if candidate.is_multiple_of(2) {
        candidate -= 1;
    }
    while !is_prime(candidate) {
        candidate -= 2;
    }
    candidate
}

/// The prime closest to `n` (ties are broken downwards).
fn nearest_prime(n: u64) -> u64 {
    if n <= 2 {
        return 2;
    }
    let lo = prev_prime(n);
    let hi = next_prime(n);
    if n - lo <= hi - n { lo } else { hi }
}

/// The `k` consecutive primes ending at `base`, in ascending order.
fn consecutive_primes_ending_at(base: u64, k: usize) -> Vec<u64> {
    let mut primes = Vec::with_capacity(k);
    primes.push(base);
    let mut p = base;
    while primes.len() < k && p > 2 {
        p = prev_prime(p - 1);
        primes.push(p);
    }
    if primes.len() < k {
        // Not enough primes below `base`; fall back to the first `k` primes.
        primes = Vec::with_capacity(k);
        let mut p = 2;
        while primes.len() < k {
            primes.push(p);
            p = next_prime(p + 1);
        }
    } else {
        primes.reverse();
    }
    primes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        let primes = [2, 3, 5, 7, 11, 13, 97, 7919, 104_729];
        for p in primes {
            assert!(is_prime(p), "{p} should be prime");
        }
        let composites = [0, 1, 4, 9, 25, 91, 561, 7917, 104_730];
        for c in composites {
            assert!(!is_prime(c), "{c} should not be prime");
        }
    }

    #[test]
    fn test_partition_lengths_table_ii() {
        // Table II of the paper, k = 10.
        let expected: [(usize, &[usize]); 8] = [
            (
                10_000,
                &[971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031],
            ),
            (
                20_000,
                &[1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027],
            ),
            (
                40_000,
                &[3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027],
            ),
            (
                80_000,
                &[7949, 7951, 7963, 7993, 8009, 8011, 8017, 8039, 8053, 8059],
            ),
            (
                160_000,
                &[
                    15937, 15959, 15971, 15973, 15991, 16001, 16007, 16033, 16057, 16061,
                ],
            ),
            (
                320_000,
                &[
                    31957, 31963, 31973, 31981, 31991, 32003, 32009, 32027, 32029, 32051,
                ],
            ),
            (
                640_000,
                &[
                    63929, 63949, 63977, 63997, 64007, 64013, 64019, 64033, 64037, 64063,
                ],
            ),
            (
                1_280_000,
                &[
                    127_931, 127_951, 127_973, 127_979, 127_997, 128_021, 128_033, 128_047,
                    128_053, 128_099,
                ],
            ),
        ];

        for (planned, expected_parts) in expected {
            assert_eq!(partition_lengths(planned, 10), expected_parts);
            let actual: usize = expected_parts.iter().sum();
            // The deviation from the planned length stays tiny (Table II).
            assert!(actual.abs_diff(planned) < 1_000);
        }
    }

    #[test]
    fn test_partition_lengths_properties() {
        for planned in [16, 100, 1_000, 10_000, 1_000_000, 16_777_216] {
            for k in [1, 2, 3, 4, 8, 10, 16, 32] {
                let parts = partition_lengths(planned, k);
                assert_eq!(parts.len(), k);
                assert!(
                    parts.windows(2).all(|w| w[0] < w[1]),
                    "primes should be ascending: {parts:?}"
                );
                assert!(
                    are_pairwise_coprime(&parts),
                    "primes should be pairwise coprime: {parts:?}"
                );
                let total: usize = parts.iter().sum();
                if planned >= 100_000 {
                    // Deviation stays small relative to the filter size.
                    assert!(total.abs_diff(planned) as f64 / (planned as f64) < 0.01);
                }
            }
        }
    }

    #[test]
    fn test_one_hashing_structure() {
        let hash_fn = |item: &u64| -> u64 { item.wrapping_mul(0x5bd1e9955bd1e995) };
        let ohbf = OneHashingBloomFilter::new(1_000_000, 10, hash_fn);

        assert_eq!(ohbf.partition_lengths().len(), 10);
        assert_eq!(ohbf.offsets.len(), 11);
        assert_eq!(*ohbf.offsets.last().unwrap(), ohbf.size_in_bits());
        for i in 0..10 {
            assert_eq!(
                ohbf.offsets[i + 1] - ohbf.offsets[i],
                ohbf.partition_lengths()[i]
            );
        }
    }

    #[test]
    fn test_one_hashing_insert_and_contains() {
        let hash_fn = |item: &u64| -> u64 { item.wrapping_mul(0x5bd1e9955bd1e995) };
        let mut ohbf = OneHashingBloomFilter::new(1024, 3, hash_fn);

        assert!(!ohbf.contains(&1));
        ohbf.insert(&1);
        assert!(ohbf.contains(&1));
        assert!(!ohbf.contains(&2));
    }

    #[test]
    fn test_one_hashing_no_false_negatives() {
        let hash_fn = |item: &u64| -> u64 { item.wrapping_mul(0x5bd1e9955bd1e995) };
        let mut ohbf = OneHashingBloomFilter::new(100_000, 7, hash_fn);

        for i in 0..2_000 {
            ohbf.insert(&i);
        }
        for i in 0..2_000 {
            assert!(ohbf.contains(&i), "Should contain item {i}");
        }
    }

    #[test]
    fn test_with_partition_lengths() {
        let hash_fn = |item: &u64| -> u64 { *item };
        let ohbf = OneHashingBloomFilter::with_partition_lengths(&[11, 13, 15], hash_fn);
        assert_eq!(ohbf.partition_lengths(), &[11, 13, 15]);
        assert_eq!(ohbf.offsets, vec![0, 11, 24, 39]);
        assert_eq!(ohbf.size_in_bits(), 39);
    }

    #[test]
    fn test_false_positive_rate_close_to_standard_bloom_filter() {
        // Compare the OHBF estimate with the standard Bloom filter formula for
        // the same number of bits and hash functions (n = 1000).
        let num_items = 1_000;
        for k in [3usize, 10] {
            for planned in [10_000, 20_000, 40_000, 80_000, 160_000] {
                let ohbf = OneHashingBloomFilter::new(planned, k, |item: &u64| *item);
                let actual_bits = ohbf.size_in_bits() as f64;
                let standard =
                    (1.0 - (-(num_items as f64) * k as f64 / actual_bits).exp()).powi(k as i32);
                let relative_diff =
                    (ohbf.false_positive_rate(num_items) - standard).abs() / standard;
                assert!(
                    relative_diff < 0.01,
                    "k={k} m={planned}: OHBF {} vs SBF {}, relative diff {}",
                    ohbf.false_positive_rate(num_items),
                    standard,
                    relative_diff
                );
            }
        }
    }

    #[test]
    fn test_empirical_false_positive_rate() {
        let hash_fn = |item: &u64| -> u64 { item.wrapping_mul(0x5bd1e9955bd1e995) };
        let num_items = 1_000;
        let planned = 20_000;
        let k = 3;
        let mut ohbf = OneHashingBloomFilter::new(planned, k, hash_fn);
        for i in 0..num_items {
            ohbf.insert(&(i as u64));
        }

        let num_queries = 100_000;
        let false_positives = (0..num_queries)
            .filter(|i| ohbf.contains(&((num_items + i) as u64)))
            .count();
        let observed = false_positives as f64 / num_queries as f64;
        let expected = ohbf.false_positive_rate(num_items);
        assert!(
            observed < expected * 2.0 + 0.001,
            "observed {observed} vs expected {expected}"
        );
    }
}
