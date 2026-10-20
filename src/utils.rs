use num_traits::{One, Zero};
use std::ops::{BitAnd, Not, Shl, Sub};

/// Masks a fingerprint value to use only the specified number of effective bits.
///
/// # Arguments
/// * `effective_bits` - Number of bits to keep from the fingerprint
/// * `fp` - The fingerprint value to mask
///
/// # Returns
/// The fingerprint masked to `effective_bits`
///
/// # Panics
/// In debug mode, panics if `effective_bits` exceeds the total bits in `F`.
pub fn mask_fingerprint<F>(effective_bits: usize, fp: F) -> F
where
    F: Copy
        + Zero
        + One
        + Not<Output = F>
        + BitAnd<Output = F>
        + Shl<usize, Output = F>
        + Sub<Output = F>,
{
    let full_bits = std::mem::size_of::<F>() * 8;
    debug_assert!(
        effective_bits <= full_bits,
        "effective_bits ({}) exceeds fingerprint bit width ({})",
        effective_bits,
        full_bits
    );

    let mask = if effective_bits >= full_bits {
        !F::zero()
    } else {
        (F::one() << effective_bits) - F::one()
    };
    fp & mask
}

#[derive(Clone)]
pub struct SetQuery {
    count: usize,
    field0: usize,
}

impl Default for SetQuery {
    fn default() -> Self {
        Self::new()
    }
}

impl SetQuery {
    pub fn new() -> Self {
        SetQuery {
            count: 0,
            field0: 0,
        }
    }

    pub fn insert(&mut self, field0: usize) {
        self.count += 1;
        self.field0 ^= field0;
    }

    pub fn remove(&mut self, field0: usize) {
        debug_assert!(self.count > 0);
        self.count -= 1;
        self.field0 ^= field0;
    }

    pub fn query(&self) -> (usize,) {
        debug_assert!(self.count == 1);
        (self.field0,)
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Calculate the optimal number of bits for a Bloom-style filter.
///
/// `m = -n * ln(p) / (ln 2)^2`
pub fn optimal_num_bits(expected_items: usize, false_positive_rate: f64) -> usize {
    if expected_items == 0 || false_positive_rate <= 0.0 || false_positive_rate >= 1.0 {
        return 1;
    }
    let ln_2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
    let bits = (-(expected_items as f64) * false_positive_rate.ln() / ln_2_squared).ceil() as usize;
    bits.max(1)
}

/// Calculate the optimal number of hash functions for a Bloom-style filter.
///
/// `k = (m / n) * ln 2`
pub fn optimal_num_hashes(num_bits: usize, expected_items: usize) -> usize {
    if expected_items == 0 {
        return 1;
    }
    let hashes = (num_bits as f64 / expected_items as f64 * std::f64::consts::LN_2).ceil() as usize;
    hashes.max(1)
}

// Sample [0, size-1] from hash in a fast way
pub fn reduce(hash: u32, size: u32) -> u32 {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    ((hash as u64 * size as u64) >> 32) as u32
}

pub fn hash_with_seed(item: &u64, seed: u64) -> u64 {
    // murmur hash 3 finalizer
    let mut h = seed.wrapping_add(*item);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}
