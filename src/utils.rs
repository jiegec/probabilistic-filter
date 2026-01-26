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

// Sample [0, size-1] from hash in a fast way
pub fn reduce(hash: u32, size: u32) -> u32 {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    ((hash as u64 * size as u64) >> 32) as u32
}
