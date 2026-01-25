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
