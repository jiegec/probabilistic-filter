/// Base trait for all probabilistic filters.
pub trait ProbabilisticFilter<T> {
    /// Check if an item is likely in the filter.
    /// Returns false if the item is definitely not in the filter.
    /// Returns true if the item is probably in the filter (with some false positive rate).
    fn contains(&self, item: &T) -> bool;

    /// Get the actual space usage of the filter in bits.
    fn size_in_bits(&self) -> usize;

    /// Get the actual bits per key, assuming `num_keys` items were inserted.
    fn bits_per_key(&self, num_keys: usize) -> f64 {
        if num_keys == 0 {
            0.0
        } else {
            self.size_in_bits() as f64 / num_keys as f64
        }
    }
}

/// Trait for filters that support insertion.
pub trait Insertable<T>: ProbabilisticFilter<T> {
    /// Insert an item into the filter.
    /// Returns true if the item was inserted successfully.
    /// Returns false if insertion failed (e.g., filter is full).
    fn insert(&mut self, item: &T) -> bool;
}

/// Trait for filters that support deletion.
pub trait Deletable<T>: Insertable<T> {
    /// Delete an item from the filter.
    /// Returns true if the item was likely in the filter and was deleted.
    /// Returns false if the item was definitely not in the filter.
    /// Note: Deleting an item that was never inserted may cause other items to be falsely deleted.
    fn delete(&mut self, item: &T) -> bool;
}

/// Trait for filters that must be constructed at once from a set of items.
pub trait Constructible<T>: ProbabilisticFilter<T> {
    /// Configuration type for constructing the filter.
    type Config;

    /// Construct the filter from an iterator of items with configuration.
    /// This should consume the items and build the filter in one pass.
    fn from_iter_with<U>(items: U, config: Self::Config) -> Self
    where
        Self: Sized,
        U: IntoIterator<Item = T>;
}
