# Probabilistic Filters

A simpmle Rust library implementing various probabilistic data structures for membership testing with tunable false positive rates.

## Features

- **Bloom Filter**: Classic space-efficient probabilistic data structure
- **One-Hashing Bloom Filter**: Bloom filter using a single base hash function plus `k` modulo operations (Lu et al., IWQoS 2015)
- **Cuckoo Filter**: Similar to Bloom but supports deletions
- **XOR Filter**: Fast construction, smaller space overhead than Bloom
- **Binary Fuse Filter**: Improved version of XOR filter with even faster construction

## License

MIT
