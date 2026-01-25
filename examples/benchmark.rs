use probabilistic_filter::bloom::BloomFilter;
use probabilistic_filter::cuckoo::CuckooFilter;
use probabilistic_filter::traits::{Constructible, Insertable, ProbabilisticFilter};
use probabilistic_filter::xor::XorFilter;
use rand::Rng;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

const NUM_KEYS: usize = 1_000_000;
const NUM_QUERIES: usize = 1_000_000;
const FRACTIONS: &[f64] = &[0.0, 0.25, 0.5, 0.75, 1.0];

#[derive(Debug, Clone)]
struct BenchmarkResult {
    bits_per_key: f64,
    filter_type: String,
    fp_rate: f64,
}

fn main() {
    println!("Benchmarking probabilistic filters...");
    println!("Keys in set: {} (64-bit integers)", NUM_KEYS);
    println!("Queries: {} (64-bit integers)", NUM_QUERIES);
    println!();

    let mut rng = rand::rng();
    let mut inserted_set: HashSet<u64> = HashSet::new();
    let inserted_keys: Vec<u64> = generate_unique_keys(&mut rng, NUM_KEYS);
    inserted_keys.iter().for_each(|&k| {
        inserted_set.insert(k);
    });

    println!("Generated {} unique keys\n", inserted_keys.len());

    let bits_per_keys = vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
    let mut all_results: Vec<BenchmarkResult> = Vec::new();

    for &bits_per_key in &bits_per_keys {
        println!("========================================");
        println!("Space: {:.1} bits per key", bits_per_key);
        println!("========================================");

        for &fraction in FRACTIONS {
            println!("\nFraction from original set: {:.0}%", fraction * 100.0);

            let query_keys = generate_query_keys(&inserted_keys, &inserted_set, fraction, &mut rng);

            // Benchmark Bloom filter
            let (fp_rate, actual_bits_per_key) =
                benchmark_bloom(&inserted_keys, &query_keys, &inserted_set, bits_per_key);
            if fraction == 0.0 {
                all_results.push(BenchmarkResult {
                    bits_per_key: actual_bits_per_key,
                    filter_type: "Bloom".to_string(),
                    fp_rate,
                });
            }

            // Benchmark Cuckoo filter
            let (fp_rate, actual_bits_per_key) = benchmark_cuckoo(
                &inserted_keys,
                &query_keys,
                &inserted_set,
                bits_per_key,
                &mut rng,
            );
            if fraction == 0.0 {
                all_results.push(BenchmarkResult {
                    bits_per_key: actual_bits_per_key,
                    filter_type: "Cuckoo".to_string(),
                    fp_rate,
                });
            }

            // Benchmark XOR filter
            let (fp_rate, actual_bits_per_key) =
                benchmark_xor(&inserted_keys, &query_keys, &inserted_set, bits_per_key);
            if fraction == 0.0 {
                all_results.push(BenchmarkResult {
                    bits_per_key: actual_bits_per_key,
                    filter_type: "XOR".to_string(),
                    fp_rate,
                });
            }
        }
    }

    println!("\nSaving results to benchmark_results.csv");
    save_results_to_csv(&all_results, "benchmark_results.csv");
    println!("Results saved to benchmark_results.csv");
    println!("Run: python3 plot_results.py");
}

fn generate_unique_keys(rng: &mut impl Rng, count: usize) -> Vec<u64> {
    let mut keys: Vec<u64> = Vec::with_capacity(count);
    let mut seen: HashSet<u64> = HashSet::new();

    while keys.len() < count {
        let key = rng.random::<u64>();
        if seen.insert(key) {
            keys.push(key);
        }
    }

    keys
}

fn generate_query_keys(
    inserted_keys: &[u64],
    inserted_set: &HashSet<u64>,
    fraction: f64,
    rng: &mut impl Rng,
) -> Vec<u64> {
    let num_from_inserted = (NUM_QUERIES as f64 * fraction) as usize;
    let mut query_keys: Vec<u64> = Vec::with_capacity(NUM_QUERIES);

    for _ in 0..num_from_inserted {
        query_keys.push(inserted_keys[rng.random_range(0..inserted_keys.len())]);
    }

    let mut not_inserted_seen: HashSet<u64> = HashSet::new();
    while query_keys.len() < NUM_QUERIES {
        let key = rng.random::<u64>();
        if !inserted_set.contains(&key) && not_inserted_seen.insert(key) {
            query_keys.push(key);
        }
    }

    for i in (1..query_keys.len()).rev() {
        let j = rng.random_range(0..=i);
        query_keys.swap(i, j);
    }

    query_keys
}

fn benchmark_bloom(
    inserted_keys: &[u64],
    query_keys: &[u64],
    inserted_set: &HashSet<u64>,
    bits_per_key: f64,
) -> (f64, f64) {
    let num_bits = (NUM_KEYS as f64 * bits_per_key).ceil() as usize;

    // Calculate optimal number of hash functions
    let num_hashes = BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_hashes(num_bits, NUM_KEYS);

    // Generate seeds for hash functions
    let mut rng = rand::rng();
    let seeds: Vec<u64> = (0..num_hashes).map(|_| rng.random()).collect();

    // Create hash function that takes a seed
    fn hash_with_seed(item: &u64, seed: u64) -> u64 {
        let mut h = seed.wrapping_add(*item);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    // Create hash functions using match arms - each closure captures its own seed
    let mut hash_funcs: Vec<Box<dyn Fn(&u64) -> u64>> = Vec::new();
    for i in 0..num_hashes {
        let seed = seeds[i];
        hash_funcs.push(Box::new(move |item: &u64| hash_with_seed(item, seed)));
    }

    let start = Instant::now();
    let mut bloom = BloomFilter::new(num_bits, hash_funcs);
    for key in inserted_keys {
        bloom.insert(key);
    }
    let construct_time = start.elapsed();

    // Validate that all inserted keys are positive (contains returns true)
    assert!(
        inserted_keys.iter().all(|&k| bloom.contains(&k)),
        "Bloom filter: all inserted keys should return true for contains()"
    );

    let start = Instant::now();
    let mut positive_count = 0usize;
    for key in query_keys {
        if bloom.contains(key) {
            positive_count += 1;
        }
    }
    let query_time = start.elapsed();

    let mut true_negatives = 0usize;
    let mut false_positives = 0usize;
    for key in query_keys {
        let is_actually_in = inserted_set.contains(key);
        let is_reported_in = bloom.contains(key);
        if !is_actually_in {
            true_negatives += 1;
            if is_reported_in {
                false_positives += 1;
            }
        }
    }
    let fp_rate = if true_negatives > 0 {
        false_positives as f64 / true_negatives as f64
    } else {
        0.0
    };

    let actual_bits_per_key = bloom.bits_per_key(NUM_KEYS);

    let query_throughput = NUM_QUERIES as f64 / query_time.as_secs_f64();

    println!("  Bloom Filter:");
    println!(
        "    Construction: {:?} ({:.3}M ops/s)",
        construct_time,
        (NUM_KEYS as f64 / construct_time.as_secs_f64()) / 1e6
    );
    println!(
        "    Query: {:?} ({:.3}M ops/s)",
        query_time,
        query_throughput / 1e6
    );
    println!("    Positive queries: {} / {}", positive_count, NUM_QUERIES);
    println!(
        "    False positives: {} / {} ({:.6}%)",
        false_positives,
        true_negatives,
        fp_rate * 100.0
    );
    println!(
        "    Space: {:.1} MB ({:.1} bits/key)",
        (bloom.size_in_bits() as f64 / 8.0) / (1024.0 * 1024.0),
        actual_bits_per_key
    );
    println!("    Parameters: {} hash functions", num_hashes);

    (fp_rate, actual_bits_per_key)
}

fn benchmark_cuckoo(
    inserted_keys: &[u64],
    query_keys: &[u64],
    inserted_set: &HashSet<u64>,
    bits_per_key: f64,
    rng: &mut impl Rng,
) -> (f64, f64) {
    let items_per_bucket = 4;
    // load factor of 0.955
    let load_factor = 0.955;
    let num_buckets = ((NUM_KEYS as f64 / load_factor / items_per_bucket as f64).ceil() as usize)
        .next_power_of_two();
    let fingerprint_bits =
        (NUM_KEYS as f64 * bits_per_key / num_buckets as f64 / items_per_bucket as f64).ceil()
            as usize;

    // Create hash function that takes a seed using MurmurHash3 finalizer
    fn hash_with_seed(item: &u64, seed: u64) -> u64 {
        let mut h = seed.wrapping_add(*item);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    let seed: u64 = rng.random();
    let hash = move |item: &u64| -> u64 { hash_with_seed(item, seed) };

    let fp_seed: u64 = rng.random();
    let fingerprint_func = move |item: &u64| -> u64 {
        let h = hash_with_seed(item, fp_seed);
        h | 1
    };

    let alt_index = move |index: usize, fp: u64| -> usize {
        let hashed_fp = hash_with_seed(&fp, fp_seed);
        index ^ hashed_fp as usize
    };

    let (cuckoo, construct_time) = loop {
        let start = Instant::now();
        let mut cuckoo = CuckooFilter::new(
            num_buckets,
            items_per_bucket,
            500,
            hash,
            fingerprint_func,
            alt_index,
            fingerprint_bits as usize,
        );
        let mut insert_failures = 0usize;
        for key in inserted_keys {
            if !cuckoo.insert(key) {
                insert_failures += 1;
                break;
            }
        }
        let construct_time = start.elapsed();

        if insert_failures > 0 {
            // retry
            continue;
        }

        // Validate that all inserted keys are positive (contains returns true)
        assert!(
            inserted_keys.iter().all(|&k| cuckoo.contains(&k)),
            "Cuckoo filter: all inserted keys should return true for contains()"
        );
        break (cuckoo, construct_time);
    };

    let start = Instant::now();
    let mut positive_count = 0usize;
    for key in query_keys {
        if cuckoo.contains(key) {
            positive_count += 1;
        }
    }
    let query_time = start.elapsed();

    let mut true_negatives = 0usize;
    let mut false_positives = 0usize;
    for key in query_keys {
        let is_actually_in = inserted_set.contains(key);
        let is_reported_in = cuckoo.contains(key);
        if !is_actually_in {
            true_negatives += 1;
            if is_reported_in {
                false_positives += 1;
            }
        }
    }
    let fp_rate = if true_negatives > 0 {
        false_positives as f64 / true_negatives as f64
    } else {
        0.0
    };

    let actual_bits_per_key = cuckoo.bits_per_key(NUM_KEYS);

    let query_throughput = NUM_QUERIES as f64 / query_time.as_secs_f64();

    println!("  Cuckoo Filter:");
    println!(
        "    Construction: {:?} ({:.3}M ops/s)",
        construct_time,
        (NUM_KEYS as f64 / construct_time.as_secs_f64()) / 1e6
    );
    println!(
        "    Query: {:?} ({:.3}M ops/s)",
        query_time,
        query_throughput / 1e6
    );
    println!("    Positive queries: {} / {}", positive_count, NUM_QUERIES);
    println!(
        "    False positives: {} / {} ({:.6}%)",
        false_positives,
        true_negatives,
        fp_rate * 100.0
    );
    println!(
        "    Space: {:.1} MB ({:.1} bits/key)",
        (cuckoo.size_in_bits() as f64 / 8.0) / (1024.0 * 1024.0),
        actual_bits_per_key
    );
    println!(
        "    Parameters: {} buckets, {} items per bucket, {} bit fingerprint",
        num_buckets, items_per_bucket, fingerprint_bits
    );

    (fp_rate, actual_bits_per_key)
}

fn benchmark_xor(
    inserted_keys: &[u64],
    query_keys: &[u64],
    inserted_set: &HashSet<u64>,
    bits_per_key: f64,
) -> (f64, f64) {
    // Calculate effective_bits from target bits per key
    // Effective bits determines fingerprint size used for space calculation
    let effective_bits = (bits_per_key / 1.23).ceil() as usize;

    let hash = move |item: &u64, seed: u64| -> usize { (*item ^ seed) as usize };

    let fp_mask = (1u64 << effective_bits) - 1;
    let fingerprint_func =
        move |item: &u64| -> u64 { (item.wrapping_mul(0x9e3779b97f4a7c15) & fp_mask) | 1 };

    let start = Instant::now();
    let xor = XorFilter::from_iter_with(
        inserted_keys.iter().copied(),
        (hash, fingerprint_func, effective_bits),
    );
    let construct_time = start.elapsed();

    // Validate that all inserted keys are positive (contains returns true)
    assert!(
        inserted_keys.iter().all(|&k| xor.contains(&k)),
        "XOR filter: all inserted keys should return true for contains()"
    );

    let start = Instant::now();
    let mut positive_count = 0usize;
    for key in query_keys {
        if xor.contains(key) {
            positive_count += 1;
        }
    }
    let query_time = start.elapsed();

    let mut true_negatives = 0usize;
    let mut false_positives = 0usize;
    for key in query_keys {
        let is_actually_in = inserted_set.contains(key);
        let is_reported_in = xor.contains(key);
        if !is_actually_in {
            true_negatives += 1;
            if is_reported_in {
                false_positives += 1;
            }
        }
    }
    let fp_rate = if true_negatives > 0 {
        false_positives as f64 / true_negatives as f64
    } else {
        0.0
    };

    let actual_bits_per_key = xor.bits_per_key(NUM_KEYS);

    let query_throughput = NUM_QUERIES as f64 / query_time.as_secs_f64();

    println!("  XOR Filter:");
    println!(
        "    Construction: {:?} ({:.3}M ops/s)",
        construct_time,
        (NUM_KEYS as f64 / construct_time.as_secs_f64()) / 1e6
    );
    println!(
        "    Query: {:?} ({:.3}M ops/s)",
        query_time,
        query_throughput / 1e6
    );
    println!("    Positive queries: {} / {}", positive_count, NUM_QUERIES);
    println!(
        "    False positives: {} / {} ({:.6}%)",
        false_positives,
        true_negatives,
        fp_rate * 100.0
    );
    println!(
        "    Space: {:.1} MB ({:.1} bits/key)",
        (xor.size_in_bits() as f64 / 8.0) / (1024.0 * 1024.0),
        actual_bits_per_key
    );

    (fp_rate, actual_bits_per_key)
}

fn save_results_to_csv(results: &[BenchmarkResult], output: &str) {
    let mut file = File::create(output).expect("Failed to create CSV file");

    // Write header
    writeln!(file, "filter_type,bits_per_key,fp_rate").expect("Failed to write header");

    // Write data rows
    for result in results {
        writeln!(
            file,
            "{},{:.6},{:.10e}",
            result.filter_type, result.bits_per_key, result.fp_rate
        )
        .expect("Failed to write row");
    }
}
