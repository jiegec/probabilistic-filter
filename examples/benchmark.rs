use probabilistic_filter::binaryfuse::{BinaryFuseConfig, BinaryFuseFilter3, BinaryFuseFilter4};
use probabilistic_filter::bloom::BloomFilter;
use probabilistic_filter::cuckoo::CuckooFilter;
use probabilistic_filter::traits::{Constructible, Insertable, ProbabilisticFilter};
use probabilistic_filter::xor::XorFilter;
use rand::Rng;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

const NUM_KEYS: usize = 10_000_000;
const NUM_QUERIES: usize = 10_000_000;
const FRACTIONS: &[f64] = &[0.0, 0.25, 0.5, 0.75, 1.0];

#[derive(Debug, Clone)]
struct BenchmarkResult {
    bits_per_key: f64,
    filter_type: String,
    fp_rate: f64,
}

struct FilterBenchmark<'a> {
    name: &'a str,
    construct_time: Duration,
    actual_bits_per_key: f64,
    size_in_bits: usize,
    parameters: String,
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

        // Build all filters once
        // Build Bloom filter
        let bloom_num_bits = (NUM_KEYS as f64 * bits_per_key).ceil() as usize;
        let bloom_num_hashes =
            BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_hashes(bloom_num_bits, NUM_KEYS);
        let bloom_seeds: Vec<u64> = (0..bloom_num_hashes).map(|_| rng.random()).collect();
        let mut bloom_hash_funcs: Vec<Box<dyn Fn(&u64) -> u64>> = Vec::new();
        for i in 0..bloom_num_hashes {
            let seed = bloom_seeds[i];
            bloom_hash_funcs.push(Box::new(move |item| hash_with_seed(item, seed)));
        }
        let bloom_start = Instant::now();
        let mut bloom = BloomFilter::new(bloom_num_bits, bloom_hash_funcs);
        for key in &inserted_keys {
            bloom.insert(key);
        }
        let bloom_construct_time = bloom_start.elapsed();
        assert!(
            inserted_keys.iter().all(|&k| bloom.contains(&k)),
            "Bloom filter: all inserted keys should return true for contains()"
        );
        let bloom_actual_bits_per_key = bloom.bits_per_key(NUM_KEYS);
        let bloom_size_in_bits = bloom.size_in_bits();
        let bloom_bench = FilterBenchmark {
            name: "Bloom",
            construct_time: bloom_construct_time,
            actual_bits_per_key: bloom_actual_bits_per_key,
            size_in_bits: bloom_size_in_bits,
            parameters: format!("{} hash functions", bloom_num_hashes),
        };
        print_construction_results(&bloom_bench);

        // Build Cuckoo filter
        let cuckoo_items_per_bucket = 4;
        let cuckoo_load_factor = 0.955;
        let cuckoo_num_buckets =
            ((NUM_KEYS as f64 / cuckoo_load_factor / cuckoo_items_per_bucket as f64).ceil()
                as usize)
                .next_power_of_two();
        let cuckoo_actual_load_factor =
            NUM_KEYS as f64 / cuckoo_num_buckets as f64 / cuckoo_items_per_bucket as f64;
        let cuckoo_fingerprint_bits = (NUM_KEYS as f64 * bits_per_key
            / cuckoo_num_buckets as f64
            / cuckoo_items_per_bucket as f64)
            .ceil() as usize;
        let cuckoo_seed: u64 = rng.random();
        let cuckoo_fp_seed: u64 = rng.random();
        let cuckoo_hash = |item: &u64| -> u64 { hash_with_seed(item, cuckoo_seed) };
        let cuckoo_fingerprint_func = |item: &u64| -> u64 { hash_with_seed(item, cuckoo_fp_seed) };
        let cuckoo_alt_index = |index: usize, fp: u64| -> usize {
            let hashed_fp = hash_with_seed(&fp, cuckoo_fp_seed);
            index ^ hashed_fp as usize
        };
        let (cuckoo, cuckoo_construct_time) = loop {
            let start = Instant::now();
            let mut cuckoo = CuckooFilter::new(
                cuckoo_num_buckets,
                cuckoo_items_per_bucket,
                500,
                cuckoo_hash,
                cuckoo_fingerprint_func,
                cuckoo_alt_index,
                cuckoo_fingerprint_bits as usize,
            );
            let insert_failures = inserted_keys.iter().any(|&key| !cuckoo.insert(&key));
            let construct_time = start.elapsed();

            if insert_failures {
                continue;
            }

            assert!(
                inserted_keys.iter().all(|&k| cuckoo.contains(&k)),
                "Cuckoo filter: all inserted keys should return true for contains()"
            );
            break (cuckoo, construct_time);
        };
        let cuckoo_actual_bits_per_key = cuckoo.bits_per_key(NUM_KEYS);
        let cuckoo_size_in_bits = cuckoo.size_in_bits();
        let cuckoo_bench = FilterBenchmark {
            name: "Cuckoo",
            construct_time: cuckoo_construct_time,
            actual_bits_per_key: cuckoo_actual_bits_per_key,
            size_in_bits: cuckoo_size_in_bits,
            parameters: format!(
                "{} buckets, {} items per bucket, {} bit fingerprint, {:.3} load factor",
                cuckoo_num_buckets,
                cuckoo_items_per_bucket,
                cuckoo_fingerprint_bits,
                cuckoo_actual_load_factor
            ),
        };
        print_construction_results(&cuckoo_bench);

        // Build XOR filter
        let xor_effective_bits = (bits_per_key / 1.23).ceil() as usize;
        let xor_hash = |item: &u64, seed: u64| -> usize { hash_with_seed(item, seed) as usize };
        let xor_fp_seed: u64 = rng.random();
        let xor_fingerprint_func = |item: &u64| -> u64 { hash_with_seed(item, xor_fp_seed) };
        let xor_start = Instant::now();
        let xor = XorFilter::from_iter_with(
            inserted_keys.iter().copied(),
            (xor_hash, xor_fingerprint_func, xor_effective_bits),
        );
        let xor_construct_time = xor_start.elapsed();
        assert!(
            inserted_keys.iter().all(|&k| xor.contains(&k)),
            "XOR filter: all inserted keys should return true for contains()"
        );
        let xor_actual_bits_per_key = xor.bits_per_key(NUM_KEYS);
        let xor_size_in_bits = xor.size_in_bits();
        let xor_bench = FilterBenchmark {
            name: "XOR",
            construct_time: xor_construct_time,
            actual_bits_per_key: xor_actual_bits_per_key,
            size_in_bits: xor_size_in_bits,
            parameters: format!(
                "{} bit fingerprint, {} fingerprints",
                xor_effective_bits,
                xor.fingerprint_count()
            ),
        };
        print_construction_results(&xor_bench);

        // Build BinaryFuse 3-way filter
        let bf3_effective_bits = (bits_per_key / 1.125).ceil() as usize;
        let bf3_fp_seed: u64 = rng.random();
        let bf3_fingerprint_func = |item: &u64| -> u64 { hash_with_seed(item, bf3_fp_seed) };
        let bf3_hash = |item: &u64, seed: u64| -> usize { hash_with_seed(item, seed) as usize };
        let bf3_config = BinaryFuseConfig::new(bf3_hash, bf3_fingerprint_func, bf3_effective_bits);
        let bf3_start = Instant::now();
        let bf3 = BinaryFuseFilter3::from_iter_with(inserted_keys.iter().copied(), bf3_config);
        let bf3_construct_time = bf3_start.elapsed();
        assert!(
            inserted_keys.iter().all(|&k| bf3.contains(&k)),
            "BinaryFuse filter: all inserted keys should return true for contains()"
        );
        let bf3_actual_bits_per_key = bf3.bits_per_key(NUM_KEYS);
        let bf3_size_in_bits = bf3.size_in_bits();
        let bf3_bench = FilterBenchmark {
            name: "BinaryFuse3",
            construct_time: bf3_construct_time,
            actual_bits_per_key: bf3_actual_bits_per_key,
            size_in_bits: bf3_size_in_bits,
            parameters: format!(
                "3 way, {} bit fingerprint, {} segments × {} size",
                bf3_effective_bits,
                bf3.total_segments(),
                bf3.segment_size()
            ),
        };
        print_construction_results(&bf3_bench);

        // Build BinaryFuse 4-way filter
        let bf4_effective_bits = (bits_per_key / 1.075).ceil() as usize;
        let bf4_fp_seed: u64 = rng.random();
        let bf4_fingerprint_func = |item: &u64| -> u64 { hash_with_seed(item, bf4_fp_seed) };
        let bf4_hash = |item: &u64, seed: u64| -> usize { hash_with_seed(item, seed) as usize };
        let bf4_config = BinaryFuseConfig::new(bf4_hash, bf4_fingerprint_func, bf4_effective_bits);
        let bf4_start = Instant::now();
        let bf4 = BinaryFuseFilter4::from_iter_with(inserted_keys.iter().copied(), bf4_config);
        let bf4_construct_time = bf4_start.elapsed();
        assert!(
            inserted_keys.iter().all(|&k| bf4.contains(&k)),
            "BinaryFuse filter: all inserted keys should return true for contains()"
        );
        let bf4_actual_bits_per_key = bf4.bits_per_key(NUM_KEYS);
        let bf4_size_in_bits = bf4.size_in_bits();
        let bf4_bench = FilterBenchmark {
            name: "BinaryFuse4",
            construct_time: bf4_construct_time,
            actual_bits_per_key: bf4_actual_bits_per_key,
            size_in_bits: bf4_size_in_bits,
            parameters: format!(
                "4 way, {} bit fingerprint, {} segments × {} size",
                bf4_effective_bits,
                bf4.total_segments(),
                bf4.segment_size()
            ),
        };
        print_construction_results(&bf4_bench);

        // Run queries for each fraction against the same filters
        for &fraction in FRACTIONS {
            println!("\nFraction from original set: {:.0}%", fraction * 100.0);

            let query_keys = generate_query_keys(&inserted_keys, &inserted_set, fraction, &mut rng);

            // Query each filter
            let bloom_fp = query_filter(&bloom, &query_keys, &inserted_set, &bloom_bench);
            let cuckoo_fp = query_filter(&cuckoo, &query_keys, &inserted_set, &cuckoo_bench);
            let xor_fp = query_filter(&xor, &query_keys, &inserted_set, &xor_bench);
            let bf3_fp = query_filter(&bf3, &query_keys, &inserted_set, &bf3_bench);
            let bf4_fp = query_filter(&bf4, &query_keys, &inserted_set, &bf4_bench);

            if fraction == 0.0 {
                all_results.push(BenchmarkResult {
                    bits_per_key: bloom_bench.actual_bits_per_key,
                    filter_type: "Bloom".to_string(),
                    fp_rate: bloom_fp,
                });
                all_results.push(BenchmarkResult {
                    bits_per_key: cuckoo_bench.actual_bits_per_key,
                    filter_type: "Cuckoo".to_string(),
                    fp_rate: cuckoo_fp,
                });
                all_results.push(BenchmarkResult {
                    bits_per_key: xor_bench.actual_bits_per_key,
                    filter_type: "XOR".to_string(),
                    fp_rate: xor_fp,
                });
                all_results.push(BenchmarkResult {
                    bits_per_key: bf3_bench.actual_bits_per_key,
                    filter_type: "BinaryFuse3".to_string(),
                    fp_rate: bf3_fp,
                });
                all_results.push(BenchmarkResult {
                    bits_per_key: bf4_bench.actual_bits_per_key,
                    filter_type: "BinaryFuse4".to_string(),
                    fp_rate: bf4_fp,
                });
            }
        }
    }

    println!("\nSaving results to benchmark_results.csv");
    save_results_to_csv(&all_results, "benchmark_results.csv");
    println!("Results saved to benchmark_results.csv");
    println!("Run: python3 plot_results.py");
}

fn query_filter<F>(
    filter: &F,
    query_keys: &[u64],
    inserted_set: &HashSet<u64>,
    benchmark: &FilterBenchmark,
) -> f64
where
    F: ProbabilisticFilter<u64>,
{
    let (query_time, positive_count) = run_queries(filter, query_keys);
    let (fp_rate, true_negatives, false_positives) =
        calculate_fp_rate(filter, query_keys, inserted_set);

    print_query_results(
        benchmark.name,
        query_time,
        positive_count,
        false_positives,
        true_negatives,
        fp_rate,
    );

    fp_rate
}

fn print_construction_results(benchmark: &FilterBenchmark) {
    let construct_throughput = (NUM_KEYS as f64 / benchmark.construct_time.as_secs_f64()) / 1e6;

    println!("  {} Filter:", benchmark.name);
    println!(
        "    Construction: {:?} ({:.3}M ops/s, {:.3} ns/op)",
        benchmark.construct_time,
        construct_throughput,
        benchmark.construct_time.as_nanos() as f64 / NUM_KEYS as f64
    );
    println!(
        "    Space: {:.1} MB ({:.1} bits/key)",
        (benchmark.size_in_bits as f64 / 8.0) / (1024.0 * 1024.0),
        benchmark.actual_bits_per_key
    );
    println!("    Parameters: {}", benchmark.parameters);
}

fn print_query_results(
    filter_name: &str,
    query_time: Duration,
    positive_count: usize,
    false_positives: usize,
    total_negatives: usize,
    fp_rate: f64,
) {
    let query_throughput = NUM_QUERIES as f64 / query_time.as_secs_f64();

    println!("  {} Filter:", filter_name);
    println!(
        "    Query: {:?} ({:.3}M ops/s, {:.3} ns/op)",
        query_time,
        query_throughput / 1e6,
        query_time.as_nanos() as f64 / NUM_QUERIES as f64
    );
    println!("    Positive queries: {} / {}", positive_count, NUM_QUERIES);
    println!(
        "    False positives: {} / {} ({:.6}%)",
        false_positives,
        total_negatives,
        fp_rate * 100.0
    );
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

/// Shared hash function with seed support using MurmurHash3 finalizer
fn hash_with_seed(item: &u64, seed: u64) -> u64 {
    let mut h = seed.wrapping_add(*item);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

/// Run queries against a filter and return query time and positive count
fn run_queries<F>(filter: &F, query_keys: &[u64]) -> (Duration, usize)
where
    F: ProbabilisticFilter<u64>,
{
    let start = Instant::now();
    let mut positive_count = 0usize;
    for key in query_keys {
        if filter.contains(key) {
            positive_count += 1;
        }
    }
    (start.elapsed(), positive_count)
}

/// Calculate false positive rate from query results
/// Returns (fp_rate, total_negatives, false_positives)
fn calculate_fp_rate<F>(
    filter: &F,
    query_keys: &[u64],
    inserted_set: &HashSet<u64>,
) -> (f64, usize, usize)
where
    F: ProbabilisticFilter<u64>,
{
    let mut total_negatives = 0usize;
    let mut false_positives = 0usize;
    for key in query_keys {
        let is_actually_in = inserted_set.contains(key);
        let is_reported_in = filter.contains(key);
        if !is_actually_in {
            total_negatives += 1;
            if is_reported_in {
                false_positives += 1;
            }
        }
    }
    let fp_rate = if total_negatives > 0 {
        false_positives as f64 / total_negatives as f64
    } else {
        0.0
    };
    (fp_rate, total_negatives, false_positives)
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
