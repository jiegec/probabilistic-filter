use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use probabilistic_filter::XorFilterConfig;
use probabilistic_filter::binaryfuse::{BinaryFuseConfig, BinaryFuseFilter3, BinaryFuseFilter4};
use probabilistic_filter::bloom::BloomFilter;
use probabilistic_filter::cuckoo::CuckooFilter;
use probabilistic_filter::traits::{Constructible, Insertable, ProbabilisticFilter};
use probabilistic_filter::xor::XorFilter;
use rand::Rng;
use std::hint::black_box;

const SIZES: &[usize] = &[1000, 10_000, 100_000];

fn bench_bloom_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_insert");

    for &size in SIZES {
        let false_positive_rate = 0.01;
        let num_bits =
            BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_bits(size, false_positive_rate);
        let num_hashes = BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_hashes(num_bits, size);
        let keys: Vec<u64> = (0..size).map(|i| i as u64).collect();
        let mut rng = rand::rng();
        let seeds: Vec<u64> = (0..num_hashes).map(|_| rng.random()).collect();
        let mut hash_funcs: Vec<Box<dyn Fn(&u64) -> u64>> = Vec::new();
        for i in 0..num_hashes {
            let seed = seeds[i];
            hash_funcs.push(Box::new(move |item: &u64| hash_with_seed(item, seed)));
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(keys, hash_funcs),
            |b, (keys, hash_funcs)| {
                b.iter(|| {
                    let mut filter = BloomFilter::new(num_bits, hash_funcs);
                    for key in keys {
                        std::hint::black_box(filter.insert(std::hint::black_box(key)));
                    }
                    filter
                })
            },
        );
    }

    group.finish();
}

fn bench_bloom_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_query");

    for &size in SIZES {
        let false_positive_rate = 0.01;
        let num_bits =
            BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_bits(size, false_positive_rate);
        let num_hashes = BloomFilter::<u64, fn(&u64) -> u64>::optimal_num_hashes(num_bits, size);
        let keys: Vec<u64> = (0..size).map(|i| i as u64).collect();

        let mut rng = rand::rng();
        let seeds: Vec<u64> = (0..num_hashes).map(|_| rng.random()).collect();
        let mut hash_funcs: Vec<Box<dyn Fn(&u64) -> u64>> = Vec::new();
        for i in 0..num_hashes {
            let seed = seeds[i];
            hash_funcs.push(Box::new(move |item: &u64| hash_with_seed(item, seed)));
        }

        let mut filter = BloomFilter::new(num_bits, hash_funcs);
        for key in &keys {
            filter.insert(key);
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &keys, |b, keys| {
            b.iter(|| {
                for key in keys {
                    std::hint::black_box(filter.contains(std::hint::black_box(key)));
                }
            })
        });
    }

    group.finish();
}

fn bench_cuckoo_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuckoo_query");

    for &size in SIZES {
        let items_per_bucket = 4;
        let load_factor = 0.955;
        let fingerprint_bits = 8;
        let num_buckets = ((size as f64 / load_factor / items_per_bucket as f64).ceil() as usize)
            .next_power_of_two();

        let keys: Vec<u64> = (0..size).map(|i| i as u64).collect();

        let mut rng = rand::rng();
        let seed: u64 = rng.random();
        let hash = move |item: &u64| -> u64 { hash_with_seed(item, seed) };

        let fp_seed: u64 = rng.random();
        let fingerprint_func = move |item: &u64| -> u64 { hash_with_seed(item, fp_seed) };

        let alt_index = move |index: usize, fp: u64| -> usize {
            let hashed_fp = hash_with_seed(&fp, fp_seed);
            (index ^ hashed_fp as usize) % num_buckets
        };

        let mut filter = CuckooFilter::new(
            num_buckets,
            items_per_bucket,
            500,
            hash,
            fingerprint_func,
            alt_index,
            fingerprint_bits,
        );
        for key in &keys {
            filter.insert(key);
        }

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &keys, |b, keys| {
            b.iter(|| {
                for key in keys {
                    std::hint::black_box(filter.contains(std::hint::black_box(key)));
                }
            })
        });
    }

    group.finish();
}

fn bench_xor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_creation");

    for &size in SIZES {
        let effective_bits = 8;
        let keys: Vec<u64> = (0..size).map(|i| i as u64).collect();

        let hash = move |item: &u64, seed: u64| -> u64 { hash_with_seed(item, seed) };

        let fingerprint_func = move |item: &u64| -> u64 { item.wrapping_mul(0x9e3779b97f4a7c15) };

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &keys, |b, keys| {
            b.iter(|| {
                XorFilter::from_iter_with(
                    keys.iter().copied(),
                    XorFilterConfig::new(hash, fingerprint_func, effective_bits),
                )
            })
        });
    }

    group.finish();
}

fn bench_xor_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("xor_query");

    for &size in SIZES {
        let effective_bits = 8;
        let keys: Vec<u64> = (0..size).map(|i| i as u64).collect();

        let hash = move |item: &u64, seed: u64| -> u64 { hash_with_seed(item, seed) };

        let fingerprint_func = move |item: &u64| -> u64 { item.wrapping_mul(0x9e3779b97f4a7c15) };

        let filter = XorFilter::from_iter_with(
            keys.iter().copied(),
            XorFilterConfig::new(hash, fingerprint_func, effective_bits),
        );

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &keys, |b, keys| {
            b.iter(|| {
                for key in keys {
                    black_box(filter.contains(black_box(key)));
                }
            })
        });
    }

    group.finish();
}

fn bench_binaryfuse3_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("binaryfuse3_creation");

    for &size in SIZES {
        let effective_bits = 8;
        let keys: Vec<u64> = (0..size).map(|i| i as u64).collect();

        let hash = move |item: &u64, seed: u64| -> u64 { hash_with_seed(item, seed) };

        let fingerprint_func = move |item: &u64| -> u64 { item.wrapping_mul(0x9e3779b97f4a7c15) };

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &keys, |b, keys| {
            b.iter(|| {
                BinaryFuseFilter3::from_iter_with(
                    keys.iter().copied(),
                    BinaryFuseConfig::new(hash, fingerprint_func, effective_bits),
                )
            })
        });
    }

    group.finish();
}

fn bench_binaryfuse3_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("binaryfuse3_query");

    for &size in SIZES {
        let effective_bits = 8;
        let keys: Vec<u64> = (0..size).map(|i| i as u64).collect();

        let hash = move |item: &u64, seed: u64| -> u64 { hash_with_seed(item, seed) };

        let fingerprint_func = move |item: &u64| -> u64 { item.wrapping_mul(0x9e3779b97f4a7c15) };

        let filter = BinaryFuseFilter3::from_iter_with(
            keys.iter().copied(),
            BinaryFuseConfig::new(hash, fingerprint_func, effective_bits),
        );

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &keys, |b, keys| {
            b.iter(|| {
                for key in keys {
                    black_box(filter.contains(black_box(key)));
                }
            })
        });
    }

    group.finish();
}

fn bench_binaryfuse4_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("binaryfuse4_creation");

    for &size in SIZES {
        let effective_bits = 8;
        let keys: Vec<u64> = (0..size).map(|i| i as u64).collect();

        let hash = move |item: &u64, seed: u64| -> u64 { hash_with_seed(item, seed) };

        let fingerprint_func = move |item: &u64| -> u64 { item.wrapping_mul(0x9e3779b97f4a7c15) };

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &keys, |b, keys| {
            b.iter(|| {
                BinaryFuseFilter4::from_iter_with(
                    keys.iter().copied(),
                    BinaryFuseConfig::new(hash, fingerprint_func, effective_bits),
                )
            })
        });
    }

    group.finish();
}

fn bench_binaryfuse4_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("binaryfuse4_query");

    for &size in SIZES {
        let effective_bits = 8;
        let keys: Vec<u64> = (0..size).map(|i| i as u64).collect();

        let hash = move |item: &u64, seed: u64| -> u64 { hash_with_seed(item, seed) };

        let fingerprint_func = move |item: &u64| -> u64 { item.wrapping_mul(0x9e3779b97f4a7c15) };

        let filter = BinaryFuseFilter4::from_iter_with(
            keys.iter().copied(),
            BinaryFuseConfig::new(hash, fingerprint_func, effective_bits),
        );

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &keys, |b, keys| {
            b.iter(|| {
                for key in keys {
                    black_box(filter.contains(black_box(key)));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(bloom, bench_bloom_insert, bench_bloom_query);
criterion_group!(cuckoo, bench_cuckoo_query,);
criterion_group!(xor, bench_xor_creation, bench_xor_query);
criterion_group!(
    binaryfuse3,
    bench_binaryfuse3_creation,
    bench_binaryfuse3_query
);
criterion_group!(
    binaryfuse4,
    bench_binaryfuse4_creation,
    bench_binaryfuse4_query
);
criterion_main!(bloom, cuckoo, xor, binaryfuse3, binaryfuse4);
