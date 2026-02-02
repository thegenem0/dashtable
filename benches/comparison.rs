use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dashtable::table::DashTable;
use rand::prelude::*;
use std::collections::HashMap;

fn bench_insert_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_sequential");

    for size in [1000, 10_000, 100_000] {
        group.bench_with_input(BenchmarkId::new("DashTable", size), &size, |b, &size| {
            b.iter(|| {
                let mut table = DashTable::new();
                for i in 0..size {
                    table.insert(i as u64, i as u64);
                }
                black_box(table)
            });
        });

        group.bench_with_input(BenchmarkId::new("HashMap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = HashMap::new();
                for i in 0..size {
                    map.insert(i as u64, i as u64);
                }
                black_box(map)
            });
        });
    }

    group.finish();
}

fn bench_insert_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_random");

    for size in [1000, 10_000, 100_000] {
        let mut rng = StdRng::seed_from_u64(42);
        let keys: Vec<u64> = (0..size).map(|_| rng.gen()).collect();

        group.bench_with_input(BenchmarkId::new("DashTable", size), &keys, |b, keys| {
            b.iter(|| {
                let mut table = DashTable::new();
                for &key in keys {
                    table.insert(key, key);
                }
                black_box(table)
            });
        });

        group.bench_with_input(BenchmarkId::new("HashMap", size), &keys, |b, keys| {
            b.iter(|| {
                let mut map = HashMap::new();
                for &key in keys {
                    map.insert(key, key);
                }
                black_box(map)
            });
        });
    }

    group.finish();
}

fn bench_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_hit");

    for size in [1000, 10_000, 100_000] {
        let mut table = DashTable::new();
        let mut map = HashMap::new();
        for i in 0..size {
            table.insert(i as u64, i as u64);
            map.insert(i as u64, i as u64);
        }

        group.bench_with_input(BenchmarkId::new("DashTable", size), &size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(table.get(&(i as u64)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("HashMap", size), &size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(map.get(&(i as u64)));
                }
            });
        });
    }

    group.finish();
}

fn bench_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_miss");

    for size in [1000, 10_000, 100_000] {
        let mut table = DashTable::new();
        let mut map = HashMap::new();
        for i in 0..size {
            table.insert(i as u64, i as u64);
            map.insert(i as u64, i as u64);
        }

        // Look up keys that don't exist
        let miss_start = size as u64;

        group.bench_with_input(BenchmarkId::new("DashTable", size), &size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(table.get(&(miss_start + i as u64)));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("HashMap", size), &size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(map.get(&(miss_start + i as u64)));
                }
            });
        });
    }

    group.finish();
}

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_80_read_20_write");

    for size in [10_000, 100_000] {
        let mut rng = StdRng::seed_from_u64(42);

        // Pre-populate with half the keys
        let prepop: Vec<u64> = (0..size / 2).map(|i| i as u64).collect();
        let operations: Vec<(bool, u64)> = (0..size)
            .map(|_| {
                let is_read = rng.gen_ratio(80, 100);
                let key = rng.gen_range(0..size as u64);
                (is_read, key)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("DashTable", size),
            &(&prepop, &operations),
            |b, (prepop, ops)| {
                b.iter(|| {
                    let mut table = DashTable::new();
                    for &key in *prepop {
                        table.insert(key, key);
                    }
                    for &(is_read, key) in *ops {
                        if is_read {
                            black_box(table.get(&key));
                        } else {
                            table.insert(key, key);
                        }
                    }
                    black_box(table)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("HashMap", size),
            &(&prepop, &operations),
            |b, (prepop, ops)| {
                b.iter(|| {
                    let mut map = HashMap::new();
                    for &key in *prepop {
                        map.insert(key, key);
                    }
                    for &(is_read, key) in *ops {
                        if is_read {
                            black_box(map.get(&key));
                        } else {
                            map.insert(key, key);
                        }
                    }
                    black_box(map)
                });
            },
        );
    }

    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");

    for size in [1000, 10_000] {
        group.bench_with_input(BenchmarkId::new("DashTable", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut table = DashTable::new();
                    for i in 0..size {
                        table.insert(i as u64, i as u64);
                    }
                    table
                },
                |mut table| {
                    for i in 0..size {
                        black_box(table.remove(&(i as u64)));
                    }
                    table
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("HashMap", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut map = HashMap::new();
                    for i in 0..size {
                        map.insert(i as u64, i as u64);
                    }
                    map
                },
                |mut map| {
                    for i in 0..size {
                        black_box(map.remove(&(i as u64)));
                    }
                    map
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_sequential,
    bench_insert_random,
    bench_lookup_hit,
    bench_lookup_miss,
    bench_mixed_workload,
    bench_remove,
);

criterion_main!(benches);
