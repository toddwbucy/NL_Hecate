/// Criterion benchmarks for edge deployment micro models.
///
/// Measures throughput (tok/s), adaptation cost, and forward latency
/// across d=64, d=128, d=256 dimension sweep.
///
/// Run: cargo +enzyme bench --features edge --bench edge_bench
/// Reports saved to: target/criterion/

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use nl_hecate_core::edge::{EdgeConfig, EdgeModel};
use nl_hecate_core::model::CompositionKind;
use nl_hecate_core::model::MemoryRuleKind;

fn make_config(d: usize) -> EdgeConfig {
    let num_heads = d / 16; // 16-dim heads
    EdgeConfig {
        d_model: d,
        num_heads,
        seq_len: 16,
        vocab_size: 256,
        k: 1,
        composition: CompositionKind::MAG,
        memory_rule: MemoryRuleKind::DeltaRule,
    }
}

fn make_input(seq_len: usize, vocab_size: usize) -> (Vec<usize>, Vec<usize>) {
    let input_ids: Vec<usize> = (0..seq_len).map(|i| i % vocab_size).collect();
    let target_ids: Vec<usize> = (1..=seq_len).map(|i| i % vocab_size).collect();
    (input_ids, target_ids)
}

/// Throughput: tokens per second for forward pass (Profile 1).
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    for d in [64, 128, 256] {
        let cfg = make_config(d);
        let (input, target) = make_input(cfg.seq_len, cfg.vocab_size);
        let mut model = EdgeModel::new_random(&cfg, 42);

        group.bench_with_input(
            BenchmarkId::new("forward", format!("d={d}")),
            &d,
            |b, _| {
                b.iter(|| {
                    model.process(&input, &target)
                });
            },
        );
    }
    group.finish();
}

/// Adaptation cost: forward + backward (Profile 2).
fn bench_adaptation(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptation");
    for d in [64, 128, 256] {
        let cfg = make_config(d);
        let (input, target) = make_input(cfg.seq_len, cfg.vocab_size);
        let mut model = EdgeModel::new_random(&cfg, 42);

        group.bench_with_input(
            BenchmarkId::new("forward_backward", format!("d={d}")),
            &d,
            |b, _| {
                b.iter(|| {
                    model.forward_backward(&input, &target)
                });
            },
        );
    }
    group.finish();
}

/// Per-token forward latency across sequence lengths.
fn bench_forward_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_latency");
    let d = 64;
    for seq_len in [16, 32, 64] {
        let cfg = EdgeConfig {
            d_model: d,
            num_heads: 4,
            seq_len,
            vocab_size: 256,
            k: 1,
            composition: CompositionKind::MAG,
            memory_rule: MemoryRuleKind::DeltaRule,
        };
        let (input, target) = make_input(seq_len, cfg.vocab_size);
        let mut model = EdgeModel::new_random(&cfg, 42);

        group.bench_with_input(
            BenchmarkId::new("d64", format!("seq={seq_len}")),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    model.process(&input, &target)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_throughput, bench_adaptation, bench_forward_latency);
criterion_main!(benches);
