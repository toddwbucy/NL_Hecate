/// Autoregressive generation via the unified forward path (spec 68).
///
/// One code path for training and generation: `gpu_stacked_forward_tokens`
/// processes tokens one at a time through decode_token. No prefill, no padding,
/// no separate eval mode.

use std::path::Path;
use std::time::Instant;

use nl_hecate_core::checkpoint::load_stacked_safetensors;
use nl_hecate_core::conductor::Conductor;
use nl_hecate_core::model::{
    MAGConfig, SWAConfig, MemoryRuleKind, CompositionKind, HopeVariant,
    LevelTapeStrategy,
};

#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_params::{GpuStackedParams, GpuStackedContext};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_forward::{
    gpu_stacked_forward_tokens, StackedDecodeWorkspace,
};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_forward::GpuKVCache;

use crate::config::Config;
use crate::sample::sample_token;

/// Generate tokens from a checkpoint given a prompt.
pub fn generate(config_path: &str, checkpoint_path: &str, prompt_tokens: &[usize],
                max_tokens: usize, temperature: f32, top_k: usize,
                stop_token: Option<usize>) {
    let cfg = Config::from_file(config_path).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    let seq_len = cfg.seq_len();
    let d = cfg.model.d_model;
    let nh = cfg.model.num_heads;
    let hd = d / nh;
    let v = cfg.model.vocab_size;
    let k = cfg.model.k;
    let n_blocks = cfg.model.n_blocks;
    let ws = cfg.model.window_size;

    // ── Load checkpoint ──────────────────────────────────────────────
    let p = Path::new(checkpoint_path);
    if !p.exists() {
        eprintln!("ERROR: checkpoint not found: {checkpoint_path}");
        std::process::exit(1);
    }

    let (host_params, loaded_cfg, loaded_n_blocks, build_state) =
        load_stacked_safetensors(p).unwrap_or_else(|e| {
            eprintln!("ERROR: failed to load checkpoint: {e}");
            std::process::exit(1);
        });

    // Validate checkpoint shape
    if loaded_cfg.swa.d_model != d {
        eprintln!("ERROR: checkpoint d_model={} but config d_model={d}", loaded_cfg.swa.d_model);
        std::process::exit(1);
    }
    if loaded_cfg.swa.vocab_size != v {
        eprintln!("ERROR: checkpoint vocab_size={} but config vocab_size={v}", loaded_cfg.swa.vocab_size);
        std::process::exit(1);
    }
    if loaded_n_blocks != n_blocks {
        eprintln!("ERROR: checkpoint n_blocks={loaded_n_blocks} but config n_blocks={n_blocks}");
        std::process::exit(1);
    }

    let step = build_state.as_ref().map(|bs| bs.global_step).unwrap_or(0);

    eprintln!("Loaded checkpoint: {checkpoint_path} (step {step})");
    eprintln!("  d={d}, heads={nh}, vocab={v}, k={k}, blocks={n_blocks}");

    // ── Build MAGConfig ──────────────────────────────────────────────
    let chunk_sizes = cfg.model.chunk_sizes.clone()
        .unwrap_or_else(|| default_chunk_sizes(k));

    let tape_strategies: Vec<LevelTapeStrategy> = cfg.model.tape_strategies
        .as_ref()
        .map(|ts| ts.iter().map(|s| match s.as_str() {
            "exact" => LevelTapeStrategy::Exact,
            _ => LevelTapeStrategy::Proxy,
        }).collect())
        .unwrap_or_else(|| {
            let mut v = vec![LevelTapeStrategy::Exact];
            v.extend((1..k).map(|_| LevelTapeStrategy::Proxy));
            v
        });

    let hope_variant = match cfg.model.hope_variant.as_str() {
        "chained" => HopeVariant::Chained,
        "independent" => HopeVariant::Independent,
        "nested" => HopeVariant::Nested,
        "sequential" => HopeVariant::Sequential,
        _ => HopeVariant::FreqGated,
    };

    let memory_rule = match cfg.model.memory_rule.as_str() {
        "titans" => MemoryRuleKind::TitansLMM,
        "delta" => MemoryRuleKind::DeltaRule,
        "hebbian" => MemoryRuleKind::HebbianRule,
        _ => MemoryRuleKind::TitansLMM,
    };

    let composition = match cfg.model.composition.as_str() {
        "mal" => CompositionKind::MAL,
        "mac" => CompositionKind::MAC,
        _ => CompositionKind::MAG,
    };

    let mut mag_cfg = MAGConfig::test_config();
    mag_cfg.swa = SWAConfig {
        d_model: d,
        num_heads: nh,
        head_dim: hd,
        seq_len,
        window_size: ws,
        vocab_size: v,
    };
    mag_cfg.memory_enabled = true;
    mag_cfg.composition = composition;
    mag_cfg.memory_rule = memory_rule;
    mag_cfg.k = k;
    mag_cfg.chunk_sizes = chunk_sizes.clone();
    mag_cfg.hope_variant = hope_variant;
    mag_cfg.residual = cfg.model.residual;
    mag_cfg.tape_multiplier = cfg.model.tape_multiplier;
    mag_cfg.tape_strategies = tape_strategies;

    if let Some(ref mnm) = cfg.model.m_norm_max {
        mag_cfg.m_norm_max = mnm.clone();
    }
    if let Some(ref ec) = cfg.model.error_clip {
        mag_cfg.error_clip = ec.clone();
    }

    // ── GPU setup ────────────────────────────────────────────────────
    #[cfg(feature = "cuda")]
    {
        let gpu_params = GpuStackedParams::from_host(&host_params);
        let mut gpu_context = GpuStackedContext::new(
            n_blocks, k, d, 1, // batch_size=1 for generation
            Some(&mag_cfg),
        );

        let mut conductor = Conductor::new(k, chunk_sizes);

        eprintln!("Generating: max_tokens={max_tokens}, temperature={temperature}, top_k={top_k}");
        eprintln!("---");

        let t_start = Instant::now();

        // ── Unified forward path (spec 68) ──────────────────────────
        // Same function for prompt processing and generation.
        // No prefill. No padding. No separate eval mode.
        let kv_len = prompt_tokens.len().max(seq_len) + max_tokens;
        let mut kv_caches: Vec<GpuKVCache> = (0..n_blocks)
            .map(|_| GpuKVCache::new(kv_len, d, 1))
            .collect();
        let mut ws = StackedDecodeWorkspace::new(n_blocks, d, v);

        // Process prompt tokens (no activation saving — pure inference)
        let logits = gpu_stacked_forward_tokens(
            &gpu_params, &mag_cfg, prompt_tokens,
            &mut conductor, &mut gpu_context, &mut kv_caches, &mut ws,
            None, // no activation window for generation
        );

        let mut generated = Vec::new();
        let mut last_logits = logits;

        for _i in 0..max_tokens {
            let next_tok = sample_token(&last_logits, temperature, top_k);
            if let Some(stop) = stop_token {
                if next_tok == stop {
                    break;
                }
            }
            print_token(next_tok);
            generated.push(next_tok);

            last_logits = gpu_stacked_forward_tokens(
                &gpu_params, &mag_cfg, &[next_tok],
                &mut conductor, &mut gpu_context, &mut kv_caches, &mut ws,
                None,
            );
        }

        let elapsed = t_start.elapsed().as_secs_f64();
        let tok_s = generated.len() as f64 / elapsed;

        eprintln!();
        eprintln!("---");
        eprintln!("{} tokens in {elapsed:.2}s ({tok_s:.1} tok/s)", generated.len());
    }
}

/// Print a token ID. Without a tokenizer, we print the raw ID.
/// When tokenizer support is added, this will decode to text.
fn print_token(token_id: usize) {
    // Raw token ID output — tokenizer decoding is a future step
    print!(" {token_id}");
    // Flush after each token for streaming output
    use std::io::Write;
    std::io::stdout().flush().ok();
}

/// Default chunk sizes for k levels.
fn default_chunk_sizes(k: usize) -> Vec<usize> {
    match k {
        1 => vec![1],
        2 => vec![1, 8],
        3 => vec![1, 8, 64],
        _ => vec![1, 8, 64, 512],
    }
}
