/// Evaluation probes: cross-exposure, within-generation, coherence samples.
/// Rust port of Python evaluation.py probes.
///
/// All probes are CS-10 compliant: no eval mode, continuous learning during
/// inference. The model's forward pass IS the optimization.

use std::path::Path;

use tokenizers::Tokenizer;
use serde_json::json;

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
    gpu_stacked_forward_tokens,
    StackedDecodeWorkspace, ActivationWindow,
};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_backward::gpu_stacked_backward;
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_optimizer::{
    GpuStackedAdamWState, gpu_stacked_adamw_update, gpu_stacked_sync_embed_weights,
};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_forward::GpuKVCache;

use crate::config::Config;
use crate::log::MetricsLogger;
use crate::sample::sample_token;

// Probing prompts matched to FineWeb-Edu domain (educational text)
const PROBE_PROMPTS: &[&str] = &[
    "The process of",
    "In mathematics,",
    "Scientists discovered that",
    "The history of",
];

/// Run all probes on a checkpoint.
pub fn run_probes(
    config_path: &str, checkpoint_path: &str, tokenizer_path: &str,
    max_tokens: usize, temperature: f32, top_k: usize,
) {
    let cfg = Config::from_file(config_path).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to load tokenizer: {e}");
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

    if loaded_cfg.swa.d_model != d {
        eprintln!("ERROR: checkpoint d_model={} but config d_model={d}", loaded_cfg.swa.d_model);
        std::process::exit(1);
    }
    if loaded_n_blocks != n_blocks {
        eprintln!("ERROR: checkpoint n_blocks={loaded_n_blocks} but config n_blocks={n_blocks}");
        std::process::exit(1);
    }

    let step = build_state.as_ref().map(|bs| bs.global_step).unwrap_or(0);

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
        d_model: d, num_heads: nh, head_dim: hd,
        seq_len, window_size: ws, vocab_size: v,
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

    let lr = cfg.build.optimizer.lr();
    let beta1 = cfg.build.optimizer.beta1();
    let beta2 = cfg.build.optimizer.beta2();
    let wd = cfg.build.optimizer.weight_decay();
    let max_grad_norm = cfg.build.max_grad_norm;

    // ── GPU setup ────────────────────────────────────────────────────
    #[cfg(feature = "cuda")]
    {
        let mut gpu_params = GpuStackedParams::from_host(&host_params);
        let mut gpu_context = GpuStackedContext::new(
            n_blocks, k, d, 1, Some(&mag_cfg),
        );
        let mut adamw_state: Option<GpuStackedAdamWState> = None;

        let sep = "=".repeat(60);
        eprintln!("{sep}");
        eprintln!("NL-Hecate Probes");
        eprintln!("{sep}");
        eprintln!("  Checkpoint: {checkpoint_path} (step {step})");
        eprintln!("  Model: d={d}, heads={nh}, k={k}, blocks={n_blocks}");
        eprintln!("  Probes: {} prompts, max_tokens={max_tokens}", PROBE_PROMPTS.len());
        eprintln!("{sep}\n");

        // Save initial state for restore between probes
        let initial_host_params = gpu_params.to_host(d, v, k);

        // ── Probe 1: Coherence Samples ───────────────────────────
        eprintln!("── Probe: Coherence Samples ──");
        for prompt_str in PROBE_PROMPTS {
            let prompt_ids = encode_prompt(&tokenizer, prompt_str);

            // Unified forward path (spec 68) — same function for prompt + generation
            let kv_len = prompt_ids.len().max(seq_len) + max_tokens;
            let mut kv_caches: Vec<GpuKVCache> = (0..n_blocks)
                .map(|_| GpuKVCache::new(kv_len, d, 1))
                .collect();
            let mut decode_ws = StackedDecodeWorkspace::new(n_blocks, d, v);
            let mut conductor = Conductor::new(k, chunk_sizes.clone());

            let mut window = ActivationWindow::new(seq_len);

            // Process prompt
            let logits = gpu_stacked_forward_tokens(
                &gpu_params, &mag_cfg, &prompt_ids,
                &mut conductor, &mut gpu_context, &mut kv_caches, &mut decode_ws,
                &mut window,
            );

            // Generate tokens (same function, one at a time)
            let mut gen = Vec::new();
            let mut last_logits = logits;
            for _ in 0..max_tokens {
                let next_tok = sample_token(&last_logits, temperature, top_k);
                gen.push(next_tok);
                last_logits = gpu_stacked_forward_tokens(
                    &gpu_params, &mag_cfg, &[next_tok],
                    &mut conductor, &mut gpu_context, &mut kv_caches, &mut decode_ws,
                    &mut window,
                );
            }

            let gen_text = decode_tokens(&tokenizer, &gen);
            eprintln!("  \"{prompt_str}\" → {gen_text}");

            // Restore state for next prompt
            gpu_params = GpuStackedParams::from_host(&initial_host_params);
            gpu_context = GpuStackedContext::new(n_blocks, k, d, 1, Some(&mag_cfg));
        }
        eprintln!();

        // ── Probe 2: Within-Generation Learning ──────────────────
        eprintln!("── Probe: Within-Generation Learning ──");
        for prompt_str in PROBE_PROMPTS {
            let prompt_ids = encode_prompt(&tokenizer, prompt_str);

            let mut conductor = Conductor::new(k, chunk_sizes.clone());

            let losses = generate_learning_losses(
                &mut gpu_params, &mag_cfg, &mut gpu_context,
                &mut adamw_state, &mut conductor,
                &prompt_ids, seq_len, v, max_tokens,
                temperature, top_k,
                lr, beta1, beta2, wd, max_grad_norm, d,
            );

            // Compute summary
            let n = losses.len();
            if n >= 10 {
                let first10: f32 = losses[..10].iter().sum::<f32>() / 10.0;
                let last10: f32 = losses[n - 10..].iter().sum::<f32>() / 10.0;
                let slope = linear_slope(&losses);
                eprintln!("  \"{prompt_str}\": first10={first10:.4} last10={last10:.4} slope={slope:.6} ({n} tokens)");
            } else {
                let avg: f32 = losses.iter().sum::<f32>() / n.max(1) as f32;
                eprintln!("  \"{prompt_str}\": avg_loss={avg:.4} ({n} tokens)");
            }

            // Restore state
            gpu_params = GpuStackedParams::from_host(&initial_host_params);
            gpu_context = GpuStackedContext::new(n_blocks, k, d, 1, Some(&mag_cfg));
            adamw_state = None;
        }
        eprintln!();

        // ── Probe 3: Cross-Exposure Adaptation ───────────────────
        eprintln!("── Probe: Cross-Exposure Adaptation ──");
        eprintln!("  (The definitive NL test — no transformer can do this)");
        for prompt_str in PROBE_PROMPTS {
            let prompt_ids = encode_prompt(&tokenizer, prompt_str);

            // Run 1: cold start
            let mut conductor = Conductor::new(k, chunk_sizes.clone());
            gpu_context = GpuStackedContext::new(n_blocks, k, d, 1, Some(&mag_cfg));

            let losses1 = generate_learning_losses(
                &mut gpu_params, &mag_cfg, &mut gpu_context,
                &mut adamw_state, &mut conductor,
                &prompt_ids, seq_len, v, max_tokens,
                temperature, top_k,
                lr, beta1, beta2, wd, max_grad_norm, d,
            );
            let avg1 = avg_loss(&losses1);

            // Run 2: reset context but KEEP updated params
            conductor = Conductor::new(k, chunk_sizes.clone());
            gpu_context = GpuStackedContext::new(n_blocks, k, d, 1, Some(&mag_cfg));

            let losses2 = generate_learning_losses(
                &mut gpu_params, &mag_cfg, &mut gpu_context,
                &mut adamw_state, &mut conductor,
                &prompt_ids, seq_len, v, max_tokens,
                temperature, top_k,
                lr, beta1, beta2, wd, max_grad_norm, d,
            );
            let avg2 = avg_loss(&losses2);

            let improvement = avg1 - avg2;
            let pct = if avg1 > 0.0 { improvement / avg1 * 100.0 } else { 0.0 };
            eprintln!("  \"{prompt_str}\": run1={avg1:.4} run2={avg2:.4} improvement={improvement:.4} ({pct:.1}%)");

            // Restore state for next prompt
            gpu_params = GpuStackedParams::from_host(&initial_host_params);
            gpu_context = GpuStackedContext::new(n_blocks, k, d, 1, Some(&mag_cfg));
            adamw_state = None;
        }

        eprintln!("\n{sep}");
        eprintln!("Probes complete.");
        eprintln!("{sep}");
    }
}

/// Run inline probes at a checkpoint during training.
///
/// Takes a host-side parameter snapshot (from `gpu_params.to_host()`) and allocates
/// its own GPU state — the caller's training state is never modified.
///
/// Probes:
/// 1. **Coherence samples**: Greedy generation from prompts (text quality check)
/// 2. **Within-generation learning**: Loss should decrease as model processes its own output
/// 3. **Cross-exposure adaptation**: Second pass on same prompt should show lower loss (NL signature)
///
/// Returns a JSON value with all probe results, suitable for `logger.log_probe_results()`.
#[cfg(feature = "cuda")]
pub fn run_inline_probes(
    host_params: &nl_hecate_core::stacked_model::StackedMAGParams,
    mag_cfg: &MAGConfig,
    tokenizer_path: &str,
    step: usize,
    d: usize,
    v: usize,
    k: usize,
    n_blocks: usize,
    chunk_sizes: &[usize],
    max_tokens: usize,
    temperature: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    wd: f32,
    max_grad_norm: f32,
) -> serde_json::Value {
    if max_tokens == 0 {
        return json!({"skipped": true});
    }
    let tokenizer = match Tokenizer::from_file(tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  [probes: tokenizer load failed: {e}]");
            return json!({"error": format!("tokenizer: {e}")});
        }
    };

    let seq_len = mag_cfg.swa.seq_len;

    // Probe-local MAGConfig with seq_len matching training (probes use batch=1)
    let probe_cfg = mag_cfg.clone();

    eprintln!("  [probes: running checkpoint probes at step {step}]");
    let t0 = std::time::Instant::now();

    let mut coherence_results = Vec::new();
    let mut within_gen_results = Vec::new();
    let mut cross_exposure_results = Vec::new();

    // ── Probe 1: Coherence Samples ───────────────────────────────────
    for prompt_str in PROBE_PROMPTS {
        let mut gpu_params = GpuStackedParams::from_host(host_params);
        let mut gpu_context = GpuStackedContext::new(n_blocks, k, d, 1, Some(&probe_cfg));

        let prompt_ids = encode_prompt(&tokenizer, prompt_str);

        // Unified forward path (spec 68)
        let kv_len = prompt_ids.len().max(seq_len) + max_tokens;
        let mut kv_caches: Vec<GpuKVCache> = (0..n_blocks)
            .map(|_| GpuKVCache::new(kv_len, d, 1))
            .collect();
        let mut decode_ws = StackedDecodeWorkspace::new(n_blocks, d, v);
        let mut conductor = Conductor::new(k, chunk_sizes.to_vec());

        let mut window = ActivationWindow::new(seq_len);

        // Process prompt
        let logits = gpu_stacked_forward_tokens(
            &gpu_params, &probe_cfg, &prompt_ids,
            &mut conductor, &mut gpu_context, &mut kv_caches, &mut decode_ws,
            &mut window,
        );

        // Generate tokens (same function)
        let mut gen = Vec::new();
        let mut last_logits = logits;
        for _ in 0..max_tokens {
            let next_tok = sample_token(&last_logits, temperature, 50);
            gen.push(next_tok);
            last_logits = gpu_stacked_forward_tokens(
                &gpu_params, &probe_cfg, &[next_tok],
                &mut conductor, &mut gpu_context, &mut kv_caches, &mut decode_ws,
                &mut window,
            );
        }

        let gen_text = decode_tokens(&tokenizer, &gen);
        coherence_results.push(json!({
            "prompt": prompt_str,
            "generation": gen_text,
        }));
        // gpu_params + gpu_context dropped here
    }

    // ── Probe 2: Within-Generation Learning ──────────────────────────
    for prompt_str in PROBE_PROMPTS {
        let mut gpu_params = GpuStackedParams::from_host(host_params);
        let mut gpu_context = GpuStackedContext::new(n_blocks, k, d, 1, Some(&probe_cfg));
        let mut adamw_state: Option<GpuStackedAdamWState> = None;

        let prompt_ids = encode_prompt(&tokenizer, prompt_str);
        let mut conductor = Conductor::new(k, chunk_sizes.to_vec());

        let losses = generate_learning_losses(
            &mut gpu_params, &probe_cfg, &mut gpu_context,
            &mut adamw_state, &mut conductor,
            &prompt_ids, seq_len, v, max_tokens,
            temperature, 50,
            lr, beta1, beta2, wd, max_grad_norm, d,
        );

        let n = losses.len();
        let (first10, last10, slope) = if n >= 10 {
            let f10: f32 = losses[..10].iter().sum::<f32>() / 10.0;
            let l10: f32 = losses[n - 10..].iter().sum::<f32>() / 10.0;
            (f10, l10, linear_slope(&losses))
        } else {
            let avg = avg_loss(&losses);
            (avg, avg, 0.0)
        };

        within_gen_results.push(json!({
            "prompt": prompt_str,
            "tokens": n,
            "first10_loss": first10,
            "last10_loss": last10,
            "slope": slope,
        }));
    }

    // ── Probe 3: Cross-Exposure Adaptation ───────────────────────────
    for prompt_str in PROBE_PROMPTS {
        let mut gpu_params = GpuStackedParams::from_host(host_params);
        let mut gpu_context = GpuStackedContext::new(n_blocks, k, d, 1, Some(&probe_cfg));
        let mut adamw_state: Option<GpuStackedAdamWState> = None;

        let prompt_ids = encode_prompt(&tokenizer, prompt_str);

        // Run 1: cold start
        let mut conductor = Conductor::new(k, chunk_sizes.to_vec());
        let losses1 = generate_learning_losses(
            &mut gpu_params, &probe_cfg, &mut gpu_context,
            &mut adamw_state, &mut conductor,
            &prompt_ids, seq_len, v, max_tokens,
            temperature, 50,
            lr, beta1, beta2, wd, max_grad_norm, d,
        );
        let avg1 = avg_loss(&losses1);

        // Run 2: reset context but KEEP updated params
        conductor = Conductor::new(k, chunk_sizes.to_vec());
        gpu_context = GpuStackedContext::new(n_blocks, k, d, 1, Some(&probe_cfg));

        let losses2 = generate_learning_losses(
            &mut gpu_params, &probe_cfg, &mut gpu_context,
            &mut adamw_state, &mut conductor,
            &prompt_ids, seq_len, v, max_tokens,
            temperature, 50,
            lr, beta1, beta2, wd, max_grad_norm, d,
        );
        let avg2 = avg_loss(&losses2);

        let improvement = avg1 - avg2;
        let pct = if avg1 > 0.0 { improvement / avg1 * 100.0 } else { 0.0 };

        cross_exposure_results.push(json!({
            "prompt": prompt_str,
            "run1_avg_loss": avg1,
            "run2_avg_loss": avg2,
            "improvement": improvement,
            "improvement_pct": pct,
        }));
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("  [probes: complete in {elapsed:.1}s]");

    json!({
        "coherence_samples": coherence_results,
        "within_gen_learning": within_gen_results,
        "cross_exposure": cross_exposure_results,
        "elapsed_secs": elapsed,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────

fn encode_prompt(tokenizer: &Tokenizer, text: &str) -> Vec<usize> {
    let enc = tokenizer.encode(text, false).expect("tokenizer encode failed");
    enc.get_ids().iter().map(|&id| id as usize).collect()
}

fn decode_tokens(tokenizer: &Tokenizer, tokens: &[usize]) -> String {
    let ids32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    tokenizer.decode(&ids32, true).unwrap_or_else(|_| String::from("???"))
}

fn pad_to_seq_len(tokens: &[usize], seq_len: usize, pad_tok: usize) -> Vec<usize> {
    if tokens.len() >= seq_len {
        tokens[tokens.len() - seq_len..].to_vec()
    } else {
        let pad_count = seq_len - tokens.len();
        let mut padded = vec![pad_tok; pad_count];
        padded.extend_from_slice(tokens);
        padded
    }
}

fn build_context(seq: &[usize], seq_len: usize, safe_pad: usize) -> Vec<usize> {
    let ctx: Vec<usize> = if seq.len() >= seq_len {
        seq[seq.len() - seq_len..].to_vec()
    } else {
        seq.to_vec()
    };
    pad_to_seq_len(&ctx, seq_len, safe_pad)
}

fn build_target(seq: &[usize], seq_len: usize, vocab_size: usize) -> Vec<usize> {
    if seq.len() >= seq_len + 1 {
        let src = &seq[seq.len() - seq_len + 1..];
        let mut target = src.to_vec();
        target.push(vocab_size); // mask last position
        while target.len() < seq_len {
            target.insert(0, vocab_size);
        }
        target
    } else if seq.len() > 1 {
        let shifted: Vec<usize> = seq[1..].to_vec();
        let n_real = seq.len().min(seq_len);
        let n_masked_prefix = seq_len - n_real;
        let mut target = vec![vocab_size; n_masked_prefix];
        let take = (seq_len - n_masked_prefix).min(shifted.len());
        target.extend_from_slice(&shifted[..take]);
        while target.len() < seq_len {
            target.push(vocab_size);
        }
        if let Some(last) = target.last_mut() {
            *last = vocab_size;
        }
        target
    } else {
        vec![vocab_size; seq_len]
    }
}

/// Run generate_learning loop via unified forward path, return per-token losses.
///
/// Unified path (spec 68): process tokens one at a time through forward_single_token,
/// accumulate activations in a window, backward through the window, update params,
/// then sample next token from the last logits. Same code path as training.
#[cfg(feature = "cuda")]
fn generate_learning_losses(
    gpu_params: &mut GpuStackedParams,
    mag_cfg: &MAGConfig,
    gpu_context: &mut GpuStackedContext,
    adamw_state: &mut Option<GpuStackedAdamWState>,
    conductor: &mut Conductor,
    prompt_ids: &[usize],
    seq_len: usize,
    vocab_size: usize,
    max_tokens: usize,
    temperature: f32,
    top_k_val: usize,
    lr: f32, beta1: f32, beta2: f32, wd: f32, max_grad_norm: f32,
    d: usize,
) -> Vec<f32> {
    let v = vocab_size;
    let n_blocks = gpu_params.n_blocks();
    let k = mag_cfg.k;
    let chunk_sizes = mag_cfg.chunk_sizes.clone();
    let mut losses = Vec::new();

    // KV caches + workspace for the unified forward path
    let mut kv_caches: Vec<GpuKVCache> = (0..n_blocks)
        .map(|_| GpuKVCache::new(seq_len + max_tokens + prompt_ids.len(), d, 1))
        .collect();
    let mut decode_ws = StackedDecodeWorkspace::new(n_blocks, d, v);

    // Activation window: gradient_window_size = seq_len
    let mut window = ActivationWindow::new(seq_len);

    // Process prompt tokens through the unified path (with activation caching)
    gpu_stacked_forward_tokens(
        gpu_params, mag_cfg, prompt_ids,
        conductor, gpu_context, &mut kv_caches, &mut decode_ws,
        &mut window,
    );

    // Generate tokens: each one gets forward → backward → update → sample
    for _ in 0..max_tokens {
        let win_len = window.len();
        if win_len < 2 { break; } // need at least 2 tokens for loss

        // Build targets: shifted by 1 (next-token prediction)
        // Window contains the last `win_len` tokens; targets are tokens[1..] + mask
        let mut target_ids: Vec<usize> = Vec::with_capacity(win_len);
        // We don't have easy access to the token IDs from window entries,
        // so reconstruct from the activation caches
        for i in 1..win_len {
            target_ids.push(window.entries[i].token_id);
        }
        target_ids.push(vocab_size); // mask the last position

        // Assemble window into GpuStackedCache for backward
        let cache = window.assemble_cache(mag_cfg, &target_ids);

        // Compute loss on host from assembled logits + targets
        let loss = host_cross_entropy_loss(&cache.logits, &target_ids, v, win_len);

        if loss.is_nan() || loss.is_infinite() { break; }
        losses.push(loss);

        // Backward through the assembled cache
        let mut grads = gpu_stacked_backward(
            gpu_params, mag_cfg, &cache, &mut None, false,
        );

        // Optimizer step — use cache.pulse (matches the window's activations),
        // not conductor.pulse() which has already advanced past the window.
        if adamw_state.is_none() {
            *adamw_state = Some(GpuStackedAdamWState::from_params(gpu_params));
        }
        let aw = adamw_state.as_mut().unwrap();
        gpu_stacked_adamw_update(
            gpu_params, &mut grads, aw, &cache.pulse,
            lr, beta1, beta2, 1e-8, wd, max_grad_norm,
            false, &mut None,
        );
        gpu_stacked_sync_embed_weights(gpu_params, d, v);

        // SPEAK: sample next token from last logits in window
        let logits = window.last_logits().unwrap();
        let next_tok = sample_token(&logits, temperature, top_k_val);

        // Forward the new token through the unified path (pushes onto window)
        gpu_stacked_forward_tokens(
            gpu_params, mag_cfg, &[next_tok],
            conductor, gpu_context, &mut kv_caches, &mut decode_ws,
            &mut window,
        );
    }

    losses
}

fn avg_loss(losses: &[f32]) -> f32 {
    let valid: Vec<f32> = losses.iter().copied()
        .filter(|l| l.is_finite())
        .collect();
    if valid.is_empty() { return 0.0; }
    valid.iter().sum::<f32>() / valid.len() as f32
}

fn linear_slope(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 2 { return 0.0; }
    let mean_x = (n - 1) as f32 / 2.0;
    let mean_y: f32 = values.iter().sum::<f32>() / n as f32;
    let num: f32 = values.iter().enumerate()
        .map(|(i, &v)| (i as f32 - mean_x) * (v - mean_y))
        .sum();
    let den: f32 = (0..n).map(|i| (i as f32 - mean_x).powi(2)).sum();
    if den > 0.0 { num / den } else { 0.0 }
}

/// Host-side cross-entropy loss from GPU logits buffer + target IDs.
/// Used by probes and chat learn mode — not on the training hot path.
#[cfg(feature = "cuda")]
pub(crate) fn host_cross_entropy_loss(
    logits_gpu: &nl_hecate_core::gpu_buf::GpuBuf<f32>,
    target_ids: &[usize],
    vocab_size: usize,
    seq_len: usize,
) -> f32 {
    let mut logits_host = vec![0.0f32; seq_len * vocab_size];
    logits_gpu.copy_to_host(&mut logits_host);

    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for t in 0..seq_len {
        let target = target_ids[t];
        if target >= vocab_size { continue; } // masked position

        let offset = t * vocab_size;
        let row = &logits_host[offset..offset + vocab_size];

        // log-sum-exp for numerical stability
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum::<f32>().ln() + max_logit;
        let log_prob = row[target] - lse;
        total_loss -= log_prob;
        count += 1;
    }

    if count > 0 { total_loss / count as f32 } else { 0.0 }
}

fn default_chunk_sizes(k: usize) -> Vec<usize> {
    match k {
        1 => vec![1],
        2 => vec![1, 8],
        3 => vec![1, 8, 64],
        _ => vec![1, 8, 64, 512],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_to_seq_len_short() {
        let result = pad_to_seq_len(&[10, 20], 5, 99);
        assert_eq!(result, vec![99, 99, 99, 10, 20]);
    }

    #[test]
    fn pad_to_seq_len_exact() {
        let result = pad_to_seq_len(&[1, 2, 3], 3, 99);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn pad_to_seq_len_long() {
        let result = pad_to_seq_len(&[1, 2, 3, 4, 5], 3, 99);
        assert_eq!(result, vec![3, 4, 5]);
    }

    #[test]
    fn build_target_short_seq() {
        // seq = [10, 20], seq_len = 4, vocab = 100
        let target = build_target(&[10, 20], 4, 100);
        assert_eq!(target.len(), 4);
        // Should have masked prefix + shifted content + masked last
        assert_eq!(*target.last().unwrap(), 100); // last always masked
    }

    #[test]
    fn build_target_long_seq() {
        // seq longer than seq_len + 1
        let seq: Vec<usize> = (0..10).collect();
        let target = build_target(&seq, 4, 100);
        assert_eq!(target.len(), 4);
        assert_eq!(*target.last().unwrap(), 100); // last masked
    }

    #[test]
    fn linear_slope_positive() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let slope = linear_slope(&values);
        assert!((slope - 1.0).abs() < 0.01);
    }

    #[test]
    fn linear_slope_flat() {
        let values = vec![3.0, 3.0, 3.0, 3.0];
        let slope = linear_slope(&values);
        assert!(slope.abs() < 0.001);
    }

    #[test]
    fn avg_loss_filters_nan() {
        let losses = vec![1.0, f32::NAN, 3.0, f32::INFINITY, 5.0];
        let avg = avg_loss(&losses);
        assert!((avg - 3.0).abs() < 0.01); // (1 + 3 + 5) / 3
    }
}
