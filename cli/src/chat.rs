/// Interactive multi-turn chat with ChatML formatting (spec 70).
///
/// Two memory modes:
/// - Stateful (default): CMS memory carries conversation context.
///   Only the new user message is fed each turn. Constant prompt size.
/// - Stateless (--stateless): Full conversation history re-sent each turn.
///   No CMS memory persistence. Traditional transformer-style chat.
///
/// The model ALWAYS learns (forward → backward → update) on every token it sees.
/// There is no --learn flag — disabling learning produces a broken NLM (CS-10).
/// Generation uses deferred backward: the model learns from its own output.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

use tokenizers::Tokenizer;

use nl_hecate_core::checkpoint::load_stacked_safetensors;
use nl_hecate_core::conductor::Conductor;
use nl_hecate_core::model::{
    MAGConfig, SWAConfig, MemoryRuleKind, CompositionKind, HopeVariant,
    LevelTapeStrategy,
};

#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_params::{GpuStackedParams, GpuStackedContext};
#[cfg(feature = "cuda")]
use nl_hecate_core::gpu_stacked_optimizer::GpuStackedAdamWState;

use crate::config::Config;
#[cfg(feature = "cuda")]
use crate::step::generate;

// ChatML special token IDs (must match tokenizer training in prepare_sharegpt.py)
const IM_START: u32 = 0; // <|im_start|>
const IM_END: u32 = 1;   // <|im_end|>

/// Encode a single ChatML turn: <|im_start|>role\ncontent<|im_end|>\n
fn chatml_encode_turn(tokenizer: &Tokenizer, role: &str, content: &str) -> Vec<usize> {
    let mut ids = vec![IM_START as usize];
    let role_enc = tokenizer.encode(format!("{role}\n"), false)
        .expect("tokenizer encode failed");
    ids.extend(role_enc.get_ids().iter().map(|&id| id as usize));
    let content_enc = tokenizer.encode(content, false)
        .expect("tokenizer encode failed");
    ids.extend(content_enc.get_ids().iter().map(|&id| id as usize));
    ids.push(IM_END as usize);
    let nl_enc = tokenizer.encode("\n", false)
        .expect("tokenizer encode failed");
    ids.extend(nl_enc.get_ids().iter().map(|&id| id as usize));
    ids
}

/// Encode the start of a turn (no content): <|im_start|>role\n
fn chatml_encode_prompt(tokenizer: &Tokenizer, role: &str) -> Vec<usize> {
    let mut ids = vec![IM_START as usize];
    let role_enc = tokenizer.encode(format!("{role}\n"), false)
        .expect("tokenizer encode failed");
    ids.extend(role_enc.get_ids().iter().map(|&id| id as usize));
    ids
}

/// Decode token IDs back to text.
fn decode_tokens(tokenizer: &Tokenizer, tokens: &[usize]) -> String {
    let ids32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    tokenizer.decode(&ids32, true).unwrap_or_else(|_| String::from("???"))
}

/// Run interactive chat. The model learns from every token — no --learn flag needed.
pub fn chat(
    config_path: &str, checkpoint_path: &str, tokenizer_path: &str,
    max_tokens: usize, temperature: f32, top_k: usize,
    stateless: bool,
) {
    let cfg = Config::from_file(config_path).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });

    // Load tokenizer
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

    // Use conductor.step for replay (per-token counter), not global_step (optimizer steps)
    let loaded_conductor_step = build_state.as_ref().map(|bs| bs.conductor.step).unwrap_or(0);

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
        let mut gpu_params = GpuStackedParams::from_host(&host_params);
        let mut gpu_context = GpuStackedContext::new(
            n_blocks, k, d, 1, // batch_size=1 for chat
            Some(&mag_cfg),
        );

        let mut conductor = Conductor::new(k, chunk_sizes.clone());
        for _ in 0..loaded_conductor_step {
            conductor.advance();
        }

        let mut adamw_state: Option<GpuStackedAdamWState> = None;

        let lr = cfg.build.optimizer.lr();
        let max_grad_norm = cfg.build.max_grad_norm;

        // ── Banner ───────────────────────────────────────────────
        let sep = "\u{2500}".repeat(60);
        let mode_label = if stateless {
            "stateless (full history)"
        } else {
            "stateful (CMS memory)"
        };

        eprintln!("\n{sep}");
        eprintln!("  NL-Hecate Chat");
        eprintln!("  Model: d={d}, heads={nh}, k={k}, blocks={n_blocks} (step {loaded_conductor_step})");
        eprintln!("  Mode: {mode_label} + learning");
        eprintln!("  temp={temperature}, top_k={top_k}, max_tokens={max_tokens}");
        eprintln!("{sep}");
        eprintln!("  Commands: /quit  /clear  /mode  /stats");
        eprintln!("{sep}\n");

        let mut history_tokens: Vec<usize> = Vec::new();
        let mut turn_count: usize = 0;

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            // Prompt
            print!("\x1b[1;36mYou:\x1b[0m ");
            stdout.flush().ok();

            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => { // EOF
                    eprintln!("\nBye!");
                    break;
                }
                Ok(_) => {}
                Err(_) => {
                    eprintln!("\nBye!");
                    break;
                }
            }

            let stripped = line.trim();
            if stripped.is_empty() {
                continue;
            }

            // ── Slash commands ──
            let lower = stripped.to_lowercase();
            if lower == "/quit" || lower == "/exit" || lower == "/q" {
                eprintln!("Bye!");
                break;
            }

            if lower == "/clear" {
                history_tokens.clear();
                turn_count = 0;
                if !stateless {
                    conductor = Conductor::new(k, chunk_sizes.clone());
                    for _ in 0..loaded_conductor_step {
                        conductor.advance();
                    }
                    gpu_context = GpuStackedContext::new(
                        n_blocks, k, d, 1, Some(&mag_cfg),
                    );
                }
                eprintln!("  [conversation cleared]\n");
                continue;
            }

            if lower == "/mode" {
                eprintln!("  Mode: {mode_label} + learning");
                eprintln!("  History: {} tokens, {turn_count} turns", history_tokens.len());
                if !stateless {
                    eprintln!("  CMS: memory persists across turns (constant prompt size)");
                } else {
                    eprintln!("  No memory: full history re-sent each turn");
                }
                eprintln!();
                continue;
            }

            if lower == "/stats" {
                eprintln!("  Turns: {turn_count}");
                eprintln!("  History tokens: {}", history_tokens.len());
                eprintln!();
                continue;
            }

            // ── Build prompt tokens ──
            let user_turn = chatml_encode_turn(&tokenizer, "user", stripped);
            let assistant_start = chatml_encode_prompt(&tokenizer, "assistant");

            let prompt_tokens: Vec<usize> = if stateless {
                history_tokens.extend_from_slice(&user_turn);
                let mut prompt = history_tokens.clone();
                prompt.extend_from_slice(&assistant_start);
                if prompt.len() > seq_len {
                    prompt = prompt[prompt.len() - seq_len..].to_vec();
                }
                prompt
            } else {
                let mut prompt = user_turn.clone();
                prompt.extend_from_slice(&assistant_start);
                prompt
            };

            // Truncate to seq_len if needed (no padding — unified path handles variable-length)
            let ctx: Vec<usize> = if prompt_tokens.len() > seq_len {
                prompt_tokens[prompt_tokens.len() - seq_len..].to_vec()
            } else {
                prompt_tokens.clone()
            };

            let t0 = Instant::now();

            // ── generate() handles both LEARN and SPEAK ──────────
            // generate() forwards the prompt (memory updates), samples response tokens,
            // then runs deferred backward on the FULL sequence (prompt + generated).
            // No separate LEARN step needed — generate() learns from everything.
            let gen_result = generate(
                &mut gpu_params, &mag_cfg, &mut gpu_context,
                &mut adamw_state, &mut conductor,
                &cfg.build.optimizer,
                lr, max_grad_norm, d, v,
                &ctx, max_tokens, temperature, top_k,
                Some(IM_END as usize), // stop on end-of-turn
                &mut None, // no profiling
            );

            let elapsed = t0.elapsed().as_secs_f64();
            let response_text = decode_tokens(&tokenizer, &gen_result.tokens);

            // Accumulate history for stateless mode
            if stateless {
                let response_turn = chatml_encode_turn(&tokenizer, "assistant", response_text.trim());
                history_tokens.extend_from_slice(&response_turn);
            }

            turn_count += 1;
            let tps = gen_result.tokens.len() as f64 / elapsed.max(1e-9);

            println!("\x1b[1;32mAssistant:\x1b[0m {}", response_text.trim());
            eprintln!("  \x1b[2m[{} tokens, {tps:.0} tok/s, loss={:.4}, gnorm={:.4}, prompt={} tokens]\x1b[0m\n",
                gen_result.tokens.len(), gen_result.loss, gen_result.grad_norm, prompt_tokens.len());
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn load_test_tokenizer() -> Option<Tokenizer> {
        // Try to load a real tokenizer for testing; skip if not available
        let paths = [
            "python/data/tokenizers/tokenizer.json",
            "../python/data/tokenizers/tokenizer.json",
        ];
        for path in &paths {
            if Path::new(path).exists() {
                return Tokenizer::from_file(path).ok();
            }
        }
        None
    }

    #[test]
    fn chatml_special_tokens() {
        // Verify IM_START and IM_END constants match Python spec
        assert_eq!(IM_START, 0);
        assert_eq!(IM_END, 1);
    }

    #[test]
    fn chatml_encode_turn_structure() {
        if let Some(tok) = load_test_tokenizer() {
            let turn = chatml_encode_turn(&tok, "user", "hello");
            // Must start with IM_START
            assert_eq!(turn[0], IM_START as usize);
            // Must contain IM_END
            assert!(turn.contains(&(IM_END as usize)));
            // IM_END should be near the end (before the \n encoding)
            let im_end_pos = turn.iter().rposition(|&t| t == IM_END as usize).unwrap();
            assert!(im_end_pos >= turn.len() - 3); // IM_END + \n tokens at end
        }
    }

    #[test]
    fn chatml_encode_prompt_no_end_token() {
        if let Some(tok) = load_test_tokenizer() {
            let prompt = chatml_encode_prompt(&tok, "assistant");
            assert_eq!(prompt[0], IM_START as usize);
            // Should NOT contain IM_END (it's an incomplete turn)
            assert!(!prompt.contains(&(IM_END as usize)));
        }
    }
}
