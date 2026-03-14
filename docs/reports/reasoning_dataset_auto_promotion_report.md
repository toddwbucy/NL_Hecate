# Auto-Promotion Pyramid on Reasoning Dataset (GPU0)

**Date**: 2026-03-07
**Status**: IN PROGRESS
**GPU**: 0 (NVIDIA RTX A6000, 48GB)

## Hypothesis

NL models with persistent memory M can learn structured reasoning directly from raw text without the scaffolding transformers require. Specifically:

1. **No role/turn markup needed**: Transformers need `<|user|>`, `<|assistant|>`, turn delimiters because every forward pass re-reads the entire conversation from scratch. NL's memory M carries conversation state persistently — the model doesn't need formatting cues to know "who said what."

2. **No externalized chain-of-thought needed**: Transformer CoT is a structural hack — models have no persistent internal state, so reasoning must be externalized to the output token stream. Research confirms CoT is frequently unfaithful (13% post-hoc rationalization in GPT-4o-mini, Oxford 2025). In NL, M IS the reasoning substrate — continuous, high-bandwidth (d×d per level), persistent across steps. Training on Q→A where A requires multi-step reasoning forces the model to use M as an internal scratchpad.

3. **Mixed reasoning data from step 0**: The hypothesis is that NL models don't need the conventional easy→hard curriculum. With M available for internal deliberation from the first step, the model should be able to engage with math, logic, and Socratic dialogue immediately.

This experiment tests all three hypotheses simultaneously by training on a reasoning-heavy dataset with all markup stripped, using the same auto-promotion curriculum that proved successful on Dolmino 100B.

## Dataset: Reasoning Mix (277M tokens)

All sources preprocessed to strip transformer scaffolding: `<think>` blocks, role markers (`<|user|>`, `### Assistant:`), turn delimiters, DeepSeek-specific tags. Text concatenated as continuous stream separated by EOT tokens.

| Source | Tokens | Share | Description |
|---|---|---|---|
| am-r1 (DeepSeek-R1) | 75M | 27% | Reasoning traces with think blocks stripped. Only Q→A remains — the reasoning gap forces M to internalize the steps. |
| big-deepseek | 60M | 22% | 677K additional DeepSeek reasoning traces (parquet), same stripping. |
| general-thought-430k | 45M | 16% | Diverse Q&A with model reasoning. Questions span medicine, physics, math, coding. |
| SocraTeach | 22M | 8% | Multi-turn Socratic math tutoring. Teacher guides student through problems via leading questions, never giving answers directly. Flattened to continuous text — M must track the pedagogical state. |
| cot_collection | 45M | 16% | 1.8M CoT rationale chains (source→rationale→answer). Diverse tasks: NLI, QA, summarization. |
| flan_cot | 30M | 11% | NLI/reasoning with chain-of-thought labels from Flan. |

**Design choices:**
- SocraTeach weighted at 15% budget but exhausted at 22M (only 31K dialogues available). The Socratic signal is high-density — every turn requires reasoning and builds on prior context — so even 8% of the final mix provides disproportionate value.
- Think blocks stripped from DeepSeek traces intentionally: we want the model to learn the mapping Q→A, not Q→externalized_reasoning→A. The "reasoning gap" between Q and A is the learning signal for M.
- No loss masking — the model learns from everything (both questions and answers). Unlike SFT where only assistant turns are learnable, pre-training on raw text lets M build representations from the full context.

## Configuration

| Parameter | Value |
|---|---|
| Composition | MAG (parallel gating) |
| Memory rule | Titans LMM |
| d_model | 512 |
| num_heads | 8 |
| seq_len | 512 |
| window_size | 512 |
| k (initial) | 1 |
| target_k | 4 |
| chunk_sizes | [1] → [1,8] → [1,8,64] → [1,8,64,512] |
| Momentum | EMA |
| M-norm clamp | 100.0 per level (extended at promotion) |
| Data | Reasoning mix (263M train tokens, sharegpt BPE) |
| LR | 0.0006 (linear warmup 500 steps, no decay) |
| Optimizer | AdamW (b1=0.9, b2=0.999, wd=0.1) |
| Grad clip | max_norm=1.0 |
| Steps | 100,000 max |
| Promotion cooldown | 2,000 steps between promotions |
| Stability window | 50 ratio samples |
| Stability streak | 50 consecutive low-stdev samples |
| Stability threshold | 0.025 trimmed stdev |
| **log_every** | **8** (aligned to L1 chunk_size) |
| **eval_every** | **4096** (LCM of [1,8,64,512], all levels fire) |
| **save_every** | **4096** (aligned to eval) |
| **tape_every** | **4096** |

**Logging alignment rationale**: With chunk_sizes [1, 8, 64, 512], evaluating at multiples of 512 ensures all 4 levels fire during eval. Using 4096 = 8×512 reduces eval overhead to ~1% of wall time while still capturing every level. log_every=8 catches every L1 fire; every 8th log step (64) catches L2; every 64th (512) catches L3.

## Comparison with Dolmino Experiment

| Aspect | Dolmino (GPU1) | Reasoning (GPU0) |
|---|---|---|
| Dataset | Dolmino 100B (general web, 950M tokens) | Reasoning mix (277M tokens, structured) |
| Data difficulty | Mixed — easy boilerplate + hard passages | Uniformly challenging — math, logic, Socratic |
| Markup | Already stripped (pre-tokenized BPE) | Explicitly stripped (think blocks, roles, turns) |
| Hypothesis | Auto-promotion works | Auto-promotion works + M handles reasoning |
| Log alignment | log_every=10, eval_every=500 (misaligned) | log_every=8, eval_every=4096 (aligned) |
| Expected promotions | ~step 4580, ~5690, pending k=4 | TBD — may differ due to data characteristics |

**Key comparison question**: Does the reasoning dataset produce different promotion timing? The Dolmino corpus has mixed difficulty (gnorm bounces 0.2-8.4), which inflated stdev and delayed the first promotion to step 4580. The reasoning dataset has more uniformly challenging content — this could mean either faster saturation (if M converges quickly on structured patterns) or slower (if the difficulty sustains gradient signal longer).

## Early Observations

### Warmup Phase (steps 0-500)
- Dead-level detector produced false positive warnings during warmup (per-level gnorms near-zero when lr < 0.001). Not fatal — warnings only.
- Loss dropped from 10.37 (random init, near log(32000)=10.37) to ~6.6 by step 280 (lr=0.000336)
- Per-level gnorms became nonzero around step 280

## Results

_To be completed when run finishes. Will include:_
- Loss trajectory comparison with Dolmino run
- Promotion timing and trigger values
- Level activity at k=4 — are higher levels more active on structured reasoning data?
- Learning probe trajectories — does M show stronger within-generation improvement on reasoning text?
- Cross-exposure improvement — does the model retain reasoning patterns across chunks?
- Qualitative: sample generations at k=1 vs k=4 on math/logic prompts
- Comparison of saturation dynamics: does uniformly hard data change the gnorm bounce pattern?

## Connection to Internal Monologue Research Direction

This experiment is a prerequisite for the M-state interpretability research direction (documented in HADES: `m_state_interpretability_vs_cot_2026_03_07`). If the model successfully learns to produce correct multi-step answers from Q→A training (with think blocks stripped), it demonstrates that M functions as an internal reasoning substrate — supporting the hypothesis that NL models don't need externalized CoT. Future work would add M-state probing (snapshots, trajectory visualization, level attribution) to verify that M's internal state actually encodes the intermediate reasoning steps.
