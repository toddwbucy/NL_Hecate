---
CONTRACT
  Purpose:   Ablate head_dim on k=4 chained SmolLM d=768 6-block config
  Expects:   Completed baseline run (k4_chain_smollm_d768_6b, num_heads=12, head_dim=64)
  Guarantees: Matched config except num_heads=24 (head_dim=32); same seed, data, steps
  Cost:      ~48h A6000 GPU time (200K steps at ~128 tok/s)
  Trade-off: 2× heads halves per-head M capacity (32×32 vs 64×64) but doubles specialization slots
  Position:  Investigates whether transformer-inherited head_dim=64 is optimal for NLM memory
  Source:    Empirical — no paper prescribes head_dim for outer-product memory rules
CONTRACT
---

# Spec 41: head_dim Ablation — 24 Heads (hd=32) vs 12 Heads (hd=64)

## Hypothesis

Transformer attention optimizes head_dim=64 for dot-product similarity lookups.
NLM memory matrices store associative mappings via outer products (M ← M + η·eₜ⊗kₜ).
The capacity/specialization tradeoff differs:

- **head_dim=64 (12 heads)**: Each head stores a 64×64 = 4096-element memory matrix.
  Fewer, deeper drawers — each head covers a wider feature subspace.
- **head_dim=32 (24 heads)**: Each head stores a 32×32 = 1024-element memory matrix.
  More, shallower drawers — each head specializes on a narrower subspace.

Total parameter count is identical: 12 × 64² = 24 × 32² = 49,152 M elements per level.

## Design

### Controlled variables
- d_model: 768
- k: 4, chunk_sizes: [1, 8, 64, 512]
- n_blocks: 6
- memory_rule: titans, composition: mag, hope_variant: chained
- optimizer: adamw_gpu_stacked, lr: 3e-4, warmup: 500, weight_decay: 0.1
- tape_strategies: [exact, proxy, proxy, proxy]
- m_norm_max: [100, 100, 100, 100]
- seed: 42, data: smollm_corpus, batch_size: 1, seq_len: 512

### Independent variable
- **Baseline**: num_heads=12, head_dim=64 (run: k4_chain_smollm_d768_6b)
- **Treatment**: num_heads=24, head_dim=32 (run: k4_chain_smollm_d768_6b_hd32)

### Metrics
1. **Loss curve**: primary signal — does hd=32 converge faster, to a lower floor, or neither?
2. **Per-head M norms** (spec 50): do 24 heads show more variance (specialization) than 12?
3. **CMS level evolution** (spec 49 sidecar): any interaction between head count and level dynamics?
4. **Throughput**: confirm tok/s roughly equivalent (same total FLOPs)

### Expected outcome
Three plausible results:
- **hd=32 wins**: More heads → better specialization for outer-product memory. Would motivate further ablation at hd=16 (48 heads).
- **hd=64 wins**: Deeper per-head capacity matters more than specialization count. Transformer default is coincidentally correct.
- **No significant difference**: Head dim is not a first-order hyperparameter for NLM. Other knobs matter more.

## Run Configuration

Config: `python/configs/k4_chain_smollm_d768_6b_hd32.json`
Run dir: `python/runs/k4_chain_smollm_d768_6b_hd32/`
GPU: 0 (A6000)
Steps: 200,000 (matching baseline target)
Resume: from scratch (no checkpoint transfer — head dim change makes weights incompatible)
