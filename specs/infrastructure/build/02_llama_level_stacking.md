# Llama-3.2-1B Ad-hoc Level Stacking

```text
CONTRACT
  Purpose:    Initialize CMS levels with pre-trained Llama-3.2-1B MLP blocks
              per HOPE paper section 7.3. This converts a pre-trained transformer
              into a continual learner by placing its MLP blocks at different CMS
              frequencies, then running continual pre-training.
  Expects:    Llama-3.2-1B weights (HuggingFace), FineWeb-Edu tokenized with
              Llama 3 tokenizer (128K vocab), single RTX 3090 (24GB VRAM).
  Guarantees: CMS levels initialized with known-good MLP weights. No random
              initialization for memory levels -- every level starts from a
              block that already knows useful transformations.
  Cost:       ~1.15GB model weights (fp32 MLP subset) + embeddings (128K vocab
              ~1GB) + AdamW moments. Total VRAM ~10GB. Fits single 3090.
  Tokenizer:  Llama 3 (tiktoken, 128,256 vocab). Matches donor model. FineWeb-Edu
              must be re-tokenized. vocab_size=32000 config removed.
  Trade-off:  Requires d_model=2048 to match Llama dimensions (no projection).
              Larger model than our current 512-dim builds (~500M vs ~40M).
  Position:   specs/infrastructure/build/02_llama_level_stacking.md
  Source:     HOPE paper (2512.24695) section 7.3, Eq 71, Eq 72
```

## Paper Reference

HOPE section 7.3 "Ad-hoc Level Stacking: Initializing CMS with Pre-Trained Models":

Given a CMS with {MLP^(f_i)}_{i=1}^{k}, and a set of pre-trained MLP
blocks {MLP_pretrained_i}_{i=1}^{k}, use Equation 71 to update
{MLP^(f_i)} in different levels; use the trained parameters
of {MLP_pretrained_i} as the initial state of CMS blocks:

```text
MLP_0^(f_i) = MLP_pretrained_i
```

The key insight: setting eta_t^(l) toward 0 keeps updated memory close to
initial state (directly using pre-trained blocks without adaptation). Higher
eta allows the block to adapt to its level's context flow. This is the
mechanism that converts a static transformer into a continual learner.

The paper uses Llama3-8B and Llama-3B as backbone, applies section 7.3 to
place MLP blocks at different CMS frequencies, then runs continual pre-training
with 15B tokens (section 9.1). We adapt this to Llama-3.2-1B.

### Traced Equations

- **Eq 71** (hope_equations/eq-071-arch-variant2): CMS conditional parameter
  update. theta^(f_l) updated every C^(f) steps by accumulating optimizer
  error over the chunk.
- **Eq 72** (hope_equations/eq-072-arch-variant3): Nested CMS initialization.
  Initial state of MLP block at level s+1 is meta-learned in level s.
- **Eq 28** (hope_equations/eq-028-maml-init): MAML initialization (knowledge
  transfer via init). Higher-level block learns the best initial value for
  the lower-level problem over all possible contexts.

## Architectural Mapping

### Llama-3.2-1B Architecture

| Parameter | Value |
|---|---|
| hidden_size (d_model) | 2048 |
| intermediate_size | 8192 |
| num_hidden_layers | 16 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 (GQA 4:1) |
| head_dim | 64 |
| vocab_size | 128,256 |
| activation | SiLU (SwiGLU) |
| mlp_bias | false |
| tie_word_embeddings | true |

Each Llama MLP block has three weight matrices (no bias):
- `gate_proj`: (8192, 2048) -- SiLU-gated path
- `up_proj`: (8192, 2048) -- linear path
- `down_proj`: (2048, 8192) -- projection back

Forward: `output = down_proj(silu(gate_proj(x)) * up_proj(x))`  <!-- Llama-3.2-1B architecture (Meta AI, 2024); SwiGLU activation from Shazeer 2020 -->

### The Transplant: What Goes Where

The paper says CMS levels ARE MLP blocks. Our current implementation uses
TitansLMM (matrix memory d x d with L2 attentional bias). These are
fundamentally different structures:

- **TitansLMM**: M in R^{d x d}, update via outer product, query via M @ q_t
- **Llama MLP**: Three matrices (gate/up/down), SwiGLU nonlinearity

For the transplant to follow the paper, the CMS levels must use MLP-based
memory rules. In the HOPE framing, a transformer's MLP block IS already a
memory system at frequency 0 (updated during pre-training, frozen during
inference). The transplant simply changes the frequency -- some blocks now
update at f=1, f=8, f=64, f=512 instead of all at f=0.

**Concrete mapping per CMS level**:

```text
Level 0 (f=1, every chunk):    Llama layer  0 MLP
Level 1 (f=8, every 8 chunks): Llama layer  5 MLP
Level 2 (f=64):                Llama layer 10 MLP
Level 3 (f=512):               Llama layer 15 MLP
```

Why these layers? Spread across the 16-layer stack to get diverse
representations. L0 = earliest (most token-level), L15 = latest (most
abstract). Aligns with CMS frequency semantics: fast level gets low-level
features, slow level gets high-level abstractions.

**What we DO transplant**:
- 4 MLP blocks (gate_proj, up_proj, down_proj) as CMS levels 0-3
- Cast bf16 to fp32 (inner-loop operations MUST be fp32)

**What we DO NOT transplant**:
- Attention weights (our SWA is architecturally different from Llama's GQA)
- Embeddings (fresh init; but we USE Llama's 128K vocab -- see Tokenizer below)
- Layer norms (different positions in our architecture)
- Positional encoding (we use SWA's implicit position, not RoPE)

## Model Assembly

The full model structure after transplant:

```text
Input tokens (Llama 3 tokenizer, 128K vocab)
    |
Embeddings (fresh init, d=2048, vocab=128,256)
    |
[For each token position t:]
    |
SWA (sliding window attention, fresh init)
    |
CMS Level 0 MLP (from Llama layer 0, updates every chunk)
CMS Level 1 MLP (from Llama layer 5, updates every 8 chunks)
CMS Level 2 MLP (from Llama layer 10, updates every 64 chunks)
CMS Level 3 MLP (from Llama layer 15, updates every 512 chunks)
    |
Output = 1/sqrt(k) * sum(level_outputs)  [CMS normalization; HOPE 2512.24695 §4]
    |
Unembed logits
```

## Rust Implementation Requirements

### New Memory Rule: SwiGluMlp

```rust
pub struct SwiGluMlp {
    pub intermediate_size: usize,  // 8192 for Llama-3.2-1B
}

// Parameters for one CMS level (one MLP block):
// gate_proj: [intermediate_size x d_model] = [8192 x 2048]
// up_proj:   [intermediate_size x d_model] = [8192 x 2048]
// down_proj: [d_model x intermediate_size] = [2048 x 8192]
// Total per level: 3 x 8192 x 2048 = 50,331,648 f32 values (~192MB)

impl MemoryRule for SwiGluMlp {
    fn step(&self, params, embedded, seq_len, d, prev_m) -> (Vec<f32>, Cache) {
        // For each token t in [0..seq_len]:
        //   gate = silu(embedded[t] @ gate_proj.T)
        //   up   = embedded[t] @ up_proj.T
        //   y[t] = (gate * up) @ down_proj.T
        // No inner-loop state (M) -- MLP is stateless per-token.
        // Gradients flow to gate_proj, up_proj, down_proj via tape.
    }

    fn step_backward(&self, params, cache, d_y, embedded) -> (Grads, Vec<f32>) {
        // Standard MLP backward through SwiGLU:
        // d_down_proj, d_gate_proj, d_up_proj, d_embedded
    }
}
```

### MemoryLevelParams Extension

Current `MemoryLevelParams` stores `w_k_mem`, `w_v_mem`, `b_alpha`, `b_theta`,
`b_eta` (matrix memory rule params). For SwiGLU MLP, we need:

```rust
pub struct MlpLevelParams {
    pub gate_proj: Vec<f32>,  // [intermediate x d]
    pub up_proj: Vec<f32>,    // [intermediate x d]
    pub down_proj: Vec<f32>,  // [d x intermediate]
}
```

Recommended: Generalize `MemoryLevelParams` to hold either matrix-rule params
or MLP-rule params as an enum variant. The MIRAS trait system already abstracts
over memory rule types -- the params just need to match the rule.

### MAGConfig Changes

```rust
pub enum MemoryRuleKind {
    DeltaRule,
    TitansLMM,
    Hebbian,
    // ... existing variants ...
    SwiGluMlp,  // NEW
}

// New field on MAGConfig:
pub intermediate_size: usize,  // 0 for matrix rules, 8192 for SwiGLU
```

## Weight Extraction (Python, one-time)

`donor_layers` is Python-configurable in the build config (see Build Config below).
The extraction script reads this list and maps CMS level index → Llama layer index.
No Rust changes needed when `donor_layers` changes -- Rust receives weight tensors
and does not know their origin layer.

```python
from transformers import AutoModelForCausalLM
import torch, json, sys

# donor_layers is read from build config JSON:
#   "donor_layers": [0, 5, 10, 15]
# Default: evenly spaced across 16-layer stack.
# Alternatives:
#   [0, 1, 2, 3]   -- first four (most token-level features)
#   [12, 13, 14, 15] -- last four (most abstract features)
#   [0, 5, 10, 15] -- evenly spaced (DEFAULT: diverse representations)

cfg = json.load(open(sys.argv[1]))
donor_layers = cfg.get("donor_layers", [0, 5, 10, 15])

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B",   # ungated mirror (no license gate)
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

extracted = {}
for cms_level, llama_layer in enumerate(donor_layers):
    mlp = model.model.layers[llama_layer].mlp
    extracted[f"level_{cms_level}"] = {
        "gate_proj": mlp.gate_proj.weight.float(),  # bf16 -> fp32 (inner loop req)
        "up_proj":   mlp.up_proj.weight.float(),
        "down_proj": mlp.down_proj.weight.float(),
    }

torch.save(extracted, "checkpoints/llama_mlp_donor.pt")
# ~768MB on disk (4 levels x 3 matrices x 8192 x 2048 x 4 bytes)
```

### Tokenizer Setup (one-time)

```python
# Re-tokenize FineWeb-Edu with Llama 3 tokenizer.
# Data pipeline uses AutoTokenizer from the donor model.
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
# vocab_size = 128,256 (tiktoken-based Llama 3 tokenizer)
```

The tokenized data is stored separately from the 32K BPE data used by the 60M
model -- these are two independent datasets on disk.

## Checkpoint Loading

Python function to load Llama MLP weights into NL_Hecate params:

```python
def load_llama_donor(donor_path, params, cfg, k):
    """Load extracted Llama MLP weights into MAGParams."""
    donor = torch.load(donor_path, weights_only=True)

    for level in range(k):
        level_data = donor[f"level_{level}"]
        params.set_level_mlp(
            cfg,
            level,
            level_data["gate_proj"].float().numpy().flatten().tolist(),
            level_data["up_proj"].float().numpy().flatten().tolist(),
            level_data["down_proj"].float().numpy().flatten().tolist(),
        )
```

## Build Config

```json
{
  "description": "Llama-3.2-1B level stacking -- continual pre-training",
  "model": {
    "d_model": 2048,
    "num_heads": 32,
    "seq_len": 512,
    "window_size": 512,
    "vocab_size": 128256,
    "k": 4,
    "chunk_sizes": [1, 8, 64, 512],
    "memory_rule": "swiglu_mlp",
    "composition": "mag",
    "intermediate_size": 8192
  },
  "build": {
    "lr": 0.0003,
    "steps": 50000,
    "optimizer": "adamw_gpu",
    "warmup_steps": 500,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "save_path": "checkpoints/llama_stacking.json",
    "save_every": 5000,
    "log_every": 10,
    "eval_every": 5000,
    "eval_max_chunks": 50
  },
  "data": {
    "path": "data/fineweb_edu_llama3",
    "format": "sharegpt",
    "tokenizer": "unsloth/Llama-3.2-1B"
  },
  "donor_layers": [0, 5, 10, 15],
  "donor_weights": "checkpoints/llama_mlp_donor.pt"
}
```

LR choice: Lower than our 60M runs (0.0003 vs 0.0006) because we are
fine-tuning known-good weights, not training from scratch. The paper uses 15B
tokens of continual pre-training; we scale proportionally to our data budget.

## VRAM Budget (Single RTX 3090, 24GB)

| Component | Size (fp32) | Notes |
|---|---|---|
| Embeddings (128K x 2048) | ~1,000 MB | Fresh init, tied embed/unembed |
| SWA (Q/K/V/O) | 64 MB | 32 heads x 4 x 2048 x 64 |
| 4 MLP levels | 768 MB | 4 x 3 x 8192 x 2048 x 4B |
| Gate biases (k=4) | < 1 MB | b_alpha, b_theta, b_eta |
| Unembed (tied) | 0 MB | Shares embedding weight |
| **Model params total** | **~1.8 GB** | Embedding dominates at 128K vocab |
| AdamW moments (2x) | ~3.8 GB | m + v buffers |
| Activations + gradients | ~3-4 GB | seq_len=512, batch=1 |
| **Total VRAM** | **~9-10 GB** | Fits single 3090 (24GB) |

## CUDA Kernel Requirements

### Why CUDA is Required

At d=2048, intermediate=8192, seq_len=512, each forward pass through one MLP level:
- gate_proj matmul: 512 × 2048 × 8192 = ~8.6G FLOPs
- up_proj matmul: same = ~8.6G FLOPs
- SiLU + elementwise multiply: 512 × 8192 = ~4M ops (negligible)
- down_proj matmul: 512 × 8192 × 2048 = ~8.6G FLOPs
- Per level: ~25.8G FLOPs. Four levels: ~103G FLOPs per chunk.

Backward is ~2x forward = ~206G FLOPs per chunk. On the Rust CPU reference path
this would take seconds per chunk -- unusable for 50K+ step training. cuBLAS
matmuls + thin CUDA fusion kernels for the SiLU gate make this practical.

### Kernel Design: cuBLAS + Thin Fusion

The heavy lifting (gate/up/down projections) uses cuBLAS sgemm -- already
optimized for large matmuls on RTX 3090. The only custom CUDA kernels cover the
elementwise SiLU gating step, which cuBLAS cannot express:

**Forward fusion** (`core/kernels/swiglu_forward.cu`):
```cuda
// Fuses: output[i] = silu(gate[i]) * up[i]
// Caches sigmoid for backward (avoids recomputing expf).
__global__ void swiglu_fuse_forward(
    const float* gate, const float* up,
    float* fused, float* gate_cache, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        fused[i]      = gate[i] * sig * up[i];
        gate_cache[i] = sig;
    }
}
```

**Backward fusion** (`core/kernels/swiglu_backward.cu`):
```cuda
// d_silu/dx = sigmoid(x) * (1 + x*(1 - sigmoid(x)))
__global__ void swiglu_fuse_backward(
    const float* d_fused, const float* gate, const float* up,
    const float* gate_cache, float* d_gate, float* d_up, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sig     = gate_cache[i];
        float silu    = gate[i] * sig;
        float dsilu   = sig * (1.0f + gate[i] * (1.0f - sig));
        d_up[i]   = d_fused[i] * silu;
        d_gate[i] = d_fused[i] * up[i] * dsilu;
    }
}
```

### Forward/Backward Orchestration (per level, all tokens batched)

**Forward** -- X is [seq_len × d_model]:
```text
gate_out  = cuBLAS_sgemm(X, gate_proj.T)          // [seq_len × intermediate]
up_out    = cuBLAS_sgemm(X, up_proj.T)             // [seq_len × intermediate]
fused     = swiglu_fuse_forward(gate_out, up_out)  // [seq_len × intermediate]
Y         = cuBLAS_sgemm(fused, down_proj.T)        // [seq_len × d_model]
```

**Backward** -- given d_Y [seq_len × d_model]:
```text
d_fused      = cuBLAS_sgemm(d_Y, down_proj)        // [seq_len × intermediate]
d_down_proj  = cuBLAS_sgemm(fused.T, d_Y)          // [intermediate × d_model]
d_gate, d_up = swiglu_fuse_backward(d_fused, ...)
d_gate_proj  = cuBLAS_sgemm(d_gate.T, X)           // [intermediate × d_model]
d_up_proj    = cuBLAS_sgemm(d_up.T, X)             // [intermediate × d_model]
d_X          = cuBLAS_sgemm(d_gate, gate_proj)      // [seq_len × d_model]
d_X         += cuBLAS_sgemm(d_up, up_proj)          // accumulate
```

### Dimension Constraints

Unlike existing Titans/Delta/Hebbian kernels (head_dim ≤ 32 warp constraint),
SwiGLU kernels have **no hard dimension limits**:
- cuBLAS handles arbitrary matrix sizes natively
- Fusion kernels are pure elementwise: grid = ceil(N/256), block = 256
- No warp reduction, no shared memory, no atomicAdd contention

### New Files

- `core/kernels/swiglu_forward.cu` -- forward fusion kernel
- `core/kernels/swiglu_backward.cu` -- backward fusion kernel
- `core/src/swiglu_mlp.rs` -- SwiGluMlp rule + cuBLAS orchestration + OpaqueVjp

## Implementation Sequence

1. **Extract Llama MLP weights** -- Python script, run once, save to `.pt` file
2. **Add SwiGLU fusion CUDA kernels** -- `swiglu_forward.cu` + `swiglu_backward.cu`
3. **Add cuBLAS orchestration** -- Rust: link cuBLAS, batched sgemm wrappers
4. **Add `SwiGluMlp` memory rule** -- Rust: forward + backward + OpaqueVjp
5. **Add `MlpLevelParams`** -- Rust: parameter storage for MLP-based levels
6. **Wire `MemoryRuleKind::SwiGluMlp`** -- Rust: dispatch in mag.rs
7. **Add `intermediate_size` to MAGConfig** -- Rust + PyO3
8. **Add `load_llama_donor()` to Python** -- Load extracted weights into params
9. **Create config JSON** -- d=2048, k=4, memory_rule=swiglu_mlp
10. **Smoke test** -- Forward pass with Llama MLP weights, verify loss is finite
11. **Launch continual pre-training** -- GPU1, FineWeb-Edu

## Falsifiable Predictions

1. **Initial loss below random init**: The Llama MLP weights should produce
   meaningful token predictions even before any NL training, because the
   MLP blocks already encode useful transformations. Expected: initial loss
   below 8.0 (vs ~10.0 for random init at this scale).

2. **Faster convergence**: The model should reach loss 6.0 in fewer steps
   than a randomly initialized d=2048 model, because the MLP weights
   provide a warm start.

3. **Cross-exposure adaptation (Probe 2) positive within 5K steps**:
   Unlike our current 60M k=4 model where higher levels barely learn, the
   Llama-initialized levels should show parameter transfer from run 1 to
   run 2 early in training.

4. **Level theta gates differentiate**: With known-good MLP weights, the
   model should learn different theta values per level (fast vs slow
   learning rates) rather than all collapsing to the same value.

## Design Decisions

### Resolved at Spec Time

1. **Which Llama layers to use**: Python-configurable via `donor_layers` in build
   config JSON. Default `[0, 5, 10, 15]` (evenly spaced, diverse representations).
   No Rust changes needed when this is adjusted -- Rust just receives weight tensors.

2. **Tokenizer**: Llama 3 (tiktoken, 128,256 vocab) for this experiment to match
   the donor model. Python-configurable via `"tokenizer"` and `"vocab_size"` in
   the build config; different experiments can swap to a smaller vocabulary
   (e.g., 32K Mistral tokenizer saves ~750MB embedding VRAM) with no Rust changes.
   FineWeb-Edu must be re-tokenized before the run. Data at `data/fineweb_edu_llama3/`.

### Deferred to Implementation (Python-tier, no Rust changes)

1. **Donor weight freeze strategy**: All options are Python optimizer logic --
   Rust computes gradients, Python decides what to do with them. Deferred until
   we see what GPU0 (Option 2 theta-floor run) shows about level learning dynamics.

   Available knobs (can be combined):
   - `"donor_freeze_steps": N` -- hard freeze donor params for first N steps
   - `"donor_lr_warmup_steps": N` -- soft LR ramp from 0 to full over N steps
   - `"donor_warmup_levels": [0,1]` -- apply warmup only to specified levels
   - No key needed -- CMS frequency provides implicit curriculum (Level 3 gets
     ~20 updates in 10K steps vs Level 0's ~10K; slow levels are naturally protected)

   Default for this experiment: no explicit freeze. Instrument per-level weight
   drift (`||W_t - W_0||`) as a learning probe and revisit if Level 0 destabilizes.

## Interaction with Existing Specs

- **CS-10 (No mode distinction)**: Unchanged -- MLP blocks update during
  forward+backward like any memory rule. No freeze/thaw switches.
- **CS-18 (Forward pass IS the API)**: SwiGLU forward is the only path.
- **CS-32 (Observe then advance)**: Conductor pulse still gates which
  levels receive outer-loop updates -- this IS the implicit curriculum.
- **CS-39 (Clamp learnable decay)**: Theta floor/ceil from Option 2
  applies to SwiGluMlp levels too.
- **Eval methodology**: Learning probes work unchanged -- generate_learning()
  will update MLP weights during generation.
