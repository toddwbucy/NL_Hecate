# Infrastructure Code Smells (CS-39 through CS-50)

```
CONTRACT
  Purpose:    These smells were discovered during Track A implementation
              (HOPE on SmolLM3-3B) and through analysis of the NL papers'
              implications for GPU computation. They are empirical lessons,
              not theoretical axioms. Each one caused a real bug or
              performance problem.
  Expects:    All infrastructure code: AD system, GPU management,
              distribution, compilation, runtime state management.
  Guarantees: Known pitfalls are documented and enforceable.
              Each smell has a specific test or enforcement mechanism.
  Cost:       Zero runtime cost — infrastructure design constraints only.
  Trade-off:  Some GPU optimization techniques (checkpointing, DDP) are off-limits.
  Position:   specs/constraints/code_smells/06_infrastructure.md
  Source:     Track A experience; nl_code_smells CS-39 through CS-47;
              committee review cycle (CS-48);
              FineWeb-Edu k=4 NaN crash (CS-49)
```

## CS-39: Learnable Decay Parameters Must Be Clamped [CRITICAL]

```
SMELL: self.lambda = nn.Parameter(torch.tensor(0.99))
       // lambda can go negative, >1, or NaN during optimization
WHY:   Retention mechanisms use learnable decay parameters (lambda).
       If lambda > 1: memory GROWS instead of decaying (divergence).
       If lambda < 0: memory oscillates (instability).
       If lambda = 0: all memory is instantly forgotten.
       Unclamped learnable decay caused NaN losses in Track B (DGD stability).

FIX:   Always clamp: lambda = sigmoid(raw_lambda) * (max - min) + min
       Typical range: [0.9, 0.999] for outer-loop decay
       Or: lambda = raw_lambda.clamp(min=0.9, max=0.999)
       The raw parameter is unconstrained; the USED value is clamped.
SEVERITY: CRITICAL — unclamped decay = guaranteed divergence
TRACE: Track B DGD stability issue (memory_init_std, alpha clamping)
```

## CS-40: Differentiation Is Opt-In, Not Opt-Out [CRITICAL]

```
SMELL: #[autodiff]  // on everything by default
       #[no_autodiff]  // only on things we explicitly exclude
WHY:   In PyTorch, autograd is opt-OUT (everything tracked unless you
       detach or use no_grad). This is dangerous for NL because:
       - Inner-loop operations look like they should be tracked but shouldn't
         be (they provide their own backward via #[custom_vjp])
       - Debug/logging operations get accidentally included in the AD graph
       - Memory grows unboundedly if everything is tracked

       NL_Hecate uses opt-IN: nothing is differentiated unless explicitly
       annotated with #[autodiff] or #[custom_vjp].

FIX:   Three explicit annotations:
       #[autodiff]     — Enzyme traces (Rust code in gradient path)
       #[custom_vjp]   — provides own backward (CUDA kernel pairs)
       #[no_autodiff]  — severed from gradient chain (debug only)
       Unannotated functions are NOT in the gradient path by default.
SEVERITY: CRITICAL — opt-out AD led to the #[no_autodiff] bug in contract v0.1.0
TRACE: Contract v0.2.0 Section 4; differentiation spec
```

## CS-41: GPU Utilization != Throughput

```
SMELL: println!("GPU utilization: 98% — good!")
WHY:   CMS creates asymmetric workloads. At most steps, only level 0
       is active — the GPU processes one block. At step 512, all 4 blocks
       activate — the GPU is 4x busier.
       "98% utilization" can mean the GPU is busy doing NOTHING USEFUL
       (waiting for allreduce, running empty kernels for frozen levels).
       Throughput (tokens/sec) is the real metric, not utilization (% busy).

FIX:   Report tokens/sec, not GPU utilization.
       Report both average and worst-case (all-levels-active step).
SEVERITY: Warning — misleading metric, not a bug
TRACE: CS-43 (DDP inflates); CS-45 (can't fill high-end GPUs)
```

## CS-42: Gradient Checkpointing Hurts NL

```
SMELL: model = torch.utils.checkpoint(model)  // save memory via recomputation
WHY:   Gradient checkpointing recomputes activations in the backward pass.
       For NL, this means RE-RUNNING the inner loop. But the inner loop
       is STATEFUL — recomputation gives different results if random state
       (dropout, initialization) differs between forward and recomputation.
       Even with fixed seeds, the recomputed inner loop may diverge from
       the original due to floating-point non-associativity.
       Result: non-reproducible gradients (CS-47).

FIX:   Don't use gradient checkpointing. Instead:
       - CMS itself saves memory (frozen levels = zero activations)
       - Smaller chunk sizes (fewer tokens in flight)
       - Mixed precision (bf16 activations, fp32 accumulation)
       If absolutely needed: make inner loop deterministic (fixed seed per chunk).
SEVERITY: Warning — performance trap, not crash
TRACE: CS-47 (reproducibility); state lifecycle spec
```

## CS-43: DDP Inflates Reported Throughput

```
SMELL: throughput = total_tokens / elapsed_time  // across all ranks
WHY:   DDP reports aggregate throughput across all GPUs.
       With 2 GPUs: "21,000 tokens/sec" sounds great.
       Per-GPU: 10,500 tokens/sec. With overhead: ~9,500 useful tokens/sec.
       DDP hides the communication cost in the aggregate number.
       For NL, CMS's asymmetric allreduce makes this worse — the rare
       all-levels-active steps dominate wall-clock time.

FIX:   Report per-GPU throughput. Include communication overhead.
       Report effective throughput (tokens that contribute to learning).
SEVERITY: Warning — misleading metric
TRACE: Track A measurement (DDP throughput vs single-GPU)
```

## CS-44: Optimization Polarity Flips Per Hardware

```
SMELL: // "bf16 is always faster than fp32"
       // "Larger batch is always faster"
WHY:   NL's workload is unusual — many small matrix operations (inner loop
       per token) rather than few large ones (conventional batch matmul).
       On A6000 (Ampere): bf16 tensor cores help for projections but
       inner-loop operations may be memory-bandwidth-bound, not compute-bound.
       What's faster on A100 may be slower on A6000 and vice versa.
       Batch size effects flip: larger batch helps conventional models
       (better GPU utilization) but hurts NL (more inner-loop steps in flight).

FIX:   Profile on TARGET hardware before choosing precision/batch settings.
       Don't assume optimizations transfer between GPU architectures.
SEVERITY: Warning — performance trap
TRACE: Track A experience (bf16 vs fp32 on A6000)
```

## CS-45: NL Cannot Fill High-End GPUs

```
SMELL: // "We need H100s for this workload"
WHY:   NL's inner loop is inherently sequential per token within a chunk.
       Chunk-wise parallelism helps but chunks are small (64-512 tokens).
       An H100 has massive parallelism (16,896 CUDA cores) that NL cannot
       saturate — the inner loop doesn't have enough independent work.
       A6000 (10,752 cores) may be MORE efficient for NL than H100
       because NL wastes fewer idle cores.

FIX:   Right-size the hardware. Mid-range GPUs may outperform high-end
       for NL workloads (better $/token). The dispatch layer allows
       optimizing kernels for the target hardware regardless.
SEVERITY: Warning — cost trap
TRACE: NL computation pattern; Track A experience (A6000 vs hypothetical H100)
```

## CS-46: Graph Tracing Cannot Trace NL Inner Loops

```
SMELL: compiled_model = torch.compile(model)
       // Expects graph breaks at every inner-loop step
WHY:   Graph tracing (torch.compile, TorchScript, JAX jit) captures a
       static computation graph by running the model once.
       NL's inner loop MUTATES state at every token — the graph changes
       at every step. Tracing sees graph breaks everywhere.
       Result: no compilation benefit, possible slowdown from tracing overhead.

FIX:   Don't use graph tracing for NL models.
       Use the two-domain compilation strategy from compilation spec:
       - Rust code: compiled by rustc + Enzyme (static)
       - CUDA kernels: pre-compiled, dispatched at runtime
       - No graph tracing needed
SEVERITY: Info — this is expected, not a bug
TRACE: Compilation spec; contract v0.2.0 "The Three Layers"
```

## CS-47: In-Place Modification Destroys Reproducibility [CRITICAL]

```
SMELL: self.memory.add_(gradient, alpha=-self.eta)  // in-place
WHY:   In-place modification of tensors that are used in multiple places
       creates order-dependent computation. If tensor A is modified in-place
       while tensor B still holds a reference to the same storage,
       B's value changes silently. This makes builds non-reproducible
       (same data, different random order → different result).

       In NL, this is especially dangerous because:
       - Inner-loop memory is modified per token (many in-place ops)
       - Outer-loop params reference memory for gradient computation
       - If inner-loop modifies memory in-place, Enzyme sees wrong values

FIX:   Never modify shared state in place. Produce a NEW tensor:
       let new_memory = memory - eta * gradient;
       Assign to the local mutable reference, don't modify the storage.
       Rust's ownership model helps: &mut self means only YOU can modify it.
SEVERITY: CRITICAL — non-reproducibility is a show-stopper
TRACE: CS-46 (compile can't trace — related to mutation); state lifecycle spec
```

## CS-48: Shared Retention Parameters Across CMS Levels

```
SMELL: let retention_lambda = nn::Parameter::new(0.99);
       // same lambda used for level 0, 1, 2, 3
WHY:   CMS levels operate at different timescales (C=1, 8, 64, 512).
       Level 0 (fast, every step) needs different retention than
       level 3 (slow, every 512 steps). If they share a single lambda:
       - Lambda optimized for level 0 → level 3 forgets too fast
         (its memory decays 512x between updates)
       - Lambda optimized for level 3 → level 0 retains too aggressively
         (it never forgets, memory saturates)
       This is RETENTION INTERFERENCE — levels fight over a shared parameter.
       The optimizer sees conflicting gradient signals and oscillates.

FIX:   Each CMS level gets its own retention parameters.
       retention_lambdas: Vec<nn::Parameter> with len == k (number of levels).
       The frequency scheduler ensures each level's lambda is optimized
       independently by its own gradient stream.
SEVERITY: Warning — silent performance degradation, not crash
TRACE: Committee review cycle; frequency scheduler error buffer analysis
```

## CS-49: Generation Through Training Model Corrupts Context [CRITICAL]

```
SMELL: samples = generate(gpu_model, prompt, max_tokens=30)
       // autoregressive generation through the live training model
       // without saving and restoring M matrix context
WHY:   Every forward() call mutates the CMS context (M matrices) — this
       is by design (memory updates ARE the forward pass). Autoregressive
       generation runs N forward calls (one per generated token), each
       modifying M. If this happens through the training model without
       context isolation, the training M state is overwritten with
       generation artifacts.

       In NL, this is uniquely dangerous because:
       - M matrices are inner-loop state that accumulates across the
         entire build — they represent learned episodic memory
       - Generation injects random/sampled tokens into M, corrupting
         the carefully accumulated state
       - With k=4 (four CMS levels), four M matrices are trashed
         simultaneously — NaN propagation is near-certain
       - The corruption is SILENT: no error is raised, training
         continues on garbage state until NaN appears steps later

       This was discovered when k=4 FineWeb-Edu crashed at step 5001:
       the eval block ran coherence samples and checkpoint sample
       generation through the training gpu_model, destroying context.
       The subsequent roundtrip verification ran forward() on trashed
       M and produced NaN.

FIX:   ALWAYS bracket generation calls with context save/restore:
       ctx = gpu_model.to_host_context()
       try:
           gpu_model.reset_context()  // clean M for generation
           samples = generate(gpu_model, ...)
       finally:
           gpu_model.upload_context(ctx)  // restore training M

       This applies to ANY operation that runs forward() outside the
       training loop: eval coherence samples, checkpoint sample
       generation, interactive probes, curriculum phase probes.
       The Python tier owns this invariant — the Rust tier correctly
       mutates M on every forward() as designed (CS-18).
SEVERITY: CRITICAL — silent context corruption → NaN within steps
TRACE: FineWeb-Edu k=4 NaN crash at step 5001; gpu_forward.rs copy_final_m;
       engine/loop.py eval/checkpoint block
```

## CS-50: Left-Padding With Untrained Tokens Causes Memory Divergence [CRITICAL]

```
SMELL: ctx = [PAD] * (seq_len - len(prompt)) + prompt
       // left-pad short prompts with PAD token (id=2) to fill seq_len
WHY:   In NL, the memory inner loop processes EVERY token — there is no
       attention mask, no "padding is invisible" assumption. Each position
       updates M via the Titans recurrence: M_new = α·M - θ·dM
       (Titans 2501.00663 Eq 3; HOPE 2512.24695 §3 Eq 28).

       When 29+ identical tokens are fed into the memory, the recurrence
       becomes a fixed-point iteration on a single (q_t, k_t) pair. For
       tokens whose embeddings produce ||q_t||² large enough that
       θ·||q_t||² > (1 - α)/2, the iteration diverges — M elements grow
       without bound, producing NaN in logits at subsequent positions.

       Special tokens (0=<|im_start|>, 1=<|im_end|>, 2=<|pad|>) are
       particularly vulnerable because:
       - Their embeddings are shaped by narrow distributional context
         (always at turn boundaries in ChatML, or never seen at all for PAD)
       - The W_K_mem / W_V_mem projections of these embeddings may produce
         vectors with unusual norm profiles
       - 508 identical PADs means 508 identical memory updates — any
         instability in the recurrence is amplified 508×

       Discovered in FineWeb-Edu k=4 probes: generate_learning() left-padded
       short eval prompts with PAD=2. All probe losses were NaN. The model
       generated tokens ("Zap") from the last position (which hadn't
       diverged yet), but cross-entropy at diverged positions was NaN.

       The threshold is sharp: 28 identical special tokens → OK,
       29 identical special tokens → NaN at positions 29+. Regular tokens
       (id ≥ 3) do NOT trigger this even at 512 repetitions.

FIX:   Never left-pad with special tokens. Use the prompt's first token
       (if id ≥ 3) or a common BPE token (id=3) as padding:
       safe_pad = prompt[0] if prompt[0] >= 3 else 3
       ctx = [safe_pad] * pad_len + prompt

       This ensures padding uses tokens the model has been trained on,
       with embedding values that produce stable memory dynamics.

       Long-term: consider adding gradient clipping or norm clamping
       inside the Titans inner loop to prevent memory divergence on
       ANY repeated input, not just special tokens.
SEVERITY: CRITICAL — guaranteed NaN for any short prompt with PAD left-padding
TRACE: FineWeb-Edu k=4 probe NaN; diag_nan threshold test (28→29 transition);
       engine/generation.py generate_learning/generate_cached/generate
```
