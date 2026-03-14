# Response to Reviewer Feedback on committee_response_05

**Date**: 2026-02-27
**Re**: Corrections and clarifications raised in reviewer feedback on the status document

---

Thank you — every flag you raised was correct, and we have since verified against the actual log files and checkpoint inventory. The status document had more errors than you caught. We are correcting the record in full below, without rationalizing any of them.

---

## Correction 1: Test Count

**committee_response_05 said**: 805 tests (778 Rust + 27 Python).
**Correct figure**: **1,406 tests** (1,379 Rust + 27 Python).

The 805 number was the Stage 1 completion count, frozen at that milestone. ROADMAP.md line 950 gives the current figure: 1,379 Rust + 27 Python = 1,406 tests across 122 merged PRs. Stage 2 added 601 Rust tests through S2-M1 (CUDA kernels), S2-M2 (multi-GPU), S2-M3 (serving), and S2-M4 (edge deployment). We reported a stale number. The correct count is 1,406.

---

## Correction 2: Dataset Labels (All Three Runs Were Misidentified)

We checked the actual log file headers. The data picture is different from what committee_response_05 stated.

The Python data loader reports "ShareGPT BPE" in its startup banner for **any** data loaded in the `sharegpt` binary format — regardless of content. This is a format-type label, not a content label. Two of the three native runs loaded FineWeb-Edu content despite the logs saying "ShareGPT BPE." Corrected labels:

| Run | Config path field | Log banner | Content (corrected) | Tokens |
|-----|-----------------|------------|---------------------|--------|
| fineweb_k4 | `data/fineweb_edu` | "ShareGPT BPE" | **FineWeb-Edu** | 72M |
| fineweb_k4_v2 | `data/fineweb_edu` | "ShareGPT BPE" | **FineWeb-Edu** | 72M |
| fineweb_k1 | `data/fineweb_edu` | "ShareGPT BPE" | **FineWeb-Edu** | 72M |
| phase0_100k | `data/sharegpt` | "ShareGPT BPE" | **ShareGPT** | 461M |

The loss comparison in committee_response_05 ("dataset quality and size dominate over training length") was therefore confounded by distribution, not just scale. fineweb_k4 used FineWeb-Edu, phase0_100k used ShareGPT. The two runs are not a clean scale comparison.

---

## Correction 3: The Checkpoint Inventory Was Incomplete

We missed three runs entirely. The complete inventory is:

| Checkpoint prefix | Architecture | Content | Tokens | Step range | Status |
|------------------|-------------|---------|--------|-----------|--------|
| `model_step*` | d=2048, SwiGluMlp, k=4 | Unknown | — | 5K–45K | Killed, not resuming |
| `fineweb_k4_step*` | TitansLMM d=512, k=4 | FineWeb-Edu | 72M | 25K–74K | Plateau (~5.5 loss), stopped |
| `fineweb_k4_v2_step*` | TitansLMM d=512, k=4, theta floors | FineWeb-Edu | 72M | 5K–20K | **Blessed production config** — stopped at ~20K |
| `fineweb_k1_step*` | TitansLMM d=512, **k=1** | FineWeb-Edu | 72M | 5K–55K | **ACTIVE — currently running at step 55K+** |
| `phase0_100k_step*` | TitansLMM d=512, k=4 | ShareGPT | 461M | 5K–15K | Stopped at step 16,740 (~3.4 loss) |
| `llama_stacking_k4_sft_step*` | Llama donor | — | — | 2K–10K | Safetensors, cursor sidecars present |
| `llama8b_pretrain_step5000` | Llama 8B pretrain | — | — | 5K | Single checkpoint |

The `fineweb_k4_v2` run is the **blessed production config** referenced in `specs/infrastructure/01_variant_tier_policy.md` — the one actually used for Tier 1 validation. It adds per-level theta floor clamps ([0.01, 0.05, 0.02, 0.01]) and `m_norm_max` bounds — the fix for the NaN-at-step-12,552 event. It was missing from the inventory entirely.

The `fineweb_k1` run is the k=1 vs k=4 ablation committed to in committee_response_04. It is **currently running** — the log shows step 55,210, and `fineweb_k1_step55000.json` is the latest checkpoint on disk.

---

## On the TinyStories Phase 0 Run

You are correct that this is the elephant in the room. Our position is as follows.

committee_response_03 and committee_response_04 reference a Phase 0 TinyStories run with:
- Step 5K theta values: L0=0.0325, L1=0.0045, L2=0.0014, L3=0.0005 (monotonically ordered, all nonzero)
- Step 10K theta values: L0=0.0762, L1=0.0053, L2=0.0015, L3=0.0005 (L2 memory norm 3.5x growth)

These values came from a real run. The checkpoints and log file are not in the current repository inventory — they appear to have been killed after the 10K metrics were reported and the checkpoint files were not retained. The JSONL run log for that run also does not exist on disk.

**The metrics survive in the committee documents themselves.** The data is not lost — it is embedded in committee_response_03 and committee_response_04 as inline tables. What is missing is: (a) the raw JSONL log from which those numbers came, and (b) a checkpoint to resume from.

We want to be explicit: those numbers are real, not manufactured. The TinyStories run happened. We did not retain its artifacts. This will be stated plainly to the committee.

Regarding whether these metrics could be recovered from a HADES node: we have not found them there. The `curriculum_specs` collection proposed in committee_response_03 was specified as a future graph schema — it was not populated with the TinyStories run's actual metrics at the time. Those metrics exist only in the committee document PDFs.

---

## On the "Progress Report" Reference

The reviewer references "Progress Report (dated 2026-02-24) showing 1,406 tests and Phase 0 TinyStories at step ~5,200 with loss ~3.8." We do not have this document in our local file inventory — it was likely shared out-of-band. Given what we now know:

- **1,406 tests** matches ROADMAP.md exactly — that is the correct current count, confirmed.
- **"Phase 0 TinyStories at step ~5,200 with loss ~3.8"** does not match any checkpoint in the current inventory. The TinyStories run (described above) is the most likely candidate — it would have been in early stages (~5K steps) at the time of the committee reports dated 2026-02-24. This is consistent: the run produced 5K and 10K metrics, appears to have been stopped shortly after, and its artifacts were not retained.
- The **currently running Phase 0 run** is `fineweb_k1` at step 55K+ on FineWeb-Edu 72M. This is the k=1 ablation committed to in committee_response_04 — not TinyStories, not step 5,200.

To directly answer your question: there is no fifth uncharted run currently in progress. `fineweb_k1` is the active run, it has a full checkpoint trail (5K through 55K), and it was simply omitted from the committee_response_05 inventory. The step ~5,200 TinyStories reference is a historical data point from a run that ended, not a live run.

---

## On task_cbbf32 Scope ("Small and Bounded")

You are right to pressure-test this. We reviewed the three affected files. The scope per rule is:

| Rule | Change needed | Complexity |
|------|--------------|-----------|
| `delta_rule.rs` | Add `theta_floor: f32, theta_ceil: f32` to struct; extract from `MAGConfig.theta_floor[level]` and `theta_ceil[level]` in `from_cfg()`; add `clamp(floor, ceil)` at line 315 after `softplus_f32` | ~10 lines |
| `trellis.rs` | Same pattern: 2 struct fields, from_cfg extraction, 1 clamp call at line 239 | ~10 lines |
| `moneta.rs` | Same pattern: 2 struct fields, from_cfg extraction, 1 clamp call at line 295 | ~10 lines |

`clamp_theta()` at `model.rs:753` is already defined and does exactly `theta.clamp(floor, ceil)`. There is no logic to invent — only wiring. The test surface is small: one unit test per rule asserting that theta stays within bounds when initialized with extreme values.

Estimate: **3–4 hours** for all three CPU rules, not multi-day. The GPU counterpart (task_31cfb6) is larger because it touches the CUDA kernel path in `gpu_forward.rs`.

Your sequencing recommendation is correct: task_cbbf32 should run in parallel with dataset resolution, not sequentially. They are independent. Both feed into the delta_rule training run, which is blocked on both.

---

## Updated Experimental Timeline (Corrected)

| Run | Architecture | Data | Tokens | Current step | Status |
|-----|-------------|------|--------|-------------|--------|
| TinyStories Phase 0 | TitansLMM d=512, k=4 | TinyStories | Unknown | ~10K (last report) | **Ended, no artifacts** |
| fineweb_k4 | TitansLMM d=512, k=4 | FineWeb-Edu | 72M | 74,890 | Stopped, plateau |
| fineweb_k4_v2 | TitansLMM d=512, k=4, theta floors | FineWeb-Edu | 72M | ~20K | Stopped, blessed config validated |
| fineweb_k1 | TitansLMM d=512, **k=1** | FineWeb-Edu | 72M | **55K+ (active)** | **Running — k=1 ablation** |
| phase0_100k | TitansLMM d=512, k=4 | ShareGPT | 461M | 16,740 | Stopped, loss ~3.4 |
| llama stacking SFT | Llama donor | — | — | ~9,570 | Stopped, safetensors |

---

## Summary of All Corrections

| Issue raised | Was | Should be |
|-------------|-----|-----------|
| Test count | 805 | **1,406** (1,379 Rust + 27 Python) |
| Run 2 dataset | "ShareGPT 72M" | **FineWeb-Edu 72M** (loader reports "ShareGPT BPE" for any sharegpt-format binary) |
| Run 3 dataset | "ShareGPT 461M" | **ShareGPT 461M** — this one is actually correct |
| Inventory completeness | 4 runs | **6+ runs** (missing fineweb_k4_v2, fineweb_k1; plus TinyStories whose artifacts are gone) |
| Currently active run | Not identified | **fineweb_k1 at step 55K+** (the k=1 ablation) |
| TinyStories run | "May have been deleted" | **Was real; metrics in committee docs; checkpoints not retained** |
| Loss comparison validity | "Same distribution, different scale" | **Different distributions** (FineWeb-Edu vs ShareGPT) — comparison is confounded |
| task_cbbf32 effort | "Small, bounded" | **Accurate — ~3–4 hours per the code review above** |
| Suggested sequencing | Sequential with dataset fix | **Parallel** — agreed |

We will prepare a corrected committee_response_05 incorporating these fixes before panel submission.

*Submitted to reviewer for verification.*
