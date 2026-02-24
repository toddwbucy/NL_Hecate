# Exploration: Brain Transplant (Pre-trained Weight Conversion to HOPE)

**Status**: Archived from ROADMAP. Not on the active engineering path.
**Origin**: HOPE paper Section 7.3 discusses initializing CMS levels from pre-trained MLPs.
**Moved**: 2026-02-24, per committee recommendation to decouple ab initio path from conversion path.

---

## Context

The HOPE paper (2512.24695) Section 7.3 describes a procedure for converting a pre-trained transformer into a HOPE model by mapping existing MLP layers to CMS frequency levels. Two open questions were identified during initial graph construction:

### Open Question 1: Layer Selection

The paper says "Given k pre-trained MLPs..." without specifying which layers to select from a model like Llama-3. With k=4 CMS levels and a 32-layer model, which 4 layers map to which frequencies?

**Possible approaches** (unvalidated):
- Geometric spacing: layers [0, 8, 16, 24] for k=4
- Uniform spacing: layers [4, 12, 20, 28]
- Probing-based: select layers whose representations most closely match the target frequency dynamics

### Open Question 2: Attention Layer Handling

The paper discusses MLP-to-memory conversion but does not address what happens to the original attention layers. HOPE uses SWA (sliding window attention) independently of the CMS memory. If MLPs are extracted for CMS initialization, the attention layers need a separate handling strategy.

**Possible approaches** (unvalidated):
- Discard attention layers entirely (use HOPE's native SWA)
- Initialize SWA weights from a subset of attention layers
- Keep attention layers frozen as a residual pathway during early training

---

## Why This Is Not On The Active Path

NL_Hecate is an ab initio research platform. The primary value proposition is proving that HOPE can learn to self-modify from a blank slate. The brain transplant path:

1. Introduces architectural compromises to maintain compatibility with Llama's structure
2. Makes it impossible to attribute learned behavior to HOPE vs. inherited weights
3. Is not required for any current experimental goal

If the ab initio path succeeds, brain transplant becomes an optimization for faster deployment, not a research necessity. If it fails, brain transplant wouldn't fix the underlying issue.

---

## HADES References

- `hope_blockers/blocker-layer-selection` (deferred)
- `hope_blockers/blocker-attention-handling` (deferred)
- `hope_probes/probe-transplant-integrity` (registered, not applicable to ab initio path)
