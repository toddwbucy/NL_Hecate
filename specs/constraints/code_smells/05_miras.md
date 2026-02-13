# MIRAS Design Space Smells (CS-33 through CS-36)

```
CONTRACT
  Purpose:    MIRAS (2504.13173) proved that memory update rules have 4
              orthogonal design choices. These smells prevent hardcoding
              any particular choice as the "default" or "only" option.
              The design space is combinatorial — restricting it prematurely
              loses the MIRAS paper's core contribution.
  Expects:    All memory update rule implementations and trait definitions.
  Guarantees: The 4-knob framework is preserved in the trait system.
              No knob is restricted to a single option.
              New MIRAS variants can be added without modifying existing code.
  Position:   specs/constraints/code_smells/05_miras.md
  Source:     MIRAS (2504.13173) Table 2; MIRAS IS #1, IS NOT #2
```

## CS-33: Don't force same attentional bias across models

```
SMELL: trait MemoryUpdateRule {
         fn loss(&self, M: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
           (M @ k - v).pow(2).sum()  // L2 hardcoded
         }
       }
WHY:   The attentional bias (loss function for inner-loop gradient) is
       MIRAS Knob #2. Different models use different biases:
       - Titans LMM: L2 (squared error)
       - YAAD: Huber (bounded gradient for stability)
       - MONETA: l_p norm (parameterized p)
       - MEMORA: KL divergence (probability distributions)

       Hardcoding L2 makes the codebase Titans-only.
       The trait must parameterize the attentional bias.
USE:   trait AttentionalBias { fn loss(&self, ...) -> Tensor; fn grad(&self, ...) -> Tensor; }
       Each memory rule composes with an AttentionalBias implementation.
TRACE: MIRAS Table 2 (Knob #2); MIRAS IS #1 (orthogonal design choices)
```

## CS-34: Don't restrict memory to matrix-valued

```
SMELL: struct Memory { M: Tensor }  // assumed 2D matrix
       fn write(&mut self, k: &Tensor, v: &Tensor) {
         self.M += k.outer(v)  // only works for matrices
       }
WHY:   The memory structure (MIRAS Knob #1) is not always a matrix:
       - Vector memory: M is a 1D tensor (simplest, like DeltaNet)
       - Matrix memory: M is a 2D tensor (Titans, Atlas)
       - MLP memory: M is a multi-layer network (MONETA)

       Hardcoding matrix operations restricts the design space.
       Vector memory uses different update math (no outer product).
       MLP memory uses backprop through the MLP, not matrix operations.
USE:   trait MemoryStructure { type State; fn update(&mut self, ...); fn read(&self, ...); }
       Each structure defines its own state type and operations.
TRACE: MIRAS Table 2 (Knob #1); MIRAS IS NOT #2 (not matrix-only)
```

## CS-35: Don't assume GD is the only memory algorithm

```
SMELL: fn inner_step(&mut self, ...) {
         let grad = self.compute_gradient(...);
         self.M -= self.eta * grad;  // GD hardcoded
       }
WHY:   The memory algorithm (MIRAS Knob #4) includes:
       - Gradient Descent (Titans basic)
       - GD + EMA momentum (Titans with momentum)
       - GD + Newton-Schulz momentum (Atlas Omega rule)
       - FTRL (Follow The Regularized Leader) — proven equivalent to some configs
       - Online Mirror Descent (Lattice OSR on the sphere)

       Hardcoding gradient descent makes GD+momentum impossible without hacks.
       The update step must be parameterized by the algorithm.
USE:   trait MemoryAlgorithm { fn step(&mut self, grad: &Tensor, ...) -> Tensor; }
       GD, GD+momentum, Newton-Schulz are all implementations.
TRACE: MIRAS Table 2 (Knob #4); Atlas Eq 12-15 (multiple algorithms)
```

## CS-36: Don't restrict retention to L2 only

```
SMELL: fn retention(&self, M: &Tensor, lambda: f32) -> Tensor {
         lambda * M  // L2 weight decay hardcoded
       }
WHY:   The retention mechanism (MIRAS Knob #3) includes:
       - L2 weight decay (exponential forgetting)
       - KL divergence (simplex constraint, softmax emergence)
       - Elastic net (L1 + L2, sparse memory)
       - f-divergence (general framework)
       - Sphere normalization (implicit via projection, Lattice)

       L2 is the simplest but not always the best. KL retention makes
       memory a probability distribution. Elastic net creates sparse memory.
       Hardcoding L2 loses all of this.
USE:   trait RetentionMechanism { fn apply(&self, M: &Tensor, ...) -> Tensor; }
       Each mechanism is a separate implementation with its own math.
TRACE: MIRAS Table 2 (Knob #3); retention specs (specs/algorithms/retention_mechanisms/)
```
