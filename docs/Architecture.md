# Architecture

Lin's architecture is built to support a groundbreaking execution model: **massively parallel optimal evaluation via Reversible Polarized Interaction Combinators (PICR)**. It shifts away from traditional sequential execution, transforming code into a highly concurrent, reversible graph that resolves optimally across CPU, General Purpose GPU, and high-performance Shaders.

To lower the language, the compiler translates an optimized, compile-time Interaction Net into an MLIR dialect representing a flat, pre-allocated memory array, which is then compiled directly into native machine code.

Both architectures (CPU & GPU) treat the interaction net as a flat, addressable **atomic bit-field array** where physical memory offsets replace pointers. This allows both CPU SIMD lanes and GPU threads to execute the same "swap-and-link" logic using standardized hardware-level atomic instructions.

This architecture fundamentally avoids virtual machines, distinguishing it from earlier iterations of Lin. The approach shares conceptual similarities with HVM3, but relies entirely on MLIR for lowering rather than using custom C++ or CUDA backends. By maintaining a 128-bit memory layout per node and using polarized nets, the system achieves lock-free redexes managed through a specialized rule lookup table.

The addition of **reversibility** is a first-class design goal, not a bolt-on: every reduction can be inverted, enabling zero-cost uncomputation passes, race-free garbage collection, and a future path to formal verification, quantum simulation targets, and optimal MLIR lowering search.

---

## Compiler Pipeline

1. **Frontend Parsing:**
   - Source files (`.lin`) are parsed by a custom C11 lexer and parser.
   - The parser constructs an Abstract Syntax Tree (AST).
2. **Lowering to MLIR (Reversible Polarized Interaction Combinators):**
   - The AST is walked and lowered into an MLIR representation (`pic.graph` dialect).
   - This models the program as a set of Interaction Net Agents and Links.
   - Every agent emitted carries directional polarity; the graph is reversible by construction.
3. **Interaction Net Optimization:**
   - E-graphs serve as an IR, optimizing code continuously (via Congruence inside Boundaries).
   - Inside boundaries the optimizer can freely mutate the graph forward *and* backward to find the optimal MLIR lowering without losing source semantics.
   - Supercompilation techniques (like Souper) can further optimize via MLIR.
4. **Lowering to Runtime Dialects:**
   - `pic.graph` $\rightarrow$ `pic.reduce` (Normalization, detecting active pairs, scheduling; emits forward and reverse ops)
   - `pic.reduce` $\rightarrow$ `pic.runtime` (CPU/GPU routing via labels; includes uncompute sweep kernel)
   - Finally, `pic.runtime` $\rightarrow$ LLVM IR or SPIR-V/PTX.
5. **AOT Compilation & Execution:**
   - The LLVM IR is compiled into an object file and linked with the Lin runtime (`runtime.c`) to produce the final `.line` binary.

---

# **PICR:** Reversible Polarized Interaction Combinators

PICR extends classical Polarized Interaction Combinators (PIC) with a single foundational constraint: **no graph rewrite may reduce the total information entropy of the graph.** Every destructive step must either store its structural delta in a local boundary history log, or be paired with an explicit inverse routing pathway.

This aligns with Landauer's Principle at the architectural level: deleting a bit costs energy; restructuring it to its pre-reduction state costs nothing beyond computation time. By enforcing this invariant, Lin gains:

- **Race-free parallel GC** — threads run math backward rather than fighting over deallocation locks.
- **Lossless E-graph search** — the optimizer can explore any MLIR lowering path without discarding original semantics.
- **Deterministic reversibility** — any computation state can be reconstructed from any later state, enabling rollback, replay, and formal proof of program equivalence.

---

## The Node Set — 7 Nodes

Every node has a **principal port** (where interactions happen) and **auxiliary ports** (where wires connect). Polarity is on the principal port only.

```
γ⁺  Constructor    — principal(+), aux(left, right)
γ⁻  Destructor     — principal(-), aux(left, right)
δ   Duplicator     — principal(*), aux(left, right)
H   History        — replaces ε; folds dead subgraphs into a linear log chain
L   Log-Bond       — passive ghost node deposited at annihilation splice sites
ω   Operation      — principal(+/-), aux(inputs..., outputs...)
R   Reverse-Vector — signal node; propagates through a subgraph forcing backward rewrites
```

And one structural primitive:

```
[ ]  Boundary (β)  — not a node, a region. Contains nodes, a history log, and an interface wire set.
```

> **Why no ε (Eraser)?**  
> In classical IC, the Eraser violates Landauer's Principle — it destroys information with no record. In PICR, the History node (H) replaces it: instead of vanishing a subgraph, H folds it into a compressed sequential log held inside the nearest enclosing boundary. A localized Uncompute Pass later runs that log backward to structurally restore the subgraph to its initial state and free it cleanly.

---

## Why These Seven And Not More?

| What you need                   | How it's covered                                              |
| ------------------------------- | ------------------------------------------------------------- |
| Data construction/pattern match | γ⁺ γ⁻ annihilation                                           |
| Sharing / fan-out               | δ duplication                                                 |
| Reversible GC                   | H (history) + R (reverse-vector) + boundary uncompute pass   |
| All compute                     | ω op with MLIR label                                          |
| Effect scoping / E-classes      | Boundary `[ ]`                                               |
| Reversibility record-keeping    | L (log-bond) at annihilation splice sites                     |
| Inverse routing                 | R signal propagation through subgraph                         |
| Effects perform/handle          | op(+, eff) floating to op(-, eff)                            |
| Continuations                   | wire held by handler op                                       |
| Membranes                       | Boundary `[ ]`                                               |

Effects, continuations, and membranes aren't separate nodes. They're **emergent behaviors** of the seven primitives plus boundaries.

---

## The Rule Set — 4 Rules (Reversible Form)

### Rule 1 — Reversible Annihilation

**Forward:**
```
γ⁺ ⋈ γ⁻  →  splice auxiliary wires, deposit L (Log-Bond) at splice site

    left₁  right₁             left₁      right₁
       \   /                      |    L    |
       γ⁺  γ⁻    →               |  [log]  |
       /   \                      |         |
    left₂  right₂             left₂      right₂
```

The L node is passive — it does not fire, consumes no auxiliary ports, and incurs no scheduling cost. It is a structural breadcrumb.

**Reverse:**
```
R signal hits L  →  L snaps back into γ⁺ ⋈ γ⁻, wires restore
```

Principal ports meet, nodes vanish, auxiliary wires connect directly. This is $O(1)$ and requires **no allocation** on the forward pass.

*   **Op variant:** When `op(+, f)` meets `op(-, f)` with a matching label, the MLIR operation fires and result wires splice through.
*   **Effect variant:** `op(+, eff)` floating upward meets an `op(-, eff)` handler. The handler's auxiliary ports carry the continuation wire.

### Rule 2 — Duplication (Symmetric in Reverse)

**Forward:**
```
δ ⋈ γ⁺  →  two γ⁺ nodes, δ crosses through

       |                  |        |
       δ                 γ⁺       γ⁺
       |        →        / \     / \
      γ⁺               δ   δ   l   r
      / \               |   |
     l   r             (l) (r)
```

The δ node commutes through γ⁺, producing two copies. This is **lazy by default.** The two δ nodes don't fire until something demands them.

**Reverse:**  
Two identical backward signals arriving at a δ node merge back into a single wire. The δ fires its inverse rule: `δ⁻¹ ⋈ (γ⁺, γ⁺) → γ⁺, δ dissolves`. A **synchronization barrier** is required at each δ on the reverse pass — the reverse signal must arrive on both auxiliary ports before the merge fires, ensuring no partial merges corrupt the graph. This is the primary cost of full reversibility.

*   **Erasure variant:** `δ ⋈ H → two H nodes`. History propagates (parallel reverse-GC).

### Rule 3 — Reversible Uncomputation (replaces Erasure Propagation)

**Forward (dead branch detected):**
```
Branch root becomes unreachable  →  emit R (Reverse-Vector) into branch root
```

**R propagation:**
```
R ⋈ γ⁺  →  R on each auxiliary port  (mirror of old ε rule)
R ⋈ γ⁻  →  R on each auxiliary port
R ⋈ ω   →  cancel op, R on each auxiliary port
R ⋈ δ   →  R on each auxiliary port
R ⋈ L   →  reconstruct γ⁺ ⋈ γ⁻, continue R propagation
```

The R signal propagates through the dead subgraph, forcing every node to execute its rewrite rule in reverse. The subgraph **runs backward** until it restores itself to its pre-computation input state. At that point the boundary is empty and undergoes B3 (Collapse) at zero thermodynamic cost.

This replaces traditional GC entirely. Deallocation is localized retro-computation, not a global mark-and-sweep.

### Rule 4 — Congruence (unchanged; already lossless)

```
Inside [ ]:
If subgraph A and subgraph B have identical structure and identical interface wires
→  merge into single subgraph, redirect all wires to it
```

This is the E-graph rule. It fires inside boundaries continuously. Two expressions that reduce to the same normal form collapse to one.

**In reverse:** A merged node splits along its reference-origin wires when different backward signals demand it. Each merged node carries an `origins` attribute listing the wires that were redirected to it.

---

## Boundary Semantics — 4 Sub-Rules (Reversible Extension)

**B1 — Containment**
```
op(+, eff) inside [ ] with no matching op(-, eff) inside [ ]
→  wire crosses boundary interface, floats to outer scope
```
Unhandled effects bubble outward automatically.

**B2 — Fusion**
```
[ A ] ⋈ [ B ]  →  [ A ∪ B ]
```
Boundaries that meet fuse into a larger E-class.

**B3 — Collapse**
```
[ single normal form ]  →  that normal form, boundary dissolves
```
When everything inside a boundary normalizes to one thing, it disappears.

**B4 — Reversible Boundary Cycle** *(PICR addition)*
```
1. Isolate:    Computation executes inside [ A ]. Generates answer wire + local history log.
2. Exfiltrate: Answer wire crosses boundary interface to outer scope via B1.
3. Rollback:   Outer scope sends success token back into [ A ].
               [ A ] triggers internal localized uncompute loop — runs its history log backward,
               erasing all garbage bits structurally.
4. Dissolve:   Empty boundary undergoes B3. Dissolves with zero thermodynamic cost.
```

B4 localizes all history entropy *inside* boundaries. Forward reductions outside boundaries remain fast and non-logging. Only inside a boundary do nodes maintain L bonds and reverse-eligibility, keeping the hot path clean.

---

## The Full Computational Architecture

```
┌─────────────────────────────────────────────────────┐
│                   SOURCE LANGUAGE                   │
│         (Red/Rebol-like syntax, homoiconic)         │
└─────────────────────┬───────────────────────────────┘
                      │ parse
                      ▼
┌─────────────────────────────────────────────────────┐
│              REVERSIBLE COMBINATOR GRAPH            │
│                                                     │
│   γ⁺  γ⁻  δ  H  L  ω  R   inside  [ boundaries ]  │
│                                                     │
│   Active pairs detected via port connection scan    │
│   L-bonds deposited at annihilation sites           │
│   H nodes accumulate history inside boundaries      │
└──────────┬──────────────────────────┬───────────────┘
           │ classify by label         │
           ▼                           ▼
┌──────────────────┐       ┌──────────────────────────┐
│   CPU POOL       │       │      GPU DISPATCH         │
│                  │       │                           │
│  γ⁺ ⋈ γ⁻        │       │  op[arith.*] batches      │
│  δ commutation   │       │  op[linalg.*] batches     │
│  Boundary ops    │       │  op[gpu.*] batches        │
│  Effect routing  │       │  R propagation sweeps     │
│  Work stealing   │       │  Congruence hashing       │
│                  │       │  Reverse mirror pass      │
│  Irregular       │       │  Regular, data-parallel   │
│  Pointer-chasing │       │  SIMD-perfect             │
└──────────┬───────┘       └────────────┬─────────────┘
           │                            │
           └──────────┬─────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│              REDUCED GRAPH                          │
│                                                     │
│  Congruence fires continuously inside boundaries    │
│  Identical subgraphs collapse as they normalize     │
│  Reverse pass cleans history inside boundaries      │
│  Empty boundaries dissolve via B3                   │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│                 MLIR LOWERING                       │
│                                                     │
│  pic.graph dialect  →  pic.reduce dialect           │
│  pic.reduce dialect →  pic.runtime dialect          │
│  pic.runtime dialect→  gpu.launch + llvm            │
│  Boundaries         →  MLIR regions (w/ history log)│
│  γ annihilation     →  MLIR control flow + L-bond   │
│  R propagation      →  mirror-image GPU kernel      │
│  op[linalg.*]       →  linalg dialect (auto-tiled)  │
│  op[gpu.*]          →  SPIR-V or PTX                │
└─────────────────────────────────────────────────────┘
```

---

## The MLIR Dialect Stack (PICR Extensions)

Three dialects, each lowering into the next:

```
pic.graph
    ops: agent, wire, boundary, active_pair, registry
    agent types: CON(γ⁺/γ⁻), DUP(δ), HIS(H), LOG(L), OP(ω), RVEC(R)
    boundary carries: interface wires, history_log region, origins attribute

pic.reduce
    ops: annihilate, duplicate, uncompute, log_state, fire_op,
         merge_boundary, reverse_vector
    annihilate now deposits a pic_reduce.log_bond instead of vanishing
    uncompute: inverts specific node pairs
    log_state: pushes structural delta to local boundary history log
    reverse_vector: injects R signal into a subgraph root

pic.runtime
    ops: cpu_dispatch, gpu_batch, gpu_batch_reverse,
         uncompute_sweep, sync_point
    GPU threads execute two primary kernels:
      1. Forward Evaluation Pass
      2. Mirror-Image Backward Cleanup Pass (uncompute_sweep)
```

The `pic.graph → pic.reduce` pass is your interaction net normalizer. It detects active pairs and emits the appropriate reduction op, now including L-bond deposition and R-signal injection. The `pic.reduce → pic.runtime` pass is your CPU/GPU router and also schedules the uncompute sweep after each forward batch completes inside a boundary.

---

## How Each Feature Emerges

*   **Garbage collection:** Dead branches receive an R signal at their root. R propagates backward through the subgraph, restoring it to its initial state structurally. You never write a GC — uncomputation IS GC.
*   **Algebraic effects:** Unhandled ops float via B1. The handler holds the continuation. Resume = feed a value into that wire. Abort = attach R to it; the aborted branch uncomputates cleanly.
*   **Lazy evaluation:** δ doesn't copy until its duplicates diverge. Sharing is the default. Copying is the exception, triggered only by actual demand.
*   **Operator fusion:** Two ops that produce the same result inside a boundary collapse via Rule 4. The fused version is whatever the congruence hash resolves to — which you can bias toward efficient ops by ordering your boundary contents. The optimizer can walk this space freely forward and backward without losing semantics.
*   **Parallel execution:** Any two active pairs with no shared nodes fire simultaneously. This is a theorem about the system, not an optimization. The programmer cannot accidentally introduce a race condition because the rewrite rules are locally confluent by construction.
*   **CPU/GPU routing:** Active pair labels determine routing. The runtime maintains two queues. No programmer annotation. No pragma. No async/await. The graph structure encodes the parallelism.
*   **Reversible optimization:** Inside a boundary, the E-graph optimizer can freely forward-reduce and then reverse to explore the full reachable space of equivalent MLIR lowerings without discarding original source semantics.

---

## The Invariants

Four properties you must preserve in the implementation:

**1. Linearity of principal ports**
Every principal port connects to exactly one wire. This is what makes reductions non-interfering. If you break this, you break the parallelism guarantee.

**2. Boundary interface consistency**
When a wire crosses a boundary, both sides must agree on the wire's polarity. This is what makes effect bubbling well-defined.

**3. Confluence**
Rules 1–4 must be confluent — any order of reduction reaches the same result. This holds for standard IC. You need to verify it holds for your op labels, particularly for ops with side effects, which is why effect tokens or boundary sequencing matter. Confluence is what lets you throw the work at any number of CPU cores and GPU threads without coordination.

**4. Conservation of Topological Information** *(PICR addition)*
No graph rewrite may reduce the total information entropy of the graph. Every destructive step must store its structural delta in a local boundary history log, or be explicitly paired with an inverse routing pathway (R signal). This invariant is what enables the Reversible Boundary Cycle (B4) and race-free GC.

---

## Node Layout: 128-bit with History Extension

```
┌──────────────────────────────────────────────────────────┐
│  Current:  [ type:8 | label:24 | port0:32 | port1:32 ]  │
├──────────────────────────────────────────────────────────┤
│  PICR:     [ type:8 | label:24 | port0:32 | port1:32 ]  │
│  +         [ log_ptr:32 | flags:8 | reserved:24 ]        │
│  (second 128-bit word, only allocated inside boundaries) │
└──────────────────────────────────────────────────────────┘
```

The `log_ptr` field is a history-chain pointer used only inside boundary regions. Outside boundaries, the second word is omitted, keeping the hot path at the same 128-bit width as classic PIC.

---

# **Philosophy: Explicit is Better Than Implicit**

Lin follows a philosophy similar to C: **minimal magic, maximum transparency.**

### **1. No Implicit Imports**
The compiler does not automatically inject any standard library files (including `std/types.lin`). Every dependency must be explicitly imported by the developer. This ensures that:
- The build process is transparent.
- There are no "hidden" symbols that could conflict with user-defined names.
- The compiler remains minimal and predictable.

### **2. Manual Type Definition**
Basic types like `i32`, `f32`, etc., are not built into the compiler's frontend as keywords. Instead, they are defined in the standard library (`std/types.lin`) and must be made available to the compiler through an explicit `import` if semantic type checking is desired.

### **3. Abstraction is Opt-in**
The primary form of abstraction in Lin is the **Interaction Net scheduling**. Any further abstraction layer (higher-level types, memory management patterns, object models) is left entirely to the developer or library authors. The core language focuses exclusively on optimal, parallel, reversible evaluation of polarized interaction nets.

### **4. Reversibility is Structural, Not Annotated**
The programmer does not annotate functions as "reversible." Reversibility is a property of the graph itself, enforced by invariant 4. The only visible surface area is the `@boundary` annotation, which enables the B4 cycle for a given scope, and the `uncompute` intrinsic available in `mlir-op` blocks for custom reverse-pass logic.

---

# **Lin: The Compiler as a High-Performance Scheduler**

Unlike traditional compilers that define a set of primitive operations (like `add`, `sub`, `load`), the `linc` compiler is fundamentally a **reversible graph scheduler**.

### **Zero Built-in Operations**
There are no operations "hardcoded" into the compiler's backend logic. When you write `a + b`, the compiler does not emit an `LLVM::AddOp`. Instead:
1. It emits an `omega` agent with a label (e.g., `add`).
2. It relies on a `registry` entry (provided by the user in the standard library or local code) that defines what happens when two `omega` agents with that label meet.
3. The `registry` entry contains a raw MLIR payload that is injected into the reduction engine.
4. *(PICR)* The registry entry may also contain an inverse payload, used by `pic_reduce.uncompute` during the reverse pass.

### **The "Scheduler" Perspective**
In this sense, Lin is closer to a hardware scheduler than a traditional language. It manages the topology of the interaction net, detects active pairs (redexes), dispatches them to appropriate compute resources (CPU threads or GPU cores), and coordinates forward and reverse passes inside boundary regions. What happens *during* that dispatch is entirely defined by the user via `mlir-op` blocks.

This architecture ensures that the core language remains extremely small, stable, and transparent, while allowing the developer to leverage the full power of MLIR dialects without the compiler standing in the way.
