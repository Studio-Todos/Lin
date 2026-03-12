# Architecture

Lin's architecture is built to support a groundbreaking execution model: **massively parallel optimal evaluation via Polarized Interaction Combinators (PIC)**. It shifts away from traditional sequential execution, transforming code into a highly concurrent graph that resolves optimally across CPU, General Purpose GPU, and high-performance Shaders.

To lower the language, the compiler translates an optimized, compile-time Interaction Net into an MLIR dialect representing a flat, pre-allocated memory array, which is then compiled directly into native machine code.

Both architectures (CPU & GPU) treat the interaction net as a flat, addressable **atomic bit-field array** where physical memory offsets replace pointers. This allows both CPU SIMD lanes and GPU threads to execute the same "swap-and-link" logic using standardized hardware-level atomic instructions.

This architecture fundamentally avoids virtual machines.

## Compiler Pipeline

1. **Frontend Parsing:**
   - Source files (`.lin`) are parsed by a custom C11 lexer and parser.
   - The parser constructs an Abstract Syntax Tree (AST).
2. **Lowering to MLIR (Polarized Interaction Combinators):**
   - The AST is walked and lowered into an MLIR representation (`pic.graph` dialect).
   - This models the program as a set of Interaction Net Agents and Links.
3. **Interaction Net Optimization:**
   - E-graphs serve as an IR, optimizing code continuously (via Congruence inside Boundaries).
   - Supercompilation techniques (like Souper) can further optimize via MLIR.
4. **Lowering to Runtime Dialects:**
   - `pic.graph` $\rightarrow$ `pic.reduce` (Normalization, detecting active pairs, scheduling)
   - `pic.reduce` $\rightarrow$ `pic.runtime` (CPU/GPU routing via labels)
   - Finally, `pic.runtime` $\rightarrow$ LLVM IR or SPIR-V/PTX.
5. **AOT Compilation & Execution:**
   - The LLVM IR is compiled into an object file and linked with the Lin runtime (`runtime.c`) to produce the final `.line` binary.

---

# **PIC:** Polarized Interaction Combinators

## The Node Set — 5 Nodes

Every node has a **principal port** (where interactions happen) and **auxiliary ports** (where wires connect). Polarity is on the principal port only.

```
γ⁺  Constructor    — principal(+), aux(left, right)
γ⁻  Destructor     — principal(-), aux(left, right)
δ   Duplicator     — principal(*), aux(left, right)
ε   Eraser         — principal(*), no aux
ω   Operation      — principal(+/-), aux(inputs..., outputs...)
```

And one structural primitive:

```
[ ]  Boundary (β)  — not a node, a region. Contains nodes. Has an interface wire set.
```

The Boundary is a scope, not a node you interact with directly. The boundary tracks which wires cross its interface.

## Why These Five And Not More?

| What you need                   | How it's covered                                         |
| ------------------------------- | -------------------------------------------------------- |
| Data construction/pattern match | $\gamma^+$ $\gamma^-$ annihilation                       |
| Sharing / fan-out               | $\delta$ duplication                                     |
| Dead code / GC                  | $\varepsilon$ erasure                                    |
| All compute                     | $\omega$ op with MLIR label                              |
| Effect scoping / E-classes      | Boundary `[ ]`                                           |
| Effects perform/handle          | op(+, eff) floating to op(-, eff)                        |
| Continuations                   | wire held by handler op                                  |
| Membranes                       | Boundary `[ ]`                                           |

Effects, continuations, and membranes aren't separate nodes. They're **emergent behaviors** of the five primitives plus boundaries.

## The Rule Set — 4 Rules

### Rule 1 — Annihilation

```
γ⁺ ⋈ γ⁻  →  splice auxiliary wires

    left₁  right₁        left₁   right₁
       \   /                 |       |
       γ⁺  γ⁻    →          |       |
       /   \                 |       |
    left₂  right₂        left₂   right₂
```

Principal ports meet, nodes vanish, auxiliary wires connect directly. This is $O(1)$ and requires **no allocation**.

*   **Op variant:** When `op(+, f)` meets `op(-, f)` with a matching label, the MLIR operation fires and result wires splice through.
*   **Effect variant:** `op(+, eff)` floating upward meets an `op(-, eff)` handler. The handler's auxiliary ports carry the continuation wire—the suspended computation's principal port, held as a value.

### Rule 2 — Duplication

```
δ ⋈ γ⁺  →  two γ⁺ nodes, δ crosses through

       |                  |        |
       δ                 γ⁺       γ⁺
       |        →        / \     / \
      γ⁺               δ   δ   l   r
      / \               |   |
     l   r             (l) (r)
```

The $\delta$ node commutes through $\gamma^+$, producing two copies. This is **lazy by default.** The two $\delta$ nodes don't fire until something demands them. Sharing is preserved.

*   **Erasure variant:** $\delta \bowtie \varepsilon \rightarrow$ two $\varepsilon$ nodes. Erasure propagates (parallel GC).

### Rule 3 — Erasure Propagation

```
ε ⋈ γ⁺  →  ε on each auxiliary port
ε ⋈ γ⁻  →  ε on each auxiliary port
ε ⋈ op  →  cancel op, ε on each auxiliary port
ε ⋈ δ   →  ε on each auxiliary port
```

Erasure uniformly propagates. Every auxiliary port gets an $\varepsilon$. This makes dead code elimination and effect abortion structurally identical. **This is your GPU GC pass**, propagating simultaneously through subgraphs.

### Rule 4 — Congruence

```
Inside [ ]:
If subgraph A and subgraph B have identical structure and identical interface wires
→  merge into single subgraph, redirect all wires to it
```

This is the E-graph rule. It fires inside boundaries continuously. Two expressions that reduce to the same normal form collapse to one. When a node finishes reducing, we hash its normal form and look it up inside the boundary.

## Boundary Semantics — 3 Sub-Rules

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
│                 COMBINATOR GRAPH                    │
│                                                     │
│   γ⁺  γ⁻  δ  ε  op[...]   inside  [ boundaries ]  │
│                                                     │
│   Active pairs detected via port connection scan   │
└──────────┬──────────────────────────┬──────────────┘
           │ classify by label         │
           ▼                           ▼
┌──────────────────┐       ┌──────────────────────────┐
│   CPU POOL       │       │      GPU DISPATCH         │
│                  │       │                           │
│  γ⁺ ⋈ γ⁻        │       │  op[arith.*] batches      │
│  δ commutation   │       │  op[linalg.*] batches     │
│  Boundary ops    │       │  op[gpu.*] batches        │
│  Effect routing  │       │  ε propagation sweeps     │
│  Work stealing   │       │  Congruence hashing       │
│                  │       │                           │
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
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│                 MLIR LOWERING                       │
│                                                     │
│  pic.graph dialect  →  pic.op dialect               │
│  pic.op dialect     →  gpu.launch + llvm            │
│  Boundaries         →  MLIR regions                 │
│  γ annihilation     →  MLIR control flow            │
│  op[linalg.*]       →  linalg dialect (auto-tiled)  │
│  op[gpu.*]          →  SPIR-V or PTX                │
└─────────────────────────────────────────────────────┘
```

## How Each Feature Emerges

*   **Garbage collection:** Dead code is a subgraph with $\varepsilon$ at its root. Rule 3 propagates it in parallel.
*   **Algebraic effects:** Unhandled ops float via B1. The handler holds the continuation. Resume = feed value, Abort = feed $\varepsilon$.
*   **Lazy evaluation:** $\delta$ delays copying until copies diverge.
*   **Operator fusion:** Ops producing the same result inside a boundary collapse via Rule 4.
*   **Parallel execution:** Active pairs with no shared nodes fire simultaneously (by construction).
*   **CPU/GPU routing:** Active pair labels determine routing (two queues, zero programmer annotation).

## The Invariants

1.  **Linearity of principal ports:** Every principal port connects to exactly one wire (non-interfering).
2.  **Boundary interface consistency:** Both sides of a boundary must agree on wire polarity.
3.  **Confluence:** Rules 1-4 are confluent.

## The MLIR Dialects

Three dialects lower sequentially:

*   **`pic.graph`**: ops: agent, wire, boundary, active_pair (abstract combinator graph)
*   **`pic.reduce`**: ops: annihilate, duplicate, erase, fire_op, merge_boundary (reduction engine, scheduling-aware)
*   **`pic.runtime`**: ops: cpu_dispatch, gpu_batch, sync_point (maps to LLVM/GPU)

The `pic.graph` $\rightarrow$ `pic.reduce` pass normalizes the interaction net. The `pic.reduce` $\rightarrow$ `pic.runtime` pass acts as the CPU/GPU router based on `fire_op` labels.
