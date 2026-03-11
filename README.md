# Lin Programming Language

Welcome to the documentation for **Lin**, a multi-paradigm, statically typed language built on top of Interaction Nets to achieve massively parallel optimal evaluation.

Lin takes inspiration from the Red/Rebol syntax family while aiming for unparalleled runtime performance through a novel compiler backend. By compiling directly to a Polarized Interaction Combinator (PIC) model and generating highly optimized LLVM IR, Lin code evaluates via a lock-free, addressable 128-bit atomic array—enabling massive parallelization across CPU threads with zero locks.

## Goals
1. **Massively Parallel Optimal Evaluation:** Execute graph-based programs without locks, drastically reducing synchronization overhead and leveraging massive core counts.
2. **Predictable Performance:** Utilize lock-free Ahead-Of-Time (AOT) execution mechanisms.
3. **Clean, Expressive Syntax:** Adopt a clean, whitespace-friendly Red/Rebol-style syntax for rapid development without sacrificing the strict safety of static typing.
4. **Cutting-Edge Backend:** Heavily leverage the MLIR compiler infrastructure to enable sophisticated graph optimization, including e-graph based reductions.

## Getting Started

Check out the detailed documentation based on your role:

* **[User Guide](docs/Users.md):** Learn how to write Lin code, understand the current syntax, and see the planned roadmap of language features.
* **[Architecture](docs/Architecture.md):** Dive deep into how Lin achieves optimal evaluation through Interaction Nets, MLIR, and 128-bit atomic memory arrays.
* **[Contributor Guide](docs/Contributors.md):** Get the project running locally, build the compiler using Nix, and understand how to contribute to the AST, MLIR dialects, or runtime.

## Quick Example

```lin
fib: func [
    n [i32!]
    return: [i32!]
][
    either n < 2 [
        n
    ][
        (fib n - 1) + (fib n - 2)
    ]
]
```
