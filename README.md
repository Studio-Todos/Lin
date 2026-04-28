<p align="center">
  <img src="docs/lin.svg" alt="Lin logo" width="300" />
</p>

# Lin Programming Language

Welcome to the documentation for **Lin**, which serves as a general-purpose, heterogeneous programming language. Lin is a multi-paradigm, statically typed language built on top of Polarized Interaction Combinators (PIC) to achieve massively parallel, optimal evaluation.

Lin takes heavy inspiration from the Red/Rebol syntax family, [Daslang](https://daslang.io/), Microsoft's Verona project, and the unified algebra system while aiming for unparalleled runtime performance through a novel compiler backend. By compiling directly to an Interaction Net model and generating highly optimized LLVM IR (leveraging MLIR), Lin code evaluates via a lock-free, addressable 128-bit atomic array—enabling massive parallelization across CPU threads and GPU cores with zero locks. Because everything lowers to interaction nets, **everything that can run in parallel *will* run in parallel.**

## Goals

1. **Massively Parallel Optimal Evaluation:** Execute graph-based programs without locks, drastically reducing synchronization overhead and leveraging massive core counts on standard CPUs, general purpose GPUs, and high-performance Shaders.
2. **Predictable Performance:** Utilize lock-free Ahead-Of-Time (AOT) execution mechanisms. The compiler translates the interaction net into a dedicated MLIR `inet` dialect representing a flat, pre-allocated memory array compiled directly to native machine code.
3. **Clean, Expressive Syntax:** Adopt a clean, whitespace-friendly Red/Rebol-style syntax for rapid development. While Red/Rebol traditionally supports optional type annotations, Lin strictly enforces them to ensure the safety of static typing.
4. **Cutting-Edge Backend:** Heavily leverage the MLIR compiler infrastructure to enable sophisticated graph optimization, including E-graphs as an IR such that all code is optimal and parallel. This also allows for supercompilation via tools like Google's Souper.
5. **Language Philosophy:**
   - **Transparent** (don’t end up like Python)
   - **Minimal featureset** (unlike Java)
   - **Low abstraction** (leveraging the Red/Rebol syntax and build environment to make nearly everything configurable, carefully balancing this against the inherent complexities of interaction net abstraction)
   - **The primary form of abstraction in Lin is scheduling. Further abstraction is left to the developer.**
   - **Versatile syntax**

## Getting Started

Check out the detailed documentation based on your role:

* **[User Guide](docs/Users.md):** Learn how to write Lin code, understand the syntax, and explore planned features like the stochastic accuracy system.
* **[Architecture](docs/Architecture.md):** Dive deep into how Lin achieves optimal evaluation through Polarized Interaction Combinators (PIC), MLIR, and 128-bit atomic memory arrays.
* **[Contributor Guide](docs/Contributors.md):** Get the project running locally, build the compiler using Nix, and understand how to contribute. The compiler is primarily written in C, utilizing C++ only when necessary (e.g., for MLIR integration), and uses LLVM's LIT for testing.

## The Compiler: `linc`

The Lin compiler is called **`linc`** (pronounced "link"). It compiles Lin source code into a single static binary called a `.line` (Lin executable), taking direct inspiration from Go's approach to near-instant cross-compilation and single static binaries.

Future iterations of `linc` will feature:
* **Hot code reloading** to facilitate rapid iterative development.
* **Built-in code styling evaluation**, integrated directly as a built-in step that runs automatically for all projects during `linc test`.

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

## Examples

* **[Wellspring](https://github.com/Studio-Todos/Wellspring)**: An example project using Lin.
