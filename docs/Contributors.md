# Contributor Guide

Thank you for your interest in contributing to the Lin programming language! This guide will help you set up your development environment and explain the build and testing process.

## Prerequisites

Lin relies on a specific set of tools, managed easily via Nix Flakes. The project specifically targets **LLVM/MLIR 18**.

- [Nix Package Manager](https://nixos.org/download.html)
- Flakes enabled in your Nix configuration.

## Environment Setup

To enter the isolated development environment with all required dependencies (LLVM 18, MLIR, CMake, Ninja, Python3, lit, GTest), run:

```bash
nix develop
```

This will drop you into a shell with `LLVM_DIR` and `MLIR_DIR` exported correctly.

## Building the Compiler

The initial implementation heavily leverages MLIR to establish a robust baseline and bootstrap the compiler quickly.

Lin uses CMake and Ninja for its build system. Inside the `nix develop` shell, compile the project with:

```bash
mkdir -p build && cd build
cmake -G Ninja ..
ninja
```

This generates the `linc` compiler executable.

## Running Tests

Lin utilizes the `lit` (LLVM Integrated Tester) framework for its core testing suite. The tests verify parser behavior, AST generation, MLIR lowering, and end-to-end execution.

To run the test suite:

```bash
cd build
lit ../test
# Or, if lit is configured via CMake:
ninja check-linc
```

*Note: In the future, a built-in checkstyle test will run automatically for projects executing `linc test`.*

## Project Structure

- `src/`: Core C/C++ source code. Contains the parser (`Parser.c`), AST definition, compiler driver (`main.cpp`), and C-to-MLIR lowering logic (`Lowering.c`). The codebase is primarily written in C, utilizing C++ strictly where necessary to interface with MLIR and LLVM APIs.
- `include/lin/`: Headers for the parser and lowering modules.
- `lib/dialect/`: TableGen definitions (`.td` files) and C++ implementations for the `pic.graph`, `pic.reduce`, and `pic.runtime` MLIR dialects. These form the Polarized Interaction Combinators graph rules.
- `lib/runtime/`: C runtime implementation (linked with compiled `.line` Lin binaries).
- `test/`: `lit` test configuration and `.lin` test files.

## Contribution Guidelines

1. **AST & Parser changes:** If modifying syntax, update `src/Parser.c` and add corresponding test files in `test/`. Ensure `lit` catches any syntax errors.
2. **MLIR changes:** Modifications to the Interaction Net representation should be made in `lib/dialect/`. Re-run `ninja` to trigger TableGen and regenerate headers.
3. **C++ Standard:** The backend uses C++17. The frontend parser uses C11. Stick to C for general compiler logic where possible, relying on C++ for LLVM API boundaries.
