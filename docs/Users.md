# User Guide

Welcome to the Lin programming language! Lin is designed to be highly expressive, borrowing structural concepts from Red/Rebol, while being completely statically typed and compiled via MLIR.

## Current Syntax

The currently implemented syntax supports basic structural programming, variable assignment, function declarations, and arithmetic.

### Variable Assignment

Variables are assigned using a colon `:` suffix on the identifier:
```lin
a: 5
b: a + 1
```

### Basic Arithmetic

Lin supports basic binary operations evaluated left-to-right (or respecting parentheses):
```lin
[ (a + b) * 2 ]
```
Supported operators currently include `+`, `-`, `<`, `>`, `*`, `/`, and `!`.

### Control Flow (`either`)

Branching is done using the `either` keyword, which evaluates a condition and executes the first block if true, or the second if false:

```lin
either a < 2 [
    a
][
    a - 1
]
```

### Functions

Functions are declared using the `func` keyword, followed by argument blocks, return types, and the body. Note the strict type annotations (e.g., `[i32!]`).

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

## Planned/Future Features

While Lin currently supports a minimal core, it is fundamentally a multi-paradigm language. The following features are planned for future releases:

1. **Rich Type System Expansion:**
   - More primitives (`f32`, `f64`, `bool`, `str`).
   - Complex structured types (Objects/Structs, Arrays).
   - Polymorphism and type inference.
2. **Advanced Control Flow:**
   - Loop constructs (`while`, `foreach`).
   - Pattern matching and algebraic data types.
3. **First-Class Concurrency Constructs:**
   - Given the massively parallel Interaction Net backend, future syntax will expose elegant ways to spawn and manage parallel workloads inherently, requiring zero manual lock management from the developer.
4. **Homoiconic Metaprogramming:**
   - Embracing its Red/Rebol roots, future Lin versions aim to treat code as data, allowing for powerful macros and DSL creation directly in the language.
