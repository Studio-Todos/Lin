# Architecture

Lin's architecture is built to support a groundbreaking execution model: **massively parallel optimal evaluation via Interaction Nets**. It shifts away from traditional sequential execution, transforming code into a highly concurrent graph that resolves optimally.

## Compiler Pipeline

1. **Frontend Parsing:**
   - Source files (`.lin`) are parsed by a custom C11 lexer and parser.
   - The parser constructs an Abstract Syntax Tree (AST).
2. **Lowering to MLIR (Polarized Interaction Combinators):**
   - The AST is walked and lowered into an MLIR representation using the C API.
   - Specifically, it lowers to the `pic_graph` dialect. This dialect models the program as a set of Interaction Net Agents (nodes) and Links (edges).
   - AST nodes like numbers, binary operations, and function calls are mapped to Agents (e.g., `omega` type for data, `delta` type for duplication).
3. **Interaction Net Optimization:**
   - (Planned) The MLIR graph is optimized using e-graph based reductions.
4. **Lowering to LLVM IR:**
   - A custom pass (`inet-to-llvm`) lowers the abstract `pic_graph` dialect into low-level LLVM IR.
   - Interaction Net rules are compiled into a highly optimized, addressable atomic array representation.
5. **AOT Compilation & Execution:**
   - The LLVM IR is compiled into an object file for the host machine.
   - The object file is linked with the Lin runtime (`runtime.c`) to produce the final binary.

## Massively Parallel Optimal Evaluation

The core philosophical architecture of Lin relies on Interaction Combinators.

### Interaction Nets
Instead of managing a call stack and sequential instructions, Lin code compiles into a graph of agents. Each agent has a "principal" port and auxiliary ports. When two agents connect via their principal ports, they interact and reduce according to fixed rules, creating new agents and connections.

### Lock-Free 128-bit Atomic Array Model
To execute this graph with extreme parallelism on modern CPUs, the Lin compiler lowers the interaction net to a polarized model:

1. **Memory Representation:** The entire state of the program is represented as a flat, 128-bit addressable array.
2. **Atomic Operations:** Reductions in the interaction net are implemented via lock-free atomic compare-and-swap (CAS) operations on this 128-bit array.
3. **Lock-Free Execution:** Because the graph is reduced via atomic updates to local segments of the array, thousands of threads can continuously search for active pairs (agents connected via principal ports) and reduce them concurrently. There are zero global locks, meaning execution scales almost linearly with core count.

### The MLIR Dialects
- **`PicGraphDialect`**: Represents the topology of the interaction net (Agents, Ports, Links).
- **`PicReduceDialect`**: Represents the interaction rules (how specific agents annihilate or duplicate each other).
- **`PicRuntimeDialect`**: Interfaces the graph state with the actual hardware 128-bit atomic array layout during LLVM lowering.
