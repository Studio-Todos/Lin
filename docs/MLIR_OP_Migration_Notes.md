# MLIR-OP Migration Notes

The `mlir-op` macro system in Lin provides an incredibly powerful way to bridge the gap between the Interaction Net graph structure and MLIR dialects without modifying the core C compiler (`Lowering.c`). This keeps the compiler frontend minimal while making the language extensible.

While we determined that arbitrary lazy control flow blocks (like `while` loops executing user code) must remain within the compiler (due to Lin strictly evaluating arguments before function/macro calls), many built-in compiler primitives can and should be extracted into standard library `mlir-op` definitions.

## Systems That Should Be Migrated to `mlir-op`

1. **Control Flow & Branching (`either`)**
   - Currently, `either` is a built-in AST construct (`AST_EITHER`). It lowers to an `omega` branch agent which evaluates a condition and routes execution wires.
   - **Migration Path:** Since `either` evaluates a boolean and acts as a wire router for interaction nets, we can theoretically replace the `AST_EITHER` compiler logic with an `mlir-op` called `either` or `branch` in `std/control.lin`, provided we can pass blocks as opaque continuations/thunks.
   - **Why:** This removes conditional evaluation hardcoding from `Lowering.c`, aligning with the Red/Rebol philosophy of everything being a standard function call.

2. **Primitive Type Conversions (Casting)**
   - Converting between `i32`, `i64`, `f32`, and `f64` currently requires manual LLVM insertions or native compiler support.
   - **Migration Path:** Add `std/cast.lin` with `mlir-op` definitions (e.g., `i32-to-f32`, `f32-to-i32`) leveraging the LLVM dialect (`llvm.sitofp`, `llvm.fptosi`, `llvm.sext`, etc.).
   - **Why:** This completely offloads the complexity of MLIR bitcasting and value extension logic out of C, making numerical precision easily extensible.

3. **String Allocation & Registries**
   - Strings (`AST_STRING`) currently lower to `pic_graph.agent` nodes with `label="str"` and custom MLIR string attributes dynamically appended in `Lowering.c`.
   - **Migration Path:** While the parsing of the string literal is still required in the Lexer, the lowering logic itself can be defined via a generic `mlir-op` that maps a raw char pointer to global LLVM memory arrays.
   - **Why:** Simplifies the core compiler and allows standard library macros to handle string memory management (e.g., GC vs `malloc`).

4. **Heap Memory Access (Pointers / State Management)**
   - Future features involving dynamic structured types (Objects/Arrays) will require pointer manipulation.
   - **Migration Path:** Define memory allocators (`malloc`, `free`) and memory access ops (`load`, `store`) entirely via `mlir-op` in `std/memory.lin` using the `!llvm.ptr` dialect, while threading a linear `state` token to preserve Interaction Net linearity and parallel safety.
   - **Why:** Keeps the Interaction Net backend mathematically pure; memory safety and aliasing logic are shifted to the `mlir-op` interface and standard library verifiers rather than complex C-level graph node tracking.

5. **Fixed-Bounded Iteration (`loop_n`)**
   - While a `while` loop taking user blocks requires compiler integration, a fixed loop that repeatedly applies a function or operates over arrays can be done cleanly.
   - **Migration Path:** Implement standard iteration helpers in `std/loop.lin` utilizing `scf.for` or `scf.while` within the MLIR inline payload.
   - **Why:** Reduces the need for writing recursive interaction net boilerplate for simple matrix math or array unrolling operations.
