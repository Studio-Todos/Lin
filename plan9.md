The code review found a few bugs:
1. `std::string` is constructed from `NULL` pointers without checks, which causes Segfaults.
   - `std::string outputs(funcNode->as.mlir_op.outputs, ...)`
   - `std::string argTypeName(...)`
   - `std::string expectedType(...)`
   - `std::string(node->as.call.args[1]->as.pair.left->resolved_type, ...)`
2. Memory leak: `setResolvedType` allocates memory using `strdup`, but `freeAst` in `src/Parser.c` does not free `resolved_type`.

I will:
1. Fix `src/main.cpp` by adding a safe string construction helper or wrapping pointer accesses with checks for `nullptr`.
2. Fix `src/Parser.c` inside `freeAst` to free `node->resolved_type`.
3. Test locally again before finishing the step.
