//===- PicRuntimeDialect.h - PIC Runtime dialect definition --*- C++ -*-===//
#ifndef LINALANG_DIALECT_PIC_RUNTIME_H
#define LINALANG_DIALECT_PIC_RUNTIME_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "PicRuntimeDialect.h.inc"

#define GET_OP_CLASSES
#include "PicRuntimeDialectOps.h.inc"

namespace mlir {
namespace pic {
namespace runtime {

// Node type constants — single source of truth for the 7 PICR node types.
// Encoded in the lower 6 bits of AllocNodeOp's $type attribute (bits 6-7 = polarity).
enum NodeType : uint8_t {
    NODE_CON  = 1,  // Constructor  (γ⁺)
    NODE_DES  = 2,  // Destructor   (γ⁻)
    NODE_DUP  = 3,  // Duplicator   (δ)
    NODE_ERA  = 4,  // Eraser       (ε)
    NODE_OP   = 5,  // Operation    (ω)
    NODE_HIS  = 6,  // History      (H)
    NODE_LOG  = 7,  // Log-Bond     (L)
    NODE_RVEC = 8,  // Reverse-Vec  (R)
};

// Derives the node type from an agent type string and polarity value.
// agentType: "constructor", "destructor", "gamma", "delta", "epsilon",
//            "history"/"H", "log_bond"/"L", "reverse_vector"/"R"
// polVal: 0 = positive (+), 1 = negative (-), 2 = erase (*)
// Returns the NodeType constant (defaults to NODE_OP for unknown types).
NodeType nodeTypeForAgent(StringRef agentType, uint32_t polVal);

} // namespace runtime
} // namespace pic
} // namespace mlir

#endif // LINALANG_DIALECT_PIC_RUNTIME_H
