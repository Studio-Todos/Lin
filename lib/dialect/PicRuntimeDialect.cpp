//===- PicRuntimeDialect.cpp - PIC Runtime dialect implementation ------===//
#include "PicRuntimeDialect.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::pic::runtime;

#include "PicRuntimeDialect.cpp.inc"

#define GET_OP_CLASSES
#include "PicRuntimeDialectOps.cpp.inc"

void PicRuntimeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PicRuntimeDialectOps.cpp.inc"
      >();
}

NodeType mlir::pic::runtime::nodeTypeForAgent(StringRef agentType, uint32_t polVal) {
    if (agentType == "constructor") return NODE_CON;
    if (agentType == "destructor") return NODE_DES;
    if (agentType == "gamma") return (polVal == 0) ? NODE_CON : NODE_DES;
    if (agentType == "delta") return NODE_DUP;
    if (agentType == "epsilon") return NODE_ERA;
    if (agentType == "history" || agentType == "H") return NODE_HIS;
    if (agentType == "log_bond" || agentType == "L") return NODE_LOG;
    if (agentType == "reverse_vector" || agentType == "R") return NODE_RVEC;
    return NODE_OP;
}
