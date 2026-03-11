//===- PicGraphDialect.h - PIC Graph dialect definition ------*- C++ -*-===//
#ifndef LINALANG_DIALECT_PIC_GRAPH_H
#define LINALANG_DIALECT_PIC_GRAPH_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Auto-generated headers
#include "PicGraphDialect.h.inc"

#define GET_OP_CLASSES
#include "PicGraphDialectOps.h.inc"

namespace mlir {
namespace pic {
namespace graph {

class PortType : public mlir::Type::TypeBase<PortType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "pic_graph.port";
};

} // namespace graph
} // namespace pic
} // namespace mlir

#endif // LINALANG_DIALECT_PIC_GRAPH_H
