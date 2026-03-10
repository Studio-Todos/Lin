//===- InetDialect.h - Inet dialect definition ----------------------*- C++ -*-===//
//
// This is the declaration file for the Inet dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LINALANG_DIALECT_INET_INETDIALECT_H
#define LINALANG_DIALECT_INET_INETDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the auto-generated dialect declarations.
#include "InetDialect.h.inc"

#define GET_OP_CLASSES
#include "InetDialectOps.h.inc"

namespace mlir {
namespace inet {

class PortType : public mlir::Type::TypeBase<PortType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "inet.port";
};

} // namespace inet
} // namespace mlir

#endif // LINALANG_DIALECT_INET_INETDIALECT_H
