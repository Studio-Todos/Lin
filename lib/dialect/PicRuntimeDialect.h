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

#endif // LINALANG_DIALECT_PIC_RUNTIME_H
