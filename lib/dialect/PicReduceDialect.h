//===- PicReduceDialect.h - PIC Reduce dialect definition ----*- C++ -*-===//
#ifndef LINALANG_DIALECT_PIC_REDUCE_H
#define LINALANG_DIALECT_PIC_REDUCE_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "PicReduceDialect.h.inc"

#define GET_OP_CLASSES
#include "PicReduceDialectOps.h.inc"

#endif // LINALANG_DIALECT_PIC_REDUCE_H
