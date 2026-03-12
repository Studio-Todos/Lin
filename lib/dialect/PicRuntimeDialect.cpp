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
