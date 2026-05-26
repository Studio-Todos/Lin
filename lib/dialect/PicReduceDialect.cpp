//===- PicReduceDialect.cpp - PIC Reduce dialect implementation --------===//
#include "PicReduceDialect.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::pic::reduce;

#include "PicReduceDialect.cpp.inc"

#define GET_OP_CLASSES
#include "PicReduceDialectOps.cpp.inc"

void PicReduceDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "PicReduceDialectOps.cpp.inc"
      >();
}
