//===- PicDialectsCAPI.cpp - C API implementation for PIC Dialects ------===//
#include "lin/PicDialectsCAPI.h"
#include "PicGraphDialect.h"
#include "PicReduceDialect.h"
#include "PicRuntimeDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace mlir::pic::graph;
using namespace mlir::pic::reduce;
using namespace mlir::pic::runtime;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PicGraph, pic_graph, PicGraphDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PicReduce, pic_reduce, PicReduceDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PicRuntime, pic_runtime, PicRuntimeDialect)

void mlirContextRegisterPicDialects(MlirContext context) {
  unwrap(context)->getOrLoadDialect<PicGraphDialect>();
  unwrap(context)->getOrLoadDialect<PicReduceDialect>();
  unwrap(context)->getOrLoadDialect<PicRuntimeDialect>();
}
