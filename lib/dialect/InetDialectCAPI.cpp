//===- InetDialectCAPI.cpp - C API implementation for Inet Dialect ------===//
//
// This is the implementation of the C API for the Inet dialect.
//
//===----------------------------------------------------------------------===//

#include "lin/InetDialectCAPI.h"
#include "InetDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace mlir::inet;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Inet, inet, InetDialect)

void mlirContextRegisterInetDialect(MlirContext context) {
  unwrap(context)->getOrLoadDialect<InetDialect>();
}
