//===- InetDialectCAPI.h - C API for Inet Dialect -------------------*- C -*-===//
//
// This is the C API declaration for the Inet dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LINALANG_DIALECT_INET_CAPI_H
#define LINALANG_DIALECT_INET_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers the Inet dialect with the given context.
MLIR_CAPI_EXPORTED void mlirContextRegisterInetDialect(MlirContext context);

/// Returns the namespace for the Inet dialect.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Inet, inet);

#ifdef __cplusplus
}
#endif

#endif // LINALANG_DIALECT_INET_CAPI_H
