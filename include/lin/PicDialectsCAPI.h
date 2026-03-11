//===- PicDialectsCAPI.h - C API for PIC Dialects -------------------*- C -*-===//
#ifndef LINALANG_DIALECT_PIC_CAPI_H
#define LINALANG_DIALECT_PIC_CAPI_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers the PIC dialects with the given context.
MLIR_CAPI_EXPORTED void mlirContextRegisterPicDialects(MlirContext context);

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PicGraph, pic_graph);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PicReduce, pic_reduce);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PicRuntime, pic_runtime);

#ifdef __cplusplus
}
#endif

#endif // LINALANG_DIALECT_PIC_CAPI_H
