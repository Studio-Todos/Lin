#ifndef LINALANG_LOWER_TO_LLVM_CAPI_H
#define LINALANG_LOWER_TO_LLVM_CAPI_H

#include "mlir-c/Pass.h"

#ifdef __cplusplus
extern "C" {
#endif

MlirPass mlirCreateInetToLLVMLoweringPass();

#ifdef __cplusplus
}
#endif

#endif // LINALANG_LOWER_TO_LLVM_CAPI_H
