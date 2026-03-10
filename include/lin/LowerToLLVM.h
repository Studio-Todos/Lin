#ifndef LINALANG_LOWER_TO_LLVM_H
#define LINALANG_LOWER_TO_LLVM_H

#include <memory>
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createInetToLLVMLoweringPass();

#endif // LINALANG_LOWER_TO_LLVM_H
