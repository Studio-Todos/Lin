#ifndef LINALANG_LOWERING_PASSES_H
#define LINALANG_LOWERING_PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createPicGraphToReducePass();
std::unique_ptr<mlir::Pass> createPicReduceToRuntimePass();
std::unique_ptr<mlir::Pass> createPicRuntimeToLLVMPass();
std::unique_ptr<mlir::Pass> createPicRuntimeToGPUPass();

#endif // LINALANG_LOWERING_PASSES_H
