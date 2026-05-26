#ifndef LINALANG_LOWERING_PASSES_H
#define LINALANG_LOWERING_PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"

#include <string>

std::unique_ptr<mlir::Pass> createPicGraphToReducePass();
std::unique_ptr<mlir::Pass> createPicReduceToRuntimePass(bool enableGPU, std::string spirvPath);
std::unique_ptr<mlir::Pass> createPicRuntimeToLLVMPass(bool enableGPU, std::string spirvPath);

#endif // LINALANG_LOWERING_PASSES_H
