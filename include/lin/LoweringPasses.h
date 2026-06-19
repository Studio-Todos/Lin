#ifndef LINALANG_LOWERING_PASSES_H
#define LINALANG_LOWERING_PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"
#include "PicReduceUtils.h"

#include <string>

std::unique_ptr<mlir::Pass> createPicGraphVerifyPass();
std::unique_ptr<mlir::Pass> createPicGraphToReducePass();
std::unique_ptr<mlir::Pass> createPicReduceToRuntimePass(TargetBackend target = TargetBackend::CPU);
std::unique_ptr<mlir::Pass> createPicReduceLoweringPass(TargetBackend target = TargetBackend::CPU);
std::unique_ptr<mlir::Pass> createPicRuntimeToLLVMPass(TargetBackend target, std::string spirvPath);
std::unique_ptr<mlir::Pass> createPicRuntimeToSPIRVPass();

#endif // LINALANG_LOWERING_PASSES_H