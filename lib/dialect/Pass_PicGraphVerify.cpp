#include "lin/LoweringPasses.h"
#include "PicGraphDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::pic::graph;

namespace {

struct PicGraphVerifyPass : public PassWrapper<PicGraphVerifyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicGraphVerifyPass)

  StringRef getArgument() const override { return "pic-graph-verify"; }
  StringRef getDescription() const override {
    return "Verifies Invariant 1: each principal port connects to at most one active pair";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    int errorCount = 0;

    module.walk([&](func::FuncOp funcOp) {
      DenseMap<Value, int> activePairCount;

      funcOp.walk([&](ActivePairOp ap) {
        activePairCount[ap.getP0()]++;
        activePairCount[ap.getP1()]++;
      });

      funcOp.walk([&](AgentOp agentOp) {
        Value p0 = agentOp.getP0();
        int count = activePairCount.lookup(p0);
        if (count > 1) {
          errorCount++;
          agentOp.emitOpError()
            << "principal port (p0) of " << agentOp.getAgentType() << " \""
            << agentOp.getLabel() << "\" participates in " << count
            << " active pairs, expected at most 1";
        }
      });
    });

    if (errorCount > 0)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createPicGraphVerifyPass() {
  return std::make_unique<PicGraphVerifyPass>();
}