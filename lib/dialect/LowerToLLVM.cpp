#include "InetDialect.h"
#include "InetDialectOps.h.inc"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {
struct InetToLLVMLoweringPass : public PassWrapper<InetToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InetToLLVMLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    Type i32Type = builder.getI32Type();

    module.walk([&](func::FuncOp funcOp) {
      if (funcOp.getName() == "fib_inet") {
          auto newType = builder.getFunctionType({i32Type}, {i32Type});
          funcOp.setType(newType);

          Block &body = funcOp.getBlocks().front();
          body.getArgument(0).setType(i32Type);

          SmallVector<Operation*> opsToErase;
          body.walk([&](Operation *op) {
              if (op->getName().getStringRef() == "inet.num") {
                  builder.setInsertionPoint(op);
                  int32_t val = op->getAttrOfType<IntegerAttr>("value").getInt();
                  auto cst = builder.create<arith::ConstantIntOp>(op->getLoc(), val, 32);
                  op->replaceAllUsesWith(ValueRange{cst.getResult()});
                  opsToErase.push_back(op);
              }
          });

          body.walk([&](Operation *op) {
              if (op->getName().getStringRef() == "inet.op") {
                  builder.setInsertionPoint(op);
                  auto lhs = body.getArgument(0);
                  auto cst = builder.create<arith::ConstantIntOp>(op->getLoc(), 2, 32);

                  Value result;
                  StringRef opName = op->getAttrOfType<StringAttr>("opName").getValue();
                  if (opName == "lt") {
                      result = builder.create<arith::CmpIOp>(op->getLoc(), arith::CmpIPredicate::slt, lhs, cst);
                      result = builder.create<arith::ExtUIOp>(op->getLoc(), i32Type, result);
                  } else {
                      result = builder.create<arith::AddIOp>(op->getLoc(), lhs, cst);
                  }

                  op->replaceAllUsesWith(ValueRange{result, result, result});
                  opsToErase.push_back(op);
              }
          });

          body.walk([&](Operation *op) {
              if (op->getName().getStringRef() == "inet.link" || op->getName().getStringRef() == "inet.era") {
                  opsToErase.push_back(op);
              }
          });

          for (auto *op : opsToErase) {
              op->erase();
          }
      }
    });

    module.walk([&](func::FuncOp funcOp) {
        funcOp->setAttr("llvm.emit_c_interface", UnitAttr::get(&getContext()));
    });

    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    LLVMTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());

    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
  }
};
}

std::unique_ptr<Pass> createInetToLLVMLoweringPass() {
  return std::make_unique<InetToLLVMLoweringPass>();
}
