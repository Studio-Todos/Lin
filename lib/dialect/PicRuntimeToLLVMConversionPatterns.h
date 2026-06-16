#ifndef PIC_RUNTIME_TO_LLVM_CONVERSION_PATTERNS_H
#define PIC_RUNTIME_TO_LLVM_CONVERSION_PATTERNS_H

#include "PicReduceUtils.h"
#include "PicRuntimeDialect.h"
#include "PicRuntimeToLLVMConversions.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::pic::runtime;

static func::FuncOp getParentFunc(Operation *op) {
  return op->getParentOfType<func::FuncOp>();
}

static Value getStateArg(func::FuncOp f) {
  return f.getArgument(f.getNumArguments() - 1);
}

struct AllocNodeOpPattern : public OpRewritePattern<pic::runtime::AllocNodeOp> {
  AllocNodeOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::AllocNodeOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    Value result = convertAllocNodeOp(rewriter, op, stateArg, f);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SetPortOpPattern : public OpRewritePattern<pic::runtime::SetPortOp> {
  SetPortOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::SetPortOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    convertSetPortOp(rewriter, op, stateArg);
    rewriter.eraseOp(op);
    return success();
  }
};

struct GetPortOpPattern : public OpRewritePattern<pic::runtime::GetPortOp> {
  GetPortOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::GetPortOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    Value result = convertGetPortOp(rewriter, op, stateArg);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GetPortDynamicOpPattern : public OpRewritePattern<pic::runtime::GetPortDynamicOp> {
  GetPortDynamicOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::GetPortDynamicOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    Value result = convertGetPortDynamicOp(rewriter, op, stateArg);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LinkOpPattern : public OpRewritePattern<pic::runtime::LinkOp> {
  LinkOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::LinkOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    convertLinkOp(rewriter, op, stateArg, f);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PushRedexOpPattern : public OpRewritePattern<pic::runtime::PushRedexOp> {
  PushRedexOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::PushRedexOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    convertPushRedexOp(rewriter, op, stateArg);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PopRedexOpPattern : public OpRewritePattern<pic::runtime::PopRedexOp> {
  PopRedexOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::PopRedexOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    auto results = convertPopRedexOp(rewriter, op, stateArg, f);
    rewriter.replaceOp(op, {results[0], results[1], results[2]});
    return success();
  }
};

struct GetHistoryOpPattern : public OpRewritePattern<pic::runtime::GetHistoryOp> {
  GetHistoryOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::GetHistoryOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    Value result = convertGetHistoryOp(rewriter, op, stateArg);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SetHistoryOpPattern : public OpRewritePattern<pic::runtime::SetHistoryOp> {
  SetHistoryOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::SetHistoryOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    convertSetHistoryOp(rewriter, op, stateArg);
    rewriter.eraseOp(op);
    return success();
  }
};

struct UncomputeSweepOpPattern : public OpRewritePattern<pic::runtime::UncomputeSweepOp> {
  UncomputeSweepOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::UncomputeSweepOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    convertUncomputeSweepOp(rewriter, op, stateArg, f);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CheckpointBoundaryOpPattern : public OpRewritePattern<pic::runtime::CheckpointBoundaryOp> {
  CheckpointBoundaryOpPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(pic::runtime::CheckpointBoundaryOp op, PatternRewriter &rewriter) const override {
    func::FuncOp f = getParentFunc(op);
    Value stateArg = getStateArg(f);
    convertCheckpointBoundaryOp(rewriter, op, stateArg);
    rewriter.eraseOp(op);
    return success();
  }
};

static void populatePicRuntimeToLLVMConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<AllocNodeOpPattern, SetPortOpPattern, GetPortOpPattern,
               GetPortDynamicOpPattern, LinkOpPattern, PushRedexOpPattern,
               PopRedexOpPattern, GetHistoryOpPattern, SetHistoryOpPattern,
               UncomputeSweepOpPattern, CheckpointBoundaryOpPattern>(
      patterns.getContext());
}

#endif // PIC_RUNTIME_TO_LLVM_CONVERSION_PATTERNS_H