#include "lin/LoweringPasses.h"
#include "PicGraphDialect.h"
#include "PicReduceDialect.h"
#include "PicRuntimeDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/IRMapping.h"
#include "PicReduceUtils.h"
#include <set>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <map>

using namespace mlir;

namespace {

struct PicReduceToRuntimePass : public PassWrapper<PicReduceToRuntimePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicReduceToRuntimePass)

  TargetBackend target = TargetBackend::CPU;
  PicReduceToRuntimePass() : target(TargetBackend::CPU) {}
  PicReduceToRuntimePass(TargetBackend target) : target(target) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());

    Location loc = module.getLoc();
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    auto i8Type = builder.getI8Type();
    auto f32Type = builder.getF32Type();
    auto f64Type = builder.getF64Type();

    auto declFunc = [&](StringRef name, Type ret, ArrayRef<Type> args) {
        if (!module.lookupSymbol(name)) {
            auto fType = builder.getFunctionType(args, ret ? TypeRange{ret} : TypeRange{});
            auto f = builder.create<func::FuncOp>(loc, name, fType);
            f.setPrivate();
        }
    };

    declFunc("lookup_rule", i32Type, {i32Type, i32Type});
    declFunc("get_num_args", i32Type, {i32Type});
    declFunc("is_gpu_op", i1Type, {i32Type});
    declFunc("dispatch_user_op", i64Type, {i32Type, i64Type, i64Type, i64Type, i64Type, i32Type, i32Type, i64Type});

    auto wType = builder.getFunctionType({i64Type}, {i64Type});
    auto wFunc = builder.create<func::FuncOp>(loc, "worker_thread", wType);
    Block *wEntry = wFunc.addEntryBlock();
    Value stateArg = wEntry->getArgument(0);

    Block *lHead = wFunc.addBlock();
    Block *lBody = wFunc.addBlock();
    Block *lEnd = wFunc.addBlock();

    builder.setInsertionPointToStart(wEntry);
    // Helpers
    Value c0_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0));
    Value c1_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(1));
    Value c2_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2));
    Value c3_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(3));
    Value c7_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(7));
    Value c8_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(8));
    Value c24_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(24));
    Value c30_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(30));
    Value c0xFFFFFF_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0xFFFFFF));
    Value c0x3F_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x3F));

    Value c0_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0));
    Value c1_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1));
    Value c2_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2));
    Value c32_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(32));
    Value c0xFFFFFFFF_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0xFFFFFFFF));
    Value c0x100000000_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0x100000000ULL));
    Value c0xFFFFFFFF00000000_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0xFFFFFFFF00000000ULL));

    std::vector<uint32_t> literalHashes;
    for (const auto &lit : {"num", "i1", "i8", "i16", "i32", "i64", "f32", "f64", "bool", "str"}) {
        literalHashes.push_back(opcodeForLabel(lit));
    }
    module.walk([&](pic::graph::RegistryOp op) {
        StringRef key = op.getOpName();
        if (key.starts_with("STR_")) {
            literalHashes.push_back(opcodeForLabel(key));
        }
    });

    auto isLiteralLabel = [&](Value label) -> Value {
        Value isLit = builder.create<arith::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));
        for (uint32_t hash : literalHashes) {
            Value hashConst = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(hash));
            Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, label, hashConst);
            isLit = builder.create<arith::OrIOp>(loc, isLit, cmp);
        }
        return isLit;
    };

    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(lHead);
    auto popOp = builder.create<pic::runtime::PopRedexOp>(loc, i1Type, i32Type, i32Type);
    Value valid = popOp.getValid();
    Value nodeA = popOp.getNodeA();
    Value nodeB = popOp.getNodeB();
    builder.create<cf::CondBranchOp>(loc, valid, lBody, ValueRange{nodeA, nodeB}, lEnd, ValueRange{});

    // Body block setup
    Value bodyNodeA = lBody->addArgument(i32Type, loc);
    Value bodyNodeB = lBody->addArgument(i32Type, loc);
    builder.setInsertionPointToStart(lBody);

    auto makePortVal = [&](Value idx, int p) {
        Value sh = builder.create<arith::ShLIOp>(loc, i32Type, idx, c2_i32);
        return builder.create<arith::OrIOp>(loc, i32Type, sh, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(p)));
    };

    Value metaA = builder.create<pic::runtime::GetPortOp>(loc, i32Type, bodyNodeA, builder.getI8IntegerAttr(3));
    Value metaB = builder.create<pic::runtime::GetPortOp>(loc, i32Type, bodyNodeB, builder.getI8IntegerAttr(3));

    Value polA = builder.create<arith::ShRUIOp>(loc, metaA, c30_i32);
    Value polB = builder.create<arith::ShRUIOp>(loc, metaB, c30_i32);

    Value labelA = builder.create<arith::AndIOp>(loc, metaA, c0xFFFFFF_i32);
    Value labelB = builder.create<arith::AndIOp>(loc, metaB, c0xFFFFFF_i32);

    Value typeValA = builder.create<arith::ShRUIOp>(loc, metaA, c24_i32);
    Value typeValB = builder.create<arith::ShRUIOp>(loc, metaB, c24_i32);

    Value nodeTypeA = builder.create<arith::AndIOp>(loc, typeValA, c0x3F_i32);
    Value nodeTypeB = builder.create<arith::AndIOp>(loc, typeValB, c0x3F_i32);

    Value isRvecA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nodeTypeA, c8_i32);
    Value isRvecB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nodeTypeB, c8_i32);
    Value hasRvec = builder.create<arith::OrIOp>(loc, isRvecA, isRvecB);

    Block *rvecCase = wFunc.addBlock();
    Block *nonRvecCase = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, hasRvec, rvecCase, nonRvecCase);

    StringAttr gpuFlag = (target == TargetBackend::GPU) ? builder.getStringAttr("gpu") : StringAttr();
    StringAttr cpuFlag = (target == TargetBackend::GPU) ? builder.getStringAttr("cpu") : StringAttr();

    builder.setInsertionPointToStart(rvecCase);
    builder.create<mlir::pic::reduce::ReverseVectorOp>(loc, bodyNodeA, bodyNodeB, stateArg, gpuFlag);
    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(nonRvecCase);
    Value isEraA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("era"))));
    Value isEraB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("era"))));
    Value hasEra = builder.create<arith::OrIOp>(loc, isEraA, isEraB);

    Block *eraCase = wFunc.addBlock();
    Block *checkDispatch = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, hasEra, eraCase, checkDispatch);

    builder.setInsertionPointToStart(eraCase);
    builder.create<mlir::pic::reduce::EraseOp>(loc, bodyNodeA, bodyNodeB, stateArg, gpuFlag);
    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(checkDispatch);
    Value implA = builder.create<func::CallOp>(loc, i32Type, "lookup_rule", ValueRange{labelA, labelB}).getResult(0);
    Value implB = builder.create<func::CallOp>(loc, i32Type, "lookup_rule", ValueRange{labelB, labelA}).getResult(0);
    Value hasRuleA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, implA, c0_i32);
    Value hasRuleB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, implB, c0_i32);
    Value hasDispatch = builder.create<arith::OrIOp>(loc, hasRuleA, hasRuleB);

    Block *dispatchCase = wFunc.addBlock();
    Block *genericCase = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, hasDispatch, dispatchCase, genericCase);

    builder.setInsertionPointToStart(dispatchCase);
    builder.create<mlir::pic::reduce::FireOpOp>(loc, bodyNodeA, bodyNodeB, stateArg, cpuFlag);
    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(genericCase);
    Value labelsMatch = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, labelB);
    Value polsDiff = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, polA, polB);
    Value isAnn = builder.create<arith::AndIOp>(loc, labelsMatch, polsDiff);

    Block *annPath = wFunc.addBlock();
    Block *dupPath = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isAnn, annPath, dupPath);

    builder.setInsertionPointToStart(annPath);
    builder.create<mlir::pic::reduce::AnnihilateOp>(loc, bodyNodeA, bodyNodeB, stateArg, gpuFlag);
    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(dupPath);
    builder.create<mlir::pic::reduce::DuplicateOp>(loc, bodyNodeA, bodyNodeB, stateArg, gpuFlag);
    builder.create<cf::BranchOp>(loc, lHead);

    // ==========================================
    // lEnd
    // ==========================================
    builder.setInsertionPointToStart(lEnd);
    builder.create<func::ReturnOp>(loc, ValueRange{c0_i64});
  }
};

} // namespace

std::unique_ptr<Pass> createPicReduceToRuntimePass(TargetBackend target) { return std::make_unique<PicReduceToRuntimePass>(target); }
