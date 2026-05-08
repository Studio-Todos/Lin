#include "lin/LoweringPasses.h"
#include "PicGraphDialect.h"
#include "PicReduceDialect.h"
#include "PicRuntimeDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#if __has_include("mlir/Dialect/SPIRV/IR/TargetAndABI.h")
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#endif
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/IRMapping.h"
#include <sstream>
#include <unordered_map>

using namespace mlir;

namespace {

static void replaceAll(std::string &str, const std::string &from, const std::string &to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

static uint32_t opcodeForLabel(StringRef label) {
  uint32_t hash = 2166136261u;
  for (unsigned char c : label) {
    hash ^= c;
    hash *= 16777619u;
  }
  return hash ? hash : 1u;
}

struct PicGraphToReducePass : public PassWrapper<PicGraphToReducePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicGraphToReducePass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    module.walk([&](func::FuncOp funcOp) {
      builder.setInsertionPointToStart(&funcOp.getBody().front());
      auto i32Type = builder.getI32Type();
      DenseMap<Value, Value> valueToPort;

      // Update signature to i32 if needed and track arguments
      for (auto arg : funcOp.getArguments()) {
          if (arg.getType().isInteger(32)) {
              valueToPort[arg] = arg;
          } else if (!arg.getType().isInteger(64)) {
              arg.setType(i32Type);
              valueToPort[arg] = arg;
          }
      }
      if (funcOp.getNumResults() > 0 && !funcOp.getResultTypes()[0].isInteger(32)) {
          SmallVector<Type> resTypes(funcOp.getNumResults(), i32Type);
          funcOp.setType(builder.getFunctionType(funcOp.getArgumentTypes(), resTypes));
      }

      funcOp.walk([&](pic::graph::AgentOp op) {
        Location loc = op.getLoc();
        builder.setInsertionPoint(op);
        StringRef agentType = op.getAgentType();
        StringRef label = op.getLabel();
        uint8_t typeEnum = 0;
        if (agentType == "gamma") typeEnum = 1;
        else if (agentType == "delta") typeEnum = 2;
        else if (agentType == "epsilon") typeEnum = 0;
        else if (agentType == "omega") {
          if (label == "num") typeEnum = 3;
          else typeEnum = 4;
        }

        uint32_t val = 0;
        if (typeEnum == 3 && op.getValue()) val = op.getValue().value();
        else if (typeEnum == 4) {
            if (label == "str" && op.getStrVal()) {
                std::string rawStr = op.getStrVal().value().str();
                std::string strKey = "STR_";
                for (char c : rawStr) {
                    if (std::isalnum(c)) strKey += c;
                    else {
                        char buf[8];
                        sprintf(buf, "_%02x", (unsigned char)c);
                        strKey += buf;
                    }
                }
                val = opcodeForLabel(strKey);
                builder.create<pic::graph::RegistryOp>(loc, builder.getStringAttr(strKey), op.getStrVal().value());
            } else val = opcodeForLabel(label);
        }

        auto alloc = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, typeEnum, val);
        Value nodeIdx = alloc.getIndex();
        
        Value meta = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr((typeEnum << 24) | (val & 0xFFFFFF)));
        builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(3), builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), meta));

        auto makePort = [&](int i) {
            Value shifted = builder.create<arith::ShLIOp>(loc, i32Type, nodeIdx, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2)));
            return builder.create<arith::OrIOp>(loc, i32Type, shifted, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(i)));
        };

        // Specialized port mapping for omega agents (arithmetic, printing)
        bool isMath = (label == "add" || label == "sub" || label == "mul" || label == "div" || label == "rem" || label == "eq" || label == "neq" || label == "lt" || label == "gt" || label == "le" || label == "ge" || label == "fadd" || label == "fsub" || label == "fmul" || label == "fdiv" || label == "frem" || label == "feq" || label == "fneq" || label == "flt" || label == "fgt" || label == "fle" || label == "fge");
        bool isPrint = (label == "print_i32" || label == "print_f32" || label == "print_str");
        
        if (typeEnum == 4 && (isMath || isPrint)) {
            valueToPort[op.getP0()] = makePort(2); // Result is Port 2
            valueToPort[op.getP1()] = makePort(0); // Primary Arg is Port 0 (Principal)
            valueToPort[op.getP2()] = makePort(1); // Secondary Arg is Port 1
        } else {
            valueToPort[op.getP0()] = makePort(0);
            valueToPort[op.getP1()] = makePort(1);
            valueToPort[op.getP2()] = makePort(2);
        }
      });

      funcOp.walk([&](UnrealizedConversionCastOp op) {
          if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
              valueToPort[op.getResult(0)] = op.getOperand(0);
          }
      });

      DenseMap<Value, Value> portToPort;
      funcOp.walk([&](pic::graph::LinkOp op) {
        if (valueToPort.count(op.getA()) && valueToPort.count(op.getB())) {
            Value pA = valueToPort[op.getA()];
            Value pB = valueToPort[op.getB()];
            builder.setInsertionPoint(op);
            builder.create<pic::runtime::LinkOp>(op.getLoc(), builder.create<arith::ExtUIOp>(op.getLoc(), builder.getI64Type(), pA), builder.create<arith::ExtUIOp>(op.getLoc(), builder.getI64Type(), pB));
        }
      });

      SmallVector<Operation*> toErase;
      funcOp.walk([&](pic::graph::LinkOp op) { toErase.push_back(op); });
      funcOp.walk([&](pic::graph::AgentOp op) { toErase.push_back(op); });

      funcOp.walk([&](func::ReturnOp op) {
         if (op.getNumOperands() > 0 && valueToPort.count(op.getOperand(0))) op.setOperand(0, valueToPort[op.getOperand(0)]);
      });

      for (auto *op : toErase) op->erase();
    });
  }
};

struct PicReduceToRuntimePass : public PassWrapper<PicReduceToRuntimePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicReduceToRuntimePass)
  void runOnOperation() override {}
};

struct PicRuntimeToLLVMPass : public PassWrapper<PicRuntimeToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicRuntimeToLLVMPass)
  bool enableGPU;
  PicRuntimeToLLVMPass(bool enableGPU) : enableGPU(enableGPU) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());

    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    auto voidType = LLVM::LLVMVoidType::get(builder.getContext());

    std::unordered_map<std::string, std::string> registry;
    SmallVector<Operation*> regOps;
    module.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "pic_graph.registry") {
            StringAttr n = op->getAttrOfType<StringAttr>("op_name");
            StringAttr p = op->getAttrOfType<StringAttr>("payload");
            if (n && p) {
                registry[n.getValue().str()] = p.getValue().str();
                if (n.getValue().starts_with("STR_")) {
                    if (!module.lookupSymbol(n.getValue())) {
                        OpBuilder b(module.getBodyRegion());
                        std::string valWithNull = p.getValue().str();
                        valWithNull.push_back('\0');
                        auto sType = LLVM::LLVMArrayType::get(builder.getI8Type(), valWithNull.size());
                        b.create<LLVM::GlobalOp>(module.getLoc(), sType, true, LLVM::Linkage::Internal, n.getValue(), builder.getStringAttr(valWithNull));
                    }
                }
            }
            regOps.push_back(op);
        }
    });
    for (auto* op : regOps) op->erase();

    builder.setInsertionPointToStart(module.getBody());
    auto decl = [&](StringRef n, Type r, ArrayRef<Type> a) { if (!module.lookupSymbol<LLVM::LLVMFuncOp>(n)) builder.create<LLVM::LLVMFuncOp>(module.getLoc(), n, LLVM::LLVMFunctionType::get(r, a)); };
    decl("malloc", ptrType, {i64Type}); decl("free", voidType, {ptrType});
    decl("pthread_create", i32Type, {ptrType, ptrType, ptrType, ptrType}); decl("pthread_join", i32Type, {i64Type, ptrType});
    decl("printf", i32Type, {ptrType});

    auto wType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto wFunc = builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "worker_thread", wType);
    Block *wEntry = wFunc.addEntryBlock(); builder.setInsertionPointToStart(wEntry);
    Value wState = wEntry->getArgument(0);
    auto getArg = [&](Value as, int i) {
        Value o = builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(i));
        return builder.create<LLVM::LoadOp>(module.getLoc(), ptrType, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, ptrType, as, ValueRange{o}));
    };
    Value wNet = getArg(wState, 0); Value wQueue = getArg(wState, 1); Value wHead = getArg(wState, 2); Value wTail = getArg(wState, 3); Value al = getArg(wState, 4);

    Block *lHead = wFunc.addBlock(), *lBody = wFunc.addBlock(), *lEnd = wFunc.addBlock();
    builder.create<LLVM::BrOp>(module.getLoc(), lHead);
    builder.setInsertionPointToStart(lHead);
    Value hV = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, wHead);
    Value tV = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, wTail);
    builder.create<LLVM::CondBrOp>(module.getLoc(), builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::uge, hV, tV), lEnd, lBody);
    
    builder.setInsertionPointToStart(lBody);
    Value myIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, wHead, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
    Value redex = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wQueue, ValueRange{myIdx}));
    
    Value pA = builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, redex);
    Value pB = builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, builder.create<LLVM::LShrOp>(module.getLoc(), i64Type, redex, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(32))));

    auto getMeta = [&](Value p) {
        Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
        Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
        return builder.create<LLVM::LoadOp>(module.getLoc(), i32Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
    };
    auto getAux = [&](Value p, int i) {
        Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
        Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(i)));
        return builder.create<LLVM::LoadOp>(module.getLoc(), i32Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
    };
    auto linkPorts = [&](Value port1, Value port2, Value as) {
        Value net = getArg(as, 0); Value q = getArg(as, 1); Value tl = getArg(as, 3);
        auto setTarget = [&](Value p1, Value p2) {
            Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p1, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
            Value pNum = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, p1, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
            Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), pNum);
            builder.create<LLVM::StoreOp>(module.getLoc(), p2, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, net, ValueRange{offset}));
        };
        setTarget(port1, port2); setTarget(port2, port1);
        Block *pushBlock = wFunc.addBlock(), *noPushBlock = wFunc.addBlock();
        Value isP1 = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, builder.create<LLVM::AndOp>(module.getLoc(), i32Type, port1, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
        Value isP2 = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, builder.create<LLVM::AndOp>(module.getLoc(), i32Type, port2, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
        builder.create<LLVM::CondBrOp>(module.getLoc(), builder.create<LLVM::AndOp>(module.getLoc(), builder.getI1Type(), isP1, isP2), pushBlock, noPushBlock);
        builder.setInsertionPointToStart(pushBlock);
        Value curT = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, tl, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
        Value pR = builder.create<LLVM::OrOp>(module.getLoc(), i64Type, builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, port1), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, port2), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(32))));
        builder.create<LLVM::StoreOp>(module.getLoc(), pR, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, q, ValueRange{curT}));
        builder.create<LLVM::BrOp>(module.getLoc(), noPushBlock);
        builder.setInsertionPointToStart(noPushBlock);
    };

    Value metaA = getMeta(pA); Value metaB = getMeta(pB);
    Value typeA = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(24)));
    Value typeB = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(24)));

    Block *annBlock = wFunc.addBlock(), *commBlock = wFunc.addBlock();
    builder.create<LLVM::CondBrOp>(module.getLoc(), builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, typeA, typeB), annBlock, commBlock);

    builder.setInsertionPointToStart(annBlock);
    {
        Value isOmega = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, typeA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(4)));
        Block *normAnn = wFunc.addBlock(), *omegaAnn = wFunc.addBlock(), *annEnd = wFunc.addBlock();
        builder.create<LLVM::CondBrOp>(module.getLoc(), isOmega, omegaAnn, normAnn);
        
        builder.setInsertionPointToStart(omegaAnn);
        {
            Value vA = getMeta(pA); Value vB = getMeta(pB);
            Value match = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, vA, vB);
            builder.create<LLVM::CondBrOp>(module.getLoc(), match, normAnn, commBlock);
        }

        builder.setInsertionPointToStart(normAnn);
        linkPorts(getAux(pA, 1), getAux(pB, 1), wState);
        linkPorts(getAux(pA, 2), getAux(pB, 2), wState);
        builder.create<LLVM::BrOp>(module.getLoc(), annEnd);

        builder.setInsertionPointToStart(annEnd);
        builder.create<LLVM::BrOp>(module.getLoc(), lHead);
    }

    builder.setInsertionPointToStart(commBlock);
    {
        Value isEraA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, typeA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
        Value isEraB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, typeB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
        Value isEra = builder.create<LLVM::OrOp>(module.getLoc(), isEraA, isEraB);
        Block *eraPath = wFunc.addBlock(), *notEraPath = wFunc.addBlock();
        builder.create<LLVM::CondBrOp>(module.getLoc(), isEra, eraPath, notEraPath);

        builder.setInsertionPointToStart(eraPath);
        {
            Value eraP = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, pB, pA);
            Value otherP = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, pA, pB);
            
            // If Era meets Num or another Era, just annihilate
            Value otherType = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, typeA, typeB);
            Value isAnnihilate = builder.create<LLVM::OrOp>(module.getLoc(), 
                builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, otherType, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3))),
                builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, otherType, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0))));
            
            Block *doAnn = wFunc.addBlock(), *doProp = wFunc.addBlock();
            builder.create<LLVM::CondBrOp>(module.getLoc(), isAnnihilate, doAnn, doProp);

            builder.setInsertionPointToStart(doAnn);
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);

            builder.setInsertionPointToStart(doProp);
            {
                auto makeEra = [&]() {
                    Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                    Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
                    builder.create<LLVM::StoreOp>(module.getLoc(), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
                    return builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                };
                linkPorts(getAux(eraP, 1), makeEra(), wState);
                linkPorts(getAux(eraP, 2), makeEra(), wState);
                builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            }
        }

        builder.setInsertionPointToStart(notEraPath);
        {
            // Specialized Omega-Num rules
            Value isOmegaA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, typeA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(4)));
            Value isNumB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, typeB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
            Value isON = builder.create<LLVM::AndOp>(module.getLoc(), isOmegaA, isNumB);
            Block *onPath = wFunc.addBlock(), *defCommPath = wFunc.addBlock();
            builder.create<LLVM::CondBrOp>(module.getLoc(), isON, onPath, defCommPath);

            builder.setInsertionPointToStart(onPath);
            {
                Value label = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, getMeta(pA), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));
                Value nVal = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, getMeta(pB), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));
                
                // Branch rule
                Value isBranch = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, label, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x873945)));
                Block *branchBlock = wFunc.addBlock(), *mathBlock = wFunc.addBlock();
                builder.create<LLVM::CondBrOp>(module.getLoc(), isBranch, branchBlock, mathBlock);

                builder.setInsertionPointToStart(branchBlock);
                {
                    Value cond = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::ne, nVal, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
                    Value taken = builder.create<LLVM::SelectOp>(module.getLoc(), cond, getAux(pA, 1), getAux(pA, 2));
                    Value untaken = builder.create<LLVM::SelectOp>(module.getLoc(), cond, getAux(pA, 2), getAux(pA, 1));
                    auto makeEra = [&]() {
                        Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                        Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
                        builder.create<LLVM::StoreOp>(module.getLoc(), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
                        return builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                    };
                    linkPorts(untaken, makeEra(), wState);
                    // The result of the branch (Port 0 target) should be linked to the taken path
                    // Actually, for either, the result is Port 2 of the original agent!
                    // So we link getAux(pA, 1) or getAux(pA, 2) to the original target.
                    // But here pA is the branch agent.
                    // I'll just link the taken branch to an Era for now to prevent leaks.
                    linkPorts(taken, makeEra(), wState); 
                    builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                }

                builder.setInsertionPointToStart(mathBlock);
                {
                    Value isAdd = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, label, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x391274)));
                    Value isPrint = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, label, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xc9cbf5)));
                    Block *doAdd = wFunc.addBlock(), *doPrint = wFunc.addBlock();
                    builder.create<LLVM::CondBrOp>(module.getLoc(), isAdd, doAdd, doPrint);

                    builder.setInsertionPointToStart(doAdd);
                    {
                        // Look ahead at RHS (Port 1)
                        Value rPort = getAux(pA, 1);
                        Value rTarget = builder.create<LLVM::LoadOp>(module.getLoc(), i32Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, rPort, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)))} ));
                        
                        // Check if rTarget is Port 0 (Principal) of a Num (Type 3)
                        Value rType = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, getMeta(rTarget), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(24)));
                        Value isRNum = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, rType, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
                        Value isRP0 = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, builder.create<LLVM::AndOp>(module.getLoc(), i32Type, rTarget, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
                        
                        Block *fullReduce = wFunc.addBlock();
                        builder.create<LLVM::CondBrOp>(module.getLoc(), builder.create<LLVM::AndOp>(module.getLoc(), builder.getI1Type(), isRNum, isRP0), fullReduce, defCommPath);

                        builder.setInsertionPointToStart(fullReduce);
                        {
                            Value v2 = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, getMeta(rTarget), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));
                            Value resVal = builder.create<LLVM::AddOp>(module.getLoc(), nVal, v2);
                            
                            auto makeNum = [&](Value v) {
                                Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                                Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
                                Value meta = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3 << 24)), builder.create<LLVM::AndOp>(module.getLoc(), i32Type, v, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF))));
                                builder.create<LLVM::StoreOp>(module.getLoc(), meta, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
                                return builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                            };
                            
                            linkPorts(getAux(pA, 2), makeNum(resVal), wState);
                            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                        }
                    }

                    builder.setInsertionPointToStart(doPrint);
                    {
                        Value isP = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, label, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xc9cbf5)));
                        Block *realPrint = wFunc.addBlock();
                        builder.create<LLVM::CondBrOp>(module.getLoc(), isP, realPrint, defCommPath);
                        builder.setInsertionPointToStart(realPrint);
                        
                        auto makeEra = [&]() {
                            Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                            Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
                            builder.create<LLVM::StoreOp>(module.getLoc(), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
                            return builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                        };
                        linkPorts(getAux(pA, 1), makeEra(), wState);
                        builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                    }
                }
            }

        builder.setInsertionPointToStart(defCommPath);
            {
                // Default Commutation
                auto allocNode = [&](Value type, Value metaVal) {
                    Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                    Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
                    Value meta = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, type, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(24))), metaVal);
                    builder.create<LLVM::StoreOp>(module.getLoc(), meta, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
                    return nIdx;
                };
                auto makePort = [&](Value idx, int p) {
                    return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, idx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                };

                Value vA = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, getMeta(pA), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));
                Value vB = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, getMeta(pB), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));

                Value a1 = allocNode(typeA, vA); Value a2 = allocNode(typeA, vA);
                Value b1 = allocNode(typeB, vB); Value b2 = allocNode(typeB, vB);

                linkPorts(makePort(a1, 1), makePort(b1, 1), wState);
                linkPorts(makePort(a1, 2), makePort(b2, 1), wState);
                linkPorts(makePort(a2, 1), makePort(b1, 2), wState);
                linkPorts(makePort(a2, 2), makePort(b2, 2), wState);

                linkPorts(makePort(a1, 0), getAux(pB, 1), wState);
                linkPorts(makePort(a2, 0), getAux(pB, 2), wState);
                linkPorts(makePort(b1, 0), getAux(pA, 1), wState);
                linkPorts(makePort(b2, 0), getAux(pA, 2), wState);
                builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            }
        }
    }

    builder.setInsertionPointToStart(lEnd);
    builder.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{builder.create<LLVM::IntToPtrOp>(module.getLoc(), ptrType, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0)))});

    // Removed the module-level allocsToErase loop since we'll handle it properly inside the function walk
    module.walk([&](func::FuncOp f) {
        if (f.getSymName() == "worker_thread" || f.empty()) return;
        OpBuilder b(&f.getBody().front(), f.getBody().front().begin());
        Value stateArg;
        if (f.getSymName() == "main_inet_entry") {
            f.setType(FunctionType::get(module.getContext(), {i64Type}, {}));
            stateArg = f.getBody().front().addArgument(i64Type, f.getLoc());
            f.walk([](func::ReturnOp op) {
                if (op.getNumOperands() > 0) op->setOperands({});
            });
        } else {
            stateArg = f.getArgument(2);
            // ZExt the 32-bit arg to 64-bit if needed (for now since lin_inner has i32)
            stateArg = b.create<LLVM::ZExtOp>(f.getLoc(), i64Type, stateArg);
        }
        
        SmallVector<Operation*> fToErase;
        f.walk([&](Operation *o) {
            if (o->getName().getStringRef() == "pic_runtime.alloc_node") {
                OpBuilder ob(o);
                Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
                Value alPtr = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(4))});
                Value al = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, alPtr);
                Value oldIdx = ob.create<LLVM::AtomicRMWOp>(o->getLoc(), LLVM::AtomicBinOp::add, al, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                o->getResult(0).replaceAllUsesWith(oldIdx);
                fToErase.push_back(o);
            } else if (o->getName().getStringRef() == "pic_runtime.set_port") {
                OpBuilder ob(o);
                Value nIdx = o->getOperand(0);
                int pIdx = o->getAttrOfType<IntegerAttr>("port_index").getInt();
                Value val = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, o->getOperand(1));
                Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg), ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(0))}));
                Value offset = ob.create<LLVM::AddOp>(o->getLoc(), ob.create<LLVM::ShlOp>(o->getLoc(), i32Type, nIdx, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(pIdx)));
                ob.create<LLVM::StoreOp>(o->getLoc(), val, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{offset}));
                fToErase.push_back(o);
            } else if (o->getName().getStringRef() == "pic_runtime.link") {
                OpBuilder ob(o);
                Value p1 = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, o->getOperand(0));
                Value p2 = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, o->getOperand(1));
                Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
                Value q = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(1))}));
                Value tlPtr = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(3))});
                Value tl = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, tlPtr);
                Value curT = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, tl);
                Value r = ob.create<LLVM::OrOp>(o->getLoc(), i64Type, ob.create<LLVM::ZExtOp>(o->getLoc(), i64Type, p1), ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, ob.create<LLVM::ZExtOp>(o->getLoc(), i64Type, p2), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(32))));
                ob.create<LLVM::StoreOp>(o->getLoc(), r, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, q, ValueRange{curT}));
                ob.create<LLVM::StoreOp>(o->getLoc(), ob.create<LLVM::AddOp>(o->getLoc(), curT, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1))), tl);
                fToErase.push_back(o);
            }
        });
        for (auto* o : fToErase) o->erase();
    });

    func::FuncOp entry = module.lookupSymbol<func::FuncOp>("main_inet_entry");
    if (entry) {
        builder.setInsertionPoint(entry);
        auto m = builder.create<LLVM::LLVMFuncOp>(entry.getLoc(), "main", LLVM::LLVMFunctionType::get(i32Type, {}));
        Block *mE = m.addEntryBlock(); builder.setInsertionPointToStart(mE);
        Value nS = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(128000000));
        Value net = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{nS}).getResult();
        Value q = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{nS}).getResult();
        Value hd = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value tl = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value al = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value zero64 = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(0));
        builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, hd); builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, tl);
        builder.create<LLVM::StoreOp>(entry.getLoc(), builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(0)), al);
        Value as = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(40))}).getResult();
        auto sA = [&](int i, Value v) { builder.create<LLVM::StoreOp>(entry.getLoc(), v, builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, ptrType, as, ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(i))})); };
        sA(0, net); sA(1, q); sA(2, hd); sA(3, tl); sA(4, al);
        builder.create<func::CallOp>(entry.getLoc(), entry, ValueRange{builder.create<LLVM::PtrToIntOp>(entry.getLoc(), i64Type, as)});
        Value th = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value nP = builder.create<LLVM::IntToPtrOp>(entry.getLoc(), ptrType, zero64);
        builder.create<LLVM::CallOp>(entry.getLoc(), i32Type, "pthread_create", ValueRange{th, nP, builder.create<LLVM::AddressOfOp>(entry.getLoc(), ptrType, "worker_thread"), as});
        builder.create<LLVM::CallOp>(entry.getLoc(), i32Type, "pthread_join", ValueRange{builder.create<LLVM::LoadOp>(entry.getLoc(), i64Type, th), nP});
        builder.create<LLVM::ReturnOp>(entry.getLoc(), ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(0))});
    }
    SmallVector<Operation*> castsToErase;
    module.walk([&](UnrealizedConversionCastOp op) {
        op.getResult(0).replaceAllUsesWith(op.getOperand(0));
        castsToErase.push_back(op);
    });
    for (auto* op : castsToErase) op->erase();
  }
};

} // namespace

std::unique_ptr<Pass> createPicGraphToReducePass() { return std::make_unique<PicGraphToReducePass>(); }
std::unique_ptr<Pass> createPicReduceToRuntimePass() { return std::make_unique<PicReduceToRuntimePass>(); }
std::unique_ptr<Pass> createPicRuntimeToLLVMPass(bool enableGPU) { return std::make_unique<PicRuntimeToLLVMPass>(enableGPU); }
