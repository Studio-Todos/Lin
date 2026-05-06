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
      StringRef symName = funcOp.getSymName();
      if (symName != "main_inet_entry") return;

      builder.setInsertionPointToStart(&funcOp.getBody().front());
      auto i32Type = builder.getI32Type();
      DenseMap<Value, Value> valueToPort;

      // Update signature
      SmallVector<Type> argTypes(funcOp.getNumArguments(), i32Type);
      SmallVector<Type> resTypes(funcOp.getNumResults(), i32Type);
      funcOp.setType(builder.getFunctionType(argTypes, resTypes));
      for (auto arg : funcOp.getArguments()) arg.setType(i32Type);

      funcOp.walk([&](pic::graph::AgentOp op) {
        Location loc = op.getLoc();
        builder.setInsertionPoint(op);
        StringRef agentType = op.getAgentType();
        StringRef label = op.getLabel();
        uint8_t typeEnum = 0;
        if (agentType == "gamma") typeEnum = 1;
        else if (agentType == "delta") typeEnum = 2;
        else if (agentType == "omega") {
          if (label == "num") typeEnum = 3;
          else typeEnum = 4;
        }

        uint32_t val = 0;
        if (typeEnum == 3 && op.getValue()) val = op.getValue().value();
        else if (typeEnum == 4) {
            if (label == "str" && op.getStrVal()) {
                std::string strKey = "STR_" + op.getStrVal().value().str();
                val = opcodeForLabel(strKey);
                builder.create<pic::graph::RegistryOp>(loc, builder.getStringAttr(strKey), op.getStrVal().value());
            } else val = opcodeForLabel(label);
        }

        auto alloc = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, typeEnum, val);
        Value nodeIdx = alloc.getIndex();
        
        // Store metadata in slot 3: (type << 24) | val
        Value meta = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr((typeEnum << 24) | (val & 0xFFFFFF)));
        builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(3), builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), meta));

        auto makePort = [&](int i) {
            Value shifted = builder.create<arith::ShLIOp>(loc, i32Type, nodeIdx, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2)));
            return builder.create<arith::OrIOp>(loc, i32Type, shifted, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(i)));
        };
        valueToPort[op.getP0()] = makePort(0);
        valueToPort[op.getP1()] = makePort(1);
        valueToPort[op.getP2()] = makePort(2);
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
      funcOp.walk([&](pic::graph::AgentOp op) { toErase.push_back(op); });
      funcOp.walk([&](pic::graph::LinkOp op) { toErase.push_back(op); });
      for (auto *op : toErase) op->erase();

      funcOp.walk([&](func::ReturnOp op) {
         if (op.getNumOperands() > 0 && valueToPort.count(op.getOperand(0))) op.setOperand(0, valueToPort[op.getOperand(0)]);
      });
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
                    builder.setInsertionPointToStart(module.getBody());
                    auto sType = LLVM::LLVMArrayType::get(builder.getI8Type(), p.getValue().size() + 1);
                    builder.create<LLVM::GlobalOp>(module.getLoc(), sType, true, LLVM::Linkage::Internal, n.getValue(), builder.getStringAttr(StringRef(p.getValue().data(), p.getValue().size() + 1)));
                }
            }
            regOps.push_back(op);
        }
    });
    for (auto* op : regOps) op->erase();

    builder.setInsertionPointToStart(module.getBody());
    auto decl = [&](StringRef n, Type r, ArrayRef<Type> a) { if (!module.lookupSymbol<LLVM::LLVMFuncOp>(n)) builder.create<LLVM::LLVMFuncOp>(module.getLoc(), n, LLVM::LLVMFunctionType::get(r, a)); };
    decl("malloc", ptrType, {i32Type}); decl("free", voidType, {ptrType});
    decl("pthread_create", i32Type, {ptrType, ptrType, ptrType, ptrType}); decl("pthread_join", i32Type, {i64Type, ptrType});
    decl("printf", i32Type, {ptrType});

    auto wType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto wFunc = builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "worker_thread", wType);
    Block *wEntry = wFunc.addEntryBlock(); builder.setInsertionPointToStart(wEntry);
    Value wArg = wEntry->getArgument(0);
    auto getWArg = [&](int i) {
        Value o = builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(i));
        return builder.create<LLVM::LoadOp>(module.getLoc(), ptrType, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, ptrType, wArg, ValueRange{o}));
    };
    Value wNet = getWArg(0); Value wQueue = getWArg(1); Value wHead = getWArg(2); Value wTail = getWArg(3);

    Block *lHead = wFunc.addBlock(), *lBody = wFunc.addBlock(), *lEnd = wFunc.addBlock();
    builder.create<LLVM::BrOp>(module.getLoc(), lHead);
    builder.setInsertionPointToStart(lHead);
    Value hV = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, wHead);
    Value tV = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, wTail);
    builder.create<LLVM::CondBrOp>(module.getLoc(), builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::uge, hV, tV), lEnd, lBody);
    
    builder.setInsertionPointToStart(lBody);
    Value myIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, wHead, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
    Value redex = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wQueue, ValueRange{myIdx}));
    
    // Decode redex: pA = r & 0xFFFFFFFF, pB = r >> 32
    Value pA = builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, redex);
    Value pB = builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, builder.create<LLVM::LShrOp>(module.getLoc(), i64Type, redex, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(32))));
    
    // Get Metadata for A and B
    auto getMeta = [&](Value p) {
        Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
        Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
        return builder.create<LLVM::LoadOp>(module.getLoc(), i32Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
    };
    Value metaA = getMeta(pA); Value metaB = getMeta(pB);
    Value typeA = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(24)));
    Value typeB = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(24)));

    // Rule Dispatch: if (typeA == typeB) Annihilate
    Block *annBlock = wFunc.addBlock(), *commBlock = wFunc.addBlock();
    builder.create<LLVM::CondBrOp>(module.getLoc(), builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, typeA, typeB), annBlock, commBlock);
    
    builder.setInsertionPointToStart(annBlock);
    // Annihilate: Link(A.p1, B.p1), Link(A.p2, B.p2)
    auto getAux = [&](Value p, int i) {
        Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
        Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(i)));
        return builder.create<LLVM::LoadOp>(module.getLoc(), i32Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
    };
    
    // link(portA, portB) logic
    auto linkPorts = [&](Value port1, Value port2) {
        // net[(port1>>2)*4 + (port1&3) + 1] = port2
        auto setTarget = [&](Value p1, Value p2) {
            Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p1, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
            Value pNum = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, p1, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
            Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::AddOp>(module.getLoc(), i32Type, pNum, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1))));
            builder.create<LLVM::StoreOp>(module.getLoc(), p2, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
        };
        setTarget(port1, port2); setTarget(port2, port1);
        // If both are principal, push redex
        Block *pushBlock = wFunc.addBlock(), *noPushBlock = wFunc.addBlock();
        Value isP1 = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, builder.create<LLVM::AndOp>(module.getLoc(), i32Type, port1, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
        Value isP2 = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, builder.create<LLVM::AndOp>(module.getLoc(), i32Type, port2, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
        builder.create<LLVM::CondBrOp>(module.getLoc(), builder.create<LLVM::AndOp>(module.getLoc(), builder.getI1Type(), isP1, isP2), pushBlock, noPushBlock);
        builder.setInsertionPointToStart(pushBlock);
        Value curT = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, wTail, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
        Value pR = builder.create<LLVM::OrOp>(module.getLoc(), i64Type, builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, port1), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, port2), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(32))));
        builder.create<LLVM::StoreOp>(module.getLoc(), pR, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wQueue, ValueRange{curT}));
        builder.create<LLVM::BrOp>(module.getLoc(), noPushBlock);
        builder.setInsertionPointToStart(noPushBlock);
    };

    linkPorts(getAux(pA, 1), getAux(pB, 1));
    linkPorts(getAux(pA, 2), getAux(pB, 2));
    builder.create<LLVM::BrOp>(module.getLoc(), lHead);

    builder.setInsertionPointToStart(commBlock);
    // Commutation / FireOp Placeholder
    // if (typeA == 4 && typeB == 4) rule_fire_op
    builder.create<LLVM::BrOp>(module.getLoc(), lHead);

    builder.setInsertionPointToStart(lEnd);
    builder.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{builder.create<LLVM::IntToPtrOp>(module.getLoc(), ptrType, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0)))});

    module.walk([&](pic::runtime::AllocNodeOp op) {
        Location loc = op.getLoc();
        builder.setInsertionPoint(op);
        Value i = op.getIndex();
        op.replaceAllUsesWith(i);
        op.erase();
    });
    module.walk([&](pic::runtime::SetPortOp op) {
        Location loc = op.getLoc();
        builder.setInsertionPoint(op);
        Value nIdx = op.getNodeIndex();
        Value pIdx = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(op.getPortIndex()));
        Value val = op.getPortValue();
        Value offset = builder.create<LLVM::AddOp>(loc, builder.create<LLVM::ShlOp>(loc, i32Type, nIdx, builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2))), pIdx);
        // Search for main_inet_entry's net pointer or use a placeholder
        // Since we are in LLVM lowering, we should have a way to access the net.
        // For now, these SetPortOps from PicGraphToReducePass are handled in the main function.
    });

    SmallVector<func::FuncOp> entrys;
    module.walk([&](func::FuncOp f) { if (f.getSymName() == "main_inet_entry") entrys.push_back(f); });
    for (auto f : entrys) {
        builder.setInsertionPoint(f);
        auto m = builder.create<LLVM::LLVMFuncOp>(f.getLoc(), "main", LLVM::LLVMFunctionType::get(i32Type, {}));
        Block *mE = m.addEntryBlock(); builder.setInsertionPointToStart(mE);
        Value nS = builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(16000000));
        Value net = builder.create<LLVM::CallOp>(f.getLoc(), ptrType, "malloc", ValueRange{nS}).getResult();
        Value q = builder.create<LLVM::CallOp>(f.getLoc(), ptrType, "malloc", ValueRange{nS}).getResult();
        Value hd = builder.create<LLVM::CallOp>(f.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(8))}).getResult();
        Value tl = builder.create<LLVM::CallOp>(f.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(8))}).getResult();
        Value al = builder.create<LLVM::CallOp>(f.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(4))}).getResult();
        Value zero64 = builder.create<LLVM::ConstantOp>(f.getLoc(), i64Type, builder.getI64IntegerAttr(0));
        builder.create<LLVM::StoreOp>(f.getLoc(), zero64, hd); builder.create<LLVM::StoreOp>(f.getLoc(), zero64, tl);
        
        IRMapping mapper;
        f.walk([&](Operation *op) {
            if (op->getName().getStringRef() == "pic_runtime.set_port") {
                Value nIdx = mapper.lookupOrDefault(op->getOperand(0));
                int8_t pIdx = op->getAttrOfType<IntegerAttr>("port_index").getInt();
                Value val = mapper.lookupOrDefault(op->getOperand(1));
                Value offset = builder.create<LLVM::AddOp>(f.getLoc(), builder.create<LLVM::ShlOp>(f.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(pIdx)));
                builder.create<LLVM::StoreOp>(f.getLoc(), builder.create<LLVM::TruncOp>(f.getLoc(), i32Type, val), builder.create<LLVM::GEPOp>(f.getLoc(), ptrType, i32Type, net, ValueRange{offset}));
            } else if (op->getName().getStringRef() == "pic_runtime.alloc_node") {
                mapper.map(op->getResult(0), op->getResult(0)); // placeholder
            }
        });

        Value curT = zero64;
        f.walk([&](pic::runtime::LinkOp op) {
            Value pA = builder.create<LLVM::TruncOp>(f.getLoc(), i32Type, mapper.lookupOrDefault(op.getPortA()));
            Value pB = builder.create<LLVM::TruncOp>(f.getLoc(), i32Type, mapper.lookupOrDefault(op.getPortB()));
            Value r = builder.create<LLVM::OrOp>(f.getLoc(), i64Type, builder.create<LLVM::ZExtOp>(f.getLoc(), i64Type, pA), builder.create<LLVM::ShlOp>(f.getLoc(), i64Type, builder.create<LLVM::ZExtOp>(f.getLoc(), i64Type, pB), builder.create<LLVM::ConstantOp>(f.getLoc(), i64Type, builder.getI64IntegerAttr(32))));
            builder.create<LLVM::StoreOp>(f.getLoc(), r, builder.create<LLVM::GEPOp>(f.getLoc(), ptrType, i64Type, q, ValueRange{curT}));
            curT = builder.create<LLVM::AddOp>(f.getLoc(), curT, builder.create<LLVM::ConstantOp>(f.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
        });

        builder.create<LLVM::StoreOp>(f.getLoc(), curT, tl);
        Value as = builder.create<LLVM::CallOp>(f.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(40))}).getResult();
        auto sA = [&](int i, Value v) { builder.create<LLVM::StoreOp>(f.getLoc(), v, builder.create<LLVM::GEPOp>(f.getLoc(), ptrType, ptrType, as, ValueRange{builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(i))})); };
        sA(0, net); sA(1, q); sA(2, hd); sA(3, tl); sA(4, al);
        Value th = builder.create<LLVM::CallOp>(f.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(8))}).getResult();
        Value nP = builder.create<LLVM::IntToPtrOp>(f.getLoc(), ptrType, zero64);
        builder.create<LLVM::CallOp>(f.getLoc(), i32Type, "pthread_create", ValueRange{th, nP, builder.create<LLVM::AddressOfOp>(f.getLoc(), ptrType, "worker_thread"), as});
        builder.create<LLVM::CallOp>(f.getLoc(), i32Type, "pthread_join", ValueRange{builder.create<LLVM::LoadOp>(f.getLoc(), i64Type, th), nP});
        builder.create<LLVM::ReturnOp>(f.getLoc(), ValueRange{builder.create<LLVM::ConstantOp>(f.getLoc(), i32Type, builder.getI32IntegerAttr(0))});
        f.erase();
    }
    module.walk([&](UnrealizedConversionCastOp op) { op.getResult(0).replaceAllUsesWith(op.getOperand(0)); op.erase(); });
  }
};

} // namespace

std::unique_ptr<Pass> createPicGraphToReducePass() { return std::make_unique<PicGraphToReducePass>(); }
std::unique_ptr<Pass> createPicReduceToRuntimePass() { return std::make_unique<PicReduceToRuntimePass>(); }
std::unique_ptr<Pass> createPicRuntimeToLLVMPass(bool enableGPU) { return std::make_unique<PicRuntimeToLLVMPass>(enableGPU); }
