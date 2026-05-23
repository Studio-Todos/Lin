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
#include <set>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/IRMapping.h"
#include <sstream>
#include <unordered_map>

using namespace mlir;

namespace {

static Value safeZExt(OpBuilder &b, Location loc, Type targetType, Value val) {
    if (!val || !val.getType().isa<IntegerType>()) return val;
    unsigned srcW = val.getType().cast<IntegerType>().getWidth();
    unsigned tgtW = targetType.cast<IntegerType>().getWidth();
    if (srcW < tgtW)
        return b.create<LLVM::ZExtOp>(loc, targetType, val);
    if (srcW > tgtW)
        return b.create<LLVM::TruncOp>(loc, targetType, val);
    return val;
}

static void replaceAll(std::string &str, const std::string &from, const std::string &to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

static std::string extractType(const std::string& payload, const std::string& varName) {
    size_t pos = 0;
    std::string targetLine = "";
    bool foundVar = false;
    while (true) {
        pos = payload.find(varName, pos);
        if (pos == std::string::npos) break;
        size_t next = pos + varName.size();
        while (next < payload.size() && std::isspace(static_cast<unsigned char>(payload[next]))) next++;
        if (next < payload.size() && payload[next] == '=') {
            size_t endOfLine = payload.find('\n', pos);
            if (endOfLine == std::string::npos) endOfLine = payload.size();
            targetLine = payload.substr(pos, endOfLine - pos);
            foundVar = true;
            break;
        }
        pos += varName.size();
    }
    
    if (!foundVar) {
        size_t lastLinePos = payload.find_last_not_of(" \t\r\n");
        if (lastLinePos != std::string::npos) {
            size_t startOfLastLine = payload.find_last_of("\n", lastLinePos);
            if (startOfLastLine == std::string::npos) startOfLastLine = 0;
            else startOfLastLine++;
            targetLine = payload.substr(startOfLastLine, lastLinePos - startOfLastLine + 1);
        } else {
            targetLine = payload;
        }
    }
    
    size_t colon = targetLine.rfind(':');
    if (colon != std::string::npos) {
        std::string typePart = targetLine.substr(colon + 1);
        typePart.erase(0, typePart.find_first_not_of(" \t\r\n"));
        typePart.erase(typePart.find_last_not_of(" \t\r\n") + 1);
        
        size_t toPos = typePart.rfind(" to ");
        if (toPos != std::string::npos) {
            std::string t = typePart.substr(toPos + 4);
            t.erase(0, t.find_first_not_of(" \t\r\n"));
            t.erase(t.find_last_not_of(" \t\r\n") + 1);
            return t;
        }
        
        size_t arrowPos = typePart.rfind("->");
        if (arrowPos != std::string::npos) {
            std::string t = typePart.substr(arrowPos + 2);
            t.erase(0, t.find_first_not_of(" \t\r\n"));
            t.erase(t.find_last_not_of(" \t\r\n") + 1);
            return t;
        }
        
        if (targetLine.find("cmpi") != std::string::npos || 
            targetLine.find("cmpf") != std::string::npos ||
            targetLine.find("icmp") != std::string::npos ||
            targetLine.find("fcmp") != std::string::npos) {
            return "i1";
        }
        
        return typePart;
    }
    
    return "i64";
}

static uint32_t opcodeForLabel(StringRef label) {
  uint32_t hash = 2166136261u;
  for (unsigned char c : label) {
    hash ^= c;
    hash *= 16777619u;
  }
  return (hash & 0xFFFFFF) ? (hash & 0xFFFFFF) : 1u;
}

struct PicGraphToReducePass : public PassWrapper<PicGraphToReducePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicGraphToReducePass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    std::vector<std::string> userOpLabels;
    module.walk([&](pic::graph::RegistryOp op) {
        if (auto attr = op->getAttrOfType<StringAttr>("op_name")) {
            userOpLabels.push_back(attr.getValue().str());
        }
    });

    module.walk([&](func::FuncOp funcOp) {
      builder.setInsertionPointToStart(&funcOp.getBody().front());
      auto i32Type = builder.getI32Type();
      auto i64Type = builder.getI64Type();
      DenseMap<Value, Value> valueToPort;

      // Update signature to i64 if needed and track arguments
      for (auto arg : funcOp.getArguments()) {
          if (!arg.getType().isInteger(64)) {
              arg.setType(i64Type);
          }
          valueToPort[arg] = arg;
      }
      SmallVector<Type> argTypes;
      for (auto arg : funcOp.getArguments()) argTypes.push_back(arg.getType());
      if (funcOp.getNumResults() > 0 && !funcOp.getResultTypes()[0].isInteger(64)) {
          SmallVector<Type> resTypes(funcOp.getNumResults(), i64Type);
          funcOp.setType(builder.getFunctionType(argTypes, resTypes));
      } else {
          funcOp.setType(builder.getFunctionType(argTypes, funcOp.getResultTypes()));
      }

      funcOp.walk([&](pic::graph::AgentOp op) {
        Location loc = op.getLoc();
        builder.setInsertionPoint(op);
        StringRef agentType = op.getAgentType();
        StringRef label = op.getLabel();
        StringRef polarity = op.getPolarity();
        // All agents are now unified. We use the 24-bit label hash and polarity.
        // Polarity: + (0), - (1), * (2)
        uint32_t polVal = 2; // Default to *
        if (polarity == "+") polVal = 0;
        else if (polarity == "-") polVal = 1;

        uint32_t val = 0;
        if (label == "num" && op.getValue()) val = op.getValue().value();
        else if (label == "str" && op.getStrVal()) {
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
            builder.create<pic::graph::RegistryOp>(loc, builder.getStringAttr(strKey), op.getStrVal().value(), StringAttr{});
        } else val = opcodeForLabel(label);

        auto alloc = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, 4, val);
        Value nodeIdx = alloc.getIndex();
        
        uint32_t metaValue = (polVal << 30) | (val & 0xFFFFFF);
        if (label == "num") {
            metaValue = (polVal << 30) | (opcodeForLabel("num") & 0xFFFFFF);
            Value litVal = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(op.getValue().value()));
            builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(1), litVal);
        }
        
        Value meta = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(metaValue));
        Value meta64 = (meta.getType() == i64Type) ? meta : builder.create<arith::ExtUIOp>(loc, i64Type, meta);
        builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(3), meta64);

        auto makePort = [&](int i) {
            Value nodeIdx64 = (nodeIdx.getType() == i64Type) ? nodeIdx : builder.create<arith::ExtUIOp>(loc, i64Type, nodeIdx);
            Value shifted = builder.create<arith::ShLIOp>(loc, i64Type, nodeIdx64, builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2)));
            return builder.create<arith::OrIOp>(loc, i64Type, shifted, builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(i)));
        };

        if (agentType == "omega") {
            bool isUserOp = (label == "call");
            for (const auto &uLabel : userOpLabels) {
                if (label == uLabel) {
                    isUserOp = true;
                    break;
                }
            }
            if (isUserOp) {
                valueToPort[op.getP0()] = makePort(0); // Callee/Function -> Port 0 (Principal)
                valueToPort[op.getP1()] = makePort(1); // Arg/Bundle      -> Port 1
                valueToPort[op.getP2()] = makePort(2); // Result          -> Port 2
            } else {
                valueToPort[op.getP0()] = makePort(2); // Result -> Port 2
                valueToPort[op.getP1()] = makePort(0); // Arg0   -> Port 0 (Principal)
                valueToPort[op.getP2()] = makePort(1); // Arg1   -> Port 1
            }
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
            auto safeExt = [&](Value v) {
                return (v.getType() == i64Type) ? v : builder.create<arith::ExtUIOp>(op.getLoc(), i64Type, v);
            };
            builder.create<pic::runtime::LinkOp>(op.getLoc(), safeExt(pA), safeExt(pB));
        }
      });

      SmallVector<Operation*> toErase;
      funcOp.walk([&](pic::graph::LinkOp op) { toErase.push_back(op); });
      funcOp.walk([&](pic::graph::AgentOp op) { toErase.push_back(op); });

      funcOp.walk([&](func::ReturnOp op) {
         if (op.getNumOperands() > 0 && valueToPort.count(op.getOperand(0))) {
             Value resPort = valueToPort[op.getOperand(0)];
             if (funcOp.getSymName() == "main_inet_entry") {
                 Location loc = op.getLoc();
                 builder.setInsertionPoint(op);
                 auto eraAlloc = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, 4, 0x979115);
                 Value eraIdx = eraAlloc.getIndex();
                 auto safeExt = [&](Value v) {
                     return (v.getType() == i64Type) ? v : builder.create<arith::ExtUIOp>(loc, i64Type, v);
                 };
                 builder.create<pic::runtime::SetPortOp>(loc, eraIdx, builder.getI8IntegerAttr(3), safeExt(builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr((2u << 30) | 0x979115u))));
                 Value eraPort = builder.create<arith::OrIOp>(loc, i64Type, builder.create<arith::ShLIOp>(loc, i64Type, safeExt(eraIdx), builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2))), builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1)));
                 builder.create<pic::runtime::LinkOp>(loc, safeExt(resPort), eraPort);
             }
             op.setOperand(0, resPort);
         }
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
  std::string spirvPath;
  PicRuntimeToLLVMPass(bool enableGPU, std::string spirvPath) : enableGPU(enableGPU), spirvPath(spirvPath) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());

    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    auto voidType = LLVM::LLVMVoidType::get(builder.getContext());

    if (!module.lookupSymbol<LLVM::GlobalOp>("GPU_SPIRV_PATH")) {
        OpBuilder b(module.getBodyRegion());
        std::string pathVal = spirvPath;
        pathVal.push_back('\0');
        b.create<LLVM::GlobalOp>(module.getLoc(), LLVM::LLVMArrayType::get(builder.getI8Type(), pathVal.size()), true, LLVM::Linkage::Internal, "GPU_SPIRV_PATH", builder.getStringAttr(pathVal));
    }

    auto decl = [&](StringRef n, Type r, ArrayRef<Type> a, bool var = false) { 
        if (!module.lookupSymbol<LLVM::LLVMFuncOp>(n)) 
            builder.create<LLVM::LLVMFuncOp>(module.getLoc(), n, LLVM::LLVMFunctionType::get(r, a, var)); 
    };
    decl("malloc", ptrType, {i64Type}); decl("free", voidType, {ptrType});
    decl("pthread_create", i32Type, {ptrType, ptrType, ptrType, ptrType}); decl("pthread_join", i32Type, {i64Type, ptrType});
    decl("printf", i32Type, {ptrType}, true);

    module.walk([&](func::FuncOp f) {
        if (f.getSymName() == "worker_thread" || f.empty()) return;
        OpBuilder b(&f.getBody().front(), f.getBody().front().begin());
        Value stateArg;
        if (f.getSymName() == "worker_thread") {
            stateArg = f.getArgument(0);
        } else if (f.getSymName() == "main_inet_entry") {
            f.setType(builder.getFunctionType({i64Type}, {}));
            f.getBody().front().eraseArguments([](BlockArgument){return true;});
            stateArg = f.getBody().front().addArgument(i64Type, f.getLoc());

            f.walk([](func::ReturnOp op) {
                if (op.getNumOperands() > 0) op->setOperands({});
            });
        } else {
            stateArg = f.getArgument(f.getNumArguments() - 1);
        }
        
        SmallVector<Operation*> nodes, setPorts, links;
        f.walk([&](Operation *o) {
            if (o->getName().getStringRef() == "pic_runtime.alloc_node") nodes.push_back(o);
            else if (o->getName().getStringRef() == "pic_runtime.set_port") setPorts.push_back(o);
            else if (o->getName().getStringRef() == "pic_runtime.link") links.push_back(o);
        });

        for (auto* o : nodes) {
            OpBuilder ob(o);
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value alPtr = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(4))});
            Value alL = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, alPtr);
            Value nIdx = ob.create<LLVM::AtomicRMWOp>(o->getLoc(), LLVM::AtomicBinOp::add, alL, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
            
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
            Value base = ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2)));
            
            auto store = [&](int i, Value v) {
                Value off = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, base, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                Value v64 = safeZExt(ob, o->getLoc(), i64Type, v);
                ob.create<LLVM::StoreOp>(o->getLoc(), v64, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, net, ValueRange{off}));
            };
            
            auto makePort = [&](int p) {
                Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
                return ob.create<LLVM::OrOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(p)));
            };
            
            store(0, makePort(0)); store(1, makePort(1)); store(2, makePort(2)); store(3, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0)));

            o->getResult(0).replaceAllUsesWith(nIdx);
            o->erase();
        }
        for (auto* o : setPorts) {
            OpBuilder ob(o);
            Value nIdx = o->getOperand(0);
            int pIdx = o->getAttrOfType<IntegerAttr>("port_index").getInt();
            Value val = o->getOperand(1);
            Value val64 = safeZExt(ob, o->getLoc(), i64Type, val);
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg), ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
            Value offset = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(pIdx)));
            ob.create<LLVM::StoreOp>(o->getLoc(), val64, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, net, ValueRange{offset}));
            o->erase();
        }
        for (auto* o : links) {
            OpBuilder ob(o);
            Value p1 = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, o->getOperand(0));
            Value p2 = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, o->getOperand(1));
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            auto setT = [&](Value v1, Value v2) {
                Value v1_32 = safeZExt(ob, o->getLoc(), i32Type, v1);
                Value nIdx = ob.create<LLVM::LShrOp>(o->getLoc(), i32Type, v1_32, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                Value pNum = ob.create<LLVM::AndOp>(o->getLoc(), i32Type, v1_32, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(3)));
                Value offset = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, safeZExt(ob, o->getLoc(), i64Type, nIdx), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))), safeZExt(ob, o->getLoc(), i64Type, pNum));
                Value v2_64 = safeZExt(ob, o->getLoc(), i64Type, v2);
                ob.create<LLVM::StoreOp>(o->getLoc(), v2_64, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, net, ValueRange{offset}));
            };
            setT(p1, p2); setT(p2, p1);
            Value isP1 = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::eq, ob.create<LLVM::AndOp>(o->getLoc(), i32Type, safeZExt(ob, o->getLoc(), i32Type, p1), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(3))), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(0)));
            Value isP2 = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::eq, ob.create<LLVM::AndOp>(o->getLoc(), i32Type, safeZExt(ob, o->getLoc(), i32Type, p2), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(3))), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(0)));
            Value isR = ob.create<LLVM::AndOp>(o->getLoc(), builder.getI1Type(), isP1, isP2);
            Block *curr = ob.getBlock();
            Block *push = curr->splitBlock(o);
            Block *cont = push->splitBlock(push->begin());
            ob.setInsertionPointToEnd(curr);
            ob.create<LLVM::CondBrOp>(o->getLoc(), isR, push, cont);
            ob.setInsertionPointToStart(push);
            Value q = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1))}));
            Value tlPtrPtr = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(3))});
            Value tlPtr = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, tlPtrPtr);
            Value curT = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, tlPtr);
            Value r = ob.create<LLVM::OrOp>(o->getLoc(), i64Type, safeZExt(ob, o->getLoc(), i64Type, p1), ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, safeZExt(ob, o->getLoc(), i64Type, p2), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(32))));
            ob.create<LLVM::StoreOp>(o->getLoc(), r, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, q, ValueRange{curT}));
            ob.create<LLVM::StoreOp>(o->getLoc(), ob.create<LLVM::AddOp>(o->getLoc(), curT, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1))), tlPtr);
            ob.create<LLVM::BrOp>(o->getLoc(), cont);
            ob.setInsertionPointToStart(cont);
            o->erase();
        }
    });

    module.walk([&](pic::graph::RegistryOp op) {
        if (op.getPayload()) {
            std::string mlirSnippet = op.getPayload().str();
            if (mlirSnippet.find("llvm.func") != std::string::npos) {
                auto tempModule = parseSourceString<ModuleOp>(mlirSnippet, module.getContext());
                if (tempModule) {
                    OpBuilder b(module.getBodyRegion());
                    for (auto f : tempModule->getOps<LLVM::LLVMFuncOp>()) {
                        if (!module.lookupSymbol(f.getName())) {
                            b.clone(*f.getOperation());
                        }
                    }
                }
            }
        }
    });

    struct UserOp {
        uint32_t hash;
        std::string label;
        std::string funcName;
        int numArgs;
        SmallVector<std::string> argTypes;
        std::string dispatch;
    };
    std::vector<UserOp> userOps;
    SmallVector<pic::graph::RegistryOp> regOps;
    module.walk([&](pic::graph::RegistryOp op) {
        regOps.push_back(op);
    });

    SmallVector<std::string> existingDecls;
    for (auto func : module.getOps<func::FuncOp>()) {
        std::string s;
        llvm::raw_string_ostream os(s);
        func.print(os);
        auto pos = s.find("{");
        if (pos != std::string::npos) s = s.substr(0, pos);
        // Trim trailing whitespace
        s.erase(s.find_last_not_of(" \n\r\t") + 1);
        
#ifdef ENABLE_DEBUG_LOGS
        llvm::errs() << "Captured Decl (func): [" << s << "]\n";
#endif
        existingDecls.push_back(s);
    }
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
        std::string decl;
        llvm::raw_string_ostream os(decl);
        func.print(os);
        std::string s = os.str();
        auto pos = s.find("{");
        if (pos != std::string::npos) s = s.substr(0, pos);
#ifdef ENABLE_DEBUG_LOGS
        llvm::errs() << "Captured Decl (llvm): [" << s << "]\n";
#endif
        existingDecls.push_back(s);
    }

    for (auto op : regOps) {
            StringAttr n = op->getAttrOfType<StringAttr>("op_name");
            StringAttr p = op->getAttrOfType<StringAttr>("payload");
            StringAttr argsAttr = op->getAttrOfType<StringAttr>("arg_names");
            if (n && p) {
                std::string label = n.getValue().str();
                if (label.compare(0, 4, "STR_") == 0) {
                    if (!module.lookupSymbol(label)) {
                        OpBuilder b(module.getBodyRegion());
                        std::string valWithNull = p.getValue().str();
                        valWithNull.push_back('\0');
                        auto sType = LLVM::LLVMArrayType::get(builder.getI8Type(), valWithNull.size());
                        b.create<LLVM::GlobalOp>(module.getLoc(), sType, true, LLVM::Linkage::Internal, label, builder.getStringAttr(valWithNull));
                    }
                } else {
                    // Generate a specialized LLVM function for this mlir-op
                    std::string funcName = "user_op_" + label;
                    replaceAll(funcName, "-", "_");
                    
                    int numArgs = 0;
                    if (argsAttr) {
#ifdef ENABLE_DEBUG_LOGS
                        llvm::errs() << "RegistryOp: " << label << ", arg_names attribute: [" << argsAttr.getValue() << "]\n";
#endif
                        std::string argsStr = argsAttr.getValue().str();
                        for (size_t i = 0; i < argsStr.size(); i++) {
                            if (argsStr[i] == '[') numArgs++;
                        }
                    } else {
                        // Fallback for user-defined functions which always have 2 args in the payload for binary ops
                        numArgs = 2;
                    }

                    // Create the user_op_* wrapper as an LLVM func.
                    // Before building snippet bodies, all func::FuncOps (lin_*) are
                    // forward-declared as external LLVM funcs (see pre-pass below),
                    // so llvm.call inside user_op bodies can resolve them.
                    if (!module.lookupSymbol(funcName)) {
                        OpBuilder b(module.getBodyRegion());
                        auto fType = LLVM::LLVMFunctionType::get(i64Type, SmallVector<Type>(numArgs, i64Type));
                        auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, fType);
                        Block *fEntry = f.addEntryBlock();
                        b.setInsertionPointToStart(fEntry);

                        SmallVector<std::string> argNamesList;
                        SmallVector<StringRef, 2> argNames;
                        StringRef(argsAttr.getValue()).split(argNames, "][");
                        for (auto name : argNames) {
                            std::string cleanName = name.str();
                            cleanName.erase(std::remove(cleanName.begin(), cleanName.end(), '['), cleanName.end());
                            cleanName.erase(std::remove(cleanName.begin(), cleanName.end(), ']'), cleanName.end());
                            if (cleanName.empty()) continue;
                            if (cleanName[0] != '%') cleanName = "%" + cleanName;
                            argNamesList.push_back(cleanName);
                        }
                        for (unsigned i = 0; i < argNamesList.size(); ++i) {
#ifdef ENABLE_DEBUG_LOGS
                             llvm::errs() << "Outer loop: argNamesList[" << i << "] = [" << argNamesList[i] << "]\n";
#endif
                        }

                        std::string payload = p.getValue().str();
                        std::string filteredDecls = "";
                        for (auto &d : existingDecls) {
                            auto atPos = d.find("@");
                            if (atPos != std::string::npos) {
                                auto parenPos = d.find("(", atPos);
                                if (parenPos != std::string::npos) {
                                    std::string sym = d.substr(atPos, parenPos - atPos);
                                    if (payload.find(sym) != std::string::npos) filteredDecls += d + "\n";
                                }
                            }
                        }
                        OwningOpRef<ModuleOp> tempModule = ModuleOp::create(module.getLoc());
                        {
                            OpBuilder b(tempModule->getBodyRegion());
                            IRMapping m;
                            for (auto &op : module.getBody()->getOperations()) {
                                if (isa<func::FuncOp>(op) || isa<LLVM::LLVMFuncOp>(op) || isa<LLVM::GlobalOp>(op)) {
                                    b.clone(op, m);
                                }
                            }
                        }
                        
                        std::string argS = "";
                        SmallVector<std::string> argTypes;
                        for (unsigned i = 0; i < argNamesList.size(); ++i) {
                            std::string type = "i64";
                            size_t underscorePos = argNamesList[i].rfind('_');
                            if (underscorePos != std::string::npos && underscorePos > 0) {
                                type = argNamesList[i].substr(underscorePos + 1);
                                argNamesList[i] = argNamesList[i].substr(0, underscorePos);
                            }
                            argTypes.push_back(type);
#ifdef ENABLE_DEBUG_LOGS
                            llvm::errs() << "Outer loop: Inferred type for " << argNamesList[i] << " is [" << type << "]\n";
#endif
                            std::string mlirType = type;
                            if (mlirType != "i1" && mlirType != "i8" && mlirType != "i16" &&
                                mlirType != "i32" && mlirType != "i64" &&
                                mlirType != "f32" && mlirType != "f64" &&
                                mlirType != "ptr") {
                                mlirType = "i32";
                            }
                            argS += argNamesList[i] + " : " + mlirType;
                            if (i < argNamesList.size() - 1) argS += ", ";
                        }
                        
                        std::string pStr = p.getValue().str();
                        std::string snippetDecls = "";
                        auto addDecl = [&](std::string name, std::string fullDecl) {
                            bool found = false;
                            for (auto &d : existingDecls) {
                                if (d.find("@" + name + "(") != std::string::npos || d.find("@\"" + name + "\"(") != std::string::npos) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) snippetDecls += fullDecl + "\n";
                        };
                        addDecl("printf", "llvm.func @printf(!llvm.ptr, ...) -> i32");
                        addDecl("putchar", "llvm.func @putchar(i32) -> i32");
                        addDecl("getchar", "llvm.func @getchar() -> i32");
                        addDecl("scanf", "llvm.func @scanf(!llvm.ptr, ...) -> i32");
                        addDecl("malloc", "llvm.func @malloc(i64) -> !llvm.ptr");
                        addDecl("free", "llvm.func @free(!llvm.ptr)");
                        addDecl("strlen", "llvm.func @strlen(!llvm.ptr) -> i64");
                        addDecl("strcmp", "llvm.func @strcmp(!llvm.ptr, !llvm.ptr) -> i32");
                        addDecl("sin", "llvm.func @sin(f64) -> f64");
                        addDecl("cos", "llvm.func @cos(f64) -> f64");
                        addDecl("tan", "llvm.func @tan(f64) -> f64");
                        addDecl("asin", "llvm.func @asin(f64) -> f64");
                        addDecl("acos", "llvm.func @acos(f64) -> f64");
                        addDecl("atan", "llvm.func @atan(f64) -> f64");
                        addDecl("exp", "llvm.func @exp(f64) -> f64");
                        addDecl("log", "llvm.func @log(f64) -> f64");
                        addDecl("sqrt", "llvm.func @sqrt(f64) -> f64");
                        addDecl("pow", "llvm.func @pow(f64, f64) -> f64");
                        for (auto &d : existingDecls) {
                            auto atPos = d.find("@");
                            if (atPos != std::string::npos) {
                                auto parenPos = d.find("(", atPos);
                                if (parenPos != std::string::npos) {
                                    std::string sym = d.substr(atPos, parenPos - atPos);
                                    std::string unquoted = sym;
                                    if (unquoted.size() > 2 && unquoted[1] == '\"' && unquoted.back() == '\"') {
                                        unquoted.erase(unquoted.size() - 1);
                                        unquoted.erase(1, 1);
                                    }
                                    if (pStr.find(sym) != std::string::npos || pStr.find(unquoted) != std::string::npos) {
                                        std::string decl = d;
                                        if (decl.find("func.func") != std::string::npos && decl.find("private") == std::string::npos) {
                                            auto atPos2 = decl.find("@");
                                            if (atPos2 != std::string::npos) {
                                                decl.insert(atPos2, "private ");
                                            }
                                        }
                                        snippetDecls += decl + "\n";
                                    }
                                }
                            }
                        }
                        
                        std::string pStrRenamed = pStr;

                        std::string argS2 = "";
                        for (unsigned i = 0; i < argNamesList.size(); ++i) {
                            std::string mlirType = argTypes[i];
                            if (mlirType != "i1" && mlirType != "i8" && mlirType != "i16" &&
                                mlirType != "i32" && mlirType != "i64" &&
                                mlirType != "f32" && mlirType != "f64" &&
                                mlirType != "ptr") {
                                mlirType = "i32";
                            }
                            argS2 += argNamesList[i] + " : " + mlirType;
                            if (i < argNamesList.size() - 1) argS2 += ", ";
                        }

                        std::string tempModuleStr = "module {\n" + snippetDecls + "func.func @temp(" + argS2 + ") -> i64 {\n";
                        tempModuleStr += pStrRenamed + "\n";
                        
                        // Determine the result variable name
                        std::string resVar = "%res";
                        if (pStrRenamed.find("%res ") == std::string::npos && 
                            pStrRenamed.find("%res\n") == std::string::npos &&
                            pStrRenamed.find("%res=") == std::string::npos &&
                            pStrRenamed.find("%res_state") != std::string::npos) {
                            resVar = "%res_state";
                        }
                        
                        std::string actualType = extractType(pStrRenamed, resVar);
                        
                        // If no result variable found in payload, assume the payload expression ITSELF is the result
                        if (pStrRenamed.find(resVar) == std::string::npos) {
                            tempModuleStr += "  %res_auto = " + pStrRenamed + "\n";
                            tempModuleStr += "  %res_coerced = \"lin.coerce\"(%res_auto) : (" + actualType + ") -> i64\n";
                            tempModuleStr += "  func.return %res_coerced : i64\n";
                        } else {
                            // We'll use a custom op to do the coercion after parsing
                            tempModuleStr += "  %res_coerced = \"lin.coerce\"(" + resVar + ") : (" + actualType + ") -> i64\n";
                            tempModuleStr += "  func.return %res_coerced : i64\n";
                        }
                        tempModuleStr += "}\n}\n";
                        
                        
#ifdef ENABLE_DEBUG_LOGS
                        llvm::errs() << "Final Snippet Module:\n" << tempModuleStr << "\n";
#endif
                        MLIRContext *ctx = module.getContext();
                        ctx->allowUnregisteredDialects();
                        auto parsedSnippet = parseSourceString<ModuleOp>(tempModuleStr, ctx);
                         if (parsedSnippet) {
                             for (auto &op : parsedSnippet->getBody()->getOperations()) {
                                 if (auto sym = dyn_cast<SymbolOpInterface>(op)) {
                                     // Never clone @temp into the parent module - we use
                                     // it only to read the body into user_op_* directly.
                                     if (sym.getName() == "temp") continue;
                                     if (!module.lookupSymbol(sym.getName())) {
                                         module.push_back(op.clone());
                                     }
                                 }
                             }
                         }

                         func::FuncOp tempFunc = parsedSnippet ? parsedSnippet->lookupSymbol<func::FuncOp>("temp") : nullptr;
                            if (tempFunc && !tempFunc.getBody().empty()) {
                                IRMapping mapper;
                                OpBuilder bBody = OpBuilder::atBlockBegin(fEntry);
                                 for (unsigned i = 0; i < argNamesList.size(); ++i) {
                                     if (i < f.getNumArguments() && i < tempFunc.getNumArguments()) {
                                         Value arg = fEntry->getArgument(i);
                                          if (argTypes[i] == "f32") {
                                              Value trunc = bBody.create<LLVM::TruncOp>(module.getLoc(), builder.getI32Type(), arg);
                                              arg = bBody.create<LLVM::BitcastOp>(module.getLoc(), builder.getF32Type(), trunc);
                                          } else if (argTypes[i] == "f64") {
                                              arg = bBody.create<LLVM::BitcastOp>(module.getLoc(), builder.getF64Type(), arg);
                                          } else if (argTypes[i] == "i1") {
                                              arg = bBody.create<LLVM::TruncOp>(module.getLoc(), builder.getI1Type(), arg);
                                          } else if (argTypes[i] == "i32") {
                                              arg = bBody.create<LLVM::TruncOp>(module.getLoc(), builder.getI32Type(), arg);
                                          } else if (argTypes[i] == "i64") {
                                              arg = safeZExt(bBody, module.getLoc(), i64Type, arg);
                                          } else {
                                              arg = bBody.create<LLVM::TruncOp>(module.getLoc(), builder.getI32Type(), arg);
                                          }
                                         mapper.map(tempFunc.getArgument(i), arg);
                                     }
                                 }
                                
                                Value finalRetVal = nullptr;
                                for (auto &op : tempFunc.getBody().front().getOperations()) {
                                    if (op.hasTrait<OpTrait::IsTerminator>()) {
                                        if (op.getNumOperands() > 0) {
                                            finalRetVal = mapper.lookupOrDefault(op.getOperand(0));
                                        }
                                        continue;
                                    }
                                     if (op.getName().getStringRef() == "lin.coerce") {
                                         Value src = mapper.lookupOrDefault(op.getOperand(0));
                                         Value coerced = src;
                                         if (src.getType() != i64Type) {
                                             if (src.getType().isa<IntegerType>()) {
                                                 if (src.getType().getIntOrFloatBitWidth() < 64) {
                                                     coerced = safeZExt(bBody, module.getLoc(), i64Type, src);
                                                 } else if (src.getType().getIntOrFloatBitWidth() > 64) {
                                                     coerced = bBody.create<LLVM::TruncOp>(module.getLoc(), i64Type, src);
                                                 }
                                             } else if (src.getType().isa<FloatType>()) {
                                                 if (src.getType().isF64()) {
                                                     coerced = bBody.create<LLVM::BitcastOp>(module.getLoc(), i64Type, src);
                                                 } else if (src.getType().isF32()) {
                                                     Value ext = bBody.create<LLVM::FPExtOp>(module.getLoc(), builder.getF64Type(), src);
                                                     coerced = bBody.create<LLVM::BitcastOp>(module.getLoc(), i64Type, ext);
                                                 } else {
                                                     // Default fallback for other float types
                                                     Value ext = bBody.create<LLVM::FPExtOp>(module.getLoc(), builder.getF64Type(), src);
                                                     coerced = bBody.create<LLVM::BitcastOp>(module.getLoc(), i64Type, ext);
                                                 }
                                             } else if (src.getType().isa<LLVM::LLVMPointerType>()) {
                                                 coerced = bBody.create<LLVM::PtrToIntOp>(module.getLoc(), i64Type, src);
                                             } else if (src.getType().isa<IndexType>()) {
                                                 coerced = bBody.create<arith::IndexCastOp>(module.getLoc(), i64Type, src);
                                             }
                                         }
                                         mapper.map(op.getResult(0), coerced);
                                         finalRetVal = coerced;
                                         continue;
                                     }
                                     // Special-case func::CallOp: the snippet was parsed as
                                     // func.call inside a func.func @temp, but we're cloning
                                     // into an LLVM::LLVMFuncOp body. Convert the referenced
                                     // func::FuncOp callee to LLVM on-the-spot so llvm.call
                                     // can find it, then emit llvm.call.
                                     if (auto callOp = dyn_cast<func::CallOp>(op)) {
                                         StringRef callee = callOp.getCallee();
                                         SmallVector<Value> callArgs;
                                         for (auto operand : callOp.getOperands())
                                             callArgs.push_back(mapper.lookupOrDefault(operand));
                                         Type retTy = i64Type;
                                         if (callOp.getNumResults() > 0)
                                             retTy = callOp.getResult(0).getType();
                                         // If callee is a func::FuncOp, convert it to LLVM in-place
                                         if (auto calleeFuncOp = module.lookupSymbol<func::FuncOp>(callee)) {
                                             if (!module.lookupSymbol<LLVM::LLVMFuncOp>(callee)) {
                                                 // Build equivalent LLVM func
                                                 SmallVector<Type> llvmArgTys(calleeFuncOp.getNumArguments(), i64Type);
                                                 OpBuilder llvmB(module.getBodyRegion());
                                                 auto llvmFty = LLVM::LLVMFunctionType::get(i64Type, llvmArgTys);
                                                 auto llvmF = llvmB.create<LLVM::LLVMFuncOp>(
                                                     module.getLoc(), callee, llvmFty);
                                                 // Move all blocks from the func::FuncOp into the LLVM func
                                                 llvmF.getBody().takeBody(calleeFuncOp.getBody());
                                                 // Fix terminators: func::ReturnOp → LLVM::ReturnOp
                                                 llvmF.walk([&](func::ReturnOp retOp) {
                                                     OpBuilder rb(retOp);
                                                     if (retOp.getNumOperands() > 0)
                                                         rb.create<LLVM::ReturnOp>(module.getLoc(), retOp.getOperand(0));
                                                     else
                                                         rb.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{});
                                                     retOp.erase();
                                                 });
                                                 calleeFuncOp.erase();
                                             }
                                         }
                                         auto llvmCall = bBody.create<LLVM::CallOp>(
                                             module.getLoc(), retTy, callee, callArgs);
                                         if (callOp.getNumResults() > 0) {
                                             mapper.map(callOp.getResult(0), llvmCall.getResult());
                                             finalRetVal = llvmCall.getResult();
                                         }
                                         continue;
                                     }
                                     Operation *cloned = bBody.clone(op, mapper);
                                     if (op.getNumResults() > 0) {
                                         finalRetVal = mapper.lookupOrDefault(op.getResult(0));
                                     }
                                 }
                                 
                                 // No need for separate finalRetVal processing anymore as it's handled by lin.coerce or the loop
                                 if (finalRetVal) {
                                     if (finalRetVal.getType() != i64Type) {
                                         if (finalRetVal.getType().isa<IntegerType>()) {
                                             finalRetVal = safeZExt(bBody, module.getLoc(), i64Type, finalRetVal);
                                         } else if (finalRetVal.getType().isa<FloatType>()) {
                                             if (finalRetVal.getType().isF64()) {
                                                 finalRetVal = bBody.create<LLVM::BitcastOp>(module.getLoc(), i64Type, finalRetVal);
                                             } else {
                                                 Value ext = bBody.create<LLVM::FPExtOp>(module.getLoc(), bBody.getF64Type(), finalRetVal);
                                                 finalRetVal = bBody.create<LLVM::BitcastOp>(module.getLoc(), i64Type, ext);
                                             }
                                         } else if (finalRetVal.getType().isa<LLVM::LLVMPointerType>()) {
                                             finalRetVal = bBody.create<LLVM::PtrToIntOp>(module.getLoc(), i64Type, finalRetVal);
                                         } else if (finalRetVal.getType().isa<IndexType>()) {
                                             finalRetVal = bBody.create<arith::IndexCastOp>(module.getLoc(), i64Type, finalRetVal);
                                         }
                                     }
                                     bBody.create<LLVM::ReturnOp>(module.getLoc(), finalRetVal);
                                 } else {
                                     bBody.create<LLVM::ReturnOp>(module.getLoc(), bBody.create<LLVM::ConstantOp>(module.getLoc(), i64Type, bBody.getI64IntegerAttr(0)));
                                 }
                            }
                        
                        // Always ensure a terminator
                        if (fEntry->empty() || !fEntry->back().hasTrait<OpTrait::IsTerminator>()) {
                            OpBuilder bTerm = OpBuilder::atBlockEnd(fEntry);
                            bTerm.create<LLVM::ReturnOp>(module.getLoc(), bTerm.create<LLVM::ConstantOp>(module.getLoc(), i64Type, bTerm.getI64IntegerAttr(0)));
                        } else if (fEntry->back().hasTrait<OpTrait::IsTerminator>()) {
                            // Replace func::ReturnOp with LLVM::ReturnOp if present
                            auto &termOp = fEntry->back();
                            if (isa<func::ReturnOp>(termOp)) {
                                OpBuilder bTerm(&termOp);
                                if (termOp.getNumOperands() > 0)
                                    bTerm.create<LLVM::ReturnOp>(module.getLoc(), termOp.getOperand(0));
                                else {
                                    Value zeroRet = bTerm.create<LLVM::ConstantOp>(module.getLoc(), i64Type, bTerm.getI64IntegerAttr(0));
                                    bTerm.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{zeroRet});
                                }
                                termOp.erase();
                            }
                        }
                        std::string dispatch = "";
                        if (auto dispAttr = op->getAttrOfType<StringAttr>("dispatch")) {
                            dispatch = dispAttr.getValue().str();
                        }
                        userOps.push_back({opcodeForLabel(label), label, funcName, (int)numArgs, argTypes, dispatch});
                    }
                }
            }
        }
    for (auto op : regOps) op.erase();

    builder.setInsertionPointToStart(module.getBody());
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        
        auto rulesType = LLVM::LLVMArrayType::get(i32Type, 300);
        builder.create<LLVM::GlobalOp>(module.getLoc(), rulesType, false, LLVM::Linkage::Internal, "runtime_rules", builder.getZeroAttr(rulesType));
        builder.create<LLVM::GlobalOp>(module.getLoc(), i32Type, false, LLVM::Linkage::Internal, "runtime_num_rules", builder.getI32IntegerAttr(0));
        
        auto genRegisterRule = [&]() {
            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(builder.getContext()), {i32Type, i32Type, i32Type});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "register_rule", fType);
            Block *entry = f.addEntryBlock();
            Value op = entry->getArgument(0);
            Value type = entry->getArgument(1);
            Value impl = entry->getArgument(2);
            
            OpBuilder eb(entry, entry->end());
            Location loc = module.getLoc();
            
            Value countAddr = eb.create<LLVM::AddressOfOp>(loc, ptrType, "runtime_num_rules");
            Value count = eb.create<LLVM::LoadOp>(loc, i32Type, countAddr);
            
            Value maxRules = eb.create<LLVM::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(100));
            Value cmp = eb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, count, maxRules);
            
            Block *thenBlock = f.addBlock();
            Block *mergeBlock = f.addBlock();
            eb.create<LLVM::CondBrOp>(loc, cmp, thenBlock, mergeBlock);
            
            OpBuilder tb(thenBlock, thenBlock->end());
            Value three = tb.create<LLVM::ConstantOp>(loc, i32Type, tb.getI32IntegerAttr(3));
            Value offsetBase = tb.create<LLVM::MulOp>(loc, i32Type, count, three);
            Value rulesAddr = tb.create<LLVM::AddressOfOp>(loc, ptrType, "runtime_rules");
            
            auto storeRuleField = [&](Value val, int fieldOffset) {
                Value offset = tb.create<LLVM::ConstantOp>(loc, i32Type, tb.getI32IntegerAttr(fieldOffset));
                Value fullIdx = tb.create<LLVM::AddOp>(loc, i32Type, offsetBase, offset);
                Value gep = tb.create<LLVM::GEPOp>(loc, ptrType, i32Type, rulesAddr, ValueRange{fullIdx});
                tb.create<LLVM::StoreOp>(loc, val, gep);
            };
            
            storeRuleField(op, 0);
            storeRuleField(type, 1);
            storeRuleField(impl, 2);
            
            Value one = tb.create<LLVM::ConstantOp>(loc, i32Type, tb.getI32IntegerAttr(1));
            Value newCount = tb.create<LLVM::AddOp>(loc, i32Type, count, one);
            tb.create<LLVM::StoreOp>(loc, newCount, countAddr);
            tb.create<LLVM::BrOp>(loc, mergeBlock);
            
            OpBuilder mb(mergeBlock, mergeBlock->end());
            mb.create<LLVM::ReturnOp>(loc, ValueRange{});
        };
        genRegisterRule();
        
        auto genLookupRule = [&]() {
            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(i32Type, {i32Type, i32Type});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "lookup_rule", fType);
            Block *entry = f.addEntryBlock();
            Value op = entry->getArgument(0);
            Value type = entry->getArgument(1);
            
            OpBuilder eb(entry, entry->end());
            Location loc = module.getLoc();
            
            Value countAddr = eb.create<LLVM::AddressOfOp>(loc, ptrType, "runtime_num_rules");
            Value count = eb.create<LLVM::LoadOp>(loc, i32Type, countAddr);
            
            Block *loopCond = f.addBlock();
            Block *loopBody = f.addBlock();
            Block *loopInc = f.addBlock();
            Block *retZero = f.addBlock();
            
            Value zero = eb.create<LLVM::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(0));
            Value iAlloc = eb.create<LLVM::AllocaOp>(loc, ptrType, eb.getI32Type(), eb.create<LLVM::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(1)));
            eb.create<LLVM::StoreOp>(loc, zero, iAlloc);
            eb.create<LLVM::BrOp>(loc, loopCond);
            
            OpBuilder cb(loopCond, loopCond->end());
            Value iVal = cb.create<LLVM::LoadOp>(loc, i32Type, iAlloc);
            Value loopCmp = cb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, iVal, count);
            cb.create<LLVM::CondBrOp>(loc, loopCmp, loopBody, retZero);
            
            OpBuilder lb(loopBody, loopBody->end());
            Value three = lb.create<LLVM::ConstantOp>(loc, i32Type, lb.getI32IntegerAttr(3));
            Value offsetBase = lb.create<LLVM::MulOp>(loc, i32Type, iVal, three);
            Value rulesAddr = lb.create<LLVM::AddressOfOp>(loc, ptrType, "runtime_rules");
            
            auto loadRuleField = [&](int fieldOffset) {
                Value offset = lb.create<LLVM::ConstantOp>(loc, i32Type, lb.getI32IntegerAttr(fieldOffset));
                Value fullIdx = lb.create<LLVM::AddOp>(loc, i32Type, offsetBase, offset);
                Value gep = lb.create<LLVM::GEPOp>(loc, ptrType, i32Type, rulesAddr, ValueRange{fullIdx});
                return lb.create<LLVM::LoadOp>(loc, i32Type, gep);
            };
            
            Value ruleOp = loadRuleField(0);
            Value ruleType = loadRuleField(1);
            Value ruleImpl = loadRuleField(2);
            
            Value opMatch = lb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ruleOp, op);
            Value typeMatch = lb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ruleType, type);
            Value match = lb.create<LLVM::AndOp>(loc, opMatch, typeMatch);
            
            Block *foundBlock = f.addBlock();
            lb.create<LLVM::CondBrOp>(loc, match, foundBlock, loopInc);
            
            OpBuilder fb(foundBlock, foundBlock->end());
            fb.create<LLVM::ReturnOp>(loc, ruleImpl);
            
            OpBuilder ib(loopInc, loopInc->end());
            Value one = ib.create<LLVM::ConstantOp>(loc, i32Type, ib.getI32IntegerAttr(1));
            Value nextI = ib.create<LLVM::AddOp>(loc, i32Type, iVal, one);
            ib.create<LLVM::StoreOp>(loc, nextI, iAlloc);
            ib.create<LLVM::BrOp>(loc, loopCond);
            
            OpBuilder rb(retZero, retZero->end());
            rb.create<LLVM::ReturnOp>(loc, zero);
        };
        genLookupRule();
    }
    auto wType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto wFunc = builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "worker_thread", wType);
    Block *wEntry = wFunc.addEntryBlock(); builder.setInsertionPointToStart(wEntry);
    Value wState = wEntry->getArgument(0);
    auto getArg = [&](Value as, int i) {
        Value o = builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(i));
        return builder.create<LLVM::LoadOp>(module.getLoc(), ptrType, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, ptrType, as, ValueRange{o}));
    };
    Value wNet = getArg(wState, 0); Value wQueue = getArg(wState, 1); Value wHead = getArg(wState, 2); Value wTail = getArg(wState, 3); Value al = getArg(wState, 4);

    auto genDispatcher = [&]() {
        OpBuilder b(module.getBodyRegion());
        auto fType = LLVM::LLVMFunctionType::get(i64Type, {i32Type, i64Type, i64Type, i64Type, i64Type, i32Type, i32Type, ptrType});
        auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "dispatch_user_op", fType);
        Block *entry = f.addEntryBlock();
        Value hash = entry->getArgument(0);
        Value arg0 = entry->getArgument(1);
        Value arg1 = entry->getArgument(2);
        Value arg2 = entry->getArgument(3);
        Value arg3 = entry->getArgument(4);
        Value nodeA = entry->getArgument(5);
        Value nodeB = entry->getArgument(6);
        Value dState = entry->getArgument(7);
        OpBuilder eb(entry, entry->end());
        Value wState = eb.create<LLVM::PtrToIntOp>(module.getLoc(), i64Type, dState);

        Block *defaultBlock = f.addBlock();
        {
            OpBuilder ob(defaultBlock, defaultBlock->end());
            Value zero = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(0));
            ob.create<LLVM::ReturnOp>(module.getLoc(), zero);
        }

        SmallVector<int32_t> caseValues;
        SmallVector<Block*> caseBlocks;
        SmallVector<ValueRange> caseOperands;
        for (auto &op : userOps) {
            caseValues.push_back(op.hash);
            Block *block = f.addBlock();
            caseBlocks.push_back(block);
            caseOperands.push_back(ValueRange{});
            OpBuilder ob(block, block->end());
            Value res;
            if (op.dispatch == "gpu") {
                Value two32 = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(2));
                Value pairBuf = ob.create<LLVM::AllocaOp>(module.getLoc(), ptrType, i32Type, two32);
                
                Value zeroIdx = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(0));
                Value ptr0 = ob.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, pairBuf, zeroIdx);
                ob.create<LLVM::StoreOp>(module.getLoc(), nodeA, ptr0);
                
                Value oneIdx = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(1));
                Value ptr1 = ob.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, pairBuf, oneIdx);
                ob.create<LLVM::StoreOp>(module.getLoc(), nodeB, ptr1);

                Value oZero = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(0));
                Value gepNet = ob.create<LLVM::GEPOp>(module.getLoc(), ptrType, ptrType, dState, ValueRange{oZero});
                Value dNet = ob.create<LLVM::LoadOp>(module.getLoc(), ptrType, gepNet);

                auto gpuDispatchFty = LLVM::LLVMFunctionType::get(
                    LLVM::LLVMVoidType::get(ob.getContext()),
                    {ptrType, ptrType, i32Type, ptrType}
                );
                auto gpuDispatchFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("pic_gpu_dispatch");
                if (!gpuDispatchFunc) {
                    OpBuilder moduleBuilder(module.getBodyRegion());
                    gpuDispatchFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(module.getLoc(), "pic_gpu_dispatch", gpuDispatchFty);
                }
                
                Value one32 = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(1));
                Value pathPtr = ob.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, "GPU_SPIRV_PATH");
                ob.create<LLVM::CallOp>(module.getLoc(), TypeRange{}, "pic_gpu_dispatch", ValueRange{dNet, pairBuf, one32, pathPtr});

                res = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(0));
            } else {
                Value zeroVal = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(0));
                SmallVector<Value> callArgs;
                if (op.numArgs >= 1) callArgs.push_back(arg0);
                if (op.numArgs >= 2) callArgs.push_back(arg1);
                if (op.numArgs >= 3) callArgs.push_back(wState);
                if (op.numArgs >= 4) callArgs.push_back(arg3);
                for (int i = 5; i <= op.numArgs; i++) {
                    callArgs.push_back(zeroVal);
                }
                res = ob.create<LLVM::CallOp>(module.getLoc(), i64Type, op.funcName, callArgs).getResult();
            }
            ob.create<LLVM::ReturnOp>(module.getLoc(), res);
        }

        eb.create<LLVM::SwitchOp>(module.getLoc(), hash, defaultBlock, ValueRange{}, caseValues, caseBlocks, caseOperands);
    };
    genDispatcher();

    auto genIsGpuOp = [&]() {
        OpBuilder b(module.getBodyRegion());
        auto fType = LLVM::LLVMFunctionType::get(i1Type, {i32Type});
        auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "is_gpu_op", fType);
        Block *entry = f.addEntryBlock();
        Value hash = entry->getArgument(0);

        Block *defaultBlock = f.addBlock();
        {
            OpBuilder ob(defaultBlock, defaultBlock->end());
            Value zero = ob.create<LLVM::ConstantOp>(module.getLoc(), i1Type, ob.getBoolAttr(false));
            ob.create<LLVM::ReturnOp>(module.getLoc(), zero);
        }
        
        SmallVector<int32_t> caseValues;
        SmallVector<Block*> caseBlocks;
        SmallVector<ValueRange> caseOperands;
        for (auto &op : userOps) {
            if (op.dispatch == "gpu") {
                caseValues.push_back(op.hash);
                Block *block = f.addBlock();
                caseBlocks.push_back(block);
                caseOperands.push_back(ValueRange{});
                OpBuilder ob(block, block->end());
                Value one = ob.create<LLVM::ConstantOp>(module.getLoc(), i1Type, ob.getBoolAttr(true));
                ob.create<LLVM::ReturnOp>(module.getLoc(), one);
            }
        }
        if (!caseValues.empty()) {
            OpBuilder eb(entry, entry->end());
            eb.create<LLVM::SwitchOp>(module.getLoc(), hash, defaultBlock, ValueRange{}, caseValues, caseBlocks, caseOperands);
        } else {
            OpBuilder eb(entry, entry->end());
            Value zero = eb.create<LLVM::ConstantOp>(module.getLoc(), i1Type, eb.getBoolAttr(false));
            eb.create<LLVM::ReturnOp>(module.getLoc(), zero);
        }
    };
    genIsGpuOp();


    auto genGetNumArgs = [&]() {
        OpBuilder b(module.getBodyRegion());
        auto fType = LLVM::LLVMFunctionType::get(i32Type, {i32Type});
        auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "get_num_args", fType);
        Block *entry = f.addEntryBlock();
        Value hash = entry->getArgument(0);

        Block *defaultBlock = f.addBlock();
        {
            OpBuilder ob(defaultBlock, defaultBlock->end());
            Value zero = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(0));
            ob.create<LLVM::ReturnOp>(module.getLoc(), zero);
        }
        
        SmallVector<int32_t> caseValues;
        SmallVector<Block*> caseBlocks;
        SmallVector<ValueRange> caseOperands;
        for (auto &op : userOps) {
            caseValues.push_back(op.hash);
            Block *block = f.addBlock();
            caseBlocks.push_back(block);
            caseOperands.push_back(ValueRange{});
            OpBuilder ob(block, block->end());
            int32_t netArgs = op.numArgs;
            if (netArgs == 3) netArgs = 2; // User-defined functions have 3 C args but 2 net args
            Value n = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(netArgs));
            ob.create<LLVM::ReturnOp>(module.getLoc(), n);
        }
        OpBuilder eb(entry, entry->end());
        eb.create<LLVM::SwitchOp>(module.getLoc(), hash, defaultBlock, ValueRange{}, caseValues, caseBlocks, caseOperands);
    };
    genGetNumArgs();

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
        Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i64Type, safeZExt(builder, module.getLoc(), i64Type, p), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
        Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(3)));
        return builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{offset}));
    };
    
    // Debug logging
    if (!module.lookupSymbol("PRINT_REDEX")) {
        OpBuilder b(module.getBodyRegion());
        std::string fmt = "Redex: [%08x] meets [%08x]\n";
        fmt.push_back('\0');
        b.create<LLVM::GlobalOp>(module.getLoc(), LLVM::LLVMArrayType::get(builder.getI8Type(), fmt.size()), true, LLVM::Linkage::Internal, "PRINT_REDEX", builder.getStringAttr(fmt));
    }
    auto logRedex = [&](Value mA, Value mB) {
        Value fmtPtr = builder.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, "PRINT_REDEX");
        builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "printf", ValueRange{fmtPtr, mA, mB});
    };
    auto getAux = [&](Value p, int i) {
        Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i64Type, safeZExt(builder, module.getLoc(), i64Type, p), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
        Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(i)));
        return builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{offset}));
    };
    auto linkPorts = [&](Value port1, Value port2, Value as) {
        Value net = getArg(as, 0); Value q = getArg(as, 1); Value tl = getArg(as, 3);
        auto setTarget = [&](Value p1, Value p2) {
            Value p1_32 = safeZExt(builder, module.getLoc(), i32Type, p1);
            Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p1_32, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
            Value pNum = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, p1_32, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
            Value nIdx64 = safeZExt(builder, module.getLoc(), i64Type, nIdx);
            Value pNum64 = safeZExt(builder, module.getLoc(), i64Type, pNum);
            Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2))), pNum64);
            Value p2_64 = safeZExt(builder, module.getLoc(), i64Type, p2);
            builder.create<LLVM::StoreOp>(module.getLoc(), p2_64, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, net, ValueRange{offset}));
        };
        setTarget(port1, port2); setTarget(port2, port1);
        Block *pushBlock = wFunc.addBlock(), *noPushBlock = wFunc.addBlock();
        Value isP1 = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, builder.create<LLVM::AndOp>(module.getLoc(), i32Type, safeZExt(builder, module.getLoc(), i32Type, port1), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
        Value isP2 = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, builder.create<LLVM::AndOp>(module.getLoc(), i32Type, safeZExt(builder, module.getLoc(), i32Type, port2), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
        builder.create<LLVM::CondBrOp>(module.getLoc(), builder.create<LLVM::AndOp>(module.getLoc(), builder.getI1Type(), isP1, isP2), pushBlock, noPushBlock);
        builder.setInsertionPointToStart(pushBlock);
        Value curT = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, tl, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
        Value pR = builder.create<LLVM::OrOp>(module.getLoc(), i64Type, safeZExt(builder, module.getLoc(), i64Type, port1), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, safeZExt(builder, module.getLoc(), i64Type, port2), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(32))));
        builder.create<LLVM::StoreOp>(module.getLoc(), pR, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, q, ValueRange{curT}));
        builder.create<LLVM::BrOp>(module.getLoc(), noPushBlock);
        builder.setInsertionPointToStart(noPushBlock);
    };

    Value metaA = getMeta(pA); Value metaB = getMeta(pB);
    Value metaA32 = safeZExt(builder, module.getLoc(), i32Type, metaA);
    Value metaB32 = safeZExt(builder, module.getLoc(), i32Type, metaB);
    Value polA = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaA32, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(30)));
    Value polB = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaB32, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(30)));
    Value labelA = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, metaA32, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));
    Value labelB = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, metaB32, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));

    logRedex(labelA, labelB);

    Block *specBlock = wFunc.addBlock(), *genBlock = wFunc.addBlock();
    
    Value isEraA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x979115)));
    Value isEraB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x979115)));
    Value isBranchA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x873945)));
    Value isBranchB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x873945)));
    Value isPrintA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xc9cbf5)));
    Value isPrintB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xc9cbf5)));

    Value implA = builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "lookup_rule", ValueRange{labelA, labelB}).getResult();
    Value implB = builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "lookup_rule", ValueRange{labelB, labelA}).getResult();
    Value zero32 = builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0));
    Value hasRuleA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::ne, implA, zero32);
    Value hasRuleB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::ne, implB, zero32);

    Value isSpec = builder.create<LLVM::OrOp>(module.getLoc(),
        builder.create<LLVM::OrOp>(module.getLoc(),
            builder.create<LLVM::OrOp>(module.getLoc(), isEraA, isEraB),
            builder.create<LLVM::OrOp>(module.getLoc(), isBranchA, isBranchB)),
        builder.create<LLVM::OrOp>(module.getLoc(),
            builder.create<LLVM::OrOp>(module.getLoc(), isPrintA, isPrintB),
            builder.create<LLVM::OrOp>(module.getLoc(), hasRuleA, hasRuleB)));
    
    builder.create<LLVM::CondBrOp>(module.getLoc(), isSpec, specBlock, genBlock);

    builder.setInsertionPointToStart(genBlock);
    {
        Value labelsMatch = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, labelB);
        Value polsDiff = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::ne, polA, polB);
        Value isAnn = builder.create<LLVM::AndOp>(module.getLoc(), labelsMatch, polsDiff);
        
        Block *annPath = wFunc.addBlock(), *commPath = wFunc.addBlock();
        builder.create<LLVM::CondBrOp>(module.getLoc(), isAnn, annPath, commPath);
        
        builder.setInsertionPointToStart(annPath);
        {
            // Check for fire_op (mlir-op)
            Value isPrint = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xc9cbf5)));
            Block *firePrint = wFunc.addBlock(), *stdAnn = wFunc.addBlock();
            builder.create<LLVM::CondBrOp>(module.getLoc(), isPrint, firePrint, stdAnn);
            
            builder.setInsertionPointToStart(firePrint);
            {
                // Simple hack for print_i32: load val from Port 2, state from Port 1
                Value bundleP = getAux(pA, 1);
                Value bTarget = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{safeZExt(builder, module.getLoc(), i64Type, bundleP)}));
                Value val = getAux(bTarget, 1);
                Value state = getAux(pA, 2);
                
                linkPorts(state, getAux(pB, 1), wState); // Pass state along
                builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            }

            builder.setInsertionPointToStart(stdAnn);
            linkPorts(getAux(pA, 1), getAux(pB, 1), wState);
            linkPorts(getAux(pA, 2), getAux(pB, 2), wState);
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
        }

        builder.setInsertionPointToStart(commPath);
        {
            Value auxA1 = getAux(pA, 1); Value auxA2 = getAux(pA, 2);
            Value auxB1 = getAux(pB, 1); Value auxB2 = getAux(pB, 2);
            auto allocNode = [&](Value pol, Value label, Value a1, Value a2) {
                Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                Value nIdx64 = safeZExt(builder, module.getLoc(), i64Type, nIdx);
                Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                Value metaVal = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, pol, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(30))), label);
                
                auto makePort = [&](int p) {
                    return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                };

                builder.create<LLVM::StoreOp>(module.getLoc(), makePort(0), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{base}));
                builder.create<LLVM::StoreOp>(module.getLoc(), a1, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)))}));
                builder.create<LLVM::StoreOp>(module.getLoc(), a2, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)))}));
                builder.create<LLVM::StoreOp>(module.getLoc(), metaVal, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(3)))}));
                return nIdx;
            };
            auto makePort = [&](Value idx, int p) {
                return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, idx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
            };
            Value a1 = allocNode(polA, labelA, auxA1, auxA2); Value a2 = allocNode(polA, labelA, auxA1, auxA2);
            Value b1 = allocNode(polB, labelB, auxB1, auxB2); Value b2 = allocNode(polB, labelB, auxB1, auxB2);
            linkPorts(makePort(a1, 1), makePort(b1, 1), wState);
            linkPorts(makePort(a1, 2), makePort(b2, 1), wState);
            linkPorts(makePort(a2, 1), makePort(b1, 2), wState);
            linkPorts(makePort(a2, 2), makePort(b2, 2), wState);
            linkPorts(makePort(a1, 0), auxB1, wState);
            linkPorts(makePort(a2, 0), auxB2, wState);
            linkPorts(makePort(b1, 0), auxA1, wState);
            linkPorts(makePort(b2, 0), auxA2, wState);
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
        }
    }

    builder.setInsertionPointToStart(specBlock);
    {
        Block *eraRule = wFunc.addBlock();
        Block *dispatchRule = wFunc.addBlock();
        Block *printRule = wFunc.addBlock();
        Block *branchRule = wFunc.addBlock();
        
        Value hasEra = builder.create<LLVM::OrOp>(module.getLoc(), isEraA, isEraB);
        Value hasDispatch = builder.create<LLVM::OrOp>(module.getLoc(), hasRuleA, hasRuleB);
        Value hasPrint = builder.create<LLVM::OrOp>(module.getLoc(), isPrintA, isPrintB);
        
        Block *next1 = wFunc.addBlock();
        builder.create<LLVM::CondBrOp>(module.getLoc(), hasEra, eraRule, next1);
        
        builder.setInsertionPointToStart(next1);
        Block *next2 = wFunc.addBlock();
        builder.create<LLVM::CondBrOp>(module.getLoc(), hasDispatch, dispatchRule, next2);
        
        builder.setInsertionPointToStart(next2);
        builder.create<LLVM::CondBrOp>(module.getLoc(), hasPrint, printRule, branchRule);

        // eraRule
        builder.setInsertionPointToStart(eraRule);
        {
            Value eraP = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, pA, pB);
            Value otherP = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, pB, pA);
            Value otherLabel = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, labelB, labelA);
            Value otherIsEra = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, otherLabel, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x979115)));
            
            Value otherPol = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, polB, polA);
            Value two32 = builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2));
            Value isVal = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, otherPol, two32);
            Value isAnn = builder.create<LLVM::OrOp>(module.getLoc(), otherIsEra, isVal);
            
            Block *doAnn = wFunc.addBlock(), *doProp = wFunc.addBlock();
            builder.create<LLVM::CondBrOp>(module.getLoc(), isAnn, doAnn, doProp);
            builder.setInsertionPointToStart(doAnn);
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            
            builder.setInsertionPointToStart(doProp);
            {
                auto makeEra = [&]() {
                    Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                    Value nIdx64 = safeZExt(builder, module.getLoc(), i64Type, nIdx);
                    Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                    auto store = [&](int i, Value v) {
                        Value off = builder.create<LLVM::AddOp>(module.getLoc(), i64Type, base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                        builder.create<LLVM::StoreOp>(module.getLoc(), v, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{off}));
                    };
                    auto mP = [&](int p) {
                        return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                    };
                    store(0, mP(0)); store(1, mP(1)); store(2, mP(2));
                    store(3, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr((2u << 30) | 0x979115u)));
                    return mP(0);
                };
                linkPorts(getAux(otherP, 1), makeEra(), wState);
                linkPorts(getAux(otherP, 2), makeEra(), wState);
                builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            }
        }

        // dispatchRule
        builder.setInsertionPointToStart(dispatchRule);
        {
            Value opP = builder.create<LLVM::SelectOp>(module.getLoc(), hasRuleA, pA, pB);
            Value valP = builder.create<LLVM::SelectOp>(module.getLoc(), hasRuleA, pB, pA);
            Value opLabel = builder.create<LLVM::SelectOp>(module.getLoc(), hasRuleA, labelA, labelB);
            Value valLabel = builder.create<LLVM::SelectOp>(module.getLoc(), hasRuleA, labelB, labelA);
            Value impl = builder.create<LLVM::SelectOp>(module.getLoc(), hasRuleA, implA, implB);
            
            Value v0 = getAux(valP, 1);
            Value nArgs = builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "get_num_args", ValueRange{impl}).getResult();
            
            Value isUnary = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, nArgs, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)));
            Value isBinary = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, nArgs, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
            
            Block *doUnary = wFunc.addBlock();
            Block *checkBinary = wFunc.addBlock();
            Block *doBinary = wFunc.addBlock();
            Block *doComm = wFunc.addBlock();
            
            builder.create<LLVM::CondBrOp>(module.getLoc(), isUnary, doUnary, checkBinary);
            
            auto makeVal = [&](Value v, Value label) {
                Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                Value nIdx64 = safeZExt(builder, module.getLoc(), i64Type, nIdx);
                Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                auto store = [&](int i, Value val) {
                    Value off = builder.create<LLVM::AddOp>(module.getLoc(), i64Type, base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                    builder.create<LLVM::StoreOp>(module.getLoc(), val, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{off}));
                };
                auto mP = [&](int p) {
                    return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                };
                store(0, mP(0)); store(1, v); store(2, mP(2));
                store(3, builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2u << 30)), label));
                return mP(0);
            };
            
            // Unary
            builder.setInsertionPointToStart(doUnary);
            {
                Value zero64 = builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0));
                Value nodeA = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, valP, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                Value nodeB = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, opP, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                
                Value isGpu = builder.create<LLVM::CallOp>(module.getLoc(), i1Type, "is_gpu_op", ValueRange{impl}).getResult();
                Block *gpuBranch = wFunc.addBlock();
                Block *cpuBranch = wFunc.addBlock();
                builder.create<LLVM::CondBrOp>(module.getLoc(), isGpu, gpuBranch, cpuBranch);
                
                builder.setInsertionPointToStart(gpuBranch);
                builder.create<LLVM::CallOp>(module.getLoc(), i64Type, "dispatch_user_op", ValueRange{impl, v0, zero64, zero64, zero64, nodeA, nodeB, wState});
                builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                
                builder.setInsertionPointToStart(cpuBranch);
                Value resVal = builder.create<LLVM::CallOp>(module.getLoc(), i64Type, "dispatch_user_op", ValueRange{impl, v0, zero64, zero64, zero64, nodeA, nodeB, wState}).getResult();
                linkPorts(getAux(opP, 2), makeVal(resVal, valLabel), wState);
                builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            }
            
            // Check Binary
            builder.setInsertionPointToStart(checkBinary);
            builder.create<LLVM::CondBrOp>(module.getLoc(), isBinary, doBinary, doComm);
            
            // Binary
            builder.setInsertionPointToStart(doBinary);
            {
                Value rPort = getAux(opP, 1);
                Value rTarget = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{safeZExt(builder, module.getLoc(), i64Type, rPort)}));
                Value rLabel = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, safeZExt(builder, module.getLoc(), i32Type, getMeta(rTarget)), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));
                
                Value isCall = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, valLabel, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel("call"))));
                Value isRVal = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, rLabel, valLabel), isCall);
                
                Block *doFullBinary = wFunc.addBlock();
                builder.create<LLVM::CondBrOp>(module.getLoc(), isRVal, doFullBinary, doComm);
                
                builder.setInsertionPointToStart(doFullBinary);
                {
                    Value v1 = getAux(rTarget, 1);
                    Value firstArg = builder.create<LLVM::SelectOp>(module.getLoc(), isCall, rTarget, v1);
                    Value zero64 = builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0));
                    Value nodeA = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, valP, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                    Value nodeB = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, opP, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));

                    Value isGpu = builder.create<LLVM::CallOp>(module.getLoc(), i1Type, "is_gpu_op", ValueRange{impl}).getResult();
                    Block *gpuBranch = wFunc.addBlock();
                    Block *cpuBranch = wFunc.addBlock();
                    builder.create<LLVM::CondBrOp>(module.getLoc(), isGpu, gpuBranch, cpuBranch);
                    
                    builder.setInsertionPointToStart(gpuBranch);
                    builder.create<LLVM::CallOp>(module.getLoc(), i64Type, "dispatch_user_op", ValueRange{impl, firstArg, v0, zero64, zero64, nodeA, nodeB, wState});
                    builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                    
                    builder.setInsertionPointToStart(cpuBranch);
                    Value resVal = builder.create<LLVM::CallOp>(module.getLoc(), i64Type, "dispatch_user_op", ValueRange{impl, firstArg, v0, zero64, zero64, nodeA, nodeB, wState}).getResult();
                    linkPorts(getAux(opP, 2), makeVal(resVal, valLabel), wState);
                    builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                }
            }
            
            // Commutation
            builder.setInsertionPointToStart(doComm);
            {
                Value auxA1 = getAux(valP, 1); Value auxA2 = getAux(valP, 2);
                Value auxB1 = getAux(opP, 1); Value auxB2 = getAux(opP, 2);
                
                auto allocNode = [&](Value pol, Value label, Value a1, Value a2) {
                    Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                    Value nIdx64 = safeZExt(builder, module.getLoc(), i64Type, nIdx);
                    Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                    Value metaVal = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, pol, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(30))), label);
                    auto mP = [&](int p) {
                        return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                    };
                    builder.create<LLVM::StoreOp>(module.getLoc(), mP(0), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{base}));
                    builder.create<LLVM::StoreOp>(module.getLoc(), a1, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)))}));
                    builder.create<LLVM::StoreOp>(module.getLoc(), a2, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)))}));
                    builder.create<LLVM::StoreOp>(module.getLoc(), metaVal, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(3)))}));
                    return nIdx;
                };
                
                auto mP = [&](Value idx, int p) {
                    return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, idx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                };
                
                Value polOp = builder.create<LLVM::SelectOp>(module.getLoc(), hasRuleA, polA, polB);
                Value polVal = builder.create<LLVM::SelectOp>(module.getLoc(), hasRuleA, polB, polA);
                
                Value a1 = allocNode(polVal, valLabel, auxA1, auxA2); Value a2 = allocNode(polVal, valLabel, auxA1, auxA2);
                Value b1 = allocNode(polOp, opLabel, auxB1, auxB2); Value b2 = allocNode(polOp, opLabel, auxB1, auxB2);
                
                linkPorts(mP(a1, 1), mP(b1, 1), wState); linkPorts(mP(a1, 2), mP(b2, 1), wState);
                linkPorts(mP(a2, 1), mP(b1, 2), wState); linkPorts(mP(a2, 2), mP(b2, 2), wState);
                linkPorts(mP(a1, 0), auxB1, wState); linkPorts(mP(a2, 0), auxB2, wState);
                linkPorts(mP(b1, 0), auxA1, wState); linkPorts(mP(b2, 0), auxA2, wState);
                
                builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            }
        }

        // printRule
        builder.setInsertionPointToStart(printRule);
        {
            Value opP = builder.create<LLVM::SelectOp>(module.getLoc(), isPrintA, pA, pB);
            Value valP = builder.create<LLVM::SelectOp>(module.getLoc(), isPrintA, pB, pA);
            Value valLabel = builder.create<LLVM::SelectOp>(module.getLoc(), isPrintA, labelB, labelA);
            Value nVal = getAux(valP, 1);
            
            if (!module.lookupSymbol("PRINT_I32_FMT")) {
                OpBuilder b(module.getBodyRegion());
                std::string fmt = "%d\n";
                fmt.push_back('\0');
                b.create<LLVM::GlobalOp>(module.getLoc(), LLVM::LLVMArrayType::get(builder.getI8Type(), fmt.size()), true, LLVM::Linkage::Internal, "PRINT_I32_FMT", builder.getStringAttr(fmt));
            }
            if (!module.lookupSymbol("PRINT_F64_FMT")) {
                OpBuilder b(module.getBodyRegion());
                std::string fmt = "%f\n";
                fmt.push_back('\0');
                b.create<LLVM::GlobalOp>(module.getLoc(), LLVM::LLVMArrayType::get(builder.getI8Type(), fmt.size()), true, LLVM::Linkage::Internal, "PRINT_F64_FMT", builder.getStringAttr(fmt));
            }
            
            Block *printFloatBlock = wFunc.addBlock();
            Block *printDoubleBlock = wFunc.addBlock();
            Block *printIntBlock = wFunc.addBlock();
            Block *printMergeBlock = wFunc.addBlock();
            
            Value isF32 = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, valLabel, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel("f32"))));
            Value isF64 = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, valLabel, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel("f64"))));
            
            builder.create<LLVM::CondBrOp>(module.getLoc(), isF32, printFloatBlock, printDoubleBlock);
            
            // printFloatBlock
            builder.setInsertionPointToStart(printFloatBlock);
            {
                Value fmtPtr = builder.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, "PRINT_F64_FMT");
                Value truncVal = builder.create<LLVM::TruncOp>(module.getLoc(), builder.getI32Type(), nVal);
                Value f32Val = builder.create<LLVM::BitcastOp>(module.getLoc(), builder.getF32Type(), truncVal);
                Value f64Val = builder.create<LLVM::FPExtOp>(module.getLoc(), builder.getF64Type(), f32Val);
                builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "printf", ValueRange{fmtPtr, f64Val});
                builder.create<LLVM::BrOp>(module.getLoc(), printMergeBlock);
            }
            
            // printDoubleBlock
            Block *doPrintDouble = wFunc.addBlock();
            builder.setInsertionPointToStart(printDoubleBlock);
            builder.create<LLVM::CondBrOp>(module.getLoc(), isF64, doPrintDouble, printIntBlock);
            
            builder.setInsertionPointToStart(doPrintDouble);
            {
                Value fmtPtr = builder.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, "PRINT_F64_FMT");
                Value f64Val = builder.create<LLVM::BitcastOp>(module.getLoc(), builder.getF64Type(), nVal);
                builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "printf", ValueRange{fmtPtr, f64Val});
                builder.create<LLVM::BrOp>(module.getLoc(), printMergeBlock);
            }
            
            // printIntBlock
            builder.setInsertionPointToStart(printIntBlock);
            {
                Value fmtPtr = builder.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, "PRINT_I32_FMT");
                builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "printf", ValueRange{fmtPtr, nVal});
                builder.create<LLVM::BrOp>(module.getLoc(), printMergeBlock);
            }
            
            // printMergeBlock
            builder.setInsertionPointToStart(printMergeBlock);
            linkPorts(getAux(opP, 2), valP, wState);
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
        }

        // branchRule
        builder.setInsertionPointToStart(branchRule);
        {
            Value opP = builder.create<LLVM::SelectOp>(module.getLoc(), isBranchA, pA, pB);
            Value valP = builder.create<LLVM::SelectOp>(module.getLoc(), isBranchA, pB, pA);
            Value nVal = getAux(valP, 1);
            
            Value cond = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::ne, nVal, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0)));
            Value taken = builder.create<LLVM::SelectOp>(module.getLoc(), cond, getAux(opP, 1), getAux(opP, 2));
            Value untaken = builder.create<LLVM::SelectOp>(module.getLoc(), cond, getAux(opP, 2), getAux(opP, 1));
            
            auto makeEra = [&]() {
                Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                Value nIdx64 = safeZExt(builder, module.getLoc(), i64Type, nIdx);
                Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                auto store = [&](int i, Value v) {
                    Value off = builder.create<LLVM::AddOp>(module.getLoc(), i64Type, base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                    builder.create<LLVM::StoreOp>(module.getLoc(), v, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{off}));
                };
                auto mP = [&](int p) {
                    return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                };
                store(0, mP(0)); store(1, mP(1)); store(2, mP(2));
                store(3, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr((2u << 30) | 0x979115u)));
                return mP(0);
            };
            
            linkPorts(untaken, makeEra(), wState);
            linkPorts(taken, getAux(opP, 0), wState);
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
        }
    }

    builder.setInsertionPointToStart(lEnd);
    builder.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{builder.create<LLVM::IntToPtrOp>(module.getLoc(), ptrType, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0)))});

    func::FuncOp entry = module.lookupSymbol<func::FuncOp>("main_inet_entry");
    if (entry) {
        builder.setInsertionPoint(entry);
        auto m = builder.create<LLVM::LLVMFuncOp>(entry.getLoc(), "main", LLVM::LLVMFunctionType::get(i32Type, {}));
        Block *mE = m.addEntryBlock(); builder.setInsertionPointToStart(mE);
        for (auto &op : userOps) {
            std::string opName = "";
            std::string typeName = "";
            size_t underscore = op.label.find('_');
            if (underscore != std::string::npos) {
                opName = op.label.substr(0, underscore);
                typeName = op.label.substr(underscore + 1);
            } else {
                std::string suffix = "";
                if (op.label.size() > 2) {
                    suffix = op.label.substr(op.label.size() - 2);
                }
                if (suffix == "64" || suffix == "32") {
                    std::string base = op.label.substr(0, op.label.size() - 2);
                    bool isFloat = (base[0] == 'f');
                    if (isFloat) {
                        base = base.substr(1);
                        typeName = (suffix == "64") ? "f64" : "f32";
                    } else {
                        typeName = (suffix == "64") ? "i64" : "i32";
                    }
                    if (base == "divs" || base == "divu") opName = "div";
                    else if (base == "rems" || base == "remu") opName = "rem";
                    else if (base == "slt") opName = "lt";
                    else if (base == "sgt") opName = "gt";
                    else if (base == "sle") opName = "le";
                    else if (base == "sge") opName = "ge";
                    else if (base == "neq") opName = "ne";
                    else opName = base;
                }
            }
            if (!opName.empty() && !typeName.empty()) {
                Value opHash = builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel(opName)));
                Value typeHash = builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel(typeName)));
                Value implHash = builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel(op.label)));
                builder.create<LLVM::CallOp>(entry.getLoc(), TypeRange{}, "register_rule", ValueRange{opHash, typeHash, implHash});
            }
            // Always register user-defined closure application rule: op.label meets "call"
            {
                Value opHash = builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel(op.label)));
                Value typeHash = builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel("call")));
                Value implHash = builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel(op.label)));
                builder.create<LLVM::CallOp>(entry.getLoc(), TypeRange{}, "register_rule", ValueRange{opHash, typeHash, implHash});
            }
        }
        auto i1Type = builder.getI1Type();
        Value nS = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(128000000));
        Value net = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{nS}).getResult();
        Value q = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{nS}).getResult();
        Value hd = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value tl = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value alL = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value zero64 = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(0));
        builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, hd); builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, tl);
        builder.create<LLVM::StoreOp>(entry.getLoc(), builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(0)), alL);
        Value as = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(40))}).getResult();
        auto sA = [&](int i, Value v) { builder.create<LLVM::StoreOp>(entry.getLoc(), v, builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, ptrType, as, ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(i))})); };
        sA(0, net); sA(1, q); sA(2, hd); sA(3, tl); sA(4, alL);
        builder.create<func::CallOp>(entry.getLoc(), TypeRange{}, entry.getSymName(), ValueRange{builder.create<LLVM::PtrToIntOp>(entry.getLoc(), i64Type, as)});
        builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "worker_thread", ValueRange{as});
        builder.create<LLVM::ReturnOp>(entry.getLoc(), ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(0))});
    }

    if (enableGPU) {
        OpBuilder b(module.getBodyRegion());
        auto fType = b.getFunctionType({}, {});
        auto f = b.create<func::FuncOp>(module.getLoc(), "pic", fType);
        Block *entry = f.addEntryBlock();
        b.setInsertionPointToStart(entry);
        
        Value oneIdx = b.create<arith::ConstantIndexOp>(module.getLoc(), 1);
        auto launchOp = b.create<gpu::LaunchOp>(
            module.getLoc(),
            oneIdx, oneIdx, oneIdx,
            oneIdx, oneIdx, oneIdx,
            /*dynamicSharedMemorySize=*/nullptr,
            /*asyncTokenType=*/Type{},
            /*asyncDependencies=*/ValueRange{},
            /*workgroupAttributions=*/TypeRange{},
            /*privateAttributions=*/TypeRange{},
            /*clusterSizeX=*/nullptr,
            /*clusterSizeY=*/nullptr,
            /*clusterSizeZ=*/nullptr
        );
        b.setInsertionPointToStart(&launchOp.getBody().front());
        b.create<gpu::TerminatorOp>(module.getLoc());
        
        b.setInsertionPointAfter(launchOp);
        b.create<func::ReturnOp>(module.getLoc());
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
std::unique_ptr<Pass> createPicRuntimeToLLVMPass(bool enableGPU, std::string spirvPath) { return std::make_unique<PicRuntimeToLLVMPass>(enableGPU, spirvPath); }
