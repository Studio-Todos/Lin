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
#if __has_include("mlir/Dialect/SPIRV/IR/SPIRVOps.h")
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#endif
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"

#include <set>
#include <algorithm>
#include "mlir/IR/Builders.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/IRMapping.h"
#include <sstream>
#include <unordered_map>
#include <map>

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

struct UserOp {
    uint32_t hash;
    std::string label;
    std::string funcName;
    int numArgs;
    SmallVector<std::string> argTypes;
    std::string dispatch;
};

enum PicrNodeType {
    NODE_CON = 1,
    NODE_DES = 2,
    NODE_DUP = 3,
    NODE_ERA = 4,
    NODE_OP  = 5,
    NODE_HIS = 6,
    NODE_LOG = 7,
    NODE_RVEC = 8
};

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

      auto getInsertPoint = [&](Operation *o) -> Operation* {
          Operation *curr = o;
          while (auto parent = curr->getParentOfType<pic::graph::BoundaryOp>()) {
              curr = parent;
          }
          return curr;
      };

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

      funcOp.walk([&](pic::graph::BoundaryOp boundaryOp) {
          if (!boundaryOp.getBody().empty()) {
              auto &bodyBlock = boundaryOp.getBody().front();
              for (unsigned i = 0; i < bodyBlock.getNumArguments(); ++i) {
                  Value blockArg = bodyBlock.getArgument(i);
                  Value boundaryInput = boundaryOp.getInputs()[i];
                  if (valueToPort.count(boundaryInput)) {
                      valueToPort[blockArg] = valueToPort[boundaryInput];
                  } else {
                      valueToPort[blockArg] = boundaryInput;
                  }
              }
              if (auto terminator = bodyBlock.getTerminator()) {
                  for (unsigned i = 0; i < terminator->getNumOperands(); ++i) {
                      Value termOp = terminator->getOperand(i);
                      Value boundaryOutput = boundaryOp.getResult(i);
                      if (valueToPort.count(termOp)) {
                          valueToPort[boundaryOutput] = valueToPort[termOp];
                      } else {
                          valueToPort[boundaryOutput] = termOp;
                      }
                  }
              }
          }
      });

      funcOp.walk([&](pic::graph::AgentOp op) {
        Location loc = op.getLoc();
        builder.setInsertionPoint(getInsertPoint(op));
        StringRef agentType = op.getAgentType();
        StringRef label = op.getLabel();
        StringRef polarity = op.getPolarity();
        // All agents are now unified. We use the 24-bit label hash and polarity.
        // Polarity: + (0), - (1), * (2)
        uint32_t polVal = 2; // Default to *
        if (polarity == "+") polVal = 0;
        else if (polarity == "-") polVal = 1;

        bool insideBoundary = false;
        Operation *parent = op->getParentOp();
        while (parent && parent != funcOp) {
            if (parent->getName().getStringRef() == "pic_graph.boundary") {
                insideBoundary = true;
                break;
            }
            parent = parent->getParentOp();
        }

        uint32_t val = 0;
        static std::map<std::string, int> strIndexMap;
        static int nextStrIndex = 0;
        if (label == "num" && op.getValue()) val = op.getValue().value();
        else if (label == "str" && op.getStrVal()) {
            val = opcodeForLabel("str");
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
            builder.create<pic::graph::RegistryOp>(loc, builder.getStringAttr(strKey), op.getStrVal().value(), StringAttr{});
            if (strIndexMap.find(strKey) == strIndexMap.end()) {
                strIndexMap[strKey] = nextStrIndex++;
            }
        } else val = opcodeForLabel(label);

        uint8_t nodeType = NODE_OP; // Default to omega/op
        if (agentType == "constructor") nodeType = NODE_CON;
        else if (agentType == "destructor") nodeType = NODE_DES;
        else if (agentType == "gamma") {
            if (polVal == 0) nodeType = NODE_CON;
            else if (polVal == 1) nodeType = NODE_DES;
        }
        else if (agentType == "delta") nodeType = NODE_DUP;
        else if (agentType == "epsilon") nodeType = NODE_ERA;
        else if (agentType == "history" || agentType == "H") nodeType = NODE_HIS;
        else if (agentType == "log_bond" || agentType == "L") nodeType = NODE_LOG;
        else if (agentType == "reverse_vector" || agentType == "R") nodeType = NODE_RVEC;
        
        uint8_t allocType = (polVal << 6) | nodeType;
        Value valConst = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(val));
        auto alloc = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, allocType, valConst, builder.getBoolAttr(insideBoundary));
        Value nodeIdx = alloc.getIndex();
        
        if ((label == "num" || label == "i32" || label == "i64" || label == "f64" || label == "bool" || label == "str") && (op.getValue() || op.getStrVal())) {
            if (label == "str") {
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
                int idx = strIndexMap[strKey];
                Value litVal = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(idx));
                builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(1), litVal);
            } else {
                uint64_t fullVal = op.getValue().value();
                Value lowVal = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(fullVal & 0xFFFFFFFF));
                builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(1), lowVal);
                if (label == "f64" || label == "i64" || label == "num") {
                    Value highVal = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(fullVal >> 32));
                    builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(2), highVal);
                }
            }
        }

        auto makePort = [&](int i) {
            Value nodeIdx32 = (nodeIdx.getType() == i32Type) ? nodeIdx : builder.create<arith::TruncIOp>(loc, i32Type, nodeIdx);
            Value shifted = builder.create<arith::ShLIOp>(loc, i32Type, nodeIdx32, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2)));
            return builder.create<arith::OrIOp>(loc, i32Type, shifted, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(i)));
        };

        if (agentType == "omega") {
            bool isUserOp = (label == "call" || label == "print");
            for (const auto &uLabel : userOpLabels) {
                if (label == uLabel) {
                    isUserOp = true;
                    break;
                }
            }
            bool isLit = (label == "num" || label == "i1" || label == "i8" || label == "i16" || label == "i32" || label == "i64" || label == "f32" || label == "f64" || label == "bool" || label == "str" || label.starts_with("STR_"));
            if (isUserOp || isLit) {
                valueToPort[op.getP0()] = makePort(0); // Callee/Function/Literal -> Port 0 (Principal)
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

      auto resolvePort = [&](Value v) -> Value {
          while (valueToPort.count(v)) {
              Value next = valueToPort[v];
              if (next == v) break;
              v = next;
          }
          return v;
      };

      funcOp.walk([&](pic::graph::LinkOp op) {
          Value pA = resolvePort(op.getA());
          Value pB = resolvePort(op.getB());
          if (pA.getType().isa<IntegerType>() && pB.getType().isa<IntegerType>()) {
              builder.setInsertionPoint(getInsertPoint(op));
              auto safeTrunc = [&](Value v) {
                  return (v.getType() == i32Type) ? v : builder.create<arith::TruncIOp>(op.getLoc(), i32Type, v);
              };
              builder.create<pic::runtime::LinkOp>(op.getLoc(), safeTrunc(pA), safeTrunc(pB));
          }
      });

      SmallVector<Operation*> toErase;
      funcOp.walk([&](pic::graph::LinkOp op) {
          if (!op->getParentOfType<pic::graph::BoundaryOp>()) {
              toErase.push_back(op);
          }
      });
      funcOp.walk([&](pic::graph::AgentOp op) {
          if (!op->getParentOfType<pic::graph::BoundaryOp>()) {
              toErase.push_back(op);
          }
      });
      funcOp.walk([&](pic::graph::BoundaryOp op) {
          if (!op->getParentOfType<pic::graph::BoundaryOp>()) {
              toErase.push_back(op);
          }
      });

      funcOp.walk([&](func::ReturnOp op) {
         if (op->getBlock() == &funcOp.getBody().front() && op.getNumOperands() > 0) {
             Value resPort = resolvePort(op.getOperand(0));
             if (resPort.getType().isa<IntegerType>()) {
                 if (funcOp.getSymName() == "main_inet_entry") {
                     Location loc = op.getLoc();
                     builder.setInsertionPoint(op);
                     Value eraValConst = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0x979115));
                     auto eraAlloc = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, (2 << 6) | NODE_ERA, eraValConst, IntegerAttr{});
                     Value eraIdx = eraAlloc.getIndex();
                     Value eraPort = builder.create<arith::OrIOp>(loc, i32Type, builder.create<arith::ShLIOp>(loc, i32Type, eraIdx, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2))), builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(1)));
                     auto safeTrunc = [&](Value v) {
                         return (v.getType() == i32Type) ? v : builder.create<arith::TruncIOp>(loc, i32Type, v);
                     };
                     builder.create<pic::runtime::LinkOp>(loc, safeTrunc(resPort), eraPort);
                 }
                 auto safeExtReturn = [&](Value v) {
                     return (v.getType() == i64Type) ? v : builder.create<arith::ExtUIOp>(op.getLoc(), i64Type, v);
                 };
                 op.setOperand(0, safeExtReturn(resPort));
             }
         }
      });
      for (auto *op : toErase) op->erase();
    });
  }
};

struct PicReduceToRuntimePass : public PassWrapper<PicReduceToRuntimePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicReduceToRuntimePass)
  bool enableGPU;
  std::string spirvPath;
  PicReduceToRuntimePass(bool enableGPU, std::string spirvPath) : enableGPU(enableGPU), spirvPath(spirvPath) {}
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

    if (enableGPU) {
        // GPU path: dispatch all pairs to GPU kernel via pic_gpu_dispatch_helper.
        // The SPIR-V kernel (pic_kernel in PicRuntimeToSPIRVPass) handles all base
        // rules (annihilate, duplicate, erase, r-vector). User ops are dispatched
        // inline after GPU returns.
        if (!module.lookupSymbol<func::FuncOp>("pic_gpu_dispatch_helper")) {
            OpBuilder modB(module.getBodyRegion());
            modB.setInsertionPointToEnd(module.getBody());
            auto helperTy = modB.getFunctionType({i32Type, i32Type, i64Type}, {});
            auto helper = modB.create<func::FuncOp>(loc, "pic_gpu_dispatch_helper", helperTy);
            helper.setPrivate();
        }
        builder.create<func::CallOp>(loc, TypeRange{}, "pic_gpu_dispatch_helper",
            ValueRange{bodyNodeA, bodyNodeB, stateArg});
        builder.create<cf::BranchOp>(loc, lHead);
    } else {
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

    // ==========================================
    // rvecCase
    // ==========================================
    builder.setInsertionPointToStart(rvecCase);
    builder.create<mlir::pic::reduce::ReverseVectorOp>(loc, bodyNodeA, bodyNodeB, stateArg);
    builder.create<cf::BranchOp>(loc, lHead);

    // ==========================================
    // nonRvecCase
    // ==========================================
    builder.setInsertionPointToStart(nonRvecCase);
    Value isEraA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("era"))));
    Value isEraB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("era"))));
    Value hasEra = builder.create<arith::OrIOp>(loc, isEraA, isEraB);

    Block *eraCase = wFunc.addBlock();
    Block *checkDispatch = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, hasEra, eraCase, checkDispatch);

    // eraCase
    builder.setInsertionPointToStart(eraCase);
    builder.create<mlir::pic::reduce::EraseOp>(loc, bodyNodeA, bodyNodeB, stateArg);
    builder.create<cf::BranchOp>(loc, lHead);

    // checkDispatch
    builder.setInsertionPointToStart(checkDispatch);
    Value implA = builder.create<func::CallOp>(loc, i32Type, "lookup_rule", ValueRange{labelA, labelB}).getResult(0);
    Value implB = builder.create<func::CallOp>(loc, i32Type, "lookup_rule", ValueRange{labelB, labelA}).getResult(0);
    Value hasRuleA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, implA, c0_i32);
    Value hasRuleB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, implB, c0_i32);
    Value hasDispatch = builder.create<arith::OrIOp>(loc, hasRuleA, hasRuleB);

    Value opNode = builder.create<arith::SelectOp>(loc, hasRuleA, bodyNodeA, bodyNodeB);
    Value valNode = builder.create<arith::SelectOp>(loc, hasRuleA, bodyNodeB, bodyNodeA);
    Value valLabel = builder.create<arith::SelectOp>(loc, hasRuleA, labelB, labelA);
    Value opLabel = builder.create<arith::SelectOp>(loc, hasRuleA, labelA, labelB);
    Value impl = builder.create<arith::SelectOp>(loc, hasRuleA, implA, implB);

    Block *dispatchCase = wFunc.addBlock();
    Block *genericCase = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, hasDispatch, dispatchCase, genericCase);

    // dispatchCase — registered user op fires through FireOpOp
    builder.setInsertionPointToStart(dispatchCase);
    builder.create<mlir::pic::reduce::FireOpOp>(loc, bodyNodeA, bodyNodeB, stateArg);
    builder.create<cf::BranchOp>(loc, lHead);

    // genericCase  —  annihilation or duplication
    builder.setInsertionPointToStart(genericCase);
    Value labelsMatch = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, labelB);
    Value polsDiff = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, polA, polB);
    Value isAnn = builder.create<arith::AndIOp>(loc, labelsMatch, polsDiff);

    Block *annPath = wFunc.addBlock();
    Block *dupPath = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isAnn, annPath, dupPath);

    // annPath
    builder.setInsertionPointToStart(annPath);
    builder.create<mlir::pic::reduce::AnnihilateOp>(loc, bodyNodeA, bodyNodeB, stateArg);
    builder.create<cf::BranchOp>(loc, lHead);

    // dupPath
    builder.setInsertionPointToStart(dupPath);
    builder.create<mlir::pic::reduce::DuplicateOp>(loc, bodyNodeA, bodyNodeB, stateArg);
    builder.create<cf::BranchOp>(loc, lHead);
    }

    // ==========================================
    // lEnd
    // ==========================================
    builder.setInsertionPointToStart(lEnd);
    builder.create<func::ReturnOp>(loc, ValueRange{c0_i64});
  }
};

struct PicRuntimeToLLVMPass : public PassWrapper<PicRuntimeToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicRuntimeToLLVMPass)
  bool enableGPU;
  std::string spirvPath;
  PicRuntimeToLLVMPass(bool enableGPU, std::string spirvPath) : enableGPU(enableGPU), spirvPath(spirvPath) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    
    std::vector<UserOp> userOps;
    for (auto func : module.getOps<func::FuncOp>()) {
        if (auto labelAttr = func->getAttrOfType<StringAttr>("lin.original_label")) {
            std::string label = labelAttr.getValue().str();
            userOps.push_back({opcodeForLabel(label), label, func.getSymName().str(), (int)func.getNumArguments(), {}, ""});
        }
    }
    
    // First, convert all func.call to runtime helper functions to LLVM calls
    module.walk([&](func::CallOp callOp) {
        StringRef callee = callOp.getCallee();
        if (callee == "lookup_rule" || callee == "dispatch_user_op" ||
            callee == "is_gpu_op" || callee == "get_num_args" ||
            callee == "pic_gpu_dispatch_helper") {
            
            OpBuilder b(callOp);
            SmallVector<Value> operands;
            for (auto op : callOp.getOperands()) {
                operands.push_back(op);
            }
            SmallVector<Type, 1> retTypes;
            if (callOp.getNumResults() > 0) {
                retTypes.push_back(callOp.getResult(0).getType());
            }
            auto llvmCall = b.create<LLVM::CallOp>(callOp.getLoc(), retTypes, callee, operands);
            if (callOp.getNumResults() > 0) {
                callOp.getResult(0).replaceAllUsesWith(llvmCall.getResult());
            }
            callOp.erase();
        }
    });

    for (StringRef name : {"lookup_rule", "dispatch_user_op", "is_gpu_op", "get_num_args"}) {
        if (auto funcOp = module.lookupSymbol<func::FuncOp>(name)) {
            funcOp.erase();
        }
    }
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
    decl("malloc", ptrType, {i64Type}); decl("calloc", ptrType, {i64Type, i64Type}); decl("free", voidType, {ptrType});
    decl("pthread_create", i32Type, {ptrType, ptrType, ptrType, ptrType}); decl("pthread_join", i32Type, {i64Type, ptrType});
    decl("printf", i32Type, {ptrType}, true);

    module.walk([&](func::FuncOp f) {
        if (f.empty()) return;
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
        
        SmallVector<Operation*> nodes, setPorts, getPorts, getPortsDynamic, links, pushRedexs, popRedexs, getHistorys, setHistorys, uncomputeSweeps;
        f.walk([&](Operation *o) {
            StringRef opName = o->getName().getStringRef();
            if (opName == "pic_runtime.alloc_node") nodes.push_back(o);
            else if (opName == "pic_runtime.set_port") setPorts.push_back(o);
            else if (opName == "pic_runtime.get_port") getPorts.push_back(o);
            else if (opName == "pic_runtime.get_port_dynamic") getPortsDynamic.push_back(o);
            else if (opName == "pic_runtime.link") links.push_back(o);
            else if (opName == "pic_runtime.push_redex") pushRedexs.push_back(o);
            else if (opName == "pic_runtime.pop_redex") popRedexs.push_back(o);
            else if (opName == "pic_runtime.get_history") getHistorys.push_back(o);
            else if (opName == "pic_runtime.set_history") setHistorys.push_back(o);
            else if (opName == "pic_runtime.uncompute_sweep") uncomputeSweeps.push_back(o);
        });

        auto nonBarrierLink = [&](OpBuilder &ob, Location loc, Value p1, Value p2, Value as) {
            Value net = ob.create<LLVM::LoadOp>(loc, ptrType, ob.create<LLVM::GEPOp>(loc, ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0))}));
            Value q = ob.create<LLVM::LoadOp>(loc, ptrType, ob.create<LLVM::GEPOp>(loc, ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1))}));
            
            auto setT = [&](Value v1, Value v2) {
                Value nIdx = ob.create<LLVM::LShrOp>(loc, i32Type, v1, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2)));
                Value pNum = ob.create<LLVM::AndOp>(loc, i32Type, v1, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(3)));
                Value offset = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, safeZExt(ob, loc, i64Type, nIdx), ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2))), safeZExt(ob, loc, i64Type, pNum));
                ob.create<LLVM::StoreOp>(loc, v2, ob.create<LLVM::GEPOp>(loc, ptrType, i32Type, net, ValueRange{offset}));
            };
            setT(p1, p2); setT(p2, p1);
            
            Value isP1 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ob.create<LLVM::AndOp>(loc, i32Type, p1, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(3))), ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0)));
            Value isP2 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ob.create<LLVM::AndOp>(loc, i32Type, p2, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(3))), ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0)));
            Value isR = ob.create<LLVM::AndOp>(loc, builder.getI1Type(), isP1, isP2);
            
            Block *curr = ob.getBlock();
            Block *push = f.addBlock();
            Block *cont = f.addBlock();
            
            ob.setInsertionPointToEnd(curr);
            ob.create<LLVM::CondBrOp>(loc, isR, push, cont);
            
            ob.setInsertionPointToStart(push);
            Value tlPtrPtr = ob.create<LLVM::GEPOp>(loc, ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(3))});
            Value tlPtr = ob.create<LLVM::LoadOp>(loc, ptrType, tlPtrPtr);
            Value curT = ob.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, tlPtr, ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
            Value inBounds = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, curT, ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(16000000)));
            
            Block *doStore = f.addBlock();
            Block *contPush = f.addBlock();
            
            ob.setInsertionPointToEnd(push);
            ob.create<LLVM::CondBrOp>(loc, inBounds, doStore, contPush);
            
            ob.setInsertionPointToStart(doStore);
            Value r = ob.create<LLVM::OrOp>(loc, i64Type, safeZExt(ob, loc, i64Type, p1), ob.create<LLVM::ShlOp>(loc, i64Type, safeZExt(ob, loc, i64Type, p2), ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(32))));
            ob.create<LLVM::StoreOp>(loc, r, ob.create<LLVM::GEPOp>(loc, ptrType, i64Type, q, ValueRange{curT}));
            ob.create<LLVM::BrOp>(loc, contPush);
            
            ob.setInsertionPointToStart(contPush);
            ob.create<LLVM::BrOp>(loc, cont);
            
            ob.setInsertionPointToStart(cont);
        };

        auto allocateRvecNode = [&](OpBuilder &ob, Location loc, Value as) -> Value {
            Value alPtr = ob.create<LLVM::GEPOp>(loc, ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(4))});
            Value alL = ob.create<LLVM::LoadOp>(loc, ptrType, alPtr);
            Value nIdx = ob.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, alL, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
            
            Value net = ob.create<LLVM::LoadOp>(loc, ptrType, ob.create<LLVM::GEPOp>(loc, ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0))}));
            Value nIdx64 = safeZExt(ob, loc, i64Type, nIdx);
            Value base = ob.create<LLVM::ShlOp>(loc, i64Type, nIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2)));
            
            Value metaVal = ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x88000000));
            Value off3 = ob.create<LLVM::AddOp>(loc, i64Type, base, ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(3)));
            ob.create<LLVM::StoreOp>(loc, metaVal, ob.create<LLVM::GEPOp>(loc, ptrType, i32Type, net, ValueRange{off3}));
            
            Value port0Val = ob.create<LLVM::OrOp>(loc, i32Type, ob.create<LLVM::ShlOp>(loc, i32Type, nIdx, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0)));
            return port0Val;
        };

        auto genLinkPorts = [&](OpBuilder &ob, Location loc, Value p1, Value p2, Value as) {
            Value net = ob.create<LLVM::LoadOp>(loc, ptrType, ob.create<LLVM::GEPOp>(loc, ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0))}));
            Value historyNet = ob.create<LLVM::LoadOp>(loc, ptrType, ob.create<LLVM::GEPOp>(loc, ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(5))}));
            
            Value p1_32 = safeZExt(ob, loc, i32Type, p1);
            Value p2_32 = safeZExt(ob, loc, i32Type, p2);
            
            Value nIdx1 = ob.create<LLVM::LShrOp>(loc, i32Type, p1_32, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2)));
            Value pNum1 = ob.create<LLVM::AndOp>(loc, i32Type, p1_32, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(3)));
            
            Value nIdx2 = ob.create<LLVM::LShrOp>(loc, i32Type, p2_32, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2)));
            Value pNum2 = ob.create<LLVM::AndOp>(loc, i32Type, p2_32, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(3)));
            
            Value offsetMeta1 = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, safeZExt(ob, loc, i64Type, nIdx1), ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(3)));
            Value meta1 = ob.create<LLVM::LoadOp>(loc, i32Type, ob.create<LLVM::GEPOp>(loc, ptrType, i32Type, net, ValueRange{offsetMeta1}));
            
            Value offsetMeta2 = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, safeZExt(ob, loc, i64Type, nIdx2), ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(3)));
            Value meta2 = ob.create<LLVM::LoadOp>(loc, i32Type, ob.create<LLVM::GEPOp>(loc, ptrType, i32Type, net, ValueRange{offsetMeta2}));
            
            Value typeVal1 = ob.create<LLVM::LShrOp>(loc, i32Type, meta1, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(24)));
            Value type1 = ob.create<LLVM::AndOp>(loc, i32Type, typeVal1, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x3F)));
            
            Value typeVal2 = ob.create<LLVM::LShrOp>(loc, i32Type, meta2, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(24)));
            Value type2 = ob.create<LLVM::AndOp>(loc, i32Type, typeVal2, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x3F)));
            
            Value isRvec1 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, type1, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(NODE_RVEC)));
            Value isPNum1_0 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, pNum1, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0)));
            Value isDup2 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, type2, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(NODE_DUP)));
            Value isPNum2_gt0 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, pNum2, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0)));
            
            Value condA = ob.create<LLVM::AndOp>(loc, ob.create<LLVM::AndOp>(loc, isRvec1, isPNum1_0), ob.create<LLVM::AndOp>(loc, isDup2, isPNum2_gt0));
            
            Value isRvec2 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, type2, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(NODE_RVEC)));
            Value isPNum2_0 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, pNum2, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0)));
            Value isDup1 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, type1, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(NODE_DUP)));
            Value isPNum1_gt0 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, pNum1, ob.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0)));
            
            Value condB = ob.create<LLVM::AndOp>(loc, ob.create<LLVM::AndOp>(loc, isRvec2, isPNum2_0), ob.create<LLVM::AndOp>(loc, isDup1, isPNum1_gt0));
            
            Value isBarrier = ob.create<LLVM::OrOp>(loc, condA, condB);
            
            Value dupNodeIdx = ob.create<LLVM::SelectOp>(loc, condA, nIdx2, nIdx1);
            Value dupNodeIdx64 = safeZExt(ob, loc, i64Type, dupNodeIdx);
            
            Block *curr = ob.getBlock();
            Block *cont = curr->splitBlock(ob.getInsertionPoint());
            
            Block *barrierBlock = f.addBlock();
            Block *mergeBlock = f.addBlock();
            Block *exitBarrier = f.addBlock();
            Block *standardLink = f.addBlock();
            
            ob.setInsertionPointToEnd(curr);
            ob.create<LLVM::CondBrOp>(loc, isBarrier, barrierBlock, standardLink);
            
            // barrierBlock
            ob.setInsertionPointToStart(barrierBlock);
            Value counterOffset = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, dupNodeIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1))), ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1)));
            Value counterPtr = ob.create<LLVM::GEPOp>(loc, ptrType, i64Type, historyNet, ValueRange{counterOffset});
            Value decVal = ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1));
            Value prevCount = ob.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::sub, counterPtr, decVal, LLVM::AtomicOrdering::seq_cst);
            Value nextCount = ob.create<LLVM::SubOp>(loc, i64Type, prevCount, decVal);
            Value isZero = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, nextCount, ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0)));
            ob.create<LLVM::CondBrOp>(loc, isZero, mergeBlock, exitBarrier);
            
            // mergeBlock
            ob.setInsertionPointToStart(mergeBlock);
            Value offsetW0 = ob.create<LLVM::ShlOp>(loc, i64Type, dupNodeIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2)));
            Value w0 = ob.create<LLVM::LoadOp>(loc, i32Type, ob.create<LLVM::GEPOp>(loc, ptrType, i32Type, net, ValueRange{offsetW0}));
            Value r_out = allocateRvecNode(ob, loc, as);
            nonBarrierLink(ob, loc, r_out, w0, as);
            ob.create<LLVM::BrOp>(loc, exitBarrier);
            
            // standardLink
            ob.setInsertionPointToStart(standardLink);
            nonBarrierLink(ob, loc, p1, p2, as);
            ob.create<LLVM::BrOp>(loc, exitBarrier);
            
            // exitBarrier
            ob.setInsertionPointToStart(exitBarrier);
            ob.create<LLVM::BrOp>(loc, cont);
            
            ob.setInsertionPointToStart(cont);
        };

        for (auto* o : nodes) {
            OpBuilder ob(o);
            auto allocOp = cast<pic::runtime::AllocNodeOp>(o);
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value alPtr = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(4))});
            Value alL = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, alPtr);
            Value nIdx = ob.create<LLVM::AtomicRMWOp>(o->getLoc(), LLVM::AtomicBinOp::add, alL, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
            
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
            Value base = ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2)));
            
            auto store = [&](int i, Value v) {
                Value off = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, base, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                Value v32 = (v.getType() == i32Type) ? v : ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, v);
                ob.create<LLVM::StoreOp>(o->getLoc(), v32, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{off}));
            };
            
            auto makePort = [&](int p) {
                return ob.create<LLVM::OrOp>(o->getLoc(), i32Type, ob.create<LLVM::ShlOp>(o->getLoc(), i32Type, nIdx, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(p)));
            };
            
            uint8_t typeVal = allocOp.getType();
            Value labelOrVal = allocOp.getLabelOrVal();
            
            Value typeValConst = ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr((uint32_t)typeVal << 24));
            Value labelOrVal32 = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, labelOrVal);
            Value maskConst = ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF));
            Value labelMasked = ob.create<LLVM::AndOp>(o->getLoc(), labelOrVal32, maskConst);
            Value metaValueVal = ob.create<LLVM::OrOp>(o->getLoc(), typeValConst, labelMasked);
            
            store(0, makePort(0)); store(1, makePort(1)); store(2, makePort(2)); store(3, metaValueVal);

            bool isInsideBoundary = allocOp.getInsideBoundary().value_or(false);
            if (isInsideBoundary || typeVal == NODE_DUP) {
                Value historyNet = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(5))}));
                Value hBase = ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                Value valWord0 = ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1ULL << 32)); // bit 0 = inside_boundary is set
                
                Value gep0 = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, historyNet, ValueRange{hBase});
                ob.create<LLVM::StoreOp>(o->getLoc(), valWord0, gep0);
                
                Value hBasePlus1 = ob.create<LLVM::AddOp>(o->getLoc(), hBase, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                Value gep1 = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, historyNet, ValueRange{hBasePlus1});
                
                // If it is a DUP node, initialize the counter to 2.
                Value valWord1 = ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(typeVal == NODE_DUP ? 2 : 0));
                ob.create<LLVM::StoreOp>(o->getLoc(), valWord1, gep1);
            }

            o->getResult(0).replaceAllUsesWith(nIdx);
            o->erase();
        }

        for (auto* o : setPorts) {
            OpBuilder ob(o);
            Value nIdx = o->getOperand(0);
            int pIdx = o->getAttrOfType<IntegerAttr>("port_index").getInt();
            Value val = o->getOperand(1);
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg), ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
            Value offset = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(pIdx)));
            ob.create<LLVM::StoreOp>(o->getLoc(), val, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{offset}));
            o->erase();
        }

        for (auto* o : links) {
            OpBuilder ob(o);
            Value p1 = o->getOperand(0);
            Value p2 = o->getOperand(1);
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            genLinkPorts(ob, o->getLoc(), p1, p2, as);
            o->erase();
        }

        for (auto* o : uncomputeSweeps) {
            OpBuilder ob(o);
            Value boundaryId = o->getOperand(0);
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value alPtr = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(4))});
            Value alL = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, alPtr);
            
            Value rIdx = ob.create<LLVM::AtomicRMWOp>(o->getLoc(), LLVM::AtomicBinOp::add, alL, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
            Value rIdx64 = safeZExt(ob, o->getLoc(), i64Type, rIdx);
            Value base = ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, rIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2)));
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            
            auto makePort = [&](Value idx, int p) {
                return ob.create<LLVM::OrOp>(o->getLoc(), i32Type, ob.create<LLVM::ShlOp>(o->getLoc(), i32Type, idx, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(p)));
            };
            
            Value rPort0 = makePort(rIdx, 0);
            ob.create<LLVM::StoreOp>(o->getLoc(), rPort0, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{base}));
            Value metaVal = ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr((2U << 30) | (8U << 24)));
            Value off3 = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, base, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(3)));
            ob.create<LLVM::StoreOp>(o->getLoc(), metaVal, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{off3}));
            
            genLinkPorts(ob, o->getLoc(), rPort0, boundaryId, as);
            o->erase();
        }
        for (auto* o : getPorts) {
            OpBuilder ob(o);
            Value nIdx = o->getOperand(0);
            int pIdx = o->getAttrOfType<IntegerAttr>("port_index").getInt();
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
            Value offset = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(pIdx)));
            Value val32 = ob.create<LLVM::LoadOp>(o->getLoc(), i32Type, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{offset}));
            o->getResult(0).replaceAllUsesWith(val32);
            o->erase();
        }
        for (auto* o : getPortsDynamic) {
            OpBuilder ob(o);
            Value nIdx = o->getOperand(0);
            Value pIdx = o->getOperand(1);
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
            Value pIdx64 = safeZExt(ob, o->getLoc(), i64Type, pIdx);
            Value offset = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))), pIdx64);
            Value val32 = ob.create<LLVM::LoadOp>(o->getLoc(), i32Type, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{offset}));
            o->getResult(0).replaceAllUsesWith(val32);
            o->erase();
        }
        for (auto* o : pushRedexs) {
            OpBuilder ob(o);
            Value nA = o->getOperand(0);
            Value nB = o->getOperand(1);
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value q = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1))}));
            Value tlPtr = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(3))}));
            
            Value curT = ob.create<LLVM::AtomicRMWOp>(o->getLoc(), LLVM::AtomicBinOp::add, tlPtr, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
            Value inBounds = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::ult, curT, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(16000000)));
            
            Block *curr = ob.getBlock();
            Block *doStore = curr->splitBlock(o);
            Block *cont = doStore->splitBlock(doStore->begin());
            
            ob.setInsertionPointToEnd(curr);
            ob.create<LLVM::CondBrOp>(o->getLoc(), inBounds, doStore, cont);
            
            ob.setInsertionPointToStart(doStore);
            Value r = ob.create<LLVM::OrOp>(o->getLoc(), i64Type, safeZExt(ob, o->getLoc(), i64Type, nA), ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, safeZExt(ob, o->getLoc(), i64Type, nB), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(32))));
            ob.create<LLVM::StoreOp>(o->getLoc(), r, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, q, ValueRange{curT}));
            ob.create<LLVM::BrOp>(o->getLoc(), cont);
            
            ob.setInsertionPointToStart(cont);
            o->erase();
        }
        for (auto* o : popRedexs) {
            OpBuilder ob(o);
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value q = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1))}));
            Value hdPtr = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))}));
            Value tlPtr = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(3))}));
            Value activePtr = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(6))}));
            Value lockPtr = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(7))}));
            
            Value oneVal = ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1));
            Value zeroVal = ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0));
            
            Block *curr = ob.getBlock();
            Block *cont = curr->splitBlock(o);
            
            Block *spinStart = f.addBlock();
            Block *lockedCase = f.addBlock();
            Block *doPop = f.addBlock();
            Block *doWait = f.addBlock();
            Block *waitStart = f.addBlock();
            Block *terminate = f.addBlock();
            Block *checkQueue = f.addBlock();
            Block *wakeUp = f.addBlock();
            Block *keepWaiting = f.addBlock();
            Block *doLoad = f.addBlock();
            
            Value finalValid = cont->addArgument(i1Type, o->getLoc());
            Value finalA = cont->addArgument(i32Type, o->getLoc());
            Value finalB = cont->addArgument(i32Type, o->getLoc());
            
            // curr: branch to spinStart
            ob.setInsertionPointToEnd(curr);
            ob.create<LLVM::BrOp>(o->getLoc(), spinStart);
            
            // spinStart: try to acquire spinlock
            ob.setInsertionPointToStart(spinStart);
            Value prevLock = ob.create<LLVM::AtomicRMWOp>(o->getLoc(), LLVM::AtomicBinOp::xchg, lockPtr, oneVal, LLVM::AtomicOrdering::seq_cst);
            Value isLocked = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::eq, prevLock, oneVal);
            ob.create<LLVM::CondBrOp>(o->getLoc(), isLocked, spinStart, lockedCase);
            
            // lockedCase: check queue under lock
            ob.setInsertionPointToStart(lockedCase);
            Value curH = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, hdPtr);
            Value curT = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, tlPtr);
            Value hasElement = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::ult, curH, curT);
            ob.create<LLVM::CondBrOp>(o->getLoc(), hasElement, doPop, doWait);
            
            // doPop: claim index, release lock, branch to doLoad
            ob.setInsertionPointToStart(doPop);
            Value nextH = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, curH, oneVal);
            ob.create<LLVM::StoreOp>(o->getLoc(), nextH, hdPtr);
            ob.create<LLVM::StoreOp>(o->getLoc(), zeroVal, lockPtr);
            ob.create<LLVM::BrOp>(o->getLoc(), ValueRange{curH}, doLoad);
            
            // doWait: release lock, decrement active_count, branch to waitStart
            ob.setInsertionPointToStart(doWait);
            ob.create<LLVM::StoreOp>(o->getLoc(), zeroVal, lockPtr);
            ob.create<LLVM::AtomicRMWOp>(o->getLoc(), LLVM::AtomicBinOp::sub, activePtr, oneVal, LLVM::AtomicOrdering::seq_cst);
            ob.create<LLVM::BrOp>(o->getLoc(), waitStart);
            
            // waitStart: loop and wait
            ob.setInsertionPointToStart(waitStart);
            Value act = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, activePtr);
            Value isZero = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::eq, act, zeroVal);
            ob.create<LLVM::CondBrOp>(o->getLoc(), isZero, terminate, checkQueue);
            
            // terminate: all threads are idle and queue is empty -> return valid=false
            ob.setInsertionPointToStart(terminate);
            Value falseVal = ob.create<LLVM::ConstantOp>(o->getLoc(), i1Type, builder.getBoolAttr(false));
            Value zero32 = ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(0));
            ob.create<LLVM::BrOp>(o->getLoc(), ValueRange{falseVal, zero32, zero32}, cont);
            
            // checkQueue: check if work is available
            ob.setInsertionPointToStart(checkQueue);
            Value checkH = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, hdPtr);
            Value checkT = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, tlPtr);
            Value checkHas = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::ult, checkH, checkT);
            ob.create<LLVM::CondBrOp>(o->getLoc(), checkHas, wakeUp, keepWaiting);
            
            // wakeUp: increment active count and return to spinStart to try to pop
            ob.setInsertionPointToStart(wakeUp);
            ob.create<LLVM::AtomicRMWOp>(o->getLoc(), LLVM::AtomicBinOp::add, activePtr, oneVal, LLVM::AtomicOrdering::seq_cst);
            ob.create<LLVM::BrOp>(o->getLoc(), spinStart);
            
            // keepWaiting: spin/loop back to waitStart
            ob.setInsertionPointToStart(keepWaiting);
            ob.create<LLVM::BrOp>(o->getLoc(), waitStart);
            
            // doLoad: load values from queue and return valid=true
            Value popIdx = doLoad->addArgument(i64Type, o->getLoc());
            ob.setInsertionPointToStart(doLoad);
            Value val = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, q, ValueRange{popIdx}));
            Value pA = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, val);
            Value pB = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, ob.create<LLVM::LShrOp>(o->getLoc(), i64Type, val, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(32))));
            Value c2 = ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(2));
            Value nA = ob.create<LLVM::LShrOp>(o->getLoc(), i32Type, pA, c2);
            Value nB = ob.create<LLVM::LShrOp>(o->getLoc(), i32Type, pB, c2);
            Value trueVal = ob.create<LLVM::ConstantOp>(o->getLoc(), i1Type, builder.getBoolAttr(true));
            ob.create<LLVM::BrOp>(o->getLoc(), ValueRange{trueVal, nA, nB}, cont);
            
            // cont: replace results
            ob.setInsertionPointToStart(cont);
            o->getResult(0).replaceAllUsesWith(finalValid);
            o->getResult(1).replaceAllUsesWith(finalA);
            o->getResult(2).replaceAllUsesWith(finalB);
            o->erase();
        }
        for (auto* o : getHistorys) {
            OpBuilder ob(o);
            Value nIdx = o->getOperand(0);
            int wIdx = o->getAttrOfType<IntegerAttr>("word_index").getInt();
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value historyNet = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(5))}));
            Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
            Value offset = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1))), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(wIdx)));
            Value val = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, historyNet, ValueRange{offset}));
            o->getResult(0).replaceAllUsesWith(val);
            o->erase();
        }
        for (auto* o : setHistorys) {
            OpBuilder ob(o);
            Value nIdx = o->getOperand(0);
            int wIdx = o->getAttrOfType<IntegerAttr>("word_index").getInt();
            Value val = o->getOperand(1);
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value historyNet = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(5))}));
            Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
            Value offset = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1))), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(wIdx)));
            ob.create<LLVM::StoreOp>(o->getLoc(), val, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, historyNet, ValueRange{offset}));
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

    SmallVector<pic::graph::RegistryOp> regOps;
    module.walk([&](pic::graph::RegistryOp op) {
        regOps.push_back(op);
    });

    for (auto op : regOps) {
        StringAttr n = op->getAttrOfType<StringAttr>("op_name");
        StringAttr p = op->getAttrOfType<StringAttr>("payload");
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
            }
        }
        op.erase();
    }

    builder.setInsertionPointToStart(module.getBody());
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(module.getBody());
        
        // A2b: emit a global 512-entry runtime rule table: [keyA, keyB, impl] × 512
        // register_rule(keyA, keyB, impl) appends an entry; lookup_rule checks it after the
        // compile-time inline chain.
        constexpr int kRuleTableSize = 512;
        if (!module.lookupSymbol("__pic_rule_table")) {
            OpBuilder gb(module.getBodyRegion());
            auto tblTy = LLVM::LLVMArrayType::get(i32Type, kRuleTableSize * 3);
            // A2b fix: ArrayAttr is NOT a valid LLVM IR constant initializer.
            // Use an initializer region with LLVM::ZeroOp to produce zeroinitializer.
            auto tblGlobal = gb.create<LLVM::GlobalOp>(
                module.getLoc(), tblTy, /*isConst=*/false,
                LLVM::Linkage::Internal, "__pic_rule_table", Attribute{});
            Block *initBlk = gb.createBlock(&tblGlobal.getInitializerRegion());
            OpBuilder ib(initBlk, initBlk->end());
            ib.create<LLVM::ReturnOp>(module.getLoc(),
                ib.create<LLVM::ZeroOp>(module.getLoc(), tblTy));
        }
        if (!module.lookupSymbol("__pic_rule_count")) {
            OpBuilder gb(module.getBodyRegion());
            // Simple i32 scalar: IntegerAttr translates correctly.
            gb.create<LLVM::GlobalOp>(module.getLoc(), i32Type, /*isConst=*/false,
                LLVM::Linkage::Internal, "__pic_rule_count",
                gb.getI32IntegerAttr(0));
        }
        auto genRegisterRule = [&]() {
            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(builder.getContext()), {i32Type, i32Type, i32Type});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "register_rule", fType);
            Block *entry = f.addEntryBlock();
            OpBuilder eb(entry, entry->end());
            Location loc = module.getLoc();
            Value keyA = entry->getArgument(0);
            Value keyB = entry->getArgument(1);
            Value impl = entry->getArgument(2);
            // cnt = __pic_rule_count; if cnt >= 512 return;
            Value cntPtr = eb.create<LLVM::AddressOfOp>(loc, ptrType, "__pic_rule_count");
            Value cnt = eb.create<LLVM::LoadOp>(loc, i32Type, cntPtr);
            Value maxN = eb.create<LLVM::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(kRuleTableSize));
            Value full = eb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, cnt, maxN);
            Block *overflow = f.addBlock();
            Block *store  = f.addBlock();
            eb.create<LLVM::CondBrOp>(loc, full, overflow, store);
            // overflow block — just return
            OpBuilder ob(overflow, overflow->end());
            ob.create<LLVM::ReturnOp>(loc, ValueRange{});
            // store block — write entry and bump count
            OpBuilder sb(store, store->end());
            Value tblPtr = sb.create<LLVM::AddressOfOp>(loc, ptrType, "__pic_rule_table");
            Value three = sb.create<LLVM::ConstantOp>(loc, i32Type, sb.getI32IntegerAttr(3));
            Value base  = sb.create<LLVM::MulOp>(loc, i32Type, cnt, three);
            Value base64 = safeZExt(sb, loc, i64Type, base);
            Value off1   = sb.create<LLVM::AddOp>(loc, i64Type, base64, sb.create<LLVM::ConstantOp>(loc, i64Type, sb.getI64IntegerAttr(1)));
            Value off2   = sb.create<LLVM::AddOp>(loc, i64Type, base64, sb.create<LLVM::ConstantOp>(loc, i64Type, sb.getI64IntegerAttr(2)));
            sb.create<LLVM::StoreOp>(loc, keyA, sb.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr, ValueRange{base64}));
            sb.create<LLVM::StoreOp>(loc, keyB, sb.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr, ValueRange{off1}));
            sb.create<LLVM::StoreOp>(loc, impl, sb.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr, ValueRange{off2}));
            // ++__pic_rule_count
            Value newCnt = sb.create<LLVM::AddOp>(loc, i32Type, cnt, sb.create<LLVM::ConstantOp>(loc, i32Type, sb.getI32IntegerAttr(1)));
            sb.create<LLVM::StoreOp>(loc, newCnt, cntPtr);
            sb.create<LLVM::ReturnOp>(loc, ValueRange{});
        };
        genRegisterRule();
        
        auto genLookupRule = [&]() {
            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(i32Type, {i32Type, i32Type});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "lookup_rule", fType);
            Block *entry = f.addEntryBlock();
            Value opArg = entry->getArgument(0);
            Value typeArg = entry->getArgument(1);
            
            OpBuilder eb(entry, entry->end());
            Location loc = module.getLoc();
            
            Block *nextBlock = entry;
            
            for (auto &op : userOps) {
                std::string opName = "";
                std::string typeName = "";
                std::string originalBase = "";
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
                        originalBase = base;
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
                
                auto addRuleMatch = [&](uint32_t keyOp, uint32_t keyType, uint32_t valImpl) {
                    Block *currBlock = nextBlock;
                    Block *matchBlock = f.addBlock();
                    nextBlock = f.addBlock();
                    
                    OpBuilder cb(currBlock, currBlock->end());
                    Value cOp = cb.create<LLVM::ConstantOp>(loc, i32Type, cb.getI32IntegerAttr(keyOp));
                    Value cType = cb.create<LLVM::ConstantOp>(loc, i32Type, cb.getI32IntegerAttr(keyType));
                    Value opMatch = cb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, opArg, cOp);
                    Value typeMatch = cb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, typeArg, cType);
                    Value cond = cb.create<LLVM::AndOp>(loc, opMatch, typeMatch);
                    cb.create<LLVM::CondBrOp>(loc, cond, matchBlock, nextBlock);
                    
                    OpBuilder mb(matchBlock, matchBlock->end());
                    mb.create<LLVM::ReturnOp>(loc, ValueRange{mb.create<LLVM::ConstantOp>(loc, i32Type, mb.getI32IntegerAttr(valImpl))});
                };
                
                if (!opName.empty() && !typeName.empty()) {
                    addRuleMatch(opcodeForLabel(opName), opcodeForLabel(typeName), opcodeForLabel(op.label));
                    addRuleMatch(opcodeForLabel(op.label), opcodeForLabel(typeName), opcodeForLabel(op.label));
                    if (!originalBase.empty() && originalBase[0] == 'f') {
                        addRuleMatch(opcodeForLabel(originalBase), opcodeForLabel(typeName), opcodeForLabel(op.label));
                    }
                    if (typeName == "i64") {
                        addRuleMatch(opcodeForLabel(op.label), opcodeForLabel("i32"), opcodeForLabel(op.label));
                    }
                }
                addRuleMatch(opcodeForLabel(op.label), opcodeForLabel("call"), opcodeForLabel(op.label));
            }
            
            // A2b: after the compile-time chain, scan the runtime-registered rule table.
            // This is a linear scan over __pic_rule_table[0..cnt-1] checking (keyA==opArg && keyB==typeArg).
            Block *rtScanHead = nextBlock;  // fall-through from compile-time chain
            Block *rtScanBody = f.addBlock(); // loop body (i < cnt check + compare)
            Block *rtFound   = f.addBlock(); // match found
            Block *rtMiss    = f.addBlock(); // no match

            // rtScanHead: initialise i=0, jump into body
            {
                OpBuilder hb(rtScanHead, rtScanHead->end());
                // add loop counter as block arg to rtScanBody
                rtScanBody->addArgument(i32Type, loc);
                Value zero32rt = hb.create<LLVM::ConstantOp>(loc, i32Type, hb.getI32IntegerAttr(0));
                hb.create<LLVM::BrOp>(loc, ValueRange{zero32rt}, rtScanBody);
            }
            // rtScanBody: if i >= cnt → miss; else compare; if match → found; else i+1 → body
            {
                Value iVal = rtScanBody->getArgument(0);
                OpBuilder bb(rtScanBody, rtScanBody->end());
                Value cntPtr2 = bb.create<LLVM::AddressOfOp>(loc, ptrType, "__pic_rule_count");
                Value cntVal  = bb.create<LLVM::LoadOp>(loc, i32Type, cntPtr2);
                Value done    = bb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, iVal, cntVal);
                Block *cmpBlk = f.addBlock();
                bb.create<LLVM::CondBrOp>(loc, done, rtMiss, cmpBlk);
                // cmpBlk: load entry[i] and compare
                OpBuilder cb2(cmpBlk, cmpBlk->end());
                Value tblPtr2 = cb2.create<LLVM::AddressOfOp>(loc, ptrType, "__pic_rule_table");
                Value three2  = cb2.create<LLVM::ConstantOp>(loc, i32Type, cb2.getI32IntegerAttr(3));
                Value base2   = cb2.create<LLVM::MulOp>(loc, i32Type, iVal, three2);
                Value base64_2 = safeZExt(cb2, loc, i64Type, base2);
                Value off1_2   = cb2.create<LLVM::AddOp>(loc, i64Type, base64_2, cb2.create<LLVM::ConstantOp>(loc, i64Type, cb2.getI64IntegerAttr(1)));
                Value off2_2   = cb2.create<LLVM::AddOp>(loc, i64Type, base64_2, cb2.create<LLVM::ConstantOp>(loc, i64Type, cb2.getI64IntegerAttr(2)));
                Value entKeyA = cb2.create<LLVM::LoadOp>(loc, i32Type, cb2.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr2, ValueRange{base64_2}));
                Value entKeyB = cb2.create<LLVM::LoadOp>(loc, i32Type, cb2.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr2, ValueRange{off1_2}));
                Value entImpl = cb2.create<LLVM::LoadOp>(loc, i32Type, cb2.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr2, ValueRange{off2_2}));
                Value matchA  = cb2.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, entKeyA, opArg);
                Value matchB  = cb2.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, entKeyB, typeArg);
                Value matched = cb2.create<LLVM::AndOp>(loc, matchA, matchB);
                // pass impl along as block arg to rtFound
                rtFound->addArgument(i32Type, loc);
                Block *nextIter = f.addBlock();
                cb2.create<LLVM::CondBrOp>(loc, matched, rtFound, ValueRange{entImpl}, nextIter, ValueRange{});
                // nextIter: i + 1 → body
                OpBuilder nb(nextIter, nextIter->end());
                Value iPlusOne = nb.create<LLVM::AddOp>(loc, i32Type, iVal, nb.create<LLVM::ConstantOp>(loc, i32Type, nb.getI32IntegerAttr(1)));
                nb.create<LLVM::BrOp>(loc, ValueRange{iPlusOne}, rtScanBody);
            }
            // rtFound: return the impl from block arg
            {
                OpBuilder fb(rtFound, rtFound->end());
                fb.create<LLVM::ReturnOp>(loc, ValueRange{rtFound->getArgument(0)});
            }
            // rtMiss: return 0
            {
                OpBuilder fb(rtMiss, rtMiss->end());
                Value zero = fb.create<LLVM::ConstantOp>(loc, i32Type, fb.getI32IntegerAttr(0));
                fb.create<LLVM::ReturnOp>(loc, ValueRange{zero});
            }
        };
        genLookupRule();
        
        auto genLinPrintI32 = [&]() {
            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(module.getContext()), {i64Type});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "lin_print_i32", fType);
            Block *entry = f.addEntryBlock();
            OpBuilder ob(entry, entry->end());
            
            Value val = entry->getArgument(0);
            Value fmtVal = ob.create<LLVM::ConstantOp>(module.getLoc(), builder.getI64Type(), ob.getI64IntegerAttr(174353445LL));
            Value eight = ob.create<LLVM::ConstantOp>(module.getLoc(), builder.getI32Type(), ob.getI32IntegerAttr(8));
            Value fmtBuf = ob.create<LLVM::AllocaOp>(module.getLoc(), ptrType, builder.getI32Type(), eight);
            ob.create<LLVM::StoreOp>(module.getLoc(), fmtVal, fmtBuf);
            
            ob.create<LLVM::CallOp>(module.getLoc(), builder.getI32Type(), "printf", ValueRange{fmtBuf, val});
            ob.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{});
        };
        genLinPrintI32();

        auto genLinPrintF32 = [&]() {
            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(module.getContext()), {builder.getF32Type()});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "lin_print_f32", fType);
            Block *entry = f.addEntryBlock();
            OpBuilder ob(entry, entry->end());
            
            Value val = entry->getArgument(0);
            Value valExt = ob.create<LLVM::FPExtOp>(module.getLoc(), builder.getF64Type(), val);
            Value fmtVal = ob.create<LLVM::ConstantOp>(module.getLoc(), builder.getI64Type(), ob.getI64IntegerAttr(681509LL));
            Value eight = ob.create<LLVM::ConstantOp>(module.getLoc(), builder.getI32Type(), ob.getI32IntegerAttr(8));
            Value fmtBuf = ob.create<LLVM::AllocaOp>(module.getLoc(), ptrType, builder.getI32Type(), eight);
            ob.create<LLVM::StoreOp>(module.getLoc(), fmtVal, fmtBuf);
            
            ob.create<LLVM::CallOp>(module.getLoc(), builder.getI32Type(), "printf", ValueRange{fmtBuf, valExt});
            ob.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{});
        };
        genLinPrintF32();

        auto genLinPrintF64 = [&]() {
            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(module.getContext()), {builder.getF64Type()});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "lin_print_f64", fType);
            Block *entry = f.addEntryBlock();
            OpBuilder ob(entry, entry->end());
            
            Value val = entry->getArgument(0);
            Value fmtVal = ob.create<LLVM::ConstantOp>(module.getLoc(), builder.getI64Type(), ob.getI64IntegerAttr(681509LL));
            Value eight = ob.create<LLVM::ConstantOp>(module.getLoc(), builder.getI32Type(), ob.getI32IntegerAttr(8));
            Value fmtBuf = ob.create<LLVM::AllocaOp>(module.getLoc(), ptrType, builder.getI32Type(), eight);
            ob.create<LLVM::StoreOp>(module.getLoc(), fmtVal, fmtBuf);
            
            ob.create<LLVM::CallOp>(module.getLoc(), builder.getI32Type(), "printf", ValueRange{fmtBuf, val});
            ob.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{});
        };
        genLinPrintF64();

        auto genLinPrintStr = [&]() {
            if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>("lin_print_str")) {
                existing.erase();
            }

            SmallVector<LLVM::GlobalOp, 16> strGlobals;
            module.walk([&](LLVM::GlobalOp gop) {
                StringRef name = gop.getName();
                if (name.starts_with("STR_")) {
                    strGlobals.push_back(gop);
                }
            });

            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(i64Type, {i64Type});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "lin_print_str", fType);
            f.setPrivate();

            Block *entry = f.addEntryBlock();
            OpBuilder ob(entry, entry->end());
            Value idx = entry->getArgument(0);
            Value zeroI64 = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(0));

            Value fmtVal = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(29477));
            Value four = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(4));
            Value fmtBuf = ob.create<LLVM::AllocaOp>(module.getLoc(), ptrType, i32Type, four);
            ob.create<LLVM::StoreOp>(module.getLoc(), fmtVal, fmtBuf);

            if (strGlobals.empty()) {
                ob.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{zeroI64});
                return;
            }

            Block *fallback = f.addBlock();
            OpBuilder fb(fallback, fallback->end());
            fb.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{zeroI64});

            Block *curBlock = entry;
            for (int i = 0; i < (int)strGlobals.size(); i++) {
                Value idxVal = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(i));
                Value cmp = ob.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, idx, idxVal);

                Block *matchBlock = f.addBlock();
                Block *nextBlock = (i == (int)strGlobals.size() - 1) ? fallback : f.addBlock();

                ob.create<LLVM::CondBrOp>(module.getLoc(), cmp, matchBlock, ValueRange{}, nextBlock, ValueRange{});

                OpBuilder mb(matchBlock, matchBlock->end());
                Value strPtr = mb.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, strGlobals[i].getName());
                mb.create<LLVM::CallOp>(module.getLoc(), i32Type, "printf", ValueRange{fmtBuf, strPtr});
                mb.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{zeroI64});

                ob.setInsertionPointToStart(nextBlock);
            }
        };
        genLinPrintStr();
    }
    /*
    auto wType = LLVM::LLVMFunctionType::get(ptrType, {ptrType});
    auto wFunc = builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "worker_thread", wType);
    Block *wEntry = wFunc.addEntryBlock(); builder.setInsertionPointToStart(wEntry);
    Value wState = wEntry->getArgument(0);
    auto getArg = [&](Value as, int i) {
        Value o = builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(i));
        return builder.create<LLVM::LoadOp>(module.getLoc(), ptrType, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, ptrType, as, ValueRange{o}));
    };
    Value wNet = getArg(wState, 0); Value wQueue = getArg(wState, 1); Value wHead = getArg(wState, 2); Value wTail = getArg(wState, 3); Value al = getArg(wState, 4); Value wHistoryNet = getArg(wState, 5);
    */

    auto genGpuDispatchHelper = [&]() {
        OpBuilder b(module.getBodyRegion());
        auto fType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(builder.getContext()), {i32Type, i32Type, i64Type});
        auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "pic_gpu_dispatch_helper", fType);
        Block *entry = f.addEntryBlock();
        Value nodeA = entry->getArgument(0);
        Value nodeB = entry->getArgument(1);
        Value dState = entry->getArgument(2);
        OpBuilder ob(entry, entry->end());

        if (enableGPU) {
            Value two32 = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(2));
            Value pairBuf = ob.create<LLVM::AllocaOp>(module.getLoc(), ptrType, i32Type, two32);
            
            Value zeroIdx = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(0));
            Value ptr0 = ob.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, pairBuf, zeroIdx);
            ob.create<LLVM::StoreOp>(module.getLoc(), nodeA, ptr0);
            
            Value oneIdx = ob.create<LLVM::ConstantOp>(module.getLoc(), i64Type, ob.getI64IntegerAttr(1));
            Value ptr1 = ob.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, pairBuf, oneIdx);
            ob.create<LLVM::StoreOp>(module.getLoc(), nodeB, ptr1);

            Value oZero = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(0));
            Value dStatePtr = ob.create<LLVM::IntToPtrOp>(module.getLoc(), ptrType, dState);
            Value gepNet = ob.create<LLVM::GEPOp>(module.getLoc(), ptrType, ptrType, dStatePtr, ValueRange{oZero});
            Value dNet = ob.create<LLVM::LoadOp>(module.getLoc(), ptrType, gepNet);

            auto gpuDispatchFty = LLVM::LLVMFunctionType::get(
                LLVM::LLVMVoidType::get(ob.getContext()),
                {ptrType, ptrType, i32Type, ptrType, ptrType}
            );
            auto gpuDispatchFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("pic_gpu_dispatch");
            if (!gpuDispatchFunc) {
                OpBuilder moduleBuilder(module.getBodyRegion());
                gpuDispatchFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(module.getLoc(), "pic_gpu_dispatch", gpuDispatchFty);
            }
            
            Value one32 = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(1));
            Value oFour = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(4));
            Value gepAl = ob.create<LLVM::GEPOp>(module.getLoc(), ptrType, ptrType, dStatePtr, ValueRange{oFour});
            Value alPtrVal = ob.create<LLVM::LoadOp>(module.getLoc(), ptrType, gepAl);
            Value pathPtr = ob.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, "GPU_SPIRV_PATH");
            ob.create<LLVM::CallOp>(module.getLoc(), TypeRange{}, "pic_gpu_dispatch", ValueRange{dNet, pairBuf, one32, alPtrVal, pathPtr});
        }

        ob.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{});
    };
    
    if (module.lookupSymbol("pic_gpu_dispatch_helper")) {
        if (auto decl = module.lookupSymbol<func::FuncOp>("pic_gpu_dispatch_helper")) {
            decl.erase();
        }
        genGpuDispatchHelper();
    }

    module.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "lin.coerce") {
            Location loc = op->getLoc();
            Value input = op->getOperand(0);
            Type targetTy = op->getResult(0).getType();
            OpBuilder ob(op);
            Value coerced = input;
            
            if (input.getType() != targetTy) {
                if (input.getType().isa<IntegerType>() && targetTy.isa<IntegerType>()) {
                    unsigned srcW = input.getType().cast<IntegerType>().getWidth();
                    unsigned tgtW = targetTy.cast<IntegerType>().getWidth();
                    if (srcW < tgtW) {
                        coerced = ob.create<LLVM::ZExtOp>(loc, targetTy, input);
                    } else if (srcW > tgtW) {
                        coerced = ob.create<LLVM::TruncOp>(loc, targetTy, input);
                    }
                } else if (input.getType().isa<FloatType>() && targetTy.isa<IntegerType>()) {
                    if (input.getType().isF64() && targetTy.isInteger(64)) {
                        coerced = ob.create<LLVM::BitcastOp>(loc, targetTy, input);
                    } else if (input.getType().isF32() && targetTy.isInteger(32)) {
                        coerced = ob.create<LLVM::BitcastOp>(loc, targetTy, input);
                    }
                } else if (input.getType().isa<IntegerType>() && targetTy.isa<FloatType>()) {
                    if (input.getType().isInteger(64) && targetTy.isF64()) {
                        coerced = ob.create<LLVM::BitcastOp>(loc, targetTy, input);
                    } else if (input.getType().isInteger(32) && targetTy.isF32()) {
                        coerced = ob.create<LLVM::BitcastOp>(loc, targetTy, input);
                    }
                } else if (input.getType().isa<LLVM::LLVMPointerType>() && targetTy.isa<IntegerType>()) {
                    coerced = ob.create<LLVM::PtrToIntOp>(loc, targetTy, input);
                } else if (input.getType().isa<IntegerType>() && targetTy.isa<LLVM::LLVMPointerType>()) {
                    coerced = ob.create<LLVM::IntToPtrOp>(loc, targetTy, input);
                }
            }
            op->getResult(0).replaceAllUsesWith(coerced);
            op->erase();
        }
    });



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
        }
        auto i1Type = builder.getI1Type();
        Value nS = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(128000000));
        Value net = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{nS}).getResult();
        Value q = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{nS}).getResult();
        Value hd = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value tl = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value alL = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value nSHistory = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(64000000));
        Value oneConst = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(1));
        Value history_net = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "calloc", ValueRange{oneConst, nSHistory}).getResult();
        Value zero64 = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(0));
        builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, hd); builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, tl);
        builder.create<LLVM::StoreOp>(entry.getLoc(), builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(0)), alL);
        Value as = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(64))}).getResult();
        auto sA = [&](int i, Value v) { builder.create<LLVM::StoreOp>(entry.getLoc(), v, builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, ptrType, as, ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(i))})); };
        
        Value activePtr = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value numThreadsConst = enableGPU
            ? builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(1))
            : builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(4));
        builder.create<LLVM::StoreOp>(entry.getLoc(), numThreadsConst, activePtr);
        
        Value lockPtr = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, lockPtr);
        
        sA(0, net); sA(1, q); sA(2, hd); sA(3, tl); sA(4, alL); sA(5, history_net); sA(6, activePtr); sA(7, lockPtr);
        builder.create<func::CallOp>(entry.getLoc(), TypeRange{}, entry.getSymName(), ValueRange{builder.create<LLVM::PtrToIntOp>(entry.getLoc(), i64Type, as)});
        
        if (enableGPU) {
            Value one32 = builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(1));
            Value threads = builder.create<LLVM::AllocaOp>(entry.getLoc(), ptrType, i64Type, one32);
            Value threadAttr = builder.create<LLVM::ZeroOp>(entry.getLoc(), ptrType);
            auto wFuncType = builder.getFunctionType({i64Type}, {i64Type});
            auto wFuncConst = builder.create<func::ConstantOp>(entry.getLoc(), wFuncType, FlatSymbolRefAttr::get(builder.getContext(), "worker_thread"));
            Value wAddr = builder.create<UnrealizedConversionCastOp>(entry.getLoc(), TypeRange{ptrType}, ValueRange(static_cast<Value>(wFuncConst.getResult()))).getResult(0);

            Value idx0 = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(0));
            Value tPtr = builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, i64Type, threads, ValueRange{idx0});
            builder.create<LLVM::CallOp>(entry.getLoc(), i32Type, "pthread_create", ValueRange{tPtr, threadAttr, wAddr, as});

            Value retValPtr = builder.create<LLVM::ZeroOp>(entry.getLoc(), ptrType);
            Value tVal = builder.create<LLVM::LoadOp>(entry.getLoc(), i64Type, tPtr);
            builder.create<LLVM::CallOp>(entry.getLoc(), i32Type, "pthread_join", ValueRange{tVal, retValPtr});
        } else {
            Value four = builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(4));
            Value threads = builder.create<LLVM::AllocaOp>(entry.getLoc(), ptrType, i64Type, four);
            Value threadAttr = builder.create<LLVM::ZeroOp>(entry.getLoc(), ptrType);
            auto wFuncType = builder.getFunctionType({i64Type}, {i64Type});
            auto wFuncConst = builder.create<func::ConstantOp>(entry.getLoc(), wFuncType, FlatSymbolRefAttr::get(builder.getContext(), "worker_thread"));
            Value wAddr = builder.create<UnrealizedConversionCastOp>(entry.getLoc(), TypeRange{ptrType}, ValueRange(static_cast<Value>(wFuncConst.getResult()))).getResult(0);
            
            for (int i = 0; i < 4; ++i) {
                Value idx = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(i));
                Value tPtr = builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, i64Type, threads, ValueRange{idx});
                builder.create<LLVM::CallOp>(entry.getLoc(), i32Type, "pthread_create", ValueRange{tPtr, threadAttr, wAddr, as});
            }
            
            Value retValPtr = builder.create<LLVM::ZeroOp>(entry.getLoc(), ptrType);
            for (int i = 0; i < 4; ++i) {
                Value idx = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(i));
                Value tPtr = builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, i64Type, threads, ValueRange{idx});
                Value tVal = builder.create<LLVM::LoadOp>(entry.getLoc(), i64Type, tPtr);
                builder.create<LLVM::CallOp>(entry.getLoc(), i32Type, "pthread_join", ValueRange{tVal, retValPtr});
            }
        }
        if (enableGPU) {
            auto cleanupFty = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(module.getContext()), {});
            auto cleanupFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("pic_gpu_cleanup");
            if (!cleanupFunc) {
                OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToStart(module.getBody());
                builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "pic_gpu_cleanup", cleanupFty);
            }
            builder.create<LLVM::CallOp>(entry.getLoc(), TypeRange{}, "pic_gpu_cleanup", ValueRange{});
        }
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
        if (op.getOperand(0).getType() == op.getResult(0).getType()) {
            op.getResult(0).replaceAllUsesWith(op.getOperand(0));
            castsToErase.push_back(op);
        }
    });
    for (auto* op : castsToErase) op->erase();
  }
};

static Value genInlineDispatch(OpBuilder &builder, Location loc, Value impl, Value val0, Value val1, Value stateArg, func::FuncOp funcOp, const std::vector<UserOp> &userOps, Value c0_i64) {
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();

    Block *mergeBlock = funcOp.addBlock();
    mergeBlock->addArgument(i32Type, loc);

    Block *currentBlock = builder.getInsertionBlock();

    for (const auto &op : userOps) {
        if (op.dispatch == "gpu") continue;

        Block *matchBlock = funcOp.addBlock();
        Block *nextBlock = funcOp.addBlock();

        OpBuilder cb(currentBlock, currentBlock->end());
        Value hashConst = cb.create<arith::ConstantOp>(loc, i32Type, cb.getI32IntegerAttr(op.hash));
        Value cmp = cb.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, impl, hashConst);
        cb.create<cf::CondBranchOp>(loc, cmp, matchBlock, nextBlock);

        OpBuilder mb(matchBlock, matchBlock->end());
        Value val0_64 = (val0.getType() == i64Type) ? val0 : mb.create<arith::ExtUIOp>(loc, i64Type, val0);
        Value val1_64 = (val1.getType() == i64Type) ? val1 : mb.create<arith::ExtUIOp>(loc, i64Type, val1);
        SmallVector<Value> callArgs;
        if (op.numArgs >= 1) callArgs.push_back(val0_64);
        if (op.numArgs >= 2) callArgs.push_back(val1_64);
        if (op.numArgs >= 3) callArgs.push_back(stateArg);
        if (op.numArgs >= 4) callArgs.push_back(c0_i64);
        while (callArgs.size() < (size_t)op.numArgs) {
            callArgs.push_back(c0_i64);
        }
        
        Value res = mb.create<func::CallOp>(loc, i64Type, op.funcName, callArgs).getResult(0);
        Value res32 = mb.create<arith::TruncIOp>(loc, i32Type, res);
        mb.create<cf::BranchOp>(loc, mergeBlock, ValueRange{res32});

        currentBlock = nextBlock;
    }

    // Default fallback block if no user op matches
    OpBuilder fb(currentBlock, currentBlock->end());
    Value c0_i32 = fb.create<arith::ConstantOp>(loc, i32Type, fb.getI32IntegerAttr(0));
    fb.create<cf::BranchOp>(loc, mergeBlock, ValueRange{c0_i32});

    builder.setInsertionPointToStart(mergeBlock);
    return mergeBlock->getArgument(0);
}

struct PicReduceLoweringPass : public PassWrapper<PicReduceLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicReduceLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    auto i8Type = builder.getI8Type();

    SmallVector<pic::graph::RegistryOp> regOps;
    module.walk([&](pic::graph::RegistryOp op) {
        regOps.push_back(op);
    });

    // Add lin_print_str declaration directly as an LLVM function
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("lin_print_str")) {
        OpBuilder b(module.getBodyRegion());
        auto fType = LLVM::LLVMFunctionType::get(i64Type, {i64Type});
        b.create<LLVM::LLVMFuncOp>(module.getLoc(), "lin_print_str", fType);
    }

    // Add lin_write_ppm declaration for PPM output (returns i64 for state threading)
    if (!module.lookupSymbol<LLVM::LLVMFuncOp>("lin_write_ppm")) {
        OpBuilder b(module.getBodyRegion());
        auto fType = LLVM::LLVMFunctionType::get(i64Type, {i64Type, i64Type, i64Type});
        b.create<LLVM::LLVMFuncOp>(module.getLoc(), "lin_write_ppm", fType);
    }

    SmallVector<std::string> existingDecls;
    for (auto func : module.getOps<func::FuncOp>()) {
        std::string name = func.getSymName().str();
        auto fType = func.getFunctionType();
        std::string s = "func.func private @" + name + "(";
        for (unsigned i = 0; i < fType.getNumInputs(); ++i) {
            std::string typeStr;
            llvm::raw_string_ostream typeStream(typeStr);
            fType.getInput(i).print(typeStream);
            typeStream.flush();
            s += typeStr;
            if (i < fType.getNumInputs() - 1) s += ", ";
        }
        s += ") -> ";
        if (fType.getNumResults() == 0) {
            // no results
        } else if (fType.getNumResults() == 1) {
            std::string typeStr;
            llvm::raw_string_ostream typeStream(typeStr);
            fType.getResult(0).print(typeStream);
            typeStream.flush();
            s += typeStr;
        } else {
            s += "(";
            for (unsigned i = 0; i < fType.getNumResults(); ++i) {
                std::string typeStr;
                llvm::raw_string_ostream typeStream(typeStr);
                fType.getResult(i).print(typeStream);
                typeStream.flush();
                s += typeStr;
                if (i < fType.getNumResults() - 1) s += ", ";
            }
            s += ")";
        }
        existingDecls.push_back(s);
    }

    // Pre-populate existingDecls with known runtime function declarations
    // so they can be referenced in mlir-op payload temp modules.
    existingDecls.push_back("llvm.func @lin_print_str(i64) -> i64");
    existingDecls.push_back("llvm.func @lin_write_ppm(i64, i64, i64) -> i64");

    std::vector<UserOp> userOps;

    for (auto op : regOps) {
        StringAttr n = op->getAttrOfType<StringAttr>("op_name");
        StringAttr p = op->getAttrOfType<StringAttr>("payload");
        StringAttr argsAttr = op->getAttrOfType<StringAttr>("arg_names");
        if (n && p) {
            std::string label = n.getValue().str();
            if (label.compare(0, 4, "STR_") != 0) {
                // Generate a specialized func.func wrapper for this mlir-op
                std::string funcName = "user_op_" + label;
                replaceAll(funcName, "-", "_");
                std::string dispatch = "";
                if (auto dispAttr = op->getAttrOfType<StringAttr>("dispatch")) {
                    dispatch = dispAttr.getValue().str();
                }
                
                int numArgs = 0;
                if (argsAttr) {
                    std::string argsStr = argsAttr.getValue().str();
                    for (size_t i = 0; i < argsStr.size(); i++) {
                        if (argsStr[i] == '[') numArgs++;
                    }
                } else {
                    numArgs = 2;
                }

                SmallVector<std::string> argNamesList;
                SmallVector<StringRef, 2> argNames;
                if (argsAttr) {
                    StringRef(argsAttr.getValue()).split(argNames, "][");
                    for (auto name : argNames) {
                        std::string cleanName = name.str();
                        cleanName.erase(std::remove(cleanName.begin(), cleanName.end(), '['), cleanName.end());
                        cleanName.erase(std::remove(cleanName.begin(), cleanName.end(), ']'), cleanName.end());
                        if (cleanName.empty()) continue;
                        if (cleanName[0] != '%') cleanName = "%" + cleanName;
                        argNamesList.push_back(cleanName);
                    }
                }

                SmallVector<std::string> argTypes;
                for (unsigned i = 0; i < argNamesList.size(); ++i) {
                    std::string type = "i64";
                    size_t underscorePos = argNamesList[i].rfind('_');
                    if (underscorePos != std::string::npos && underscorePos > 0) {
                        type = argNamesList[i].substr(underscorePos + 1);
                    }
                    argTypes.push_back(type);
                }

                func::FuncOp f;
                bool isNew = false;
                if (auto existing = module.lookupSymbol<func::FuncOp>(funcName)) {
                    f = existing;
                    f->setAttr("lin.original_label", builder.getStringAttr(label));
                } else {
                    isNew = true;
                    OpBuilder b(module.getBodyRegion());
                    auto fType = builder.getFunctionType(SmallVector<Type>(numArgs, i64Type), i64Type);
                    f = b.create<func::FuncOp>(module.getLoc(), funcName, fType);
                    f->setAttr("lin.original_label", builder.getStringAttr(label));
                }

                if (isNew) {
                    Block *fEntry = f.addEntryBlock();
                    OpBuilder b(module.getContext());
                    b.setInsertionPointToStart(fEntry);

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
                    for (unsigned i = 0; i < argNamesList.size(); ++i) {
                        std::string cleanName = argNamesList[i];
                        size_t underscorePos = cleanName.rfind('_');
                        if (underscorePos != std::string::npos && underscorePos > 0) {
                            cleanName = cleanName.substr(0, underscorePos);
                        }
                        std::string mlirType = argTypes[i];
                        if (mlirType != "i1" && mlirType != "i8" && mlirType != "i16" &&
                            mlirType != "i32" && mlirType != "i64" &&
                            mlirType != "f32" && mlirType != "f64" &&
                            mlirType != "ptr") {
                            mlirType = "i32";
                        }
                        argS += cleanName + " : " + mlirType;
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
                    addDecl("lin_print_str", "llvm.func @lin_print_str(i64) -> i64");
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
                    addDecl("fopen", "llvm.func @fopen(!llvm.ptr, !llvm.ptr) -> !llvm.ptr");
                    addDecl("fclose", "llvm.func @fclose(!llvm.ptr) -> i32");
                    addDecl("fgetc", "llvm.func @fgetc(!llvm.ptr) -> i32");
                    addDecl("fputc", "llvm.func @fputc(i32, !llvm.ptr) -> i32");
                    addDecl("exit", "llvm.func @exit(i32)");
                    addDecl("abort", "llvm.func @abort()");
                    addDecl("time", "llvm.func @time(!llvm.ptr) -> i64");
                    addDecl("sleep", "llvm.func @sleep(i32) -> i32");
addDecl("lin_write_ppm", "llvm.func @lin_write_ppm(i64, i64, i64) -> i64");
addDecl("lin_create_window", "llvm.func @lin_create_window(i32, i32) -> i32");
addDecl("lin_window_should_close", "llvm.func @lin_window_should_close() -> i32");
addDecl("lin_poll_events", "llvm.func @lin_poll_events() -> i32");
addDecl("lin_display_pixels", "llvm.func @lin_display_pixels(i32, i32, i64) -> i32");
addDecl("lin_destroy_window", "llvm.func @lin_destroy_window() -> i32");
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
                        std::string cleanName = argNamesList[i];
                        size_t underscorePos = cleanName.rfind('_');
                        if (underscorePos != std::string::npos && underscorePos > 0) {
                            cleanName = cleanName.substr(0, underscorePos);
                        }
                        std::string mlirType = argTypes[i];
                        if (mlirType != "i1" && mlirType != "i8" && mlirType != "i16" &&
                            mlirType != "i32" && mlirType != "i64" &&
                            mlirType != "f32" && mlirType != "f64" &&
                            mlirType != "ptr") {
                            mlirType = "i32";
                        }
                        argS2 += cleanName + " : " + mlirType;
                        if (i < argNamesList.size() - 1) argS2 += ", ";
                    }

                    std::string tempModuleStr = "module {\n" + snippetDecls + "func.func @temp(" + argS2 + ") -> i64 {\n";
                    tempModuleStr += pStrRenamed + "\n";
                    
                    std::string resVar = "%res";
                    if (pStrRenamed.find("%res ") == std::string::npos && 
                        pStrRenamed.find("%res\n") == std::string::npos &&
                        pStrRenamed.find("%res=") == std::string::npos &&
                        pStrRenamed.find("%res_state") != std::string::npos) {
                        resVar = "%res_state";
                    }
                    
                    std::string actualType = extractType(pStrRenamed, resVar);
                    
                    if (pStrRenamed.find(resVar) == std::string::npos) {
                        tempModuleStr += "  %res_auto = " + pStrRenamed + "\n";
                        tempModuleStr += "  %res_coerced = \"lin.coerce\"(%res_auto) : (" + actualType + ") -> i64\n";
                        tempModuleStr += "  func.return %res_coerced : i64\n";
                    } else {
                        tempModuleStr += "  %res_coerced = \"lin.coerce\"(" + resVar + ") : (" + actualType + ") -> i64\n";
                        tempModuleStr += "  func.return %res_coerced : i64\n";
                    }
                    tempModuleStr += "}\n}\n";
                    
                    MLIRContext *ctx = module.getContext();
                    ctx->allowUnregisteredDialects();
                    auto parsedSnippet = parseSourceString<ModuleOp>(tempModuleStr, ctx);
                    if (parsedSnippet) {
                        for (auto &op : parsedSnippet->getBody()->getOperations()) {
                            if (auto sym = dyn_cast<SymbolOpInterface>(op)) {
                                if (sym.getName() == "temp") continue;
                                if (!module.lookupSymbol(sym.getName())) {
                                    module.push_back(op.clone());
                                }
                            }
                        }
                    } else {
                        llvm::errs() << "PARSE ERROR: Failed to parse MLIR snippet for user_op_" << label << "\n";
                        llvm::errs() << "--- tempModuleStr ---\n" << tempModuleStr << "\n--- end ---\n";
                        abort();
                    }

                    func::FuncOp tempFunc = parsedSnippet ? parsedSnippet->lookupSymbol<func::FuncOp>("temp") : nullptr;
                    if (dispatch == "gpu" && tempFunc) {
                        tempFunc.walk([&](Operation *op) {
                            if (auto* dialect = op->getDialect()) {
                                StringRef dialectName = dialect->getNamespace();
                                // GPU-legal allowlist per PIC spec: only arith, math, func, cf,
                                // memref, scf, vector, index are allowed on GPU.
                                if (dialectName != "arith" && dialectName != "math" &&
                                    dialectName != "func" && dialectName != "cf" &&
                                    dialectName != "memref" && dialectName != "scf" &&
                                    dialectName != "vector" && dialectName != "index") {
                                    op->emitError() << "Operation '" << op->getName() << "' in registry op '" << label << "' is illegal on GPU (not in GPU-legal allowlist)!";
                                    llvm::errs() << "GPU Compile Error: Operation '" << op->getName() << "' in registry op '" << label << "' is illegal on GPU!\n";
                                    abort();
                                }
                            }
                        });
                    }
                    if (tempFunc && !tempFunc.getBody().empty()) {
                        IRMapping mapper;
                        OpBuilder bBody = OpBuilder::atBlockBegin(fEntry);
                        for (unsigned i = 0; i < argNamesList.size(); ++i) {
                            if (i < f.getNumArguments() && i < tempFunc.getNumArguments()) {
                                Value arg = fEntry->getArgument(i);
                                if (argTypes[i] == "f32") {
                                    Value trunc = bBody.create<arith::TruncIOp>(module.getLoc(), builder.getI32Type(), arg);
                                    arg = bBody.create<arith::BitcastOp>(module.getLoc(), builder.getF32Type(), trunc);
                                } else if (argTypes[i] == "f64") {
                                    arg = bBody.create<arith::BitcastOp>(module.getLoc(), builder.getF64Type(), arg);
                                } else if (argTypes[i] == "i1") {
                                    arg = bBody.create<arith::TruncIOp>(module.getLoc(), builder.getI1Type(), arg);
                                } else if (argTypes[i] == "i32") {
                                    arg = bBody.create<arith::TruncIOp>(module.getLoc(), builder.getI32Type(), arg);
                                } else if (argTypes[i] == "i64") {
                                    // no coercion needed
                                } else {
                                    arg = bBody.create<arith::TruncIOp>(module.getLoc(), builder.getI32Type(), arg);
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
                                            coerced = bBody.create<arith::ExtUIOp>(module.getLoc(), i64Type, src);
                                        } else if (src.getType().getIntOrFloatBitWidth() > 64) {
                                            coerced = bBody.create<arith::TruncIOp>(module.getLoc(), i64Type, src);
                                        }
                                    } else if (src.getType().isa<FloatType>()) {
                                        if (src.getType().isF64()) {
                                            coerced = bBody.create<arith::BitcastOp>(module.getLoc(), i64Type, src);
                                        } else if (src.getType().isF32()) {
                                            Value ext = bBody.create<arith::ExtFOp>(module.getLoc(), builder.getF64Type(), src);
                                            coerced = bBody.create<arith::BitcastOp>(module.getLoc(), i64Type, ext);
                                        } else {
                                            Value ext = bBody.create<arith::ExtFOp>(module.getLoc(), builder.getF64Type(), src);
                                            coerced = bBody.create<arith::BitcastOp>(module.getLoc(), i64Type, ext);
                                        }
                                    } else if (src.getType().isa<IndexType>()) {
                                        coerced = bBody.create<arith::IndexCastOp>(module.getLoc(), i64Type, src);
                                    } else {
                                        // Clone the unregistered lin.coerce op as-is using target-independent values
                                        bBody.clone(op, mapper);
                                        coerced = mapper.lookup(op.getResult(0));
                                    }
                                }
                                mapper.map(op.getResult(0), coerced);
                                finalRetVal = coerced;
                                continue;
                            }
                            if (auto callOp = dyn_cast<func::CallOp>(op)) {
                                bBody.clone(op, mapper);
                                continue;
                            }
                            Operation *cloned = bBody.clone(op, mapper);
                            if (op.getNumResults() > 0) {
                                finalRetVal = mapper.lookupOrDefault(op.getResult(0));
                            }
                        }
                        
                        if (finalRetVal) {
                            if (finalRetVal.getType() != i64Type) {
                                if (finalRetVal.getType().isa<IntegerType>()) {
                                    finalRetVal = bBody.create<arith::ExtUIOp>(module.getLoc(), i64Type, finalRetVal);
                                } else if (finalRetVal.getType().isa<FloatType>()) {
                                    if (finalRetVal.getType().isF64()) {
                                        finalRetVal = bBody.create<arith::BitcastOp>(module.getLoc(), i64Type, finalRetVal);
                                    } else {
                                        Value ext = bBody.create<arith::ExtFOp>(module.getLoc(), builder.getF64Type(), finalRetVal);
                                        finalRetVal = bBody.create<arith::BitcastOp>(module.getLoc(), i64Type, ext);
                                    }
                                } else if (finalRetVal.getType().isa<IndexType>()) {
                                    finalRetVal = bBody.create<arith::IndexCastOp>(module.getLoc(), i64Type, finalRetVal);
                                } else {
                                    // Pointer or other type: create an unregistered lin.coerce operation
                                    OperationState coerceState(module.getLoc(), "lin.coerce");
                                    coerceState.addOperands(finalRetVal);
                                    coerceState.addTypes(i64Type);
                                    finalRetVal = bBody.create(coerceState)->getResult(0);
                                }
                            }
                            bBody.create<func::ReturnOp>(module.getLoc(), finalRetVal);
                        } else {
                            Value zeroRet = bBody.create<arith::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0));
                            bBody.create<func::ReturnOp>(module.getLoc(), zeroRet);
                        }
                    }
                    
                    if (fEntry->empty() || !fEntry->back().hasTrait<OpTrait::IsTerminator>()) {
                        OpBuilder bTerm = OpBuilder::atBlockEnd(fEntry);
                        Value zeroRet = bTerm.create<arith::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0));
                        bTerm.create<func::ReturnOp>(module.getLoc(), zeroRet);
                    }
                }
                userOps.push_back({opcodeForLabel(label), label, funcName, (int)numArgs, argTypes, dispatch});
            }
        }
    }

    for (auto op : regOps) {
        StringAttr n = op->getAttrOfType<StringAttr>("op_name");
        if (n) {
            std::string label = n.getValue().str();
            if (label.compare(0, 4, "STR_") != 0) {
                op.erase();
            }
        }
    }


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

    auto isLiteralLabel = [&](OpBuilder &b, Location loc, Value label) -> Value {
        Value isLit = b.create<arith::ConstantOp>(loc, i1Type, b.getBoolAttr(false));
        for (uint32_t hash : literalHashes) {
            Value hashConst = b.create<arith::ConstantOp>(loc, i32Type, b.getI32IntegerAttr(hash));
            Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, label, hashConst);
            isLit = b.create<arith::OrIOp>(loc, isLit, cmp);
        }
        return isLit;
    };

    SmallVector<Operation *> opsToLower;
    module.walk([&](Operation *op) {
      if (isa<mlir::pic::reduce::ReverseVectorOp,
              mlir::pic::reduce::EraseOp,
              mlir::pic::reduce::FireOpOp,
              mlir::pic::reduce::AnnihilateOp,
              mlir::pic::reduce::DuplicateOp>(op)) {
        opsToLower.push_back(op);
      }
    });

    for (auto *op : opsToLower) {
      Location loc = op->getLoc();
      builder.setInsertionPoint(op);
      
      auto funcOp = op->getParentOfType<func::FuncOp>();
      Block *block = op->getBlock();
      auto branchOp = dyn_cast<cf::BranchOp>(block->getTerminator());
      if (!branchOp) continue;
      Block *lHead = branchOp.getDest();

      // Constants
      Value c0_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0));
      Value c1_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(1));
      Value c32_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(32));
      auto loadLiteralVal = [&](Value node, Value label) -> Value {
          Value p1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, node, builder.getI8IntegerAttr(1));
          Value p2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, node, builder.getI8IntegerAttr(2));
          
          Value isF64 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, label, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("f64"))));
          Value isI64 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, label, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("i64"))));
          Value isNum = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, label, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("num"))));
          Value is64 = builder.create<arith::OrIOp>(loc, isF64, builder.create<arith::OrIOp>(loc, isI64, isNum));
          
          Value p1_64 = builder.create<arith::ExtUIOp>(loc, i64Type, p1);
          Value p2_64 = builder.create<arith::ExtUIOp>(loc, i64Type, p2);
          Value sh2 = builder.create<arith::ShLIOp>(loc, i64Type, p2_64, c32_i64);
          Value combined = builder.create<arith::OrIOp>(loc, i64Type, p1_64, sh2);
          
          return builder.create<arith::SelectOp>(loc, is64, combined, p1_64);
      };
      Value c2_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2));
      Value c3_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(3));
      Value c7_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(7));
      Value c8_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(8));
      Value c24_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(24));
      Value c30_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(30));
      Value c0xFFFFFF_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0xFFFFFF));
      Value c0x3F_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x3F));
      Value c0x979115_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x979115));

      Value c0_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0));
      Value c1_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1));
      Value c2_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2));
      Value c0xFFFFFFFF_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0xFFFFFFFF));
      Value c0x100000000_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0x100000000ULL));
      Value c0xFFFFFFFF00000000_i64 = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0xFFFFFFFF00000000ULL));

      auto makePortVal = [&](Value idx, int p) {
          Value sh = builder.create<arith::ShLIOp>(loc, i32Type, idx, c2_i32);
          return builder.create<arith::OrIOp>(loc, i32Type, sh, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(p)));
      };

      if (auto rvecOp = dyn_cast<mlir::pic::reduce::ReverseVectorOp>(op)) {
          Value nodeA = rvecOp.getNodeA();
          Value nodeB = rvecOp.getNodeB();
          Value stateArg = rvecOp.getState();

          Value metaA = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(3));
          Value metaB = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(3));
          
          Value typeValA = builder.create<arith::ShRUIOp>(loc, metaA, c24_i32);
          Value typeValB = builder.create<arith::ShRUIOp>(loc, metaB, c24_i32);
          Value nodeTypeA = builder.create<arith::AndIOp>(loc, typeValA, c0x3F_i32);
          Value nodeTypeB = builder.create<arith::AndIOp>(loc, typeValB, c0x3F_i32);
          
          Value isRvecA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nodeTypeA, c8_i32);
          Value tNode = builder.create<arith::SelectOp>(loc, isRvecA, nodeB, nodeA);
          Value tType = builder.create<arith::SelectOp>(loc, isRvecA, nodeTypeB, nodeTypeA);
          
          Block *rvecLBlock = funcOp.addBlock();
          Block *rvecStdBlock = funcOp.addBlock();
          
          Value isL = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tType, c7_i32);
          builder.create<cf::CondBranchOp>(loc, isL, rvecLBlock, rvecStdBlock);
          
          builder.setInsertionPointToStart(rvecLBlock);
          Value recNodeA = builder.create<pic::runtime::GetPortOp>(loc, i32Type, tNode, builder.getI8IntegerAttr(1));
          Value recNodeB = builder.create<pic::runtime::GetPortOp>(loc, i32Type, tNode, builder.getI8IntegerAttr(2));
          Value wordA12 = builder.create<pic::runtime::GetHistoryOp>(loc, i64Type, recNodeA, builder.getI8IntegerAttr(1));
          Value wordB12 = builder.create<pic::runtime::GetHistoryOp>(loc, i64Type, recNodeB, builder.getI8IntegerAttr(1));
          Value neighborA1 = builder.create<arith::ShRUIOp>(loc, wordA12, c32_i64);
          Value neighborA2 = builder.create<arith::AndIOp>(loc, wordA12, c0xFFFFFFFF_i64);
          Value neighborB1 = builder.create<arith::ShRUIOp>(loc, wordB12, c32_i64);
          Value neighborB2 = builder.create<arith::AndIOp>(loc, wordB12, c0xFFFFFFFF_i64);
          
          Value neighborA1_32 = builder.create<arith::TruncIOp>(loc, i32Type, neighborA1);
          Value neighborA2_32 = builder.create<arith::TruncIOp>(loc, i32Type, neighborA2);
          Value neighborB1_32 = builder.create<arith::TruncIOp>(loc, i32Type, neighborB1);
          Value neighborB2_32 = builder.create<arith::TruncIOp>(loc, i32Type, neighborB2);
          
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeA, 1), neighborA1_32);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeA, 2), neighborA2_32);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeB, 1), neighborB1_32);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeB, 2), neighborB2_32);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeA, 0), makePortVal(recNodeB, 0));
          builder.create<cf::BranchOp>(loc, lHead);
          
          builder.setInsertionPointToStart(rvecStdBlock);
          Value r1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(136), c0_i64, builder.getBoolAttr(false));
          Value r2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(136), c0_i64, builder.getBoolAttr(false));
          Value auxT1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, tNode, builder.getI8IntegerAttr(1));
          Value auxT2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, tNode, builder.getI8IntegerAttr(2));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(r1, 0), auxT1);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(r2, 0), auxT2);
          builder.create<cf::BranchOp>(loc, lHead);
      }
      else if (auto eraOp = dyn_cast<mlir::pic::reduce::EraseOp>(op)) {
          Value nodeA = eraOp.getNodeA();
          Value nodeB = eraOp.getNodeB();
          Value stateArg = eraOp.getState();

          Value metaA = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(3));
          Value metaB = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(3));
          Value labelA = builder.create<arith::AndIOp>(loc, metaA, c0xFFFFFF_i32);
          Value labelB = builder.create<arith::AndIOp>(loc, metaB, c0xFFFFFF_i32);
          Value polA = builder.create<arith::ShRUIOp>(loc, metaA, c30_i32);
          Value polB = builder.create<arith::ShRUIOp>(loc, metaB, c30_i32);

          Value isEraA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, c0x979115_i32);
          Value otherNode = builder.create<arith::SelectOp>(loc, isEraA, nodeB, nodeA);
          Value otherLabel = builder.create<arith::SelectOp>(loc, isEraA, labelB, labelA);
          Value otherPol = builder.create<arith::SelectOp>(loc, isEraA, polB, polA);
          Value otherIsEra = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, otherLabel, c0x979115_i32);
          Value isVal = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, otherPol, c2_i32);
          Value isAnn = builder.create<arith::OrIOp>(loc, otherIsEra, isVal);

          Block *doEraAnn = funcOp.addBlock();
          Block *doEraProp = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isAnn, doEraAnn, doEraProp);

          builder.setInsertionPointToStart(doEraAnn);
          builder.create<cf::BranchOp>(loc, lHead);

          builder.setInsertionPointToStart(doEraProp);
          Value otherIsLit = isLiteralLabel(builder, loc, otherLabel);
          Block *eraLitCase = funcOp.addBlock();
          Block *eraNormalProp = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, otherIsLit, eraLitCase, eraNormalProp);

          builder.setInsertionPointToStart(eraLitCase);
          builder.create<cf::BranchOp>(loc, lHead);

          builder.setInsertionPointToStart(eraNormalProp);
          Value era1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(132), c0_i64, builder.getBoolAttr(false));
          Value era2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(132), c0_i64, builder.getBoolAttr(false));
          Value auxOther1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, otherNode, builder.getI8IntegerAttr(1));
          Value auxOther2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, otherNode, builder.getI8IntegerAttr(2));
          builder.create<pic::runtime::LinkOp>(loc, auxOther1, makePortVal(era1, 0));
          builder.create<pic::runtime::LinkOp>(loc, auxOther2, makePortVal(era2, 0));
          builder.create<cf::BranchOp>(loc, lHead);
      }
      else if (auto fireOp = dyn_cast<mlir::pic::reduce::FireOpOp>(op)) {
          Value nodeA = fireOp.getNodeA();
          Value nodeB = fireOp.getNodeB();
          Value stateArg = fireOp.getState();

          Value metaA = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(3));
          Value metaB = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(3));
          Value labelA = builder.create<arith::AndIOp>(loc, metaA, c0xFFFFFF_i32);
          Value labelB = builder.create<arith::AndIOp>(loc, metaB, c0xFFFFFF_i32);
          Value polA = builder.create<arith::ShRUIOp>(loc, metaA, c30_i32);
          Value polB = builder.create<arith::ShRUIOp>(loc, metaB, c30_i32);

          if (!module.lookupSymbol("lookup_rule")) {
              OpBuilder mb(module.getBodyRegion());
              auto lookupTy = builder.getFunctionType({i32Type, i32Type}, i32Type);
              auto lookupFunc = mb.create<func::FuncOp>(loc, "lookup_rule", lookupTy);
              lookupFunc.setPrivate();
          }

          Value implA = builder.create<func::CallOp>(loc, i32Type, "lookup_rule", ValueRange{labelA, labelB}).getResult(0);
          Value implB = builder.create<func::CallOp>(loc, i32Type, "lookup_rule", ValueRange{labelB, labelA}).getResult(0);
          Value hasRuleA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, implA, c0_i32);

          Value opNode = builder.create<arith::SelectOp>(loc, hasRuleA, nodeA, nodeB);
          Value valNode = builder.create<arith::SelectOp>(loc, hasRuleA, nodeB, nodeA);
          Value valLabel = builder.create<arith::SelectOp>(loc, hasRuleA, labelB, labelA);
          Value opLabel = builder.create<arith::SelectOp>(loc, hasRuleA, labelA, labelB);
          Value impl = builder.create<arith::SelectOp>(loc, hasRuleA, implA, implB);

          Value v0 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, valNode, builder.getI8IntegerAttr(1));

          Value isUnary = builder.create<arith::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));
          Value isBinary = builder.create<arith::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));
          for (auto &op : userOps) {
              int netArgs = op.numArgs;
              if (netArgs == 3) netArgs = 2; // User-defined functions have 3 C args but 2 net args
              Value hashConst = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(op.hash));
              Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, impl, hashConst);
              if (netArgs == 1) {
                  isUnary = builder.create<arith::OrIOp>(loc, isUnary, cmp);
              } else if (netArgs == 2) {
                  isBinary = builder.create<arith::OrIOp>(loc, isBinary, cmp);
              }
          }

          Value isGpu = builder.create<arith::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));
          for (auto &op : userOps) {
              if (op.dispatch == "gpu") {
                  Value hashConst = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(op.hash));
                  Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, impl, hashConst);
                  isGpu = builder.create<arith::OrIOp>(loc, isGpu, cmp);
              }
          }

          Block *doUnary = funcOp.addBlock();
          Block *checkBinary = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isUnary, doUnary, checkBinary);

          builder.setInsertionPointToStart(doUnary);
          Block *gpuBranch = funcOp.addBlock();
          Block *cpuBranch = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isGpu, gpuBranch, cpuBranch);

          builder.setInsertionPointToStart(gpuBranch);
          if (!module.lookupSymbol("pic_gpu_dispatch_helper")) {
              OpBuilder mb(module.getBodyRegion());
              auto helperTy = builder.getFunctionType({i32Type, i32Type, i64Type}, {});
              auto helperFunc = mb.create<func::FuncOp>(loc, "pic_gpu_dispatch_helper", helperTy);
              helperFunc.setPrivate();
          }
          builder.create<func::CallOp>(loc, TypeRange{}, "pic_gpu_dispatch_helper", ValueRange{valNode, opNode, stateArg});
          builder.create<cf::BranchOp>(loc, lHead);

          builder.setInsertionPointToStart(cpuBranch);
          Value v0_64 = loadLiteralVal(valNode, valLabel);
          Value resVal = genInlineDispatch(builder, loc, impl, v0_64, c0_i32, stateArg, funcOp, userOps, c0_i64);
          Value opP_aux2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, opNode, builder.getI8IntegerAttr(2));
          Value isSame = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, resVal, opP_aux2);
          Block *skipLink = funcOp.addBlock();
          Block *doLink = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isSame, skipLink, doLink);

          builder.setInsertionPointToStart(doLink);
          Value valLabel64 = builder.create<arith::ExtUIOp>(loc, i64Type, valLabel);
          Value resNode = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(133), valLabel64, builder.getBoolAttr(false));
          builder.create<pic::runtime::SetPortOp>(loc, resNode, builder.getI8IntegerAttr(1), resVal);
          builder.create<pic::runtime::LinkOp>(loc, opP_aux2, makePortVal(resNode, 0));
          builder.create<cf::BranchOp>(loc, skipLink);

          builder.setInsertionPointToStart(skipLink);
          builder.create<cf::BranchOp>(loc, lHead);

          builder.setInsertionPointToStart(checkBinary);
          Block *doBinary = funcOp.addBlock();
          Block *doComm = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isBinary, doBinary, doComm);

          builder.setInsertionPointToStart(doBinary);
          Value rPort = builder.create<pic::runtime::GetPortOp>(loc, i32Type, opNode, builder.getI8IntegerAttr(1));
          Value rTarget = builder.create<arith::ShRUIOp>(loc, rPort, c2_i32);
          Value rMeta = builder.create<pic::runtime::GetPortOp>(loc, i32Type, rTarget, builder.getI8IntegerAttr(3));
          Value rLabel = builder.create<arith::AndIOp>(loc, rMeta, c0xFFFFFF_i32);
          Value cCallCode = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("call")));
          Value isCall = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valLabel, cCallCode);
          Value rLabelMatch = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rLabel, valLabel);
          Value isRCall = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rLabel, cCallCode);
          Value rIsLit = isLiteralLabel(builder, loc, rLabel);
          Value isRVal = builder.create<arith::OrIOp>(loc, rLabelMatch, isCall);
          isRVal = builder.create<arith::OrIOp>(loc, isRVal, isRCall);
          isRVal = builder.create<arith::OrIOp>(loc, isRVal, rIsLit);

          Block *doFullBinary = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isRVal, doFullBinary, doComm);

          builder.setInsertionPointToStart(doFullBinary);
          Value v0_64_bin = loadLiteralVal(valNode, valLabel);
          Value v1_64 = loadLiteralVal(rTarget, rLabel);
          Value rTarget_64 = builder.create<arith::ExtUIOp>(loc, i64Type, rTarget);
          Value valNode_aux2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, valNode, builder.getI8IntegerAttr(2));
          Value valNode_aux2_64 = builder.create<arith::ExtUIOp>(loc, i64Type, valNode_aux2);
          
          Value firstArg_64 = builder.create<arith::SelectOp>(loc, isCall, rTarget_64, v1_64);
          Value callArg0_64 = builder.create<arith::SelectOp>(loc, isCall, v0_64_bin, firstArg_64);
          Value callArg1_64 = builder.create<arith::SelectOp>(loc, isCall, valNode_aux2_64, v0_64_bin);

          Block *gpuBranchBin = funcOp.addBlock();
          Block *cpuBranchBin = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isGpu, gpuBranchBin, cpuBranchBin);

          builder.setInsertionPointToStart(gpuBranchBin);
          if (!module.lookupSymbol("pic_gpu_dispatch_helper")) {
              OpBuilder mb(module.getBodyRegion());
              auto helperTy = builder.getFunctionType({i32Type, i32Type, i64Type}, {});
              auto helperFunc = mb.create<func::FuncOp>(loc, "pic_gpu_dispatch_helper", helperTy);
              helperFunc.setPrivate();
          }
          builder.create<func::CallOp>(loc, TypeRange{}, "pic_gpu_dispatch_helper", ValueRange{valNode, opNode, stateArg});
          builder.create<cf::BranchOp>(loc, lHead);

          builder.setInsertionPointToStart(cpuBranchBin);
          Value resValBin = genInlineDispatch(builder, loc, impl, callArg0_64, callArg1_64, stateArg, funcOp, userOps, c0_i64);
          Value callArg1_32 = builder.create<arith::SelectOp>(loc, isCall, valNode_aux2, v0);
          Value isSameBin = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, resValBin, callArg1_32);
          Block *skipLinkBin = funcOp.addBlock();
          Block *doLinkBin = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isSameBin, skipLinkBin, doLinkBin);

          builder.setInsertionPointToStart(doLinkBin);
          Value valLabelBin64 = builder.create<arith::ExtUIOp>(loc, i64Type, valLabel);
          Value resNodeBin = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(133), valLabelBin64, builder.getBoolAttr(false));
          builder.create<pic::runtime::SetPortOp>(loc, resNodeBin, builder.getI8IntegerAttr(1), resValBin);
          Value opNode_aux2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, opNode, builder.getI8IntegerAttr(2));
          builder.create<pic::runtime::LinkOp>(loc, opNode_aux2, makePortVal(resNodeBin, 0));
          builder.create<cf::BranchOp>(loc, skipLinkBin);

          builder.setInsertionPointToStart(skipLinkBin);
          builder.create<cf::BranchOp>(loc, lHead);


          // doComm
          builder.setInsertionPointToStart(doComm);
          Value metaValVal = builder.create<arith::SelectOp>(loc, hasRuleA, metaB, metaA);
          Value metaValOp = builder.create<arith::SelectOp>(loc, hasRuleA, metaA, metaB);
          Value valIsLit = isLiteralLabel(builder, loc, valLabel);
          Block *valLitCase = funcOp.addBlock();
          Block *checkOpLit = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, valIsLit, valLitCase, checkOpLit);

          builder.setInsertionPointToStart(valLitCase);
          Value litValVal = builder.create<pic::runtime::GetPortOp>(loc, i32Type, valNode, builder.getI8IntegerAttr(1));
          Value valLit1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value valLit2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          builder.create<pic::runtime::SetPortOp>(loc, valLit1, builder.getI8IntegerAttr(3), metaValVal);
          builder.create<pic::runtime::SetPortOp>(loc, valLit2, builder.getI8IntegerAttr(3), metaValVal);
          builder.create<pic::runtime::SetPortOp>(loc, valLit1, builder.getI8IntegerAttr(1), litValVal);
          builder.create<pic::runtime::SetPortOp>(loc, valLit2, builder.getI8IntegerAttr(1), litValVal);
          Value auxB1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, opNode, builder.getI8IntegerAttr(1));
          Value auxB2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, opNode, builder.getI8IntegerAttr(2));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(valLit1, 0), auxB1);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(valLit2, 0), auxB2);
          builder.create<cf::BranchOp>(loc, lHead);

          builder.setInsertionPointToStart(checkOpLit);
          Value opIsLit = isLiteralLabel(builder, loc, opLabel);
          Block *opLitCase = funcOp.addBlock();
          Block *stdCommCase = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, opIsLit, opLitCase, stdCommCase);

          builder.setInsertionPointToStart(opLitCase);
          Value litValOp = builder.create<pic::runtime::GetPortOp>(loc, i32Type, opNode, builder.getI8IntegerAttr(1));
          Value opLit1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value opLit2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          builder.create<pic::runtime::SetPortOp>(loc, opLit1, builder.getI8IntegerAttr(3), metaValOp);
          builder.create<pic::runtime::SetPortOp>(loc, opLit2, builder.getI8IntegerAttr(3), metaValOp);
          builder.create<pic::runtime::SetPortOp>(loc, opLit1, builder.getI8IntegerAttr(1), litValOp);
          builder.create<pic::runtime::SetPortOp>(loc, opLit2, builder.getI8IntegerAttr(1), litValOp);
          Value auxA1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, valNode, builder.getI8IntegerAttr(1));
          Value auxA2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, valNode, builder.getI8IntegerAttr(2));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(opLit1, 0), auxA1);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(opLit2, 0), auxA2);
          builder.create<cf::BranchOp>(loc, lHead);

          builder.setInsertionPointToStart(stdCommCase);
          Value auxA1_std = builder.create<pic::runtime::GetPortOp>(loc, i32Type, valNode, builder.getI8IntegerAttr(1));
          Value auxA2_std = builder.create<pic::runtime::GetPortOp>(loc, i32Type, valNode, builder.getI8IntegerAttr(2));
          Value auxB1_std = builder.create<pic::runtime::GetPortOp>(loc, i32Type, opNode, builder.getI8IntegerAttr(1));
          Value auxB2_std = builder.create<pic::runtime::GetPortOp>(loc, i32Type, opNode, builder.getI8IntegerAttr(2));

          Value a1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value a2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value b1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value b2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));

          builder.create<pic::runtime::SetPortOp>(loc, a1, builder.getI8IntegerAttr(3), metaValVal);
          builder.create<pic::runtime::SetPortOp>(loc, a2, builder.getI8IntegerAttr(3), metaValVal);
          builder.create<pic::runtime::SetPortOp>(loc, b1, builder.getI8IntegerAttr(3), metaValOp);
          builder.create<pic::runtime::SetPortOp>(loc, b2, builder.getI8IntegerAttr(3), metaValOp);

          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a1, 1), makePortVal(b1, 1));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a1, 2), makePortVal(b2, 1));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a2, 1), makePortVal(b1, 2));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a2, 2), makePortVal(b2, 2));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a1, 0), auxB1_std);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a2, 0), auxB2_std);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(b1, 0), auxA1_std);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(b2, 0), auxA2_std);
          builder.create<cf::BranchOp>(loc, lHead);
      }
      else if (auto annOp = dyn_cast<mlir::pic::reduce::AnnihilateOp>(op)) {
          Value nodeA = annOp.getNodeA();
          Value nodeB = annOp.getNodeB();
          Value stateArg = annOp.getState();

          Value hWord0A = builder.create<pic::runtime::GetHistoryOp>(loc, i64Type, nodeA, builder.getI8IntegerAttr(0));
          Value inBoundaryA = builder.create<arith::AndIOp>(loc, hWord0A, c0x100000000_i64);
          Value isBoundaryA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, inBoundaryA, c0_i64);

          Block *depLBlock = funcOp.addBlock();
          Block *skipLBlock = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isBoundaryA, depLBlock, skipLBlock);

          builder.setInsertionPointToStart(depLBlock);
          Value lIdx = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(128 + 7), c0_i64, builder.getBoolAttr(false));
          Value lIdx64 = builder.create<arith::ExtUIOp>(loc, i64Type, lIdx);
          
          builder.create<pic::runtime::SetPortOp>(loc, lIdx, builder.getI8IntegerAttr(1), nodeA);
          builder.create<pic::runtime::SetPortOp>(loc, lIdx, builder.getI8IntegerAttr(2), nodeB);
          
          builder.create<pic::runtime::SetPortOp>(loc, lIdx, builder.getI8IntegerAttr(3), builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr((int32_t)2214592512U)));

          Value logPtrA = builder.create<arith::AndIOp>(loc, hWord0A, c0xFFFFFFFF_i64);
          Value valWord0 = builder.create<arith::OrIOp>(loc, c0x100000000_i64, logPtrA);
          builder.create<pic::runtime::SetHistoryOp>(loc, lIdx, builder.getI8IntegerAttr(0), valWord0);

          Value ann_auxA1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(1));
          Value ann_auxA2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(2));
          Value ann_auxB1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(1));
          Value ann_auxB2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(2));

          auto getNeighborPortVal = [&](Value auxPort) {
              Value nNode = builder.create<arith::ShRUIOp>(loc, auxPort, c2_i32);
              Value nPort = builder.create<arith::AndIOp>(loc, auxPort, c3_i32);
              Value nPort8 = builder.create<arith::TruncIOp>(loc, i8Type, nPort);
              return builder.create<pic::runtime::GetPortDynamicOp>(loc, i32Type, nNode, nPort8);
          };

          Value neighborA1_ann = getNeighborPortVal(ann_auxA1);
          Value neighborA2_ann = getNeighborPortVal(ann_auxA2);
          Value neighborB1_ann = getNeighborPortVal(ann_auxB1);
          Value neighborB2_ann = getNeighborPortVal(ann_auxB2);

          Value neighborA1_ann_64 = builder.create<arith::ExtUIOp>(loc, i64Type, neighborA1_ann);
          Value neighborA2_ann_64 = builder.create<arith::ExtUIOp>(loc, i64Type, neighborA2_ann);
          Value neighborB1_ann_64 = builder.create<arith::ExtUIOp>(loc, i64Type, neighborB1_ann);
          Value neighborB2_ann_64 = builder.create<arith::ExtUIOp>(loc, i64Type, neighborB2_ann);

          Value wordA12_ann = builder.create<arith::OrIOp>(loc, builder.create<arith::ShLIOp>(loc, i64Type, neighborA1_ann_64, c32_i64), neighborA2_ann_64);
          Value wordB12_ann = builder.create<arith::OrIOp>(loc, builder.create<arith::ShLIOp>(loc, i64Type, neighborB1_ann_64, c32_i64), neighborB2_ann_64);

          builder.create<pic::runtime::SetHistoryOp>(loc, nodeA, builder.getI8IntegerAttr(1), wordA12_ann);
          builder.create<pic::runtime::SetHistoryOp>(loc, nodeB, builder.getI8IntegerAttr(1), wordB12_ann);
          builder.create<pic::runtime::SetHistoryOp>(loc, lIdx, builder.getI8IntegerAttr(1), c0_i64);

          auto updateNeighborHistory = [&](Value auxPort) {
              Value nNode = builder.create<arith::ShRUIOp>(loc, auxPort, c2_i32);
              Value hWord0N = builder.create<pic::runtime::GetHistoryOp>(loc, i64Type, nNode, builder.getI8IntegerAttr(0));
              Value flagsN = builder.create<arith::AndIOp>(loc, hWord0N, c0xFFFFFFFF00000000_i64);
              Value newWord0N = builder.create<arith::OrIOp>(loc, flagsN, lIdx64);
              builder.create<pic::runtime::SetHistoryOp>(loc, nNode, builder.getI8IntegerAttr(0), newWord0N);
          };

          updateNeighborHistory(ann_auxA1);
          updateNeighborHistory(ann_auxA2);
          updateNeighborHistory(ann_auxB1);
          updateNeighborHistory(ann_auxB2);

          builder.create<cf::BranchOp>(loc, skipLBlock);

          builder.setInsertionPointToStart(skipLBlock);
          Value final_auxA1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(1));
          Value final_auxA2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(2));
          Value final_auxB1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(1));
          Value final_auxB2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(2));

          builder.create<pic::runtime::LinkOp>(loc, final_auxA1, final_auxB1);
          builder.create<pic::runtime::LinkOp>(loc, final_auxA2, final_auxB2);
          builder.create<cf::BranchOp>(loc, lHead);
      }
      else if (auto dupOp = dyn_cast<mlir::pic::reduce::DuplicateOp>(op)) {
          Value nodeA = dupOp.getNodeA();
          Value nodeB = dupOp.getNodeB();
          Value stateArg = dupOp.getState();

          Value metaA = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(3));
          Value metaB = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(3));
          Value labelA = builder.create<arith::AndIOp>(loc, metaA, c0xFFFFFF_i32);
          Value labelB = builder.create<arith::AndIOp>(loc, metaB, c0xFFFFFF_i32);

          Value valIsLit = isLiteralLabel(builder, loc, labelA);
          Block *valLitCase = funcOp.addBlock();
          Block *checkOpLit = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, valIsLit, valLitCase, checkOpLit);

          // valLitCase: nodeA is a literal constant, nodeB is standard
          builder.setInsertionPointToStart(valLitCase);
          Value isF64A = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("f64"))));
          Value isI64A = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("i64"))));
          Value isNumA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("num"))));
          Value is64A = builder.create<arith::OrIOp>(loc, isF64A, builder.create<arith::OrIOp>(loc, isI64A, isNumA));
          
          Value litValA = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(1));
          Value valLit1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value valLit2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          builder.create<pic::runtime::SetPortOp>(loc, valLit1, builder.getI8IntegerAttr(3), metaA);
          builder.create<pic::runtime::SetPortOp>(loc, valLit2, builder.getI8IntegerAttr(3), metaA);
          builder.create<pic::runtime::SetPortOp>(loc, valLit1, builder.getI8IntegerAttr(1), litValA);
          builder.create<pic::runtime::SetPortOp>(loc, valLit2, builder.getI8IntegerAttr(1), litValA);
          
          Value litValAHigh = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(2));
          Value valLit1_selfPort2 = makePortVal(valLit1, 2);
          Value valLit2_selfPort2 = makePortVal(valLit2, 2);
          Value writeVal1 = builder.create<arith::SelectOp>(loc, is64A, litValAHigh, valLit1_selfPort2);
          Value writeVal2 = builder.create<arith::SelectOp>(loc, is64A, litValAHigh, valLit2_selfPort2);
          builder.create<pic::runtime::SetPortOp>(loc, valLit1, builder.getI8IntegerAttr(2), writeVal1);
          builder.create<pic::runtime::SetPortOp>(loc, valLit2, builder.getI8IntegerAttr(2), writeVal2);

          Value auxB1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(1));
          Value auxB2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(2));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(valLit1, 0), auxB1);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(valLit2, 0), auxB2);
          builder.create<cf::BranchOp>(loc, lHead);

          // checkOpLit
          builder.setInsertionPointToStart(checkOpLit);
          Value opIsLit = isLiteralLabel(builder, loc, labelB);
          Block *opLitCase = funcOp.addBlock();
          Block *stdCommCase = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, opIsLit, opLitCase, stdCommCase);

          // opLitCase: nodeB is a literal constant, nodeA is standard
          builder.setInsertionPointToStart(opLitCase);
          Value isF64B = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("f64"))));
          Value isI64B = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("i64"))));
          Value isNumB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("num"))));
          Value is64B = builder.create<arith::OrIOp>(loc, isF64B, builder.create<arith::OrIOp>(loc, isI64B, isNumB));

          Value litValB = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(1));
          Value opLit1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value opLit2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          builder.create<pic::runtime::SetPortOp>(loc, opLit1, builder.getI8IntegerAttr(3), metaB);
          builder.create<pic::runtime::SetPortOp>(loc, opLit2, builder.getI8IntegerAttr(3), metaB);
          builder.create<pic::runtime::SetPortOp>(loc, opLit1, builder.getI8IntegerAttr(1), litValB);
          builder.create<pic::runtime::SetPortOp>(loc, opLit2, builder.getI8IntegerAttr(1), litValB);
          
          Value litValBHigh = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(2));
          Value opLit1_selfPort2 = makePortVal(opLit1, 2);
          Value opLit2_selfPort2 = makePortVal(opLit2, 2);
          Value writeValOp1 = builder.create<arith::SelectOp>(loc, is64B, litValBHigh, opLit1_selfPort2);
          Value writeValOp2 = builder.create<arith::SelectOp>(loc, is64B, litValBHigh, opLit2_selfPort2);
          builder.create<pic::runtime::SetPortOp>(loc, opLit1, builder.getI8IntegerAttr(2), writeValOp1);
          builder.create<pic::runtime::SetPortOp>(loc, opLit2, builder.getI8IntegerAttr(2), writeValOp2);

          Value auxA1 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(1));
          Value auxA2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(2));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(opLit1, 0), auxA1);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(opLit2, 0), auxA2);
          builder.create<cf::BranchOp>(loc, lHead);

          // stdCommCase: standard commutation
          builder.setInsertionPointToStart(stdCommCase);
          Value auxA1_std = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(1));
          Value auxA2_std = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(2));
          Value auxB1_std = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(1));
          Value auxB2_std = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(2));

          Value a1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value a2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value b1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
          Value b2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));

          builder.create<pic::runtime::SetPortOp>(loc, a1, builder.getI8IntegerAttr(3), metaA);
          builder.create<pic::runtime::SetPortOp>(loc, a2, builder.getI8IntegerAttr(3), metaA);
          builder.create<pic::runtime::SetPortOp>(loc, b1, builder.getI8IntegerAttr(3), metaB);
          builder.create<pic::runtime::SetPortOp>(loc, b2, builder.getI8IntegerAttr(3), metaB);

          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a1, 1), makePortVal(b1, 1));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a1, 2), makePortVal(b2, 1));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a2, 1), makePortVal(b1, 2));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a2, 2), makePortVal(b2, 2));
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a1, 0), auxB1_std);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(a2, 0), auxB2_std);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(b1, 0), auxA1_std);
          builder.create<pic::runtime::LinkOp>(loc, makePortVal(b2, 0), auxA2_std);
          builder.create<cf::BranchOp>(loc, lHead);
      }

      op->erase();
      branchOp->erase();
    }
  }
};

struct PicRuntimeToSPIRVPass : public PassWrapper<PicRuntimeToSPIRVPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicRuntimeToSPIRVPass)

  void runOnOperation() override {
#if __has_include("mlir/Dialect/SPIRV/IR/SPIRVOps.h")
    ModuleOp module = getOperation();
    func::FuncOp workerFunc = module.lookupSymbol<func::FuncOp>("worker_thread");
    if (!workerFunc || workerFunc.empty()) return;

    MLIRContext *ctx = module.getContext();
    Location loc = module.getLoc();
    OpBuilder builder(module.getBodyRegion());

    auto i32Type = builder.getI32Type();

    auto spirvModule = builder.create<mlir::spirv::ModuleOp>(loc);
    spirvModule->setAttr("addressing_model",
        mlir::spirv::AddressingModelAttr::get(ctx, mlir::spirv::AddressingModel::Logical));
    spirvModule->setAttr("memory_model",
        mlir::spirv::MemoryModelAttr::get(ctx, mlir::spirv::MemoryModel::GLSL450));
    auto vce = mlir::spirv::VerCapExtAttr::get(
        mlir::spirv::Version::V_1_5,
        ArrayRef<mlir::spirv::Capability>{
            mlir::spirv::Capability::Shader,
            mlir::spirv::Capability::Int64,
            mlir::spirv::Capability::AtomicStorage},
        ArrayRef<mlir::spirv::Extension>{},
        ctx);
    spirvModule->setAttr("vce_triple", vce);

    Block *spirvBody = spirvModule.getBody();
    OpBuilder sb(spirvBody, spirvBody->begin());

    auto netArrType = mlir::spirv::RuntimeArrayType::get(i32Type);
    auto pairsArrType = mlir::spirv::RuntimeArrayType::get(i32Type);

    auto makeStorageBufferPtr = [&](Type elem, int64_t offset) {
      auto structTy = mlir::spirv::StructType::get(
          {elem}, {static_cast<uint32_t>(offset)});
      return mlir::spirv::PointerType::get(structTy,
          mlir::spirv::StorageClass::StorageBuffer);
    };
    auto makePtrToArray = [&](mlir::spirv::RuntimeArrayType arrTy) {
      return makeStorageBufferPtr(arrTy, 0);
    };

    Type netPtrType = makePtrToArray(netArrType);
    Type pairsPtrType = makePtrToArray(pairsArrType);

    sb.create<mlir::spirv::GlobalVariableOp>(
        loc, netPtrType, "net",
        /*descriptorSet=*/0, /*binding=*/0);
    sb.create<mlir::spirv::GlobalVariableOp>(
        loc, pairsPtrType, "pairs",
        /*descriptorSet=*/0, /*binding=*/1);

    // State struct: {numPairs, al}
    SmallVector<mlir::spirv::StructType::OffsetInfo> stateOffsets = {0, 4};
    auto stateStructType = mlir::spirv::StructType::get(
        {i32Type, i32Type}, stateOffsets);
    Type statePtrType = mlir::spirv::PointerType::get(
        stateStructType, mlir::spirv::StorageClass::StorageBuffer);
    sb.create<mlir::spirv::GlobalVariableOp>(
        loc, statePtrType, "state",
        /*descriptorSet=*/0, /*binding=*/2);

    Type i32PtrSSBO = mlir::spirv::PointerType::get(
        i32Type, mlir::spirv::StorageClass::StorageBuffer);

    // GlobalInvocationId built-in variable
    auto vec3I32 = mlir::VectorType::get(3, i32Type);
    Type gidPtrType = mlir::spirv::PointerType::get(
        vec3I32, mlir::spirv::StorageClass::Input);
    auto gidVar = sb.create<mlir::spirv::GlobalVariableOp>(
        loc, gidPtrType, "gl_GlobalInvocationID");
    gidVar->setAttr("built_in",
        mlir::StringAttr::get(ctx, "GlobalInvocationId"));

    auto func = sb.create<mlir::spirv::FuncOp>(loc, "pic_kernel",
        FunctionType::get(ctx, {}, {}));

    SmallVector<Attribute> iface = {
        FlatSymbolRefAttr::get(ctx, "net"),
        FlatSymbolRefAttr::get(ctx, "pairs"),
        FlatSymbolRefAttr::get(ctx, "state"),
        FlatSymbolRefAttr::get(ctx, "gl_GlobalInvocationID"),
    };
    sb.create<mlir::spirv::EntryPointOp>(loc,
        mlir::spirv::ExecutionModel::GLCompute, func, iface);
    SmallVector<int32_t> wgs = {256, 1, 1};
    sb.create<mlir::spirv::ExecutionModeOp>(loc, func,
        mlir::spirv::ExecutionMode::LocalSize, wgs);

    Block *entry = func.addEntryBlock();
    OpBuilder eb(entry, entry->end());

    auto makeSSBOLoad = [&](OpBuilder &b, Value basePtr, Value index) -> Value {
      Type i32Ptr = mlir::spirv::PointerType::get(
          i32Type, mlir::spirv::StorageClass::StorageBuffer);
      auto elemPtr = b.create<mlir::spirv::AccessChainOp>(loc, i32Ptr,
          basePtr, ValueRange{
              b.create<mlir::spirv::ConstantOp>(loc, i32Type, b.getI32IntegerAttr(0)),
              index});
      return b.create<mlir::spirv::LoadOp>(loc, elemPtr);
    };

    auto makeSSBOStore = [&](OpBuilder &b, Value basePtr, Value index, Value val) {
      Type i32Ptr = mlir::spirv::PointerType::get(
          i32Type, mlir::spirv::StorageClass::StorageBuffer);
      auto elemPtr = b.create<mlir::spirv::AccessChainOp>(loc, i32Ptr,
          basePtr, ValueRange{
              b.create<mlir::spirv::ConstantOp>(loc, i32Type, b.getI32IntegerAttr(0)),
              index});
      b.create<mlir::spirv::StoreOp>(loc, elemPtr, val);
    };

    Value netPtr = eb.create<mlir::spirv::AddressOfOp>(loc, netPtrType,
        FlatSymbolRefAttr::get(ctx, "net"));
    Value pairsPtr = eb.create<mlir::spirv::AddressOfOp>(loc, pairsPtrType,
        FlatSymbolRefAttr::get(ctx, "pairs"));
    Value statePtr = eb.create<mlir::spirv::AddressOfOp>(loc, statePtrType,
        FlatSymbolRefAttr::get(ctx, "state"));
    Value gidVarPtr = eb.create<mlir::spirv::AddressOfOp>(loc, gidPtrType,
        FlatSymbolRefAttr::get(ctx, "gl_GlobalInvocationID"));

    Value gidVec = eb.create<mlir::spirv::LoadOp>(loc, gidVarPtr);
    Value id = eb.create<mlir::spirv::CompositeExtractOp>(loc, i32Type, gidVec,
        eb.getI32ArrayAttr({0}));

    Value c0 = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(0));
    Value c1 = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(1));
    Value c2 = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(2));
    Value c3 = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(3));
    Value c4 = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(4));
    Value c24 = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(24));
    Value c30 = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(30));
    Value c0x3F = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(0x3F));

    // Read numPairs from state[0] (this SSBO is shared CPU↔GPU; host writes numPairs)
    Value numPairsPtr = eb.create<mlir::spirv::AccessChainOp>(loc, i32PtrSSBO,
        statePtr, ValueRange{c0});
    Value numPairs = eb.create<mlir::spirv::LoadOp>(loc, numPairsPtr);

    // OOB check: thread id < numPairs
    Value oob = eb.create<mlir::spirv::ULessThanOp>(loc, eb.getI1Type(), numPairs, id);
    Value notOob = eb.create<mlir::spirv::LogicalNotOp>(loc, eb.getI1Type(), oob);

    // Read pair nodes from pairs[id*2], pairs[id*2+1]
    Value idTimes2 = eb.create<mlir::spirv::IMulOp>(loc, i32Type, id, c2);
    Value idTimes2Plus1 = eb.create<mlir::spirv::IAddOp>(loc, i32Type, idTimes2, c1);
    Value nodeA = makeSSBOLoad(eb, pairsPtr, idTimes2);
    Value nodeB = makeSSBOLoad(eb, pairsPtr, idTimes2Plus1);

    // Decode node types from meta word at net[node*4+3]
    Value nodeATimes4 = eb.create<mlir::spirv::IMulOp>(loc, i32Type, nodeA, c4);
    Value nodeBTimes4 = eb.create<mlir::spirv::IMulOp>(loc, i32Type, nodeB, c4);
    Value metaA = makeSSBOLoad(eb, netPtr,
        eb.create<mlir::spirv::IAddOp>(loc, i32Type, nodeATimes4, c3));
    Value metaB = makeSSBOLoad(eb, netPtr,
        eb.create<mlir::spirv::IAddOp>(loc, i32Type, nodeBTimes4, c3));

    Value typeA = eb.create<mlir::spirv::BitwiseAndOp>(loc, i32Type,
        eb.create<mlir::spirv::ShiftRightArithmeticOp>(loc, i32Type, metaA, c24), c0x3F);
    Value typeB = eb.create<mlir::spirv::BitwiseAndOp>(loc, i32Type,
        eb.create<mlir::spirv::ShiftRightArithmeticOp>(loc, i32Type, metaB, c24), c0x3F);

    Value polA = eb.create<mlir::spirv::BitwiseAndOp>(loc, i32Type,
        eb.create<mlir::spirv::ShiftRightArithmeticOp>(loc, i32Type, metaA, c30), c3);
    Value polB = eb.create<mlir::spirv::BitwiseAndOp>(loc, i32Type,
        eb.create<mlir::spirv::ShiftRightArithmeticOp>(loc, i32Type, metaB, c30), c3);

    // Node type constants matching PicrNodeType enum
    Value tCON  = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(1));
    Value tDES  = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(2));
    Value tDUP  = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(3));
    Value tERA  = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(4));
    Value tRVEC = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(8));

    // Type classification for each rule
    Value isConA = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeA, tCON);
    Value isConB = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeB, tCON);
    Value isDesA = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeA, tDES);
    Value isDesB = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeB, tDES);
    Value isAnnPair = eb.create<mlir::spirv::LogicalOrOp>(loc, eb.getI1Type(),
        eb.create<mlir::spirv::LogicalAndOp>(loc, eb.getI1Type(), isConA, isDesB),
        eb.create<mlir::spirv::LogicalAndOp>(loc, eb.getI1Type(), isDesA, isConB));

    Value isDupA   = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeA, tDUP);
    Value isDupB   = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeB, tDUP);
    Value isDupPair = eb.create<mlir::spirv::LogicalOrOp>(loc, eb.getI1Type(), isDupA, isDupB);

    Value isEraA   = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeA, tERA);
    Value isEraB   = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeB, tERA);
    Value isEraPair = eb.create<mlir::spirv::LogicalOrOp>(loc, eb.getI1Type(), isEraA, isEraB);

    Value isRvecA  = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeA, tRVEC);
    Value isRvecB  = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeB, tRVEC);
    Value isRvecPair = eb.create<mlir::spirv::LogicalOrOp>(loc, eb.getI1Type(), isRvecA, isRvecB);

    // Load aux port values (available for all paths)
    Value p1A = makeSSBOLoad(eb, netPtr, eb.create<mlir::spirv::IAddOp>(loc, i32Type, nodeATimes4, c1));
    Value p2A = makeSSBOLoad(eb, netPtr, eb.create<mlir::spirv::IAddOp>(loc, i32Type, nodeATimes4, c2));
    Value p1B = makeSSBOLoad(eb, netPtr, eb.create<mlir::spirv::IAddOp>(loc, i32Type, nodeBTimes4, c1));
    Value p2B = makeSSBOLoad(eb, netPtr, eb.create<mlir::spirv::IAddOp>(loc, i32Type, nodeBTimes4, c2));

    // Compute target net offset for a port value: ((port >> 2) * 4) + (port & 3)
    auto computeTarget = [&](OpBuilder &b, Value port) -> Value {
      Value c2Local = b.create<mlir::spirv::ConstantOp>(loc, i32Type, b.getI32IntegerAttr(2));
      Value targetNode = b.create<mlir::spirv::ShiftRightArithmeticOp>(loc, i32Type, port, c2Local);
      Value targetPort = b.create<mlir::spirv::BitwiseAndOp>(loc, i32Type, port, c3);
      return b.create<mlir::spirv::IAddOp>(loc, i32Type,
          b.create<mlir::spirv::IMulOp>(loc, i32Type, targetNode, c4), targetPort);
    };

    // =====================================================================
    // Multi-rule dispatch: sequential if-then blocks using SPIR-V structured
    // control flow. Rules are mutually exclusive (one pair = one rule type).
    // Priority order: RVEC > ERA > DUP > ANN (catch-all). Unmatched pairs
    // (omega/OP nodes) are silently skipped for CPU dispatch.
    // =====================================================================
    // 1) R-Vector propagation
    mlir::spirv::SelectionOp::createIfThen(loc,
        eb.create<mlir::spirv::LogicalAndOp>(loc, eb.getI1Type(), notOob, isRvecPair),
        [&](OpBuilder &b) {
          Value otherP1 = b.create<mlir::spirv::SelectOp>(loc, i32Type, isRvecA, p1B, p1A);
          Value otherP2 = b.create<mlir::spirv::SelectOp>(loc, i32Type, isRvecA, p2B, p2A);
          Value rvecNode = b.create<mlir::spirv::SelectOp>(loc, i32Type, isRvecA, nodeA, nodeB);
          Value rvecP0 = makeSSBOLoad(b, netPtr,
              b.create<mlir::spirv::IAddOp>(loc, i32Type,
                  b.create<mlir::spirv::IMulOp>(loc, i32Type, rvecNode, c4), c0));
          makeSSBOStore(b, netPtr, computeTarget(b, otherP1), rvecP0);
          makeSSBOStore(b, netPtr, computeTarget(b, otherP2), rvecP0);
        }, eb);

    // 2) Erasure
    mlir::spirv::SelectionOp::createIfThen(loc,
        eb.create<mlir::spirv::LogicalAndOp>(loc, eb.getI1Type(), notOob, isEraPair),
        [&](OpBuilder &b) {
          Value nonEraP1 = b.create<mlir::spirv::SelectOp>(loc, i32Type, isEraA, p1B, p1A);
          Value nonEraP2 = b.create<mlir::spirv::SelectOp>(loc, i32Type, isEraA, p2B, p2A);
          Value eraNode  = b.create<mlir::spirv::SelectOp>(loc, i32Type, isEraA, nodeA, nodeB);
          Value eraP0    = makeSSBOLoad(b, netPtr,
              b.create<mlir::spirv::IAddOp>(loc, i32Type,
                  b.create<mlir::spirv::IMulOp>(loc, i32Type, eraNode, c4), c0));
          makeSSBOStore(b, netPtr, computeTarget(b, nonEraP1), eraP0);
          makeSSBOStore(b, netPtr, computeTarget(b, nonEraP2), eraP0);
        }, eb);

    // 3) Duplication (DUP node with any partner)
    mlir::spirv::SelectionOp::createIfThen(loc,
        eb.create<mlir::spirv::LogicalAndOp>(loc, eb.getI1Type(), notOob, isDupPair),
        [&](OpBuilder &b) {
          Value otherMeta = b.create<mlir::spirv::SelectOp>(loc, i32Type, isDupA, metaB, metaA);
          Value auxA1 = b.create<mlir::spirv::SelectOp>(loc, i32Type, isDupA, p1B, p1A);
          Value auxA2 = b.create<mlir::spirv::SelectOp>(loc, i32Type, isDupA, p2B, p2A);
          Value auxB1 = b.create<mlir::spirv::SelectOp>(loc, i32Type, isDupA, p1A, p1B);
          Value auxB2 = b.create<mlir::spirv::SelectOp>(loc, i32Type, isDupA, p2A, p2B);

          // Atomic alloc: reserve 4 contiguous node indices from state[1]
          Value alPtr = b.create<mlir::spirv::AccessChainOp>(loc, i32PtrSSBO, statePtr, ValueRange{c1});
          Value four = b.create<mlir::spirv::ConstantOp>(loc, i32Type, b.getI32IntegerAttr(4));
          auto scopeA = mlir::spirv::ScopeAttr::get(ctx, mlir::spirv::Scope::Device);
          auto semA   = mlir::spirv::MemorySemanticsAttr::get(ctx, mlir::spirv::MemorySemantics::AcquireRelease);
          Value base = b.create<mlir::spirv::AtomicIAddOp>(loc, i32Type, alPtr, scopeA, semA, four);
          Value n1 = base;
          Value n2 = b.create<mlir::spirv::IAddOp>(loc, i32Type, base, c1);
          Value n3 = b.create<mlir::spirv::IAddOp>(loc, i32Type, base, c2);
          Value n4 = b.create<mlir::spirv::IAddOp>(loc, i32Type, base, c3);

          // Write meta to each new node
          auto writeMeta = [&](OpBuilder &ob, Value nIdx) {
            makeSSBOStore(ob, netPtr,
                ob.create<mlir::spirv::IAddOp>(loc, i32Type,
                    ob.create<mlir::spirv::IMulOp>(loc, i32Type, nIdx, c4), c3), otherMeta);
          };
          writeMeta(b, n1); writeMeta(b, n2); writeMeta(b, n3); writeMeta(b, n4);

          // Self-referencing port helper: (nodeIdx * 4 + port)
          auto makeSelfPort = [&](OpBuilder &ob, Value nIdx, int pNum) {
            return ob.create<mlir::spirv::IAddOp>(loc, i32Type,
                ob.create<mlir::spirv::IMulOp>(loc, i32Type, nIdx, c4),
                ob.create<mlir::spirv::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(pNum)));
          };

          // Standard commutation wiring:
          // a1.p1↔b1.p1, a1.p2↔b2.p1, a2.p1↔b1.p2, a2.p2↔b2.p2
          // a1.p0↔auxB1, a2.p0↔auxB2, b1.p0↔auxA1, b2.p0↔auxA2
          auto w = [&](Value nIdx, int p, Value val) {
            makeSSBOStore(b, netPtr,
                b.create<mlir::spirv::IAddOp>(loc, i32Type,
                    b.create<mlir::spirv::IMulOp>(loc, i32Type, nIdx, c4),
                    b.create<mlir::spirv::ConstantOp>(loc, i32Type, b.getI32IntegerAttr(p))), val);
          };
          w(n1, 1, makeSelfPort(b, n3, 1));
          w(n1, 2, makeSelfPort(b, n4, 1));
          w(n2, 1, makeSelfPort(b, n3, 2));
          w(n2, 2, makeSelfPort(b, n4, 2));
          w(n3, 1, makeSelfPort(b, n1, 1));
          w(n3, 2, makeSelfPort(b, n2, 1));
          w(n4, 1, makeSelfPort(b, n1, 2));
          w(n4, 2, makeSelfPort(b, n2, 2));
          w(n1, 0, auxB1);
          w(n2, 0, auxB2);
          w(n3, 0, auxA1);
          w(n4, 0, auxA2);
        }, eb);

    // 4) Annihilation (CON ~ DES)
    mlir::spirv::SelectionOp::createIfThen(loc,
        eb.create<mlir::spirv::LogicalAndOp>(loc, eb.getI1Type(), notOob, isAnnPair),
        [&](OpBuilder &b) {
          Value tA1 = computeTarget(b, p1A);
          Value tA2 = computeTarget(b, p2A);
          Value tB1 = computeTarget(b, p1B);
          Value tB2 = computeTarget(b, p2B);
          makeSSBOStore(b, netPtr, tA1, p1B);
          makeSSBOStore(b, netPtr, tB1, p1A);
          makeSSBOStore(b, netPtr, tA2, p2B);
          makeSSBOStore(b, netPtr, tB2, p2A);
        }, eb);

    eb.create<mlir::spirv::ReturnOp>(loc);

    // worker_thread is kept alive for the dispatch loop lowered by PicRuntimeToLLVMPass
#else
    llvm::outs() << "Info: SPIR-V dialect headers not available; PicRuntimeToSPIRVPass skipped.\n";
#endif
  }
};
} // namespace

std::unique_ptr<Pass> createPicGraphToReducePass() { return std::make_unique<PicGraphToReducePass>(); }
std::unique_ptr<Pass> createPicReduceToRuntimePass(bool enableGPU, std::string spirvPath) { return std::make_unique<PicReduceToRuntimePass>(enableGPU, spirvPath); }
std::unique_ptr<Pass> createPicReduceLoweringPass() { return std::make_unique<PicReduceLoweringPass>(); }
std::unique_ptr<Pass> createPicRuntimeToLLVMPass(bool enableGPU, std::string spirvPath) { return std::make_unique<PicRuntimeToLLVMPass>(enableGPU, spirvPath); }
std::unique_ptr<Pass> createPicRuntimeToSPIRVPass() { return std::make_unique<PicRuntimeToSPIRVPass>(); }
