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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
        
        if ((label == "num" || label == "i32" || label == "f64" || label == "bool" || label == "str") && (op.getValue() || op.getStrVal())) {
            Value litVal;
            if (label == "str") {
                litVal = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(val));
            } else {
                litVal = builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(op.getValue().value()));
            }
            builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(1), litVal);
        }

        auto makePort = [&](int i) {
            Value nodeIdx64 = (nodeIdx.getType() == i64Type) ? nodeIdx : builder.create<arith::ExtUIOp>(loc, i64Type, nodeIdx);
            Value shifted = builder.create<arith::ShLIOp>(loc, i64Type, nodeIdx64, builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2)));
            return builder.create<arith::OrIOp>(loc, i64Type, shifted, builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(i)));
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
              auto safeExt = [&](Value v) {
                  return (v.getType() == i64Type) ? v : builder.create<arith::ExtUIOp>(op.getLoc(), i64Type, v);
              };
              builder.create<pic::runtime::LinkOp>(op.getLoc(), safeExt(pA), safeExt(pB));
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
                     auto safeExt = [&](Value v) {
                         return (v.getType() == i64Type) ? v : builder.create<arith::ExtUIOp>(loc, i64Type, v);
                     };
                     Value eraPort = builder.create<arith::OrIOp>(loc, i64Type, builder.create<arith::ShLIOp>(loc, i64Type, safeExt(eraIdx), builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2))), builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(1)));
                     builder.create<pic::runtime::LinkOp>(loc, safeExt(resPort), eraPort);
                 }
                 op.setOperand(0, resPort);
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
    declFunc("lin_print_i32", nullptr, {i64Type});
    declFunc("lin_print_f32", nullptr, {f32Type});
    declFunc("lin_print_f64", nullptr, {f64Type});

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
    Value c0x979115_i32 = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x979115));

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
        Value idx64 = builder.create<arith::ExtUIOp>(loc, i64Type, idx);
        Value sh = builder.create<arith::ShLIOp>(loc, i64Type, idx64, c2_i64);
        return builder.create<arith::OrIOp>(loc, i64Type, sh, builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(p)));
    };

    Value metaA = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeA, builder.getI8IntegerAttr(3));
    Value metaB = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeB, builder.getI8IntegerAttr(3));

    Value metaA32 = builder.create<arith::TruncIOp>(loc, i32Type, metaA);
    Value metaB32 = builder.create<arith::TruncIOp>(loc, i32Type, metaB);

    Value polA = builder.create<arith::ShRUIOp>(loc, metaA32, c30_i32);
    Value polB = builder.create<arith::ShRUIOp>(loc, metaB32, c30_i32);

    Value labelA = builder.create<arith::AndIOp>(loc, metaA32, c0xFFFFFF_i32);
    Value labelB = builder.create<arith::AndIOp>(loc, metaB32, c0xFFFFFF_i32);

    Value typeValA = builder.create<arith::ShRUIOp>(loc, metaA32, c24_i32);
    Value typeValB = builder.create<arith::ShRUIOp>(loc, metaB32, c24_i32);

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
    Value rNode = builder.create<arith::SelectOp>(loc, isRvecA, bodyNodeA, bodyNodeB);
    Value tNode = builder.create<arith::SelectOp>(loc, isRvecA, bodyNodeB, bodyNodeA);
    Value tType = builder.create<arith::SelectOp>(loc, isRvecA, nodeTypeB, nodeTypeA);
    Value tPort = builder.create<arith::ShLIOp>(loc, i32Type, tNode, c2_i32);

    Block *rvecLBlock = wFunc.addBlock();
    Block *rvecStdBlock = wFunc.addBlock();
    Value isL = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tType, c7_i32);
    builder.create<cf::CondBranchOp>(loc, isL, rvecLBlock, rvecStdBlock);

    builder.setInsertionPointToStart(rvecLBlock);
    Value recNodeA = builder.create<pic::runtime::GetPortOp>(loc, i64Type, tNode, builder.getI8IntegerAttr(1));
    Value recNodeB = builder.create<pic::runtime::GetPortOp>(loc, i64Type, tNode, builder.getI8IntegerAttr(2));
    Value recNodeA_32 = builder.create<arith::TruncIOp>(loc, i32Type, recNodeA);
    Value recNodeB_32 = builder.create<arith::TruncIOp>(loc, i32Type, recNodeB);
    Value wordA12 = builder.create<pic::runtime::GetHistoryOp>(loc, i64Type, recNodeA_32, builder.getI8IntegerAttr(1));
    Value wordB12 = builder.create<pic::runtime::GetHistoryOp>(loc, i64Type, recNodeB_32, builder.getI8IntegerAttr(1));
    Value neighborA1 = builder.create<arith::ShRUIOp>(loc, wordA12, c32_i64);
    Value neighborA2 = builder.create<arith::AndIOp>(loc, wordA12, c0xFFFFFFFF_i64);
    Value neighborB1 = builder.create<arith::ShRUIOp>(loc, wordB12, c32_i64);
    Value neighborB2 = builder.create<arith::AndIOp>(loc, wordB12, c0xFFFFFFFF_i64);
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeA_32, 1), neighborA1);
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeA_32, 2), neighborA2);
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeB_32, 1), neighborB1);
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeB_32, 2), neighborB2);
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(recNodeA_32, 0), makePortVal(recNodeB_32, 0));
    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(rvecStdBlock);
    Value r1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(136), c0_i64, builder.getBoolAttr(false));
    Value r2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(136), c0_i64, builder.getBoolAttr(false));
    Value auxT1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, tNode, builder.getI8IntegerAttr(1));
    Value auxT2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, tNode, builder.getI8IntegerAttr(2));
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(r1, 0), auxT1);
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(r2, 0), auxT2);
    builder.create<cf::BranchOp>(loc, lHead);

    // ==========================================
    // nonRvecCase
    // ==========================================
    builder.setInsertionPointToStart(nonRvecCase);
    Value isEraA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, c0x979115_i32);
    Value isEraB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, c0x979115_i32);
    Value hasEra = builder.create<arith::OrIOp>(loc, isEraA, isEraB);

    Block *eraCase = wFunc.addBlock();
    Block *checkDispatch = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, hasEra, eraCase, checkDispatch);

    // eraCase
    builder.setInsertionPointToStart(eraCase);
    Value eraNode = builder.create<arith::SelectOp>(loc, isEraA, bodyNodeA, bodyNodeB);
    Value otherNode = builder.create<arith::SelectOp>(loc, isEraA, bodyNodeB, bodyNodeA);
    Value otherLabel = builder.create<arith::SelectOp>(loc, isEraA, labelB, labelA);
    Value otherPol = builder.create<arith::SelectOp>(loc, isEraA, polB, polA);
    Value otherIsEra = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, otherLabel, c0x979115_i32);
    Value isVal = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, otherPol, c2_i32);
    Value isAnn = builder.create<arith::OrIOp>(loc, otherIsEra, isVal);

    Block *doEraAnn = wFunc.addBlock();
    Block *doEraProp = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isAnn, doEraAnn, doEraProp);

    builder.setInsertionPointToStart(doEraAnn);
    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(doEraProp);
    Value otherIsLit = isLiteralLabel(otherLabel);
    Block *eraLitCase = wFunc.addBlock();
    Block *eraNormalProp = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, otherIsLit, eraLitCase, eraNormalProp);

    builder.setInsertionPointToStart(eraLitCase);
    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(eraNormalProp);
    Value era1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(132), c0_i64, builder.getBoolAttr(false));
    Value era2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(132), c0_i64, builder.getBoolAttr(false));
    Value auxOther1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, otherNode, builder.getI8IntegerAttr(1));
    Value auxOther2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, otherNode, builder.getI8IntegerAttr(2));
    builder.create<pic::runtime::LinkOp>(loc, auxOther1, makePortVal(era1, 0));
    builder.create<pic::runtime::LinkOp>(loc, auxOther2, makePortVal(era2, 0));
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
    Block *checkPrint = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, hasDispatch, dispatchCase, checkPrint);

    // dispatchCase
    builder.setInsertionPointToStart(dispatchCase);

    Value v0 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, valNode, builder.getI8IntegerAttr(1));
    Value nArgs = builder.create<func::CallOp>(loc, i32Type, "get_num_args", ValueRange{impl}).getResult(0);
    Value isUnary = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nArgs, c1_i32);
    Value isBinary = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, nArgs, c2_i32);

    Block *doUnary = wFunc.addBlock();
    Block *checkBinary = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isUnary, doUnary, checkBinary);

    builder.setInsertionPointToStart(doUnary);
    Value isGpu = builder.create<func::CallOp>(loc, i1Type, "is_gpu_op", ValueRange{impl}).getResult(0);
    Block *gpuBranch = wFunc.addBlock();
    Block *cpuBranch = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isGpu, gpuBranch, cpuBranch);

    builder.setInsertionPointToStart(gpuBranch);
    builder.create<func::CallOp>(loc, i64Type, "dispatch_user_op", ValueRange{impl, v0, c0_i64, c0_i64, c0_i64, valNode, opNode, stateArg});
    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(cpuBranch);
    Value resVal = builder.create<func::CallOp>(loc, i64Type, "dispatch_user_op", ValueRange{impl, v0, c0_i64, c0_i64, c0_i64, valNode, opNode, stateArg}).getResult(0);
    Value opP_aux2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, opNode, builder.getI8IntegerAttr(2));
    Value isSame = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, resVal, opP_aux2);
    Block *skipLink = wFunc.addBlock();
    Block *doLink = wFunc.addBlock();
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
    Block *doBinary = wFunc.addBlock();
    Block *doComm = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isBinary, doBinary, doComm);

    builder.setInsertionPointToStart(doBinary);
    Value rPort = builder.create<pic::runtime::GetPortOp>(loc, i64Type, opNode, builder.getI8IntegerAttr(1));
    Value rPort_32 = builder.create<arith::TruncIOp>(loc, i32Type, rPort);
    Value rTarget = builder.create<arith::ShRUIOp>(loc, rPort_32, c2_i32);
    Value rMeta = builder.create<pic::runtime::GetPortOp>(loc, i64Type, rTarget, builder.getI8IntegerAttr(3));
    Value rMeta_32 = builder.create<arith::TruncIOp>(loc, i32Type, rMeta);
    Value rLabel = builder.create<arith::AndIOp>(loc, rMeta_32, c0xFFFFFF_i32);
    Value cCallCode = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("call")));
    Value isCall = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, valLabel, cCallCode);
    Value rLabelMatch = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rLabel, valLabel);
    Value isRCall = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, rLabel, cCallCode);
    Value rIsLit = isLiteralLabel(rLabel);
    Value isRVal = builder.create<arith::OrIOp>(loc, rLabelMatch, isCall);
    isRVal = builder.create<arith::OrIOp>(loc, isRVal, isRCall);
    isRVal = builder.create<arith::OrIOp>(loc, isRVal, rIsLit);

    Block *doFullBinary = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isRVal, doFullBinary, doComm);

    builder.setInsertionPointToStart(doFullBinary);
    Value v1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, rTarget, builder.getI8IntegerAttr(1));
    Value rTarget_ext = builder.create<arith::ExtUIOp>(loc, i64Type, rTarget);
    Value firstArg = builder.create<arith::SelectOp>(loc, isCall, rTarget_ext, v1);
    Value callArg0 = builder.create<arith::SelectOp>(loc, isCall, v0, firstArg);
    Value valNode_aux2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, valNode, builder.getI8IntegerAttr(2));
    Value callArg1 = builder.create<arith::SelectOp>(loc, isCall, valNode_aux2, v0);

    Value isGpuBin = builder.create<func::CallOp>(loc, i1Type, "is_gpu_op", ValueRange{impl}).getResult(0);
    Block *gpuBranchBin = wFunc.addBlock();
    Block *cpuBranchBin = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isGpuBin, gpuBranchBin, cpuBranchBin);

    builder.setInsertionPointToStart(gpuBranchBin);
    builder.create<func::CallOp>(loc, i64Type, "dispatch_user_op", ValueRange{impl, callArg0, callArg1, c0_i64, c0_i64, valNode, opNode, stateArg});
    builder.create<cf::BranchOp>(loc, lHead);

    builder.setInsertionPointToStart(cpuBranchBin);
    Value resValBin = builder.create<func::CallOp>(loc, i64Type, "dispatch_user_op", ValueRange{impl, callArg0, callArg1, c0_i64, c0_i64, valNode, opNode, stateArg}).getResult(0);
    Value isSameBin = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, resValBin, callArg1);
    Block *skipLinkBin = wFunc.addBlock();
    Block *doLinkBin = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isSameBin, skipLinkBin, doLinkBin);

    builder.setInsertionPointToStart(doLinkBin);
    Value valLabelBin64 = builder.create<arith::ExtUIOp>(loc, i64Type, valLabel);
    Value resNodeBin = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(133), valLabelBin64, builder.getBoolAttr(false));
    builder.create<pic::runtime::SetPortOp>(loc, resNodeBin, builder.getI8IntegerAttr(1), resValBin);
    Value opNode_aux2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, opNode, builder.getI8IntegerAttr(2));
    builder.create<pic::runtime::LinkOp>(loc, opNode_aux2, makePortVal(resNodeBin, 0));
    builder.create<cf::BranchOp>(loc, skipLinkBin);

    builder.setInsertionPointToStart(skipLinkBin);
    builder.create<cf::BranchOp>(loc, lHead);

    // doComm
    builder.setInsertionPointToStart(doComm);
    Value metaValVal = builder.create<arith::SelectOp>(loc, hasRuleA, metaB, metaA);
    Value metaValOp = builder.create<arith::SelectOp>(loc, hasRuleA, metaA, metaB);
    Value valIsLit = isLiteralLabel(valLabel);
    Block *valLitCase = wFunc.addBlock();
    Block *checkOpLit = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, valIsLit, valLitCase, checkOpLit);

    // If valLabel is a literal constant
    builder.setInsertionPointToStart(valLitCase);
    Value litValVal = builder.create<pic::runtime::GetPortOp>(loc, i64Type, valNode, builder.getI8IntegerAttr(1));
    Value valLit1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
    Value valLit2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
    builder.create<pic::runtime::SetPortOp>(loc, valLit1, builder.getI8IntegerAttr(3), metaValVal);
    builder.create<pic::runtime::SetPortOp>(loc, valLit2, builder.getI8IntegerAttr(3), metaValVal);
    builder.create<pic::runtime::SetPortOp>(loc, valLit1, builder.getI8IntegerAttr(1), litValVal);
    builder.create<pic::runtime::SetPortOp>(loc, valLit2, builder.getI8IntegerAttr(1), litValVal);
    Value auxB1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, opNode, builder.getI8IntegerAttr(1));
    Value auxB2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, opNode, builder.getI8IntegerAttr(2));
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(valLit1, 0), auxB1);
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(valLit2, 0), auxB2);
    builder.create<cf::BranchOp>(loc, lHead);

    // Check if opLabel is a literal constant
    builder.setInsertionPointToStart(checkOpLit);
    Value opIsLit = isLiteralLabel(opLabel);
    Block *opLitCase = wFunc.addBlock();
    Block *stdCommCase = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, opIsLit, opLitCase, stdCommCase);

    // If opLabel is a literal constant
    builder.setInsertionPointToStart(opLitCase);
    Value litValOp = builder.create<pic::runtime::GetPortOp>(loc, i64Type, opNode, builder.getI8IntegerAttr(1));
    Value opLit1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
    Value opLit2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(0), c0_i64, builder.getBoolAttr(false));
    builder.create<pic::runtime::SetPortOp>(loc, opLit1, builder.getI8IntegerAttr(3), metaValOp);
    builder.create<pic::runtime::SetPortOp>(loc, opLit2, builder.getI8IntegerAttr(3), metaValOp);
    builder.create<pic::runtime::SetPortOp>(loc, opLit1, builder.getI8IntegerAttr(1), litValOp);
    builder.create<pic::runtime::SetPortOp>(loc, opLit2, builder.getI8IntegerAttr(1), litValOp);
    Value auxA1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, valNode, builder.getI8IntegerAttr(1));
    Value auxA2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, valNode, builder.getI8IntegerAttr(2));
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(opLit1, 0), auxA1);
    builder.create<pic::runtime::LinkOp>(loc, makePortVal(opLit2, 0), auxA2);
    builder.create<cf::BranchOp>(loc, lHead);

    // Standard commutation case
    builder.setInsertionPointToStart(stdCommCase);
    Value auxA1_std = builder.create<pic::runtime::GetPortOp>(loc, i64Type, valNode, builder.getI8IntegerAttr(1));
    Value auxA2_std = builder.create<pic::runtime::GetPortOp>(loc, i64Type, valNode, builder.getI8IntegerAttr(2));
    Value auxB1_std = builder.create<pic::runtime::GetPortOp>(loc, i64Type, opNode, builder.getI8IntegerAttr(1));
    Value auxB2_std = builder.create<pic::runtime::GetPortOp>(loc, i64Type, opNode, builder.getI8IntegerAttr(2));

    Value polOp = builder.create<arith::SelectOp>(loc, hasRuleA, polA, polB);
    Value polVal = builder.create<arith::SelectOp>(loc, hasRuleA, polB, polA);

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

    // checkPrint
    builder.setInsertionPointToStart(checkPrint);
    Value isPrintA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("print"))));
    Value isPrintB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("print"))));
    Value hasPrint = builder.create<arith::OrIOp>(loc, isPrintA, isPrintB);

    Block *printCase = wFunc.addBlock();
    Block *checkBranch = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, hasPrint, printCase, checkBranch);

    // printCase
    builder.setInsertionPointToStart(printCase);
    Value printOpNode = builder.create<arith::SelectOp>(loc, isPrintA, bodyNodeA, bodyNodeB);
    Value printValNode = builder.create<arith::SelectOp>(loc, isPrintA, bodyNodeB, bodyNodeA);
    Value printValLabel = builder.create<arith::SelectOp>(loc, isPrintA, labelB, labelA);
    Value nVal = builder.create<pic::runtime::GetPortOp>(loc, i64Type, printValNode, builder.getI8IntegerAttr(1));

    Value cF32Code = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("f32")));
    Value cF64Code = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opcodeForLabel("f64")));
    Value isF32 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, printValLabel, cF32Code);
    Value isF64 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, printValLabel, cF64Code);

    Block *printFloatBlock = wFunc.addBlock();
    Block *printDoubleBlock = wFunc.addBlock();
    Block *printIntBlock = wFunc.addBlock();
    Block *printMergeBlock = wFunc.addBlock();

    builder.create<cf::CondBranchOp>(loc, isF32, printFloatBlock, printDoubleBlock);

    builder.setInsertionPointToStart(printFloatBlock);
    Value truncVal = builder.create<arith::TruncIOp>(loc, builder.getI32Type(), nVal);
    Value f32Val = builder.create<arith::BitcastOp>(loc, builder.getF32Type(), truncVal);
    builder.create<func::CallOp>(loc, TypeRange{}, "lin_print_f32", ValueRange{f32Val});
    builder.create<cf::BranchOp>(loc, printMergeBlock);

    builder.setInsertionPointToStart(printDoubleBlock);
    Block *doPrintDouble = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isF64, doPrintDouble, printIntBlock);

    builder.setInsertionPointToStart(doPrintDouble);
    Value f64Val = builder.create<arith::BitcastOp>(loc, builder.getF64Type(), nVal);
    builder.create<func::CallOp>(loc, TypeRange{}, "lin_print_f64", ValueRange{f64Val});
    builder.create<cf::BranchOp>(loc, printMergeBlock);

    builder.setInsertionPointToStart(printIntBlock);
    builder.create<func::CallOp>(loc, TypeRange{}, "lin_print_i32", ValueRange{nVal});
    builder.create<cf::BranchOp>(loc, printMergeBlock);

    builder.setInsertionPointToStart(printMergeBlock);
    Value printOpNode_aux2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, printOpNode, builder.getI8IntegerAttr(1));
    builder.create<pic::runtime::LinkOp>(loc, printOpNode_aux2, makePortVal(printValNode, 0));
    builder.create<cf::BranchOp>(loc, lHead);

    // checkBranch
    builder.setInsertionPointToStart(checkBranch);
    Value isRawBranchA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x873945)));
    Value isEitherA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x80f214)));
    Value isBranchA = builder.create<arith::OrIOp>(loc, isRawBranchA, isEitherA);
    Value isRawBranchB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x873945)));
    Value isEitherB = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0x80f214)));
    Value isBranchB = builder.create<arith::OrIOp>(loc, isRawBranchB, isEitherB);
    Value hasBranch = builder.create<arith::OrIOp>(loc, isBranchA, isBranchB);

    Block *branchCase = wFunc.addBlock();
    Block *genericCase = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, hasBranch, branchCase, genericCase);

    // branchCase
    builder.setInsertionPointToStart(branchCase);
    Value branchOpNode = builder.create<arith::SelectOp>(loc, isBranchA, bodyNodeA, bodyNodeB);
    Value branchValNode = builder.create<arith::SelectOp>(loc, isBranchA, bodyNodeB, bodyNodeA);
    Value isEither = builder.create<arith::SelectOp>(loc, isBranchA, isEitherA, isEitherB);

    Value nValBr = builder.create<pic::runtime::GetPortOp>(loc, i64Type, branchValNode, builder.getI8IntegerAttr(1));
    Value condBr = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, nValBr, c0_i64);

    Value eitherPairP = builder.create<pic::runtime::GetPortOp>(loc, i64Type, branchOpNode, builder.getI8IntegerAttr(1));
    Value eitherPairP_32 = builder.create<arith::TruncIOp>(loc, i32Type, eitherPairP);
    Value eitherPairNode = builder.create<arith::ShRUIOp>(loc, eitherPairP_32, c2_i32);

    Value eitherBranch1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, eitherPairNode, builder.getI8IntegerAttr(1));
    Value eitherBranch2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, eitherPairNode, builder.getI8IntegerAttr(2));
    Value branch1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, branchOpNode, builder.getI8IntegerAttr(1));
    Value branch2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, branchOpNode, builder.getI8IntegerAttr(2));

    Value trueBranch = builder.create<arith::SelectOp>(loc, isEither, eitherBranch1, branch1);
    Value falseBranch = builder.create<arith::SelectOp>(loc, isEither, eitherBranch2, branch2);
    Value taken = builder.create<arith::SelectOp>(loc, condBr, trueBranch, falseBranch);
    Value untaken = builder.create<arith::SelectOp>(loc, condBr, falseBranch, trueBranch);

    Value eitherCont = builder.create<pic::runtime::GetPortOp>(loc, i64Type, branchOpNode, builder.getI8IntegerAttr(2));
    Value branchCont = makePortVal(branchOpNode, 0);
    Value target = builder.create<arith::SelectOp>(loc, isEither, eitherCont, branchCont);

    Value eraBr = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(132), c0_i64, builder.getBoolAttr(false));
    builder.create<pic::runtime::LinkOp>(loc, untaken, makePortVal(eraBr, 0));
    builder.create<pic::runtime::LinkOp>(loc, taken, target);
    builder.create<cf::BranchOp>(loc, lHead);

    // genericCase
    builder.setInsertionPointToStart(genericCase);
    Value labelsMatch = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, labelB);
    Value polsDiff = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, polA, polB);
    Value isAnn_gen = builder.create<arith::AndIOp>(loc, labelsMatch, polsDiff);

    Block *annPath = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isAnn_gen, annPath, doComm);

    // annPath
    builder.setInsertionPointToStart(annPath);
    Value hWord0A = builder.create<pic::runtime::GetHistoryOp>(loc, i64Type, bodyNodeA, builder.getI8IntegerAttr(0));
    Value inBoundaryA = builder.create<arith::AndIOp>(loc, hWord0A, c0x100000000_i64);
    Value isBoundaryA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, inBoundaryA, c0_i64);

    Block *depLBlock = wFunc.addBlock();
    Block *skipLBlock = wFunc.addBlock();
    builder.create<cf::CondBranchOp>(loc, isBoundaryA, depLBlock, skipLBlock);

    builder.setInsertionPointToStart(depLBlock);
    Value lIdx = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(128 + 7), c0_i64, builder.getBoolAttr(false));
    Value lIdx64 = builder.create<arith::ExtUIOp>(loc, i64Type, lIdx);
    
    Value bodyNodeA64 = builder.create<arith::ExtUIOp>(loc, i64Type, bodyNodeA);
    Value bodyNodeB64 = builder.create<arith::ExtUIOp>(loc, i64Type, bodyNodeB);
    builder.create<pic::runtime::SetPortOp>(loc, lIdx, builder.getI8IntegerAttr(1), bodyNodeA64);
    builder.create<pic::runtime::SetPortOp>(loc, lIdx, builder.getI8IntegerAttr(2), bodyNodeB64);
    
    builder.create<pic::runtime::SetPortOp>(loc, lIdx, builder.getI8IntegerAttr(3), builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(2214592512ULL)));

    Value logPtrA = builder.create<arith::AndIOp>(loc, hWord0A, c0xFFFFFFFF_i64);
    Value valWord0 = builder.create<arith::OrIOp>(loc, c0x100000000_i64, logPtrA);
    builder.create<pic::runtime::SetHistoryOp>(loc, lIdx, builder.getI8IntegerAttr(0), valWord0);

    Value ann_auxA1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeA, builder.getI8IntegerAttr(1));
    Value ann_auxA2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeA, builder.getI8IntegerAttr(2));
    Value ann_auxB1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeB, builder.getI8IntegerAttr(1));
    Value ann_auxB2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeB, builder.getI8IntegerAttr(2));

    auto getNeighborPortVal = [&](Value auxPort) {
        Value auxPort_32 = builder.create<arith::TruncIOp>(loc, i32Type, auxPort);
        Value nNode = builder.create<arith::ShRUIOp>(loc, auxPort_32, c2_i32);
        Value nPort = builder.create<arith::AndIOp>(loc, auxPort_32, c3_i32);
        Value nPort8 = builder.create<arith::TruncIOp>(loc, i8Type, nPort);
        return builder.create<pic::runtime::GetPortDynamicOp>(loc, i64Type, nNode, nPort8);
    };

    Value neighborA1_ann = getNeighborPortVal(ann_auxA1);
    Value neighborA2_ann = getNeighborPortVal(ann_auxA2);
    Value neighborB1_ann = getNeighborPortVal(ann_auxB1);
    Value neighborB2_ann = getNeighborPortVal(ann_auxB2);

    Value wordA12_ann = builder.create<arith::OrIOp>(loc, builder.create<arith::ShLIOp>(loc, i64Type, neighborA1_ann, c32_i64), neighborA2_ann);
    Value wordB12_ann = builder.create<arith::OrIOp>(loc, builder.create<arith::ShLIOp>(loc, i64Type, neighborB1_ann, c32_i64), neighborB2_ann);

    builder.create<pic::runtime::SetHistoryOp>(loc, bodyNodeA, builder.getI8IntegerAttr(1), wordA12_ann);
    builder.create<pic::runtime::SetHistoryOp>(loc, bodyNodeB, builder.getI8IntegerAttr(1), wordB12_ann);
    builder.create<pic::runtime::SetHistoryOp>(loc, lIdx, builder.getI8IntegerAttr(1), c0_i64);

    auto updateNeighborHistory = [&](Value auxPort) {
        Value auxPort_32 = builder.create<arith::TruncIOp>(loc, i32Type, auxPort);
        Value nNode = builder.create<arith::ShRUIOp>(loc, auxPort_32, c2_i32);
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
    Value final_auxA1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeA, builder.getI8IntegerAttr(1));
    Value final_auxA2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeA, builder.getI8IntegerAttr(2));
    Value final_auxB1 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeB, builder.getI8IntegerAttr(1));
    Value final_auxB2 = builder.create<pic::runtime::GetPortOp>(loc, i64Type, bodyNodeB, builder.getI8IntegerAttr(2));

    builder.create<pic::runtime::LinkOp>(loc, final_auxA1, final_auxB1);
    builder.create<pic::runtime::LinkOp>(loc, final_auxA2, final_auxB2);
    builder.create<cf::BranchOp>(loc, lHead);

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
    
    // First, convert all func.call to runtime helper functions to LLVM calls
    module.walk([&](func::CallOp callOp) {
        StringRef callee = callOp.getCallee();
        if (callee == "lookup_rule" || callee == "dispatch_user_op" ||
            callee == "is_gpu_op" || callee == "get_num_args" ||
            callee == "lin_print_i32" || callee == "lin_print_f32" ||
            callee == "lin_print_f64") {
            
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

    for (StringRef name : {"lookup_rule", "dispatch_user_op", "is_gpu_op", "get_num_args", "lin_print_i32", "lin_print_f32", "lin_print_f64"}) {
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
                Value v64 = safeZExt(ob, o->getLoc(), i64Type, v);
                ob.create<LLVM::StoreOp>(o->getLoc(), v64, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, net, ValueRange{off}));
            };
            
            auto makePort = [&](int p) {
                Value nIdx64 = safeZExt(ob, o->getLoc(), i64Type, nIdx);
                return ob.create<LLVM::OrOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(p)));
            };
            
            uint8_t typeVal = allocOp.getType();
            Value labelOrVal = allocOp.getLabelOrVal();
            
            Value typeValConst = ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr((uint64_t)typeVal << 24));
            Value labelOrVal64 = safeZExt(ob, o->getLoc(), i64Type, labelOrVal);
            Value maskConst = ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0xFFFFFF));
            Value labelMasked = ob.create<LLVM::AndOp>(o->getLoc(), labelOrVal64, maskConst);
            Value metaValueVal = ob.create<LLVM::OrOp>(o->getLoc(), typeValConst, labelMasked);
            
            store(0, makePort(0)); store(1, makePort(1)); store(2, makePort(2)); store(3, metaValueVal);

            bool isInsideBoundary = allocOp.getInsideBoundary().value_or(false);
            if (isInsideBoundary) {
                Value historyNet = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(5))}));
                Value hBase = ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                Value valWord0 = ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1ULL << 32)); // bit 0 = inside_boundary is set
                Value valWord1 = ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0));
                
                Value gep0 = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, historyNet, ValueRange{hBase});
                ob.create<LLVM::StoreOp>(o->getLoc(), valWord0, gep0);
                
                Value hBasePlus1 = ob.create<LLVM::AddOp>(o->getLoc(), hBase, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                Value gep1 = ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, historyNet, ValueRange{hBasePlus1});
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
            ob.create<LLVM::StoreOp>(o->getLoc(), safeZExt(ob, o->getLoc(), i64Type, rPort0), ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, net, ValueRange{base}));
            Value metaVal = ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr((2ULL << 30) | (8ULL << 24)));
            Value off3 = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, base, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(3)));
            ob.create<LLVM::StoreOp>(o->getLoc(), metaVal, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, net, ValueRange{off3}));
            
            auto linkPortsLLVM = [&](Value p1, Value p2) {
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
            };
            linkPortsLLVM(rPort0, boundaryId);
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
            Value val64 = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, net, ValueRange{offset}));
            o->getResult(0).replaceAllUsesWith(val64);
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
            Value val64 = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, net, ValueRange{offset}));
            o->getResult(0).replaceAllUsesWith(val64);
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
            
            Value curH = ob.create<LLVM::AtomicRMWOp>(o->getLoc(), LLVM::AtomicBinOp::add, hdPtr, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
            Value curT = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, tlPtr);
            Value inBounds = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::ult, curH, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(16000000)));
            Value hasElement = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::ult, curH, curT);
            Value isValid = ob.create<LLVM::AndOp>(o->getLoc(), inBounds, hasElement);
            
            Block *curr = ob.getBlock();
            Block *doLoad = curr->splitBlock(o);
            Block *cont = doLoad->splitBlock(doLoad->begin());
            
            Value finalA = cont->addArgument(i32Type, o->getLoc());
            Value finalB = cont->addArgument(i32Type, o->getLoc());
            
            ob.setInsertionPointToEnd(curr);
            Value c0_i32 = ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, ob.getI32IntegerAttr(0));
            ob.create<LLVM::CondBrOp>(o->getLoc(), isValid, doLoad, ValueRange{}, cont, ValueRange{c0_i32, c0_i32});
            
            ob.setInsertionPointToStart(doLoad);
            Value val = ob.create<LLVM::LoadOp>(o->getLoc(), i64Type, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, q, ValueRange{curH}));
            Value pA = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, val);
            Value pB = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, ob.create<LLVM::LShrOp>(o->getLoc(), i64Type, val, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(32))));
            Value c2 = ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(2));
            Value nA = ob.create<LLVM::LShrOp>(o->getLoc(), i32Type, pA, c2);
            Value nB = ob.create<LLVM::LShrOp>(o->getLoc(), i32Type, pB, c2);
            ob.create<LLVM::BrOp>(o->getLoc(), ValueRange{nA, nB}, cont);
            
            ob.setInsertionPointToStart(cont);
            
            o->getResult(0).replaceAllUsesWith(isValid);
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
#ifdef ENABLE_DEBUG_LOGS
        llvm::errs() << "Captured Decl (func): [" << s << "]\n";
#endif
        existingDecls.push_back(s);
    }
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
        std::string name = func.getSymName().str();
        auto fType = func.getFunctionType();
        std::string s;
        if (name.rfind("lin_", 0) == 0) {
            s = "func.func private @" + name + "(";
            for (unsigned i = 0; i < fType.getNumParams(); ++i) {
                std::string typeStr;
                llvm::raw_string_ostream typeStream(typeStr);
                fType.getParamType(i).print(typeStream);
                typeStream.flush();
                s += typeStr;
                if (i < fType.getNumParams() - 1) s += ", ";
            }
            s += ") -> ";
            Type retType = fType.getReturnType();
            if (!retType.isa<LLVM::LLVMVoidType>()) {
                std::string typeStr;
                llvm::raw_string_ostream typeStream(typeStr);
                retType.print(typeStream);
                typeStream.flush();
                s += typeStr;
            }
        } else {
            s = "llvm.func private @" + name + "(";
            for (unsigned i = 0; i < fType.getNumParams(); ++i) {
                std::string typeStr;
                llvm::raw_string_ostream typeStream(typeStr);
                fType.getParamType(i).print(typeStream);
                typeStream.flush();
                s += typeStr;
                if (i < fType.getNumParams() - 1) s += ", ";
            }
            if (fType.isVarArg()) {
                if (fType.getNumParams() > 0) s += ", ";
                s += "...";
            }
            s += ") -> ";
            std::string typeStr;
            llvm::raw_string_ostream typeStream(typeStr);
            fType.getReturnType().print(typeStream);
            typeStream.flush();
            s += typeStr;
        }
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
                    std::string dispatch = "";
                    if (auto dispAttr = op->getAttrOfType<StringAttr>("dispatch")) {
                        dispatch = dispAttr.getValue().str();
                    }
                    
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
                         if (dispatch == "gpu" && tempFunc) {
                             tempFunc.walk([&](Operation *op) {
                                 if (auto* dialect = op->getDialect()) {
                                     StringRef dialectName = dialect->getNamespace();
                                     if (dialectName == "llvm" || dialectName == "pic_runtime" || dialectName == "pic_graph") {
                                         op->emitError() << "Operation '" << op->getName() << "' in registry op '" << label << "' is illegal on GPU!";
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
        
        auto genRegisterRule = [&]() {
            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(builder.getContext()), {i32Type, i32Type, i32Type});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "register_rule", fType);
            Block *entry = f.addEntryBlock();
            OpBuilder eb(entry, entry->end());
            eb.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{});
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
                    if (typeName == "i64") {
                        addRuleMatch(opcodeForLabel(op.label), opcodeForLabel("i32"), opcodeForLabel(op.label));
                    }
                }
                addRuleMatch(opcodeForLabel(op.label), opcodeForLabel("call"), opcodeForLabel(op.label));
            }
            
            OpBuilder fb(nextBlock, nextBlock->end());
            Value zero = fb.create<LLVM::ConstantOp>(loc, i32Type, fb.getI32IntegerAttr(0));
            fb.create<LLVM::ReturnOp>(loc, ValueRange{zero});
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

    auto genDispatcher = [&]() {
        OpBuilder b(module.getBodyRegion());
        auto fType = LLVM::LLVMFunctionType::get(i64Type, {i32Type, i64Type, i64Type, i64Type, i64Type, i32Type, i32Type, i64Type});
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
        Value wState = dState;

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

    /*
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

    Value typeValA = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaA32, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(24)));
    Value nodeTypeA = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, typeValA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x3F)));
    Value typeValB = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaB32, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(24)));
    Value nodeTypeB = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, typeValB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x3F)));

    logRedex(labelA, labelB);

    Block *rvecBlock = wFunc.addBlock(), *nonRvecBlock = wFunc.addBlock();
    Value isRvecA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, nodeTypeA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(NODE_RVEC)));
    Value isRvecB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, nodeTypeB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(NODE_RVEC)));
    Value hasRvec = builder.create<LLVM::OrOp>(module.getLoc(), isRvecA, isRvecB);
    builder.create<LLVM::CondBrOp>(module.getLoc(), hasRvec, rvecBlock, nonRvecBlock);

    builder.setInsertionPointToStart(rvecBlock);
    {
        Value rPort = builder.create<LLVM::SelectOp>(module.getLoc(), isRvecA, pA, pB);
        Value tPort = builder.create<LLVM::SelectOp>(module.getLoc(), isRvecA, pB, pA);
        Value tType = builder.create<LLVM::SelectOp>(module.getLoc(), isRvecA, nodeTypeB, nodeTypeA);
        
        Block *rvecLBlock = wFunc.addBlock(), *rvecStdBlock = wFunc.addBlock();
        Value isL = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, tType, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(NODE_LOG)));
        builder.create<LLVM::CondBrOp>(module.getLoc(), isL, rvecLBlock, rvecStdBlock);
        
        builder.setInsertionPointToStart(rvecLBlock);
        {
            Value tNodeIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, safeZExt(builder, module.getLoc(), i32Type, tPort), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
            Value tNodeIdx64 = safeZExt(builder, module.getLoc(), i64Type, tNodeIdx);
            Value tBase = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, tNodeIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
            Value recNodeA64 = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), tBase, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)))}));
            Value recNodeB64 = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), tBase, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)))}));
            
            Value hIdxRecA = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, recNodeA64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
            Value hIdxRecB = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, recNodeB64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
            Value payloadAIdx = builder.create<LLVM::AddOp>(module.getLoc(), hIdxRecA, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
            Value payloadBIdx = builder.create<LLVM::AddOp>(module.getLoc(), hIdxRecB, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
            Value wordA12 = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wHistoryNet, ValueRange{payloadAIdx}));
            Value wordB12 = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wHistoryNet, ValueRange{payloadBIdx}));
            
            Value neighborA1 = builder.create<LLVM::LShrOp>(module.getLoc(), i64Type, wordA12, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(32)));
            Value neighborA2 = builder.create<LLVM::AndOp>(module.getLoc(), i64Type, wordA12, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0xFFFFFFFF)));
            Value neighborB1 = builder.create<LLVM::LShrOp>(module.getLoc(), i64Type, wordB12, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(32)));
            Value neighborB2 = builder.create<LLVM::AndOp>(module.getLoc(), i64Type, wordB12, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0xFFFFFFFF)));
            
            auto makePort = [&](Value idx, int p) {
                return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, idx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
            };
            
            linkPorts(makePort(builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, recNodeA64), 1), neighborA1, wState);
            linkPorts(makePort(builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, recNodeA64), 2), neighborA2, wState);
            linkPorts(makePort(builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, recNodeB64), 1), neighborB1, wState);
            linkPorts(makePort(builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, recNodeB64), 2), neighborB2, wState);
            linkPorts(makePort(builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, recNodeA64), 0), makePort(builder.create<LLVM::TruncOp>(module.getLoc(), i32Type, recNodeB64), 0), wState);
            
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
        }
        
        builder.setInsertionPointToStart(rvecStdBlock);
        {
            Value r1Idx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
            Value r1Idx64 = safeZExt(builder, module.getLoc(), i64Type, r1Idx);
            Value r1Base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, r1Idx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
            
            auto makePort = [&](Value idx, int p) {
                return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, idx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
            };
            
            Value r1Port0 = makePort(r1Idx, 0);
            builder.create<LLVM::StoreOp>(module.getLoc(), safeZExt(builder, module.getLoc(), i64Type, r1Port0), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{r1Base}));
            Value r1Meta = builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr((2ULL << 30) | (8ULL << 24)));
            builder.create<LLVM::StoreOp>(module.getLoc(), r1Meta, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), r1Base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(3)))}));
            
            Value r2Idx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
            Value r2Idx64 = safeZExt(builder, module.getLoc(), i64Type, r2Idx);
            Value r2Base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, r2Idx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
            Value r2Port0 = makePort(r2Idx, 0);
            builder.create<LLVM::StoreOp>(module.getLoc(), safeZExt(builder, module.getLoc(), i64Type, r2Port0), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{r2Base}));
            Value r2Meta = builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr((2ULL << 30) | (8ULL << 24)));
            builder.create<LLVM::StoreOp>(module.getLoc(), r2Meta, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), r2Base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(3)))}));
            
            linkPorts(r1Port0, getAux(tPort, 1), wState);
            linkPorts(r2Port0, getAux(tPort, 2), wState);
            
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
        }
    }

    builder.setInsertionPointToStart(nonRvecBlock);
    Block *specBlock = wFunc.addBlock(), *genBlock = wFunc.addBlock();
    
    Value isEraA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x979115)));
    Value isEraB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x979115)));
    Value isRawBranchA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x873945)));
    Value isEitherA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x80f214)));
    Value isBranchA = builder.create<LLVM::OrOp>(module.getLoc(), isRawBranchA, isEitherA);
    Value isRawBranchB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x873945)));
    Value isEitherB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x80f214)));
    Value isBranchB = builder.create<LLVM::OrOp>(module.getLoc(), isRawBranchB, isEitherB);
    Value isPrintA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel("print"))));
    Value isPrintB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel("print"))));

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
            Value isPrint = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(opcodeForLabel("print"))));
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
            {
                Value nodeAIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, safeZExt(builder, module.getLoc(), i32Type, pA), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                Value nodeBIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, safeZExt(builder, module.getLoc(), i32Type, pB), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                Value nodeAIdx64 = safeZExt(builder, module.getLoc(), i64Type, nodeAIdx);
                Value hIdxA = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nodeAIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                Value hWord0A = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wHistoryNet, ValueRange{hIdxA}));
                Value inBoundaryA = builder.create<LLVM::AndOp>(module.getLoc(), hWord0A, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1ULL << 32)));
                Value zero64 = builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0));
                Value isBoundaryA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::ne, inBoundaryA, zero64);
                
                Block *depLBlock = wFunc.addBlock(), *skipLBlock = wFunc.addBlock();
                builder.create<LLVM::CondBrOp>(module.getLoc(), isBoundaryA, depLBlock, skipLBlock);
                
                builder.setInsertionPointToStart(depLBlock);
                {
                    Value lIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                    Value lIdx64 = safeZExt(builder, module.getLoc(), i64Type, lIdx);
                    Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, lIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                    
                    auto makePort = [&](Value idx, int p) {
                        return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, idx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                    };
                    
                    builder.create<LLVM::StoreOp>(module.getLoc(), safeZExt(builder, module.getLoc(), i64Type, makePort(lIdx, 0)), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{base}));
                    builder.create<LLVM::StoreOp>(module.getLoc(), nodeAIdx64, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)))}));
                    Value nodeBIdx64 = safeZExt(builder, module.getLoc(), i64Type, nodeBIdx);
                    builder.create<LLVM::StoreOp>(module.getLoc(), nodeBIdx64, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)))}));
                    Value metaVal = builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr((2ULL << 30) | (7ULL << 24)));
                    builder.create<LLVM::StoreOp>(module.getLoc(), metaVal, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(3)))}));
                    
                    Value logPtrA = builder.create<LLVM::AndOp>(module.getLoc(), hWord0A, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0xFFFFFFFF)));
                    Value hIdxL = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, lIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                    Value valWord0 = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1ULL << 32)), logPtrA);
                    builder.create<LLVM::StoreOp>(module.getLoc(), valWord0, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wHistoryNet, ValueRange{hIdxL}));
                    
                    Value auxA1 = getAux(pA, 1);
                    Value auxA2 = getAux(pA, 2);
                    Value auxB1 = getAux(pB, 1);
                    Value auxB2 = getAux(pB, 2);
                    Value neighborA1 = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{safeZExt(builder, module.getLoc(), i64Type, auxA1)}));
                    Value neighborA2 = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{safeZExt(builder, module.getLoc(), i64Type, auxA2)}));
                    Value neighborB1 = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{safeZExt(builder, module.getLoc(), i64Type, auxB1)}));
                    Value neighborB2 = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wNet, ValueRange{safeZExt(builder, module.getLoc(), i64Type, auxB2)}));
                    
                    Value wordA12 = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, neighborA1, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(32))), neighborA2);
                    Value wordB12 = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, neighborB1, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(32))), neighborB2);
                    
                    Value payloadAIdx = builder.create<LLVM::AddOp>(module.getLoc(), hIdxA, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                    builder.create<LLVM::StoreOp>(module.getLoc(), wordA12, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wHistoryNet, ValueRange{payloadAIdx}));
                    
                    Value hIdxB = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nodeBIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                    Value payloadBIdx = builder.create<LLVM::AddOp>(module.getLoc(), hIdxB, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                    builder.create<LLVM::StoreOp>(module.getLoc(), wordB12, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wHistoryNet, ValueRange{payloadBIdx}));

                    Value hIdxLPlus1 = builder.create<LLVM::AddOp>(module.getLoc(), hIdxL, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                    builder.create<LLVM::StoreOp>(module.getLoc(), zero64, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wHistoryNet, ValueRange{hIdxLPlus1}));
                    
                    auto updateNeighborHistory = [&](Value auxPort) {
                        Value neighborIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i64Type, safeZExt(builder, module.getLoc(), i64Type, auxPort), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                        Value hIdxN = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, neighborIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)));
                        Value gepN = builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i64Type, wHistoryNet, ValueRange{hIdxN});
                        Value hWord0N = builder.create<LLVM::LoadOp>(module.getLoc(), i64Type, gepN);
                        Value flagsN = builder.create<LLVM::AndOp>(module.getLoc(), hWord0N, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0xFFFFFFFF00000000ULL)));
                        Value newWord0N = builder.create<LLVM::OrOp>(module.getLoc(), flagsN, lIdx64);
                        builder.create<LLVM::StoreOp>(module.getLoc(), newWord0N, gepN);
                    };
                    updateNeighborHistory(auxA1);
                    updateNeighborHistory(auxA2);
                    updateNeighborHistory(auxB1);
                    updateNeighborHistory(auxB2);
                    
                    builder.create<LLVM::BrOp>(module.getLoc(), skipLBlock);
                }
                
                builder.setInsertionPointToStart(skipLBlock);
                linkPorts(getAux(pA, 1), getAux(pB, 1), wState);
                linkPorts(getAux(pA, 2), getAux(pB, 2), wState);
                builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            }
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
                Value isSame = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, resVal, getAux(opP, 2));
                Block *skipLink = wFunc.addBlock();
                Block *doLink = wFunc.addBlock();
                builder.create<LLVM::CondBrOp>(module.getLoc(), isSame, skipLink, doLink);
                
                builder.setInsertionPointToStart(doLink);
                linkPorts(getAux(opP, 2), makeVal(resVal, valLabel), wState);
                builder.create<LLVM::BrOp>(module.getLoc(), skipLink);
                
                builder.setInsertionPointToStart(skipLink);
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
                    Value callArg0 = builder.create<LLVM::SelectOp>(module.getLoc(), isCall, v0, firstArg);
                    Value callArg1 = builder.create<LLVM::SelectOp>(module.getLoc(), isCall, getAux(valP, 2), v0);
                    Value zero64 = builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0));
                    Value nodeA = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, valP, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                    Value nodeB = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, opP, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));

                    Value isGpu = builder.create<LLVM::CallOp>(module.getLoc(), i1Type, "is_gpu_op", ValueRange{impl}).getResult();
                    Block *gpuBranch = wFunc.addBlock();
                    Block *cpuBranch = wFunc.addBlock();
                    builder.create<LLVM::CondBrOp>(module.getLoc(), isGpu, gpuBranch, cpuBranch);
                    
                    builder.setInsertionPointToStart(gpuBranch);
                    builder.create<LLVM::CallOp>(module.getLoc(), i64Type, "dispatch_user_op", ValueRange{impl, callArg0, callArg1, zero64, zero64, nodeA, nodeB, wState});
                    builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                    
                    builder.setInsertionPointToStart(cpuBranch);
                    Value resVal = builder.create<LLVM::CallOp>(module.getLoc(), i64Type, "dispatch_user_op", ValueRange{impl, callArg0, callArg1, zero64, zero64, nodeA, nodeB, wState}).getResult();
                    Value isSame = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, resVal, callArg1);
                    Block *skipLink = wFunc.addBlock();
                    Block *doLink = wFunc.addBlock();
                    builder.create<LLVM::CondBrOp>(module.getLoc(), isSame, skipLink, doLink);
                    
                    builder.setInsertionPointToStart(doLink);
                    linkPorts(getAux(opP, 2), makeVal(resVal, valLabel), wState);
                    builder.create<LLVM::BrOp>(module.getLoc(), skipLink);
                    
                    builder.setInsertionPointToStart(skipLink);
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
            Value labelA_val = builder.create<LLVM::SelectOp>(module.getLoc(), isBranchA, labelA, labelB);
            Value isEither = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA_val, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x80f214)));
            
            Value nVal = getAux(valP, 1);
            Value cond = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::ne, nVal, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0)));
            
            Value eitherPairP = getAux(opP, 1);
            Value zero64 = builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0));
            Value safePairP = builder.create<LLVM::SelectOp>(module.getLoc(), isEither, eitherPairP, zero64);
            
            Value eitherBranch1 = getAux(safePairP, 1);
            Value eitherBranch2 = getAux(safePairP, 2);
            Value branch1 = getAux(opP, 1);
            Value branch2 = getAux(opP, 2);
            
            Value trueBranch = builder.create<LLVM::SelectOp>(module.getLoc(), isEither, eitherBranch1, branch1);
            Value falseBranch = builder.create<LLVM::SelectOp>(module.getLoc(), isEither, eitherBranch2, branch2);
            
            Value taken = builder.create<LLVM::SelectOp>(module.getLoc(), cond, trueBranch, falseBranch);
            Value untaken = builder.create<LLVM::SelectOp>(module.getLoc(), cond, falseBranch, trueBranch);
            
            Value eitherCont = getAux(opP, 2);
            Value branchCont = getAux(opP, 0);
            Value target = builder.create<LLVM::SelectOp>(module.getLoc(), isEither, eitherCont, branchCont);
            
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
            linkPorts(taken, target, wState);
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
        }
    }

    builder.setInsertionPointToStart(lEnd);
    builder.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{builder.create<LLVM::IntToPtrOp>(module.getLoc(), ptrType, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0)))});
    */

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
        Value as = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(48))}).getResult();
        auto sA = [&](int i, Value v) { builder.create<LLVM::StoreOp>(entry.getLoc(), v, builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, ptrType, as, ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(i))})); };
        sA(0, net); sA(1, q); sA(2, hd); sA(3, tl); sA(4, alL); sA(5, history_net);
        builder.create<func::CallOp>(entry.getLoc(), TypeRange{}, entry.getSymName(), ValueRange{builder.create<LLVM::PtrToIntOp>(entry.getLoc(), i64Type, as)});
        Value asInt = builder.create<LLVM::PtrToIntOp>(entry.getLoc(), i64Type, as);
        builder.create<func::CallOp>(entry.getLoc(), TypeRange{i64Type}, "worker_thread", ValueRange{asInt});
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
        op.getResult(0).replaceAllUsesWith(op.getOperand(0));
        castsToErase.push_back(op);
    });
    for (auto* op : castsToErase) op->erase();
  }
};

} // namespace

std::unique_ptr<Pass> createPicGraphToReducePass() { return std::make_unique<PicGraphToReducePass>(); }
std::unique_ptr<Pass> createPicReduceToRuntimePass(bool enableGPU, std::string spirvPath) { return std::make_unique<PicReduceToRuntimePass>(enableGPU, spirvPath); }
std::unique_ptr<Pass> createPicRuntimeToLLVMPass(bool enableGPU, std::string spirvPath) { return std::make_unique<PicRuntimeToLLVMPass>(enableGPU, spirvPath); }
