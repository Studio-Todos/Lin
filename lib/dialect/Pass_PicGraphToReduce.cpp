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

} // namespace

std::unique_ptr<Pass> createPicGraphToReducePass() { return std::make_unique<PicGraphToReducePass>(); }