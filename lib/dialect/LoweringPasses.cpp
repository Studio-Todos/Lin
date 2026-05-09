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
  return (hash & 0xFFFFFF) ? (hash & 0xFFFFFF) : 1u;
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
            builder.create<pic::graph::RegistryOp>(loc, builder.getStringAttr(strKey), op.getStrVal().value());
        } else val = opcodeForLabel(label);

        auto alloc = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, 4, val);
        Value nodeIdx = alloc.getIndex();
        
        uint32_t metaValue = (polVal << 30) | (val & 0xFFFFFF);
        if (label == "num") {
            metaValue = (polVal << 30) | (opcodeForLabel("num") & 0xFFFFFF);
            Value litVal = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(op.getValue().value()));
            builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(1), builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), litVal));
        }
        
        Value meta = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(metaValue));
        builder.create<pic::runtime::SetPortOp>(loc, nodeIdx, builder.getI8IntegerAttr(3), builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), meta));

        auto makePort = [&](int i) {
            Value shifted = builder.create<arith::ShLIOp>(loc, i32Type, nodeIdx, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2)));
            return builder.create<arith::OrIOp>(loc, i32Type, shifted, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(i)));
        };

        // Port mapping:
        // For omega (operations), we make Port 1 (Arg 0) principal to trigger reduction.
        // Port 0 (Result) is moved to Port 2.
        // Port 2 (Arg 1) is moved to Port 1.
        if (agentType == "omega") {
            valueToPort[op.getP0()] = makePort(2); // Result -> Port 2
            valueToPort[op.getP1()] = makePort(0); // Arg0   -> Port 0 (Principal)
            valueToPort[op.getP2()] = makePort(1); // Arg1   -> Port 1
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
         if (op.getNumOperands() > 0 && valueToPort.count(op.getOperand(0))) {
             Value resPort = valueToPort[op.getOperand(0)];
             if (funcOp.getSymName() == "main_inet_entry") {
                 Location loc = op.getLoc();
                 builder.setInsertionPoint(op);
                 auto eraAlloc = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, 4, 0x979115);
                 Value eraIdx = eraAlloc.getIndex();
                 builder.create<pic::runtime::SetPortOp>(loc, eraIdx, builder.getI8IntegerAttr(3), builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr((2u << 30) | 0x979115u))));
                 builder.create<pic::runtime::LinkOp>(loc, builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), resPort), builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), builder.create<arith::ShLIOp>(loc, i32Type, eraIdx, builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2)))));
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
  PicRuntimeToLLVMPass(bool enableGPU) : enableGPU(enableGPU) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());

    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    auto voidType = LLVM::LLVMVoidType::get(builder.getContext());

    auto decl = [&](StringRef n, Type r, ArrayRef<Type> a, bool var = false) { 
        if (!module.lookupSymbol<LLVM::LLVMFuncOp>(n)) 
            builder.create<LLVM::LLVMFuncOp>(module.getLoc(), n, LLVM::LLVMFunctionType::get(r, a, var)); 
    };
    decl("malloc", ptrType, {i64Type}); decl("free", voidType, {ptrType});
    decl("pthread_create", i32Type, {ptrType, ptrType, ptrType, ptrType}); decl("pthread_join", i32Type, {i64Type, ptrType});
    decl("printf", i32Type, {ptrType}, true);

    struct UserOp { uint32_t hash; std::string label; std::string funcName; int numArgs; };
    std::vector<UserOp> userOps;
    SmallVector<pic::graph::RegistryOp> regOps;
    module.walk([&](pic::graph::RegistryOp op) {
        regOps.push_back(op);
    });

    SmallVector<std::string> existingDecls;
    for (auto func : module.getOps<func::FuncOp>()) {
        std::string decl;
        llvm::raw_string_ostream os(decl);
        func.print(os);
        existingDecls.push_back(decl);
    }
    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
        std::string decl;
        llvm::raw_string_ostream os(decl);
        func.print(os);
        existingDecls.push_back(decl);
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
                        std::string argsStr = argsAttr.getValue().str();
                        for (size_t i = 0; i < argsStr.size(); i++) {
                            if (argsStr[i] == '[') numArgs++;
                        }
                    } else {
                        // Fallback for user-defined functions which always have 2 args in the payload for binary ops
                        numArgs = 2;
                    }

                    // Create the function: (i32, i32) -> i32
                    // We always pass 2 args for binary, 1 for unary, and return 1.
                    // The values are bit-cast inside if needed.
                    if (!module.lookupSymbol(funcName)) {
                        OpBuilder b(module.getBodyRegion());
                        auto fType = LLVM::LLVMFunctionType::get(i32Type, SmallVector<Type>(numArgs, i32Type));
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

                        // Parse the snippet and insert it
                        std::string fullSnippet = "module {\n";
                        for (auto &decl : existingDecls) fullSnippet += decl + "\n";
                        fullSnippet += "func.func @temp(";
                        for (unsigned i = 0; i < argNamesList.size(); ++i) {
                            // Heuristic: check if name indicates type
                            std::string type = "i32";
                            if (argNamesList[i].find("f") != std::string::npos) {
                                if (argNamesList[i].find("64") != std::string::npos) type = "f64";
                                else type = "f32";
                            } else if (argNamesList[i].find("64") != std::string::npos) {
                                type = "i64";
                            }
                            
                            fullSnippet += argNamesList[i] + " : " + type;
                            if (i < argNamesList.size() - 1) fullSnippet += ", ";
                        }
                        fullSnippet += ") {\n";
                        fullSnippet += p.getValue().str();
                        fullSnippet += "\n  func.return\n";
                        fullSnippet += "}\n";
                        fullSnippet += "}";
 
                        OwningOpRef<ModuleOp> tempModule = parseSourceString<ModuleOp>(fullSnippet, module.getContext());
                        if (tempModule) {
                            func::FuncOp tempFunc = tempModule->lookupSymbol<func::FuncOp>("temp");
                            if (tempFunc && !tempFunc.getBody().empty()) {
                                IRMapping mapper;
                                for (unsigned i = 0; i < argNamesList.size(); ++i) {
                                    if (i < f.getNumArguments() && i < tempFunc.getNumArguments()) {
                                        mapper.map(tempFunc.getArgument(i), f.getArgument(i));
                                    }
                                }
                                OpBuilder bBody = OpBuilder::atBlockBegin(fEntry);
                                for (auto &op : tempFunc.getBody().front().getOperations()) {
                                    if (!isa<func::ReturnOp>(op)) {
                                        bBody.clone(op, mapper);
                                    }
                                }
                            }
                        }
                        // Always ensure a terminator
                        if (fEntry->empty() || !fEntry->back().hasTrait<OpTrait::IsTerminator>()) {
                            OpBuilder bTerm = OpBuilder::atBlockEnd(fEntry);
                            bTerm.create<LLVM::ReturnOp>(module.getLoc(), bTerm.create<LLVM::ConstantOp>(module.getLoc(), i32Type, bTerm.getI32IntegerAttr(0)));
                        }
                    }
                    userOps.push_back({opcodeForLabel(label), label, funcName, (int)numArgs});
                }
            }
        }
    for (auto op : regOps) op.erase();

    builder.setInsertionPointToStart(module.getBody());
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
        auto fType = LLVM::LLVMFunctionType::get(i32Type, {i32Type, i32Type, i32Type});
        auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "dispatch_user_op", fType);
        Block *entry = f.addEntryBlock();
        Value hash = entry->getArgument(0);
        Value arg0 = entry->getArgument(1);
        Value arg1 = entry->getArgument(2);

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
            Value res;
            if (op.numArgs == 1) {
                res = ob.create<LLVM::CallOp>(module.getLoc(), i32Type, op.funcName, ValueRange{arg0}).getResult();
            } else {
                res = ob.create<LLVM::CallOp>(module.getLoc(), i32Type, op.funcName, ValueRange{arg0, arg1}).getResult();
            }
            ob.create<LLVM::ReturnOp>(module.getLoc(), res);
        }
        OpBuilder eb(entry, entry->end());
        eb.create<LLVM::SwitchOp>(module.getLoc(), hash, defaultBlock, ValueRange{}, caseValues, caseBlocks, caseOperands);
    };
    genDispatcher();

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
            Value n = ob.create<LLVM::ConstantOp>(module.getLoc(), i32Type, ob.getI32IntegerAttr(op.numArgs));
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
        Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
        Value nIdx64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, nIdx);
        Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(3)));
        return builder.create<LLVM::LoadOp>(module.getLoc(), i32Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
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
        Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
        Value nIdx64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, nIdx);
        Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(i)));
        return builder.create<LLVM::LoadOp>(module.getLoc(), i32Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{offset}));
    };
    auto linkPorts = [&](Value port1, Value port2, Value as) {
        Value net = getArg(as, 0); Value q = getArg(as, 1); Value tl = getArg(as, 3);
        auto setTarget = [&](Value p1, Value p2) {
            Value nIdx = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, p1, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
            Value pNum = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, p1, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(3)));
            Value nIdx64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, nIdx);
            Value pNum64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, pNum);
            Value offset = builder.create<LLVM::AddOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2))), pNum64);
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
    Value polA = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(30)));
    Value polB = builder.create<LLVM::LShrOp>(module.getLoc(), i32Type, metaB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(30)));
    Value labelA = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, metaA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));
    Value labelB = builder.create<LLVM::AndOp>(module.getLoc(), i32Type, metaB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF)));

    Block *specBlock = wFunc.addBlock(), *genBlock = wFunc.addBlock();
    
    // Check for specialized rules (era, num)
    Value isEraA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x979115)));
    Value isEraB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x979115)));
    Value isNumA = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelA, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x725db3)));
    Value isNumB = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, labelB, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x725db3)));
    
    Value isSpec = builder.create<LLVM::OrOp>(module.getLoc(), 
        builder.create<LLVM::OrOp>(module.getLoc(), isEraA, isEraB),
        builder.create<LLVM::OrOp>(module.getLoc(), isNumA, isNumB));
    
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
                Value bTarget = builder.create<LLVM::LoadOp>(module.getLoc(), i32Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, bundleP)}));
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
                Value nIdx64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, nIdx);
                Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                Value metaVal = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, pol, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(30))), label);
                
                auto makePort = [&](int p) {
                    return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                };

                builder.create<LLVM::StoreOp>(module.getLoc(), makePort(0), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{base}));
                builder.create<LLVM::StoreOp>(module.getLoc(), a1, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)))}));
                builder.create<LLVM::StoreOp>(module.getLoc(), a2, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)))}));
                builder.create<LLVM::StoreOp>(module.getLoc(), metaVal, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(3)))}));
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
        Block *eraRule = wFunc.addBlock(), *numRule = wFunc.addBlock();
        Value hasEra = builder.create<LLVM::OrOp>(module.getLoc(), isEraA, isEraB);
        builder.create<LLVM::CondBrOp>(module.getLoc(), hasEra, eraRule, numRule);

        builder.setInsertionPointToStart(eraRule);
        {
            Value eraP = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, pA, pB);
            Value otherP = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, pB, pA);
            Value otherLabel = builder.create<LLVM::SelectOp>(module.getLoc(), isEraA, labelB, labelA);
            Value otherIsEra = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, otherLabel, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x979115)));
            Value otherIsNum = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, otherLabel, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x725db3)));
            Value isAnn = builder.create<LLVM::OrOp>(module.getLoc(), otherIsEra, otherIsNum);
            Block *doAnn = wFunc.addBlock(), *doProp = wFunc.addBlock();
            builder.create<LLVM::CondBrOp>(module.getLoc(), isAnn, doAnn, doProp);
            builder.setInsertionPointToStart(doAnn);
            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            builder.setInsertionPointToStart(doProp);
            {
                auto makeEra = [&]() {
                    Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                    Value nIdx64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, nIdx);
                    Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                    auto store = [&](int i, Value v) {
                        Value off = builder.create<LLVM::AddOp>(module.getLoc(), i64Type, base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                        builder.create<LLVM::StoreOp>(module.getLoc(), v, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{off}));
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

        builder.setInsertionPointToStart(numRule);
        {
            Value numP = builder.create<LLVM::SelectOp>(module.getLoc(), isNumA, pA, pB);
            Value otherP = builder.create<LLVM::SelectOp>(module.getLoc(), isNumA, pB, pA);
            Value otherLabel = builder.create<LLVM::SelectOp>(module.getLoc(), isNumA, labelB, labelA);
            Value nVal = getAux(numP, 1);

            Value isPrint = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, otherLabel, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xc9cbf5)));
            Block *doPrint = wFunc.addBlock(), *doUserOp = wFunc.addBlock();
            builder.create<LLVM::CondBrOp>(module.getLoc(), isPrint, doPrint, doUserOp);

            builder.setInsertionPointToStart(doPrint);
            {
                if (!module.lookupSymbol("PRINT_I32_FMT")) {
                    OpBuilder b(module.getBodyRegion());
                    std::string fmt = "%d\n";
                    fmt.push_back('\0');
                    b.create<LLVM::GlobalOp>(module.getLoc(), LLVM::LLVMArrayType::get(builder.getI8Type(), fmt.size()), true, LLVM::Linkage::Internal, "PRINT_I32_FMT", builder.getStringAttr(fmt));
                }
                Value fmtPtr = builder.create<LLVM::AddressOfOp>(module.getLoc(), ptrType, "PRINT_I32_FMT");
                builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "printf", ValueRange{fmtPtr, nVal});
                linkPorts(getAux(otherP, 2), numP, wState);
                builder.create<LLVM::BrOp>(module.getLoc(), lHead);
            }

            builder.setInsertionPointToStart(doUserOp);
            {
                Value isBranch = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, otherLabel, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x873945)));
                Block *doBranch = wFunc.addBlock(), *doGeneral = wFunc.addBlock();
                builder.create<LLVM::CondBrOp>(module.getLoc(), isBranch, doBranch, doGeneral);

                builder.setInsertionPointToStart(doBranch);
                {
                    Value cond = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::ne, nVal, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0)));
                    Value taken = builder.create<LLVM::SelectOp>(module.getLoc(), cond, getAux(otherP, 1), getAux(otherP, 2));
                    Value untaken = builder.create<LLVM::SelectOp>(module.getLoc(), cond, getAux(otherP, 2), getAux(otherP, 1));
                    auto makeEra = [&]() {
                        Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                        Value nIdx64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, nIdx);
                        Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                        auto store = [&](int i, Value v) {
                            Value off = builder.create<LLVM::AddOp>(module.getLoc(), i64Type, base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                            builder.create<LLVM::StoreOp>(module.getLoc(), v, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{off}));
                        };
                        auto mP = [&](int p) {
                            return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                        };
                        store(0, mP(0)); store(1, mP(1)); store(2, mP(2));
                        store(3, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr((2u << 30) | 0x979115u)));
                        return mP(0);
                    };
                    linkPorts(untaken, makeEra(), wState);
                    linkPorts(taken, getAux(otherP, 0), wState);
                    builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                }

                builder.setInsertionPointToStart(doGeneral);
                {
                    Value nArgs = builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "get_num_args", ValueRange{otherLabel}).getResult();
                    Value isUnary = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, nArgs, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)));
                    Value isBinary = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, nArgs, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                    
                    Block *doUnary = wFunc.addBlock(), *checkBinary = wFunc.addBlock(), *doComm = wFunc.addBlock();
                    builder.create<LLVM::CondBrOp>(module.getLoc(), isUnary, doUnary, checkBinary);

                    builder.setInsertionPointToStart(doUnary);
                    {
                        Value resVal = builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "dispatch_user_op", ValueRange{otherLabel, nVal, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0))}).getResult();
                        auto makeNum = [&](Value v) {
                            Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                            Value nIdx64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, nIdx);
                            Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                            auto store = [&](int i, Value val) {
                                Value off = builder.create<LLVM::AddOp>(module.getLoc(), i64Type, base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                                builder.create<LLVM::StoreOp>(module.getLoc(), val, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{off}));
                            };
                            auto mP = [&](int p) {
                                return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                            };
                            store(0, mP(0)); store(1, v); store(2, mP(2));
                            store(3, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr((2u << 30) | 0x725db3u)));
                            return mP(0);
                        };
                        linkPorts(getAux(otherP, 2), makeNum(resVal), wState);
                        builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                    }

                    builder.setInsertionPointToStart(checkBinary);
                    Block *doBinaryBlock = wFunc.addBlock();
                    builder.create<LLVM::CondBrOp>(module.getLoc(), isBinary, doBinaryBlock, doComm);
                    
                    builder.setInsertionPointToStart(doBinaryBlock);
                    {
                        Value rPort = getAux(otherP, 1);
                        Value rTarget = builder.create<LLVM::LoadOp>(module.getLoc(), i32Type, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, rPort)}));
                        Value rMeta = getMeta(rTarget);
                        Value isRNum = builder.create<LLVM::ICmpOp>(module.getLoc(), LLVM::ICmpPredicate::eq, builder.create<LLVM::AndOp>(module.getLoc(), i32Type, rMeta, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0xFFFFFF))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(0x725db3)));
                        
                        Block *doFullBinary = wFunc.addBlock();
                        builder.create<LLVM::CondBrOp>(module.getLoc(), isRNum, doFullBinary, doComm);
                        
                        builder.setInsertionPointToStart(doFullBinary);
                        {
                            Value v1 = getAux(rTarget, 1);
                            Value resVal = builder.create<LLVM::CallOp>(module.getLoc(), i32Type, "dispatch_user_op", ValueRange{otherLabel, v1, nVal}).getResult();
                            auto makeNum = [&](Value v) {
                                Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                                Value nIdx64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, nIdx);
                                Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                                auto store = [&](int i, Value val) {
                                    Value off = builder.create<LLVM::AddOp>(module.getLoc(), i64Type, base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                                    builder.create<LLVM::StoreOp>(module.getLoc(), val, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{off}));
                                };
                                auto mP = [&](int p) {
                                    return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                                };
                                store(0, mP(0)); store(1, v); store(2, mP(2));
                                store(3, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr((2u << 30) | 0x725db3u)));
                                return mP(0);
                            };
                            linkPorts(getAux(otherP, 2), makeNum(resVal), wState);
                            builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                        }
                    }

                    builder.setInsertionPointToStart(doComm);
                    {
                        Value auxA1 = getAux(numP, 1); Value auxA2 = getAux(numP, 2);
                        Value auxB1 = getAux(otherP, 1); Value auxB2 = getAux(otherP, 2);
                        auto allocNode = [&](Value pol, Value label, Value a1, Value a2) {
                            Value nIdx = builder.create<LLVM::AtomicRMWOp>(module.getLoc(), LLVM::AtomicBinOp::add, al, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(1)), LLVM::AtomicOrdering::seq_cst);
                            Value nIdx64 = builder.create<LLVM::ZExtOp>(module.getLoc(), i64Type, nIdx);
                            Value base = builder.create<LLVM::ShlOp>(module.getLoc(), i64Type, nIdx64, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)));
                            Value metaVal = builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, pol, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(30))), label);
                            auto mP = [&](int p) {
                                return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, nIdx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                            };
                            builder.create<LLVM::StoreOp>(module.getLoc(), mP(0), builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{base}));
                            builder.create<LLVM::StoreOp>(module.getLoc(), a1, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(1)))}));
                            builder.create<LLVM::StoreOp>(module.getLoc(), a2, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(2)))}));
                            builder.create<LLVM::StoreOp>(module.getLoc(), metaVal, builder.create<LLVM::GEPOp>(module.getLoc(), ptrType, i32Type, wNet, ValueRange{builder.create<LLVM::AddOp>(module.getLoc(), base, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(3)))}));
                            return nIdx;
                        };
                        auto mP = [&](Value idx, int p) {
                            return builder.create<LLVM::OrOp>(module.getLoc(), builder.create<LLVM::ShlOp>(module.getLoc(), i32Type, idx, builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(2))), builder.create<LLVM::ConstantOp>(module.getLoc(), i32Type, builder.getI32IntegerAttr(p)));
                        };
                        Value a1 = allocNode(polA, labelA, auxA1, auxA2); Value a2 = allocNode(polA, labelA, auxA1, auxA2);
                        Value b1 = allocNode(polB, labelB, auxB1, auxB2); Value b2 = allocNode(polB, labelB, auxB1, auxB2);
                        linkPorts(mP(a1, 1), mP(b1, 1), wState); linkPorts(mP(a1, 2), mP(b2, 1), wState);
                        linkPorts(mP(a2, 1), mP(b1, 2), wState); linkPorts(mP(a2, 2), mP(b2, 2), wState);
                        linkPorts(mP(a1, 0), auxB1, wState); linkPorts(mP(a2, 0), auxB2, wState);
                        linkPorts(mP(b1, 0), auxA1, wState); linkPorts(mP(b2, 0), auxA2, wState);
                        builder.create<LLVM::BrOp>(module.getLoc(), lHead);
                    }
                }
            }
        }
    }

    builder.setInsertionPointToStart(lEnd);
    builder.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{builder.create<LLVM::IntToPtrOp>(module.getLoc(), ptrType, builder.create<LLVM::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0)))});

    module.walk([&](func::FuncOp f) {
        if (f.getSymName() == "worker_thread" || f.empty()) return;
        OpBuilder b(&f.getBody().front(), f.getBody().front().begin());
        Value stateArg;
        if (f.getSymName() == "worker_thread") {
            stateArg = f.getArgument(0);
        } else if (f.getSymName() == "main_inet_entry") {
            f.setType(FunctionType::get(module.getContext(), {i64Type}, {}));
            stateArg = f.getBody().front().addArgument(i64Type, f.getLoc());
            f.walk([](func::ReturnOp op) {
                if (op.getNumOperands() > 0) op->setOperands({});
            });
        } else {
            stateArg = f.getArgument(2);
            stateArg = b.create<LLVM::ZExtOp>(f.getLoc(), i64Type, stateArg);
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
            Value nIdx64 = ob.create<LLVM::ZExtOp>(o->getLoc(), i64Type, nIdx);
            Value base = ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2)));
            
            auto store = [&](int i, Value v) {
                Value off = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, base, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(i)));
                ob.create<LLVM::StoreOp>(o->getLoc(), v, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{off}));
            };
            
            auto makePort = [&](int p) {
                return ob.create<LLVM::OrOp>(o->getLoc(), ob.create<LLVM::ShlOp>(o->getLoc(), i32Type, nIdx, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(p)));
            };
            
            store(0, makePort(0)); store(1, makePort(1)); store(2, makePort(2)); store(3, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(0)));

            o->getResult(0).replaceAllUsesWith(nIdx);
            o->erase();
        }
        for (auto* o : setPorts) {
            OpBuilder ob(o);
            Value nIdx = o->getOperand(0);
            int pIdx = o->getAttrOfType<IntegerAttr>("port_index").getInt();
            Value val = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, o->getOperand(1));
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg), ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            Value nIdx64 = ob.create<LLVM::ZExtOp>(o->getLoc(), i64Type, nIdx);
            Value offset = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, nIdx64, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(pIdx)));
            ob.create<LLVM::StoreOp>(o->getLoc(), val, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{offset}));
            o->erase();
        }
        for (auto* o : links) {
            OpBuilder ob(o);
            Value p1 = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, o->getOperand(0));
            Value p2 = ob.create<LLVM::TruncOp>(o->getLoc(), i32Type, o->getOperand(1));
            Value as = ob.create<LLVM::IntToPtrOp>(o->getLoc(), ptrType, stateArg);
            Value net = ob.create<LLVM::LoadOp>(o->getLoc(), ptrType, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, ptrType, as, ValueRange{ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(0))}));
            auto setT = [&](Value v1, Value v2) {
                Value nIdx = ob.create<LLVM::LShrOp>(o->getLoc(), i32Type, v1, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(2)));
                Value pNum = ob.create<LLVM::AndOp>(o->getLoc(), i32Type, v1, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(3)));
                Value offset = ob.create<LLVM::AddOp>(o->getLoc(), i64Type, ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, ob.create<LLVM::ZExtOp>(o->getLoc(), i64Type, nIdx), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(2))), ob.create<LLVM::ZExtOp>(o->getLoc(), i64Type, pNum));
                ob.create<LLVM::StoreOp>(o->getLoc(), v2, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i32Type, net, ValueRange{offset}));
            };
            setT(p1, p2); setT(p2, p1);
            Value isP1 = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::eq, ob.create<LLVM::AndOp>(o->getLoc(), i32Type, p1, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(3))), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(0)));
            Value isP2 = ob.create<LLVM::ICmpOp>(o->getLoc(), LLVM::ICmpPredicate::eq, ob.create<LLVM::AndOp>(o->getLoc(), i32Type, p2, ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(3))), ob.create<LLVM::ConstantOp>(o->getLoc(), i32Type, builder.getI32IntegerAttr(0)));
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
            Value r = ob.create<LLVM::OrOp>(o->getLoc(), i64Type, ob.create<LLVM::ZExtOp>(o->getLoc(), i64Type, p1), ob.create<LLVM::ShlOp>(o->getLoc(), i64Type, ob.create<LLVM::ZExtOp>(o->getLoc(), i64Type, p2), ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(32))));
            ob.create<LLVM::StoreOp>(o->getLoc(), r, ob.create<LLVM::GEPOp>(o->getLoc(), ptrType, i64Type, q, ValueRange{curT}));
            ob.create<LLVM::StoreOp>(o->getLoc(), ob.create<LLVM::AddOp>(o->getLoc(), curT, ob.create<LLVM::ConstantOp>(o->getLoc(), i64Type, builder.getI64IntegerAttr(1))), tlPtr);
            ob.create<LLVM::BrOp>(o->getLoc(), cont);
            ob.setInsertionPointToStart(cont);
            o->erase();
        }
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
        Value alL = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        Value zero64 = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(0));
        builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, hd); builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, tl);
        builder.create<LLVM::StoreOp>(entry.getLoc(), builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(0)), alL);
        Value as = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(40))}).getResult();
        auto sA = [&](int i, Value v) { builder.create<LLVM::StoreOp>(entry.getLoc(), v, builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, ptrType, as, ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(i))})); };
        sA(0, net); sA(1, q); sA(2, hd); sA(3, tl); sA(4, alL);
        builder.create<func::CallOp>(entry.getLoc(), entry, ValueRange{builder.create<LLVM::PtrToIntOp>(entry.getLoc(), i64Type, as)});
        builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "worker_thread", ValueRange{as});
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
