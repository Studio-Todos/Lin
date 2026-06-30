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

constexpr uint8_t ALLOC_OP   = (2 << 6) | mlir::pic::runtime::NODE_OP;
constexpr uint8_t ALLOC_ERA  = (2 << 6) | mlir::pic::runtime::NODE_ERA;
constexpr uint8_t ALLOC_RVEC = (2 << 6) | mlir::pic::runtime::NODE_RVEC;
constexpr uint8_t ALLOC_LOG  = (2 << 6) | mlir::pic::runtime::NODE_LOG;

static const std::vector<std::pair<std::string, std::string>> kRuntimeDecls = {
    {"printf", "llvm.func @printf(!llvm.ptr, ...) -> i32"},
    {"putchar", "llvm.func @putchar(i32) -> i32"},
    {"getchar", "llvm.func @getchar() -> i32"},
    {"scanf", "llvm.func @scanf(!llvm.ptr, ...) -> i32"},
    {"malloc", "llvm.func @malloc(i64) -> !llvm.ptr"},
    {"free", "llvm.func @free(!llvm.ptr)"},
    {"strlen", "llvm.func @strlen(!llvm.ptr) -> i64"},
    {"strcmp", "llvm.func @strcmp(!llvm.ptr, !llvm.ptr) -> i32"},
    {"lin_print_str", "llvm.func @lin_print_str(i64) -> i64"},
    {"sin", "llvm.func @sin(f64) -> f64"},
    {"cos", "llvm.func @cos(f64) -> f64"},
    {"tan", "llvm.func @tan(f64) -> f64"},
    {"asin", "llvm.func @asin(f64) -> f64"},
    {"acos", "llvm.func @acos(f64) -> f64"},
    {"atan", "llvm.func @atan(f64) -> f64"},
    {"exp", "llvm.func @exp(f64) -> f64"},
    {"log", "llvm.func @log(f64) -> f64"},
    {"sqrt", "llvm.func @sqrt(f64) -> f64"},
    {"pow", "llvm.func @pow(f64, f64) -> f64"},
    {"fopen", "llvm.func @fopen(!llvm.ptr, !llvm.ptr) -> !llvm.ptr"},
    {"fclose", "llvm.func @fclose(!llvm.ptr) -> i32"},
    {"fgetc", "llvm.func @fgetc(!llvm.ptr) -> i32"},
    {"fputc", "llvm.func @fputc(i32, !llvm.ptr) -> i32"},
    {"exit", "llvm.func @exit(i32)"},
    {"abort", "llvm.func @abort()"},
    {"time", "llvm.func @time(!llvm.ptr) -> i64"},
    {"sleep", "llvm.func @sleep(i32) -> i32"},
    {"lin_write_ppm", "llvm.func @lin_write_ppm(i64, i64, i64) -> i64"},
};

using namespace mlir;

namespace {

static Value genInlineDispatch(OpBuilder &builder, Location loc, Value impl, Value val0, Value val1, Value stateArg, func::FuncOp funcOp, const std::vector<UserOp> &userOps, Value c0_i64, bool useInverse = false) {
    auto i32Type = builder.getI32Type();
    auto i64Type = builder.getI64Type();

    Block *mergeBlock = funcOp.addBlock();
    mergeBlock->addArgument(i32Type, loc);

    Block *currentBlock = builder.getInsertionBlock();

    for (const auto &op : userOps) {
        if (useInverse && op.inverseFuncName.empty()) continue;

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

        const std::string &fn = useInverse ? op.inverseFuncName : op.funcName;
        Value res = mb.create<func::CallOp>(loc, i64Type, fn, callArgs).getResult(0);
        Value res32 = mb.create<arith::TruncIOp>(loc, i32Type, res);
        mb.create<cf::BranchOp>(loc, mergeBlock, ValueRange{res32});

        currentBlock = nextBlock;
    }

    OpBuilder fb(currentBlock, currentBlock->end());
    Value c0_i32 = fb.create<arith::ConstantOp>(loc, i32Type, fb.getI32IntegerAttr(0));
    fb.create<cf::BranchOp>(loc, mergeBlock, ValueRange{c0_i32});

    builder.setInsertionPointToStart(mergeBlock);
    return mergeBlock->getArgument(0);
}

struct PicReduceLoweringPass : public PassWrapper<PicReduceLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicReduceLoweringPass)

  TargetBackend target = TargetBackend::CPU;
  PicReduceLoweringPass() : target(TargetBackend::CPU) {}
  PicReduceLoweringPass(TargetBackend target) : target(target) {}

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

    NamedAttrList gpuOpAttrs;

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
        StringAttr dispatchAttr = op->getAttrOfType<StringAttr>("dispatch");
        std::string forcedTarget = dispatchAttr ? dispatchAttr.getValue().str() : "";
        bool isGpuCapable = true;
        if (n && p) {
            std::string label = n.getValue().str();
            if (label.compare(0, 4, "STR_") != 0) {
                // Generate a specialized func.func wrapper for this mlir-op
                std::string funcName = "user_op_" + label;
                replaceAll(funcName, "-", "_");
                
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
                        if (!isValidMLIRType(mlirType)) {
                            mlirType = "i32";
                        }
                        argS += cleanName + " : " + mlirType;
                        if (i < argNamesList.size() - 1) argS += ", ";
                    }
                    
                    std::string pStr = p.getValue().str();
                    std::string snippetDecls = "";
                    auto addDecl = [&](const std::vector<std::pair<std::string, std::string>> &decls) {
                        for (auto &kv : decls) {
                            bool found = false;
                            for (auto &d : existingDecls) {
                                if (d.find("@" + kv.first + "(") != std::string::npos || d.find("@\"" + kv.first + "\"(") != std::string::npos) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) snippetDecls += kv.second + "\n";
                        }
                    };
                    addDecl(kRuntimeDecls);
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
                        if (!isValidMLIRType(mlirType)) {
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
                        module.emitError() << "Failed to parse MLIR snippet for user_op_"
                            << label << "\ntempModuleStr:\n" << tempModuleStr;
                        signalPassFailure();
                        return;
                    }

                    func::FuncOp tempFunc = parsedSnippet ? parsedSnippet->lookupSymbol<func::FuncOp>("temp") : nullptr;
                    isGpuCapable = true;
                    if (tempFunc) {
                        tempFunc.walk([&](Operation *tempOp) {
                            if (auto* dialect = tempOp->getDialect()) {
                                StringRef dialectName = dialect->getNamespace();
                                if (dialectName != "arith" && dialectName != "math" &&
                                    dialectName != "func" && dialectName != "cf" &&
                                    dialectName != "memref" && dialectName != "scf" &&
                                    dialectName != "vector" && dialectName != "index") {
                                    isGpuCapable = false;
                                }
                            }
                        });
                        if (isGpuCapable) {
                            std::string serialized;
                            llvm::raw_string_ostream os(serialized);
                            tempFunc->print(os);
                            os.flush();
                            gpuOpAttrs.append(funcName, builder.getStringAttr(serialized));
                        }
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
                SmallVector<std::string> targets = {"cpu"};
                if (isGpuCapable) targets.push_back("gpu");
                
                StringAttr invPAttr = op->getAttrOfType<StringAttr>("inverse_payload");
                std::string inverseFuncName;
                if (invPAttr) {
                    std::string invLabel = label;
                    std::string invFuncName = "inverse_user_op_" + invLabel;
                    replaceAll(invFuncName, "-", "_");
                    inverseFuncName = invFuncName;
                    
                    auto existingInv = module.lookupSymbol<func::FuncOp>(invFuncName);
                    if (!existingInv) {
                        OpBuilder bMod(module.getBodyRegion());
                        auto invType = builder.getFunctionType(SmallVector<Type>(numArgs, i64Type), i64Type);
                        auto invFunc = bMod.create<func::FuncOp>(module.getLoc(), invFuncName, invType);
                        invFunc.setPrivate();
                        
                        Block *invEntry = invFunc.addEntryBlock();
                        OpBuilder bInv(invEntry, invEntry->begin());
                        
                        std::string invPayload = invPAttr.getValue().str();
                        std::string invArgS = "";
                        for (unsigned i = 0; i < argNamesList.size(); ++i) {
                            std::string cleanName = argNamesList[i];
                            size_t underscorePos = cleanName.rfind('_');
                            if (underscorePos != std::string::npos && underscorePos > 0) {
                                cleanName = cleanName.substr(0, underscorePos);
                            }
                            std::string mlirType = argTypes[i];
                            if (!isValidMLIRType(mlirType)) mlirType = "i32";
                            invArgS += cleanName + " : " + mlirType;
                            if (i < argNamesList.size() - 1) invArgS += ", ";
                        }
                        
                        std::string invDecls = "";
                        for (auto &d : existingDecls) {
                            auto atPos = d.find("@");
                            if (atPos != std::string::npos) {
                                auto parenPos = d.find("(", atPos);
                                if (parenPos != std::string::npos) {
                                    std::string sym = d.substr(atPos, parenPos - atPos);
                                    if (invPayload.find(sym) != std::string::npos) invDecls += d + "\n";
                                }
                            }
                        }
                        
                        OwningOpRef<ModuleOp> invTempModule = ModuleOp::create(module.getLoc());
                        {
                            OpBuilder b(invTempModule->getBodyRegion());
                            IRMapping m;
                            for (auto &op : module.getBody()->getOperations()) {
                                if (isa<func::FuncOp>(op) || isa<LLVM::LLVMFuncOp>(op) || isa<LLVM::GlobalOp>(op)) {
                                    b.clone(op, m);
                                }
                            }
                        }
                        
                        std::string invSnippetDecls = "";
                        auto invAddDecl = [&](const std::vector<std::pair<std::string, std::string>> &decls) {
                            for (auto &kv : decls) {
                                bool found = false;
                                for (auto &d : existingDecls) {
                                    if (d.find("@" + kv.first + "(") != std::string::npos || d.find("@\"" + kv.first + "\"(") != std::string::npos) {
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found) invSnippetDecls += kv.second + "\n";
                            }
                        };
                        invAddDecl(kRuntimeDecls);
                        
                        // Coerce arguments to their declared types
                        for (unsigned i = 0; i < argNamesList.size(); ++i) {
                            if (i < invFunc.getNumArguments()) {
                                Value arg = invEntry->getArgument(i);
                                std::string cleanName = argNamesList[i];
                                size_t underscorePos = cleanName.rfind('_');
                                if (underscorePos != std::string::npos && underscorePos > 0) {
                                    cleanName = cleanName.substr(0, underscorePos);
                                }
                                std::string mlirType = argTypes[i];
                                if (!isValidMLIRType(mlirType)) mlirType = "i32";
                                if (mlirType == "f32") {
                                    Value trunc = bInv.create<arith::TruncIOp>(module.getLoc(), builder.getI32Type(), arg);
                                    arg = bInv.create<arith::BitcastOp>(module.getLoc(), builder.getF32Type(), trunc);
                                } else if (mlirType == "f64") {
                                    arg = bInv.create<arith::BitcastOp>(module.getLoc(), builder.getF64Type(), arg);
                                } else if (mlirType == "i1") {
                                    arg = bInv.create<arith::TruncIOp>(module.getLoc(), builder.getI1Type(), arg);
                                } else if (mlirType == "i32") {
                                    arg = bInv.create<arith::TruncIOp>(module.getLoc(), builder.getI32Type(), arg);
                                }
                                // i64 needs no coercion
                            }
                        }
                        
                        // Build temp module string with the inverse payload
                        std::string invModuleStr = "module attributes {llvm.data_layout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"} {\n";
                        invModuleStr += "  func.func @temp(" + invArgS + ") -> i64 {\n";
                        invModuleStr += invSnippetDecls;
                        invModuleStr += "    " + invPayload + "\n";
                        invModuleStr += "    func.return %0 : i64\n";
                        invModuleStr += "  }\n";
                        invModuleStr += "}\n";
                        
                        auto invSnippet = parseSourceString<ModuleOp>(invModuleStr, module.getContext());
                        if (!invSnippet) {
                            llvm::errs() << "PARSE ERROR: Failed to parse inverse payload for " << label << "\n";
                            llvm::errs() << "--- invModuleStr ---\n" << invModuleStr << "\n--- end ---\n";
                            abort();
                        }
                        
                        func::FuncOp invTempFunc = invSnippet->lookupSymbol<func::FuncOp>("temp");
                            if (invTempFunc && !invTempFunc.getBody().empty()) {
                                Block *invTempEntry = &invTempFunc.getBody().front();
                                bInv.setInsertionPointToStart(invEntry);
                                
                                IRMapping invMap;
                                for (unsigned i = 0; i < invFunc.getNumArguments(); ++i) {
                                    invMap.map(invTempEntry->getArgument(i), invEntry->getArgument(i));
                                }
                                
                                Value invResult = bInv.create<arith::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0));
                                for (auto &op : *invTempEntry) {
                                    if (!isa<func::ReturnOp>(op)) {
                                        Operation *cloned = bInv.clone(op, invMap);
                                        if (op.getNumResults() > 0) {
                                            invResult = invMap.lookupOrDefault(op.getResult(0));
                                        }
                                    }
                                }
                                bInv.create<func::ReturnOp>(module.getLoc(), invResult);
                            }
                        
                        if (invEntry->empty() || !invEntry->back().hasTrait<OpTrait::IsTerminator>()) {
                            OpBuilder bTerm = OpBuilder::atBlockEnd(invEntry);
                            Value zeroRet = bTerm.create<arith::ConstantOp>(module.getLoc(), i64Type, builder.getI64IntegerAttr(0));
                            bTerm.create<func::ReturnOp>(module.getLoc(), zeroRet);
                        }
                    }
                }
                
                userOps.push_back(UserOp(opcodeForLabel(label), label, funcName, (int)numArgs, argTypes, targets, forcedTarget, inverseFuncName));
            }
        }
    }

    if (!gpuOpAttrs.empty()) {
        module->setAttr("pic.gpu_user_ops", builder.getDictionaryAttr(gpuOpAttrs));
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
    for (const auto &lit : getTypeLabels(module)) {
        literalHashes.push_back(opcodeForLabel(lit));
    }
    module.walk([&](pic::graph::RegistryOp op) {
        StringRef key = op.getOpName();
        if (key.starts_with("STR_")) {
            literalHashes.push_back(opcodeForLabel(key));
        }
    });

    std::vector<uint32_t> wideTypeHashes;
    for (const auto &lit : getTypeLabels(module)) {
        if (is64BitLabel(lit))
            wideTypeHashes.push_back(opcodeForLabel(lit));
    }

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
          
          Value is64 = builder.create<arith::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));
          for (uint32_t hash : wideTypeHashes) {
              Value hashConst = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(hash));
              Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, label, hashConst);
              is64 = builder.create<arith::OrIOp>(loc, is64, cmp);
          }
          
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

      auto handleGpuDispatch = [&](Value nA, Value nB, Value sA) -> bool {
          auto flag = op->getAttrOfType<StringAttr>("dispatch_flag");
          if (flag && flag.getValue() == "gpu") {
              if (!module.lookupSymbol("pic_gpu_dispatch_helper")) {
                  OpBuilder mb(module.getBodyRegion());
                  auto helperTy = builder.getFunctionType({i32Type, i32Type, i64Type}, {});
                  auto helperFunc = mb.create<func::FuncOp>(loc, "pic_gpu_dispatch_helper", helperTy);
                  helperFunc.setPrivate();
              }
              builder.create<func::CallOp>(loc, TypeRange{}, "pic_gpu_dispatch_helper", ValueRange{nA, nB, sA});
              builder.create<cf::BranchOp>(loc, lHead);
              op->erase();
              branchOp->erase();
              return true;
          }
          return false;
      };

      if (auto rvecOp = dyn_cast<mlir::pic::reduce::ReverseVectorOp>(op)) {
          Value nodeA = rvecOp.getNodeA();
          Value nodeB = rvecOp.getNodeB();
          Value stateArg = rvecOp.getState();
          if (handleGpuDispatch(nodeA, nodeB, stateArg)) continue;

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

          // Inverse dispatch: if the reconstructed pair has a user-op with inverse
          // payload, call the inverse function to compute original values.
          Value metaRecA = builder.create<pic::runtime::GetPortOp>(loc, i32Type, recNodeA, builder.getI8IntegerAttr(3));
          Value metaRecB = builder.create<pic::runtime::GetPortOp>(loc, i32Type, recNodeB, builder.getI8IntegerAttr(3));
          Value typeRecA = builder.create<arith::ShRUIOp>(loc, metaRecA, c24_i32);
          Value typeRecB = builder.create<arith::ShRUIOp>(loc, metaRecB, c24_i32);
          Value ntA = builder.create<arith::AndIOp>(loc, typeRecA, c0x3F_i32);
          Value ntB = builder.create<arith::AndIOp>(loc, typeRecB, c0x3F_i32);
          Value lRecA = builder.create<arith::AndIOp>(loc, metaRecA, c0xFFFFFF_i32);
          Value lRecB = builder.create<arith::AndIOp>(loc, metaRecB, c0xFFFFFF_i32);
          Value impl = builder.create<func::CallOp>(loc, i32Type, "lookup_rule",
              ValueRange{ntA, lRecA, ntB, lRecB}).getResult(0);
          Value invResult = genInlineDispatch(builder, loc, impl,
              builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0)),
              builder.create<arith::ConstantOp>(loc, i64Type, builder.getI64IntegerAttr(0)),
              stateArg, funcOp, userOps, c0_i64, true);
          builder.create<pic::runtime::SetHistoryOp>(loc, tNode, builder.getI8IntegerAttr(1),
              builder.create<arith::ExtUIOp>(loc, i64Type, invResult));

          builder.create<cf::BranchOp>(loc, lHead);
          
          builder.setInsertionPointToStart(rvecStdBlock);
          Value r1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(ALLOC_RVEC), c0_i64, builder.getBoolAttr(false));
          Value r2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(ALLOC_RVEC), c0_i64, builder.getBoolAttr(false));
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
          if (handleGpuDispatch(nodeA, nodeB, stateArg)) continue;

          auto freeNode = [&](Value nodeIdx) {
              auto fcGlobal = builder.create<memref::GetGlobalOp>(loc, MemRefType::get({}, i32Type), "__pic_free_count");
              Value oldCount = builder.create<memref::AtomicRMWOp>(loc, i32Type, arith::AtomicRMWKind::addi,
                  builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(1)),
                  fcGlobal, ValueRange{});
              auto flGlobal = builder.create<memref::GetGlobalOp>(loc, MemRefType::get({8000000}, i32Type), "__pic_free_list");
              Value storeIdx = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), oldCount);
              builder.create<memref::StoreOp>(loc, nodeIdx, flGlobal, ValueRange{storeIdx});
          };

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
          freeNode(nodeA); freeNode(nodeB);
          builder.create<cf::BranchOp>(loc, lHead);

          builder.setInsertionPointToStart(doEraProp);
          Value otherIsLit = isLiteralLabel(builder, loc, otherLabel);
          Block *eraLitCase = funcOp.addBlock();
          Block *eraNormalProp = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, otherIsLit, eraLitCase, eraNormalProp);

          builder.setInsertionPointToStart(eraLitCase);
          freeNode(nodeA); freeNode(nodeB);
          builder.create<cf::BranchOp>(loc, lHead);

          builder.setInsertionPointToStart(eraNormalProp);
          freeNode(nodeA); freeNode(nodeB);
          Value era1 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(ALLOC_ERA), c0_i64, builder.getBoolAttr(false));
          Value era2 = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(ALLOC_ERA), c0_i64, builder.getBoolAttr(false));
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
          if (handleGpuDispatch(nodeA, nodeB, stateArg)) continue;

          Value metaA = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeA, builder.getI8IntegerAttr(3));
          Value metaB = builder.create<pic::runtime::GetPortOp>(loc, i32Type, nodeB, builder.getI8IntegerAttr(3));
          Value labelA = builder.create<arith::AndIOp>(loc, metaA, c0xFFFFFF_i32);
          Value labelB = builder.create<arith::AndIOp>(loc, metaB, c0xFFFFFF_i32);
          Value polA = builder.create<arith::ShRUIOp>(loc, metaA, c30_i32);
          Value polB = builder.create<arith::ShRUIOp>(loc, metaB, c30_i32);

          Value typeValA = builder.create<arith::ShRUIOp>(loc, metaA, c24_i32);
          Value typeValB = builder.create<arith::ShRUIOp>(loc, metaB, c24_i32);
          Value nodeTypeA = builder.create<arith::AndIOp>(loc, typeValA, c0x3F_i32);
          Value nodeTypeB = builder.create<arith::AndIOp>(loc, typeValB, c0x3F_i32);

          if (!module.lookupSymbol("lookup_rule")) {
              OpBuilder mb(module.getBodyRegion());
              auto lookupTy = builder.getFunctionType({i32Type, i32Type, i32Type, i32Type}, i32Type);
              auto lookupFunc = mb.create<func::FuncOp>(loc, "lookup_rule", lookupTy);
              lookupFunc.setPrivate();
          }

          Value implA = builder.create<func::CallOp>(loc, i32Type, "lookup_rule", ValueRange{nodeTypeA, labelA, nodeTypeB, labelB}).getResult(0);
          Value implB = builder.create<func::CallOp>(loc, i32Type, "lookup_rule", ValueRange{nodeTypeB, labelB, nodeTypeA, labelA}).getResult(0);
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
          if (target == TargetBackend::GPU) {
              for (auto &gop : userOps) {
                  bool hasGpu = std::find(gop.targets.begin(), gop.targets.end(), "gpu") != gop.targets.end();
                  bool forcedGpu = (gop.forcedTarget == "gpu");
                  bool forcedOff = (gop.forcedTarget == "cpu");
                  if ((hasGpu || forcedGpu) && !forcedOff) {
                      Value hashConst = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(gop.hash));
                      Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, impl, hashConst);
                      isGpu = builder.create<arith::OrIOp>(loc, isGpu, cmp);
                  }
              }
          }

          Block *doUnary = funcOp.addBlock();
          Block *checkBinary = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isUnary, doUnary, checkBinary);

          builder.setInsertionPointToStart(doUnary);
          if (target == TargetBackend::GPU) {
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
          }
          Value v0_64 = loadLiteralVal(valNode, valLabel);
          Value resVal = genInlineDispatch(builder, loc, impl, v0_64, c0_i32, stateArg, funcOp, userOps, c0_i64);
          Value opP_aux2 = builder.create<pic::runtime::GetPortOp>(loc, i32Type, opNode, builder.getI8IntegerAttr(2));
          Value isSame = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, resVal, opP_aux2);
          Block *skipLink = funcOp.addBlock();
          Block *doLink = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isSame, skipLink, doLink);

          builder.setInsertionPointToStart(doLink);
          Value valLabel64 = builder.create<arith::ExtUIOp>(loc, i64Type, valLabel);
          Value resNode = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(ALLOC_OP), valLabel64, builder.getBoolAttr(false));
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

          if (target == TargetBackend::GPU) {
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
          }
          Value resValBin = genInlineDispatch(builder, loc, impl, callArg0_64, callArg1_64, stateArg, funcOp, userOps, c0_i64);
          Value callArg1_32 = builder.create<arith::SelectOp>(loc, isCall, valNode_aux2, v0);
          Value isSameBin = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, resValBin, callArg1_32);
          Block *skipLinkBin = funcOp.addBlock();
          Block *doLinkBin = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isSameBin, skipLinkBin, doLinkBin);

          builder.setInsertionPointToStart(doLinkBin);
          Value valLabelBin64 = builder.create<arith::ExtUIOp>(loc, i64Type, valLabel);
          Value resNodeBin = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(ALLOC_OP), valLabelBin64, builder.getBoolAttr(false));
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
          if (handleGpuDispatch(nodeA, nodeB, stateArg)) continue;

          Value hWord0A = builder.create<pic::runtime::GetHistoryOp>(loc, i64Type, nodeA, builder.getI8IntegerAttr(0));
          Value inBoundaryA = builder.create<arith::AndIOp>(loc, hWord0A, c0x100000000_i64);
          Value isBoundaryA = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, inBoundaryA, c0_i64);

          Block *depLBlock = funcOp.addBlock();
          Block *skipLBlock = funcOp.addBlock();
          builder.create<cf::CondBranchOp>(loc, isBoundaryA, depLBlock, skipLBlock);

          builder.setInsertionPointToStart(depLBlock);
          Value lIdx = builder.create<pic::runtime::AllocNodeOp>(loc, i32Type, builder.getI8IntegerAttr(ALLOC_LOG), c0_i64, builder.getBoolAttr(false));
          Value lIdx64 = builder.create<arith::ExtUIOp>(loc, i64Type, lIdx);
          
          builder.create<pic::runtime::SetPortOp>(loc, lIdx, builder.getI8IntegerAttr(1), nodeA);
          builder.create<pic::runtime::SetPortOp>(loc, lIdx, builder.getI8IntegerAttr(2), nodeB);
          
          builder.create<pic::runtime::SetPortOp>(loc, lIdx, builder.getI8IntegerAttr(3), builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr((int32_t)((uint32_t)ALLOC_LOG << 24))));

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
          if (handleGpuDispatch(nodeA, nodeB, stateArg)) continue;

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
          Value is64A = builder.create<arith::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));
          for (uint32_t hash : wideTypeHashes) {
              Value hashConst = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(hash));
              Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelA, hashConst);
              is64A = builder.create<arith::OrIOp>(loc, is64A, cmp);
          }
          
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
          Value is64B = builder.create<arith::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));
          for (uint32_t hash : wideTypeHashes) {
              Value hashConst = builder.create<arith::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(hash));
              Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, labelB, hashConst);
              is64B = builder.create<arith::OrIOp>(loc, is64B, cmp);
          }

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

} // namespace

std::unique_ptr<Pass> createPicReduceLoweringPass(TargetBackend target) { return std::make_unique<PicReduceLoweringPass>(target); }
