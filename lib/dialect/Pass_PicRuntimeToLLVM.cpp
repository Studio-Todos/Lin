#include "lin/LoweringPasses.h"
#include "PicGraphDialect.h"
#include "PicReduceDialect.h"
#include "PicRuntimeDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/IRMapping.h"
#include "PicRuntimeToLLVMConversions.h"

using namespace mlir;
using namespace mlir::pic::runtime;

namespace {

struct PicRuntimeToLLVMPass : public PassWrapper<PicRuntimeToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PicRuntimeToLLVMPass)
  TargetBackend target;
  std::string spirvPath;
  PicRuntimeToLLVMPass(TargetBackend target, std::string spirvPath) : target(target), spirvPath(spirvPath) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    
    std::vector<UserOp> userOps;
    for (auto func : module.getOps<func::FuncOp>()) {
        if (auto labelAttr = func->getAttrOfType<StringAttr>("lin.original_label")) {
            std::string label = labelAttr.getValue().str();
            userOps.push_back({opcodeForLabel(label), label, func.getSymName().str(), (int)func.getNumArguments(), {}});
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
        
        SmallVector<Operation*> nodes, setPorts, getPorts, getPortsDynamic, links, pushRedexs, popRedexs, getHistorys, setHistorys, uncomputeSweeps, checkpoints;
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
            else if (opName == "pic_runtime.checkpoint_boundary") checkpoints.push_back(o);
        });

        // ============================================================
        // Mechanical conversions: pic_runtime ops -> LLVM dialect ops
        // Each conversion is a standalone function in
        // PicRuntimeToLLVMConversions.h — making this pass a thin
        // orchestrator that dispatches to the pattern library.
        //
        // The conversion functions no longer erase ops — the caller is
        // responsible for replacement and erasure so that both the
        // direct-dispatch path (here) and the RewritePattern path
        // (PicRuntimeToLLVMConversionPatterns.h) can use them.
        // ============================================================
        for (auto* o : nodes) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::AllocNodeOp>(o);
            Value result = convertAllocNodeOp(ob, op, stateArg, f);
            op.getResult().replaceAllUsesWith(result);
            op.erase();
        }
        for (auto* o : setPorts) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::SetPortOp>(o);
            convertSetPortOp(ob, op, stateArg);
            op.erase();
        }
        for (auto* o : getPorts) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::GetPortOp>(o);
            Value result = convertGetPortOp(ob, op, stateArg);
            op.getResult().replaceAllUsesWith(result);
            op.erase();
        }
        for (auto* o : getPortsDynamic) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::GetPortDynamicOp>(o);
            Value result = convertGetPortDynamicOp(ob, op, stateArg);
            op.getResult().replaceAllUsesWith(result);
            op.erase();
        }
        for (auto* o : links) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::LinkOp>(o);
            convertLinkOp(ob, op, stateArg, f);
            op.erase();
        }
        for (auto* o : pushRedexs) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::PushRedexOp>(o);
            convertPushRedexOp(ob, op, stateArg);
            op.erase();
        }
        for (auto* o : popRedexs) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::PopRedexOp>(o);
            auto results = convertPopRedexOp(ob, op, stateArg, f);
            op.getResult(0).replaceAllUsesWith(results[0]);
            op.getResult(1).replaceAllUsesWith(results[1]);
            op.getResult(2).replaceAllUsesWith(results[2]);
            op.erase();
        }
        for (auto* o : getHistorys) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::GetHistoryOp>(o);
            Value result = convertGetHistoryOp(ob, op, stateArg);
            op.getResult().replaceAllUsesWith(result);
            op.erase();
        }
        for (auto* o : setHistorys) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::SetHistoryOp>(o);
            convertSetHistoryOp(ob, op, stateArg);
            op.erase();
        }
        for (auto* o : uncomputeSweeps) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::UncomputeSweepOp>(o);
            convertUncomputeSweepOp(ob, op, stateArg, f);
            op.erase();
        }
        for (auto* o : checkpoints) {
            OpBuilder ob(o);
            auto op = cast<pic::runtime::CheckpointBoundaryOp>(o);
            convertCheckpointBoundaryOp(ob, op, stateArg);
            op.erase();
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
            auto tblTy = LLVM::LLVMArrayType::get(i32Type, kRuleTableSize * 5);
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
            auto fType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(builder.getContext()), {i32Type, i32Type, i32Type, i32Type, i32Type});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "register_rule", fType);
            Block *entry = f.addEntryBlock();
            OpBuilder eb(entry, entry->end());
            Location loc = module.getLoc();
            Value typeA = entry->getArgument(0);
            Value keyA = entry->getArgument(1);
            Value typeB = entry->getArgument(2);
            Value keyB = entry->getArgument(3);
            Value impl = entry->getArgument(4);
            Value cntPtr = eb.create<LLVM::AddressOfOp>(loc, ptrType, "__pic_rule_count");
            Value cnt = eb.create<LLVM::LoadOp>(loc, i32Type, cntPtr);
            Value maxN = eb.create<LLVM::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(kRuleTableSize));
            Value full = eb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, cnt, maxN);
            Block *overflow = f.addBlock();
            Block *store  = f.addBlock();
            eb.create<LLVM::CondBrOp>(loc, full, overflow, store);
            OpBuilder ob(overflow, overflow->end());
            ob.create<LLVM::ReturnOp>(loc, ValueRange{});
            OpBuilder sb(store, store->end());
            Value tblPtr = sb.create<LLVM::AddressOfOp>(loc, ptrType, "__pic_rule_table");
            Value five = sb.create<LLVM::ConstantOp>(loc, i32Type, sb.getI32IntegerAttr(5));
            Value base  = sb.create<LLVM::MulOp>(loc, i32Type, cnt, five);
            Value base64 = safeZExt(sb, loc, i64Type, base);
            Value off1   = sb.create<LLVM::AddOp>(loc, i64Type, base64, sb.create<LLVM::ConstantOp>(loc, i64Type, sb.getI64IntegerAttr(1)));
            Value off2   = sb.create<LLVM::AddOp>(loc, i64Type, base64, sb.create<LLVM::ConstantOp>(loc, i64Type, sb.getI64IntegerAttr(2)));
            Value off3   = sb.create<LLVM::AddOp>(loc, i64Type, base64, sb.create<LLVM::ConstantOp>(loc, i64Type, sb.getI64IntegerAttr(3)));
            Value off4   = sb.create<LLVM::AddOp>(loc, i64Type, base64, sb.create<LLVM::ConstantOp>(loc, i64Type, sb.getI64IntegerAttr(4)));
            sb.create<LLVM::StoreOp>(loc, typeA, sb.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr, ValueRange{base64}));
            sb.create<LLVM::StoreOp>(loc, keyA, sb.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr, ValueRange{off1}));
            sb.create<LLVM::StoreOp>(loc, typeB, sb.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr, ValueRange{off2}));
            sb.create<LLVM::StoreOp>(loc, keyB, sb.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr, ValueRange{off3}));
            sb.create<LLVM::StoreOp>(loc, impl, sb.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr, ValueRange{off4}));
            Value newCnt = sb.create<LLVM::AddOp>(loc, i32Type, cnt, sb.create<LLVM::ConstantOp>(loc, i32Type, sb.getI32IntegerAttr(1)));
            sb.create<LLVM::StoreOp>(loc, newCnt, cntPtr);
            sb.create<LLVM::ReturnOp>(loc, ValueRange{});
        };
        genRegisterRule();
        
        auto genLookupRule = [&]() {
            OpBuilder b(module.getBodyRegion());
            auto fType = LLVM::LLVMFunctionType::get(i32Type, {i32Type, i32Type, i32Type, i32Type});
            auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "lookup_rule", fType);
            Block *entry = f.addEntryBlock();
            Value typeArgA = entry->getArgument(0);
            Value labelArgA = entry->getArgument(1);
            Value typeArgB = entry->getArgument(2);
            Value labelArgB = entry->getArgument(3);
            
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
                
                auto addRuleMatch = [&](uint32_t typeA, uint32_t keyOp, uint32_t typeB, uint32_t keyType, uint32_t valImpl) {
                    Block *currBlock = nextBlock;
                    Block *matchBlock = f.addBlock();
                    nextBlock = f.addBlock();
                    
                    OpBuilder cb(currBlock, currBlock->end());
                    Value cTypeA = cb.create<LLVM::ConstantOp>(loc, i32Type, cb.getI32IntegerAttr(typeA));
                    Value cLabelA = cb.create<LLVM::ConstantOp>(loc, i32Type, cb.getI32IntegerAttr(keyOp));
                    Value cTypeB = cb.create<LLVM::ConstantOp>(loc, i32Type, cb.getI32IntegerAttr(typeB));
                    Value cLabelB = cb.create<LLVM::ConstantOp>(loc, i32Type, cb.getI32IntegerAttr(keyType));
                    Value matchA1 = cb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, typeArgA, cTypeA);
                    Value matchA2 = cb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, labelArgA, cLabelA);
                    Value matchA = cb.create<LLVM::AndOp>(loc, matchA1, matchA2);
                    Value matchB1 = cb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, typeArgB, cTypeB);
                    Value matchB2 = cb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, labelArgB, cLabelB);
                    Value matchB = cb.create<LLVM::AndOp>(loc, matchB1, matchB2);
                    Value cond = cb.create<LLVM::AndOp>(loc, matchA, matchB);
                    cb.create<LLVM::CondBrOp>(loc, cond, matchBlock, nextBlock);
                    
                    OpBuilder mb(matchBlock, matchBlock->end());
                    mb.create<LLVM::ReturnOp>(loc, ValueRange{mb.create<LLVM::ConstantOp>(loc, i32Type, mb.getI32IntegerAttr(valImpl))});
                };
                
                if (!opName.empty() && !typeName.empty()) {
                    addRuleMatch(NODE_OP, opcodeForLabel(opName), NODE_OP, opcodeForLabel(typeName), opcodeForLabel(op.label));
                    addRuleMatch(NODE_OP, opcodeForLabel(op.label), NODE_OP, opcodeForLabel(typeName), opcodeForLabel(op.label));
                    if (!originalBase.empty() && originalBase[0] == 'f') {
                        addRuleMatch(NODE_OP, opcodeForLabel(originalBase), NODE_OP, opcodeForLabel(typeName), opcodeForLabel(op.label));
                    }
                    if (typeName == "i64") {
                        addRuleMatch(NODE_OP, opcodeForLabel(op.label), NODE_OP, opcodeForLabel("i32"), opcodeForLabel(op.label));
                    }
                }
                addRuleMatch(NODE_OP, opcodeForLabel(op.label), NODE_OP, opcodeForLabel("call"), opcodeForLabel(op.label));
            }
            
            Block *rtScanHead = nextBlock;
            Block *rtScanBody = f.addBlock();
            Block *rtFound   = f.addBlock();
            Block *rtMiss    = f.addBlock();

            {
                OpBuilder hb(rtScanHead, rtScanHead->end());
                rtScanBody->addArgument(i32Type, loc);
                Value zero32rt = hb.create<LLVM::ConstantOp>(loc, i32Type, hb.getI32IntegerAttr(0));
                hb.create<LLVM::BrOp>(loc, ValueRange{zero32rt}, rtScanBody);
            }
            {
                Value iVal = rtScanBody->getArgument(0);
                OpBuilder bb(rtScanBody, rtScanBody->end());
                Value cntPtr2 = bb.create<LLVM::AddressOfOp>(loc, ptrType, "__pic_rule_count");
                Value cntVal  = bb.create<LLVM::LoadOp>(loc, i32Type, cntPtr2);
                Value done    = bb.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, iVal, cntVal);
                Block *cmpBlk = f.addBlock();
                bb.create<LLVM::CondBrOp>(loc, done, rtMiss, cmpBlk);
                OpBuilder cb2(cmpBlk, cmpBlk->end());
                Value tblPtr2 = cb2.create<LLVM::AddressOfOp>(loc, ptrType, "__pic_rule_table");
                Value five2  = cb2.create<LLVM::ConstantOp>(loc, i32Type, cb2.getI32IntegerAttr(5));
                Value base2   = cb2.create<LLVM::MulOp>(loc, i32Type, iVal, five2);
                Value base64_2 = safeZExt(cb2, loc, i64Type, base2);
                Value off1_2   = cb2.create<LLVM::AddOp>(loc, i64Type, base64_2, cb2.create<LLVM::ConstantOp>(loc, i64Type, cb2.getI64IntegerAttr(1)));
                Value off2_2   = cb2.create<LLVM::AddOp>(loc, i64Type, base64_2, cb2.create<LLVM::ConstantOp>(loc, i64Type, cb2.getI64IntegerAttr(2)));
                Value off3_2   = cb2.create<LLVM::AddOp>(loc, i64Type, base64_2, cb2.create<LLVM::ConstantOp>(loc, i64Type, cb2.getI64IntegerAttr(3)));
                Value off4_2   = cb2.create<LLVM::AddOp>(loc, i64Type, base64_2, cb2.create<LLVM::ConstantOp>(loc, i64Type, cb2.getI64IntegerAttr(4)));
                Value entTypeA = cb2.create<LLVM::LoadOp>(loc, i32Type, cb2.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr2, ValueRange{base64_2}));
                Value entKeyA = cb2.create<LLVM::LoadOp>(loc, i32Type, cb2.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr2, ValueRange{off1_2}));
                Value entTypeB = cb2.create<LLVM::LoadOp>(loc, i32Type, cb2.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr2, ValueRange{off2_2}));
                Value entKeyB = cb2.create<LLVM::LoadOp>(loc, i32Type, cb2.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr2, ValueRange{off3_2}));
                Value entImpl = cb2.create<LLVM::LoadOp>(loc, i32Type, cb2.create<LLVM::GEPOp>(loc, ptrType, i32Type, tblPtr2, ValueRange{off4_2}));
                Value matchA1  = cb2.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, entTypeA, typeArgA);
                Value matchA2  = cb2.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, entKeyA, labelArgA);
                Value matchA   = cb2.create<LLVM::AndOp>(loc, matchA1, matchA2);
                Value matchB1  = cb2.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, entTypeB, typeArgB);
                Value matchB2  = cb2.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, entKeyB, labelArgB);
                Value matchB   = cb2.create<LLVM::AndOp>(loc, matchB1, matchB2);
                Value matched = cb2.create<LLVM::AndOp>(loc, matchA, matchB);
                rtFound->addArgument(i32Type, loc);
                Block *nextIter = f.addBlock();
                cb2.create<LLVM::CondBrOp>(loc, matched, rtFound, ValueRange{entImpl}, nextIter, ValueRange{});
                OpBuilder nb(nextIter, nextIter->end());
                Value iPlusOne = nb.create<LLVM::AddOp>(loc, i32Type, iVal, nb.create<LLVM::ConstantOp>(loc, i32Type, nb.getI32IntegerAttr(1)));
                nb.create<LLVM::BrOp>(loc, ValueRange{iPlusOne}, rtScanBody);
            }
            {
                OpBuilder fb(rtFound, rtFound->end());
                fb.create<LLVM::ReturnOp>(loc, ValueRange{rtFound->getArgument(0)});
            }
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

    auto genGpuDispatchHelper = [&]() {
        OpBuilder b(module.getBodyRegion());
        auto fType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(builder.getContext()), {i32Type, i32Type, i64Type});
        auto f = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "pic_gpu_dispatch_helper", fType);
        if (target != TargetBackend::GPU) return;

        auto *entry = f.addEntryBlock();
        Value nodeA = entry->getArgument(0);
        Value nodeB = entry->getArgument(1);
        Value dState = entry->getArgument(2);
        OpBuilder ob(entry, entry->end());

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

        ob.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{});
    };
    
    if (auto decl = module.lookupSymbol<func::FuncOp>("pic_gpu_dispatch_helper")) {
            decl.erase();
        }
    genGpuDispatchHelper();



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
        Value numThreadsConst = target == TargetBackend::GPU
            ? builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(1))
            : builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(4));
        builder.create<LLVM::StoreOp>(entry.getLoc(), numThreadsConst, activePtr);
        
        Value lockPtr = builder.create<LLVM::CallOp>(entry.getLoc(), ptrType, "malloc", ValueRange{builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(8))}).getResult();
        builder.create<LLVM::StoreOp>(entry.getLoc(), zero64, lockPtr);
        
        sA(0, net); sA(1, q); sA(2, hd); sA(3, tl); sA(4, alL); sA(5, history_net); sA(6, activePtr); sA(7, lockPtr);
        builder.create<func::CallOp>(entry.getLoc(), TypeRange{}, entry.getSymName(), ValueRange{builder.create<LLVM::PtrToIntOp>(entry.getLoc(), i64Type, as)});
        
        {
            Value numThreadsAlloca = target == TargetBackend::GPU
                ? builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(1))
                : builder.create<LLVM::ConstantOp>(entry.getLoc(), i32Type, builder.getI32IntegerAttr(4));
            Value threads = builder.create<LLVM::AllocaOp>(entry.getLoc(), ptrType, i64Type, numThreadsAlloca);
            Value threadAttr = builder.create<LLVM::ZeroOp>(entry.getLoc(), ptrType);
            auto wFuncType = builder.getFunctionType({i64Type}, {i64Type});
            auto wFuncConst = builder.create<func::ConstantOp>(entry.getLoc(), wFuncType, FlatSymbolRefAttr::get(builder.getContext(), "worker_thread"));
            Value wAddr = builder.create<UnrealizedConversionCastOp>(entry.getLoc(), TypeRange{ptrType}, ValueRange(static_cast<Value>(wFuncConst.getResult()))).getResult(0);
            
            int numThreads = target == TargetBackend::GPU ? 1 : 4;
            for (int i = 0; i < numThreads; ++i) {
                Value idx = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(i));
                Value tPtr = builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, i64Type, threads, ValueRange{idx});
                builder.create<LLVM::CallOp>(entry.getLoc(), i32Type, "pthread_create", ValueRange{tPtr, threadAttr, wAddr, as});
            }
            
            Value retValPtr = builder.create<LLVM::ZeroOp>(entry.getLoc(), ptrType);
            for (int i = 0; i < numThreads; ++i) {
                Value idx = builder.create<LLVM::ConstantOp>(entry.getLoc(), i64Type, builder.getI64IntegerAttr(i));
                Value tPtr = builder.create<LLVM::GEPOp>(entry.getLoc(), ptrType, i64Type, threads, ValueRange{idx});
                Value tVal = builder.create<LLVM::LoadOp>(entry.getLoc(), i64Type, tPtr);
                builder.create<LLVM::CallOp>(entry.getLoc(), i32Type, "pthread_join", ValueRange{tVal, retValPtr});
            }
        }
        if (target == TargetBackend::GPU) {
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
} // namespace

std::unique_ptr<Pass> createPicRuntimeToLLVMPass(TargetBackend target, std::string spirvPath) { return std::make_unique<PicRuntimeToLLVMPass>(target, spirvPath); }
