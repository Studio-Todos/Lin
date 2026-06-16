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
#include "PicReduceUtils.h"
#include <set>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <map>

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
