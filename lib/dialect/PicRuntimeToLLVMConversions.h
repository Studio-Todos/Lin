#ifndef PIC_RUNTIME_TO_LLVM_CONVERSIONS_H
#define PIC_RUNTIME_TO_LLVM_CONVERSIONS_H

#include "PicReduceUtils.h"
#include "PicRuntimeDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::pic::runtime;

// =========================================================================
// Helper functions used by the per-op conversion functions below.
// =========================================================================

static Value toIdx(OpBuilder &ob, Location loc, Value v) {
    return ob.create<arith::IndexCastOp>(loc, ob.getIndexType(), v);
}

// If the net slot at `idx` currently holds a link to another node (slot_flag==1),
// decrement that target's refcount (that link is dying), and clear the slot flag.
// Owner is the node index whose slot we are about to overwrite. Safe no-op when the
// slot holds a value (flag==0), points at self, is zero, or is out of bounds.
static void emitSlotRefcountClear(OpBuilder &ob, Location loc, Value idx, Value owner,
                                  Value netGlobal, Value slotFlagGlobal, Value refcountGlobal,
                                  Type i32Type) {
    Value oldVal = ob.create<memref::LoadOp>(loc, i32Type, netGlobal, ValueRange{idx});
    Value oldFlag = ob.create<memref::LoadOp>(loc, i32Type, slotFlagGlobal, ValueRange{idx});
    Value cOne = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(1));
    Value isLink = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, oldFlag, cOne);
    Value oldNode = ob.create<LLVM::LShrOp>(loc, i32Type, oldVal, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2)));
    Value cZero = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0));
    Value notSelf = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, oldNode, owner);
    Value nonZero = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, oldNode, cZero);
    Value inBounds = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, oldNode, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(16000000)));
    Value cond = ob.create<LLVM::AndOp>(loc, ob.getI1Type(), ob.create<LLVM::AndOp>(loc, ob.getI1Type(), isLink, notSelf), ob.create<LLVM::AndOp>(loc, ob.getI1Type(), nonZero, inBounds));
    Value decIdx = ob.create<LLVM::SelectOp>(loc, cond, oldNode, cZero);
    Value decAmt = ob.create<LLVM::SelectOp>(loc, cond, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(-1)), cZero);
    ob.create<memref::AtomicRMWOp>(loc, i32Type, arith::AtomicRMWKind::addi, decAmt, refcountGlobal, ValueRange{toIdx(ob, loc, decIdx)});
    // The slot now holds a value / self-port (not a link); clear the flag.
    ob.create<memref::StoreOp>(loc, cZero, slotFlagGlobal, ValueRange{idx});
}

static void genNonBarrierLink(OpBuilder &ob, Location loc, Value p1, Value p2, Value stateArg, func::FuncOp &f) {
    auto i32Type = ob.getI32Type();
    auto i64Type = ob.getI64Type();

    auto netGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_net");
    auto qGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({32000000}, i64Type), "__pic_queue");
    auto refcountGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i32Type), "__pic_refcount");
    auto slotFlagGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_slot_flag");
    auto cZero = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0));
    auto cOne = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(1));
    auto cNegOne = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(-1));
    auto cNodePool = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(16000000));

    auto setT = [&](Value v1, Value v2) {
        Value nIdx = ob.create<LLVM::LShrOp>(loc, i32Type, v1, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2)));
        Value pNum = ob.create<LLVM::AndOp>(loc, i32Type, v1, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(3)));
        Value offset = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, safeZExt(ob, loc, i64Type, nIdx), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2))), safeZExt(ob, loc, i64Type, pNum));
        Value idx = toIdx(ob, loc, offset);
        // Refcount: the overwritten slot loses its old reference, gains the new one.
        Value oldVal = ob.create<memref::LoadOp>(loc, i32Type, netGlobal, ValueRange{idx});
        Value oldFlag = ob.create<memref::LoadOp>(loc, i32Type, slotFlagGlobal, ValueRange{idx});
        Value oldNode = ob.create<LLVM::LShrOp>(loc, i32Type, oldVal, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2)));
        Value wasLink = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, oldFlag, cOne);
        Value oldNotSelf = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, oldNode, nIdx);
        Value oldNonZero = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, oldNode, cZero);
        Value oldInBounds = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, oldNode, cNodePool);
        Value decCond = ob.create<LLVM::AndOp>(loc, ob.getI1Type(), ob.create<LLVM::AndOp>(loc, ob.getI1Type(), wasLink, oldNotSelf), ob.create<LLVM::AndOp>(loc, ob.getI1Type(), oldNonZero, oldInBounds));
        Value decIdx = ob.create<LLVM::SelectOp>(loc, decCond, oldNode, cZero);
        Value decAmt = ob.create<LLVM::SelectOp>(loc, decCond, cNegOne, cZero);
        ob.create<memref::AtomicRMWOp>(loc, i32Type, arith::AtomicRMWKind::addi, decAmt, refcountGlobal, ValueRange{toIdx(ob, loc, decIdx)});
        // Increment the new target's refcount.
        Value newNode = ob.create<LLVM::LShrOp>(loc, i32Type, v2, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2)));
        Value newNotSelf = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, newNode, nIdx);
        Value newNonZero = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, newNode, cZero);
        Value newInBounds = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, newNode, cNodePool);
        Value incCond = ob.create<LLVM::AndOp>(loc, ob.getI1Type(), ob.create<LLVM::AndOp>(loc, ob.getI1Type(), newNotSelf, newNonZero), newInBounds);
        Value incIdx = ob.create<LLVM::SelectOp>(loc, incCond, newNode, cZero);
        Value incAmt = ob.create<LLVM::SelectOp>(loc, incCond, cOne, cZero);
        ob.create<memref::AtomicRMWOp>(loc, i32Type, arith::AtomicRMWKind::addi, incAmt, refcountGlobal, ValueRange{toIdx(ob, loc, incIdx)});
        // Mark the slot as holding a link, then store.
        ob.create<memref::StoreOp>(loc, cOne, slotFlagGlobal, ValueRange{idx});
        ob.create<memref::StoreOp>(loc, v2, netGlobal, ValueRange{idx});
    };
    setT(p1, p2); setT(p2, p1);

    Value isP1 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ob.create<LLVM::AndOp>(loc, i32Type, p1, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(3))), ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));
    Value isP2 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, ob.create<LLVM::AndOp>(loc, i32Type, p2, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(3))), ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));
    Value isR = ob.create<LLVM::AndOp>(loc, ob.getI1Type(), isP1, isP2);

    Block *curr = ob.getBlock();
    Block *push = f.addBlock();
    Block *cont = f.addBlock();

    ob.setInsertionPointToEnd(curr);
    ob.create<LLVM::CondBrOp>(loc, isR, push, cont);

    ob.setInsertionPointToStart(push);
    auto tailGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({}, i64Type), "__pic_queue_tail");
    Value curT = ob.create<memref::AtomicRMWOp>(loc, i64Type, arith::AtomicRMWKind::addi,
        ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1)),
        tailGlobal, ValueRange{});
    // Circular queue: index with (tail mod 32000000) so a saturated queue wraps
    // instead of overflowing. Without this the tail counter keeps growing past
    // the buffer; the worker's head<tail test then stays true forever and it
    // re-reads stale early slots (garbage node indices) until it OOB-crashes.
    Value curTMod = ob.create<LLVM::URemOp>(loc, i64Type, curT,
        ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(32000000)));

    Block *doStore = f.addBlock();
    Block *contPush = f.addBlock();

    ob.setInsertionPointToEnd(push);
    ob.create<LLVM::BrOp>(loc, doStore);

    ob.setInsertionPointToStart(doStore);
    Value r = ob.create<LLVM::OrOp>(loc, i64Type, safeZExt(ob, loc, i64Type, p1), ob.create<LLVM::ShlOp>(loc, i64Type, safeZExt(ob, loc, i64Type, p2), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(32))));
    ob.create<memref::StoreOp>(loc, r, qGlobal, ValueRange{toIdx(ob, loc, curTMod)});
    ob.create<LLVM::BrOp>(loc, contPush);

    ob.setInsertionPointToStart(contPush);
    ob.create<LLVM::BrOp>(loc, cont);

    ob.setInsertionPointToStart(cont);
}

static Value genAllocateRvecNode(OpBuilder &ob, Location loc, Value stateArg, func::FuncOp &f) {
    auto i32Type = ob.getI32Type();
    auto i64Type = ob.getI64Type();

    auto freeCountGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({}, i32Type), "__pic_free_count");
    Value freeCount = ob.create<memref::LoadOp>(loc, i32Type, freeCountGlobal, ValueRange{});
    Value hasFree = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, freeCount, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));

    Value newFreeCount = ob.create<LLVM::SubOp>(loc, i32Type, freeCount, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(1)));
    Value storeFreeCount = ob.create<LLVM::SelectOp>(loc, hasFree, newFreeCount, freeCount);
    ob.create<memref::StoreOp>(loc, storeFreeCount, freeCountGlobal, ValueRange{});
    Value safeNewCount = ob.create<LLVM::SelectOp>(loc, hasFree, newFreeCount, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));
    auto freeListGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i32Type), "__pic_free_list");
    Value freeIdx = ob.create<memref::LoadOp>(loc, i32Type, freeListGlobal, ValueRange{toIdx(ob, loc, safeNewCount)});

    auto alGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({}, i32Type), "__pic_allocator");
    Value bumpIdx = ob.create<memref::AtomicRMWOp>(loc, i32Type, arith::AtomicRMWKind::addi,
        ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(1)),
        alGlobal, ValueRange{});
    Value nIdx = ob.create<LLVM::SelectOp>(loc, hasFree, freeIdx, bumpIdx);

    auto netGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_net");
    auto slotFlagGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_slot_flag");
    auto refcountGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i32Type), "__pic_refcount");
    Value nIdx64 = safeZExt(ob, loc, i64Type, nIdx);
    Value base = ob.create<LLVM::ShlOp>(loc, i64Type, nIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2)));

    Value metaVal = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0x88000000));
    Value off3 = ob.create<LLVM::AddOp>(loc, i64Type, base, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(3)));
    Value idx3 = toIdx(ob, loc, off3);
    // A recycled node's slots may still hold links to other nodes; release them.
    emitSlotRefcountClear(ob, loc, idx3, nIdx, netGlobal, slotFlagGlobal, refcountGlobal, i32Type);
    ob.create<memref::StoreOp>(loc, metaVal, netGlobal, ValueRange{idx3});

    Value port0Val = ob.create<LLVM::OrOp>(loc, i32Type, ob.create<LLVM::ShlOp>(loc, i32Type, nIdx, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));
    return port0Val;
}

// history_net is i64[16000000], indexed as (nIdx<<1)+wIdx, so nIdx must be < 4000000.
// Clamp out-of-range node indices to 0 to avoid OOB access on corrupt ports.
static Value guardHistoryIndex(OpBuilder &ob, Location loc, Value nIdx) {
    auto i32Type = ob.getI32Type();
    Value cMax = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(7999999));
    Value oob = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, nIdx, cMax);
    Value cZero = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0));
    return ob.create<LLVM::SelectOp>(loc, i32Type, oob, cZero, nIdx);
}

static void genLinkPorts(OpBuilder &ob, Location loc, Value p1, Value p2, Value stateArg, func::FuncOp &f) {
    auto i32Type = ob.getI32Type();
    auto i64Type = ob.getI64Type();

    auto netGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_net");
    auto histGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i64Type), "__pic_history_net");

    Value p1_32 = safeZExt(ob, loc, i32Type, p1);
    Value p2_32 = safeZExt(ob, loc, i32Type, p2);

    Value nIdx1 = ob.create<LLVM::LShrOp>(loc, i32Type, p1_32, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2)));
    Value pNum1 = ob.create<LLVM::AndOp>(loc, i32Type, p1_32, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(3)));

    Value nIdx2 = ob.create<LLVM::LShrOp>(loc, i32Type, p2_32, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2)));
    Value pNum2 = ob.create<LLVM::AndOp>(loc, i32Type, p2_32, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(3)));

    // Clamp corrupt node indices before any net read so a garbage port can never
    // index past __pic_net (net holds 16000000 nodes; 7999999 keeps us well inside
    // both the net and the narrower history arrays).
    nIdx1 = guardHistoryIndex(ob, loc, nIdx1);
    nIdx2 = guardHistoryIndex(ob, loc, nIdx2);

    Value offsetMeta1 = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, safeZExt(ob, loc, i64Type, nIdx1), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(3)));
    Value meta1 = ob.create<memref::LoadOp>(loc, i32Type, netGlobal, ValueRange{toIdx(ob, loc, offsetMeta1)});

    Value offsetMeta2 = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, safeZExt(ob, loc, i64Type, nIdx2), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(3)));
    Value meta2 = ob.create<memref::LoadOp>(loc, i32Type, netGlobal, ValueRange{toIdx(ob, loc, offsetMeta2)});

    Value typeVal1 = ob.create<LLVM::LShrOp>(loc, i32Type, meta1, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(24)));
    Value type1 = ob.create<LLVM::AndOp>(loc, i32Type, typeVal1, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0x3F)));

    Value typeVal2 = ob.create<LLVM::LShrOp>(loc, i32Type, meta2, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(24)));
    Value type2 = ob.create<LLVM::AndOp>(loc, i32Type, typeVal2, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0x3F)));

    Value isRvec1 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, type1, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(NODE_RVEC)));
    Value isPNum1_0 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, pNum1, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));
    Value isDup2 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, type2, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(NODE_DUP)));
    Value isPNum2_gt0 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, pNum2, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));

    Value condA = ob.create<LLVM::AndOp>(loc, ob.create<LLVM::AndOp>(loc, isRvec1, isPNum1_0), ob.create<LLVM::AndOp>(loc, isDup2, isPNum2_gt0));

    Value isRvec2 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, type2, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(NODE_RVEC)));
    Value isPNum2_0 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, pNum2, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));
    Value isDup1 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, type1, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(NODE_DUP)));
    Value isPNum1_gt0 = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ugt, pNum1, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));

    Value condB = ob.create<LLVM::AndOp>(loc, ob.create<LLVM::AndOp>(loc, isRvec2, isPNum2_0), ob.create<LLVM::AndOp>(loc, isDup1, isPNum1_gt0));

    Value isBarrier = ob.create<LLVM::OrOp>(loc, condA, condB);

    Value dupNodeIdxSel = ob.create<LLVM::SelectOp>(loc, condA, nIdx2, nIdx1);
    // Guard: history_net is i64[16000000] indexed by (nIdx<<1)+wIdx, so a node must
    // be < 4000000. A corrupt node index reaching the barrier path would read out
    // of bounds; clamp so the barrier degrades into a no-op miss instead.
    Value dupNodeIdx = guardHistoryIndex(ob, loc, dupNodeIdxSel);
    Value dupNodeIdx64 = safeZExt(ob, loc, i64Type, dupNodeIdx);

    Block *curr = ob.getBlock();
    Block *cont = curr->splitBlock(ob.getInsertionPoint());

    Block *barrierBlock = f.addBlock();
    Block *mergeBlock = f.addBlock();
    Block *exitBarrier = f.addBlock();
    Block *standardLink = f.addBlock();

    ob.setInsertionPointToEnd(curr);
    ob.create<LLVM::CondBrOp>(loc, isBarrier, barrierBlock, standardLink);

    ob.setInsertionPointToStart(barrierBlock);
    Value counterOffset = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, dupNodeIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1))), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1)));
    Value decVal = ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(-1));
    Value prevCount = ob.create<memref::AtomicRMWOp>(loc, i64Type, arith::AtomicRMWKind::addi,
        decVal, histGlobal, ValueRange{toIdx(ob, loc, counterOffset)});
    Value nextCount = ob.create<LLVM::AddOp>(loc, i64Type, prevCount, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(-1)));
    Value isZero = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, nextCount, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(0)));
    ob.create<LLVM::CondBrOp>(loc, isZero, mergeBlock, exitBarrier);

    ob.setInsertionPointToStart(mergeBlock);
    Value offsetW0 = ob.create<LLVM::ShlOp>(loc, i64Type, dupNodeIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2)));
    Value w0 = ob.create<memref::LoadOp>(loc, i32Type, netGlobal, ValueRange{toIdx(ob, loc, offsetW0)});
    Value r_out = genAllocateRvecNode(ob, loc, stateArg, f);
    genNonBarrierLink(ob, loc, r_out, w0, stateArg, f);
    ob.create<LLVM::BrOp>(loc, exitBarrier);

    ob.setInsertionPointToStart(standardLink);
    genNonBarrierLink(ob, loc, p1, p2, stateArg, f);
    ob.create<LLVM::BrOp>(loc, exitBarrier);

    ob.setInsertionPointToStart(exitBarrier);
    ob.create<LLVM::BrOp>(loc, cont);

    ob.setInsertionPointToStart(cont);
}

// =========================================================================
// Per-op conversion functions (one per pic_runtime operation).
// These are called from Pass_PicRuntimeToLLVM.cpp and from the
// RewritePattern wrappers in PicRuntimeToLLVMConversionPatterns.h.
//
// NOTE: These functions do NOT erase the op — the caller is responsible
// for replacement and erasure via replaceAllUsesWith+erase (direct
// dispatch) or PatternRewriter::replaceOp/eraseOp (pattern path).
// =========================================================================

static Value convertAllocNodeOp(OpBuilder &ob, pic::runtime::AllocNodeOp allocOp, Value stateArg, func::FuncOp &f) {
    auto i32Type = ob.getI32Type();
    auto i64Type = ob.getI64Type();
    Location loc = allocOp.getLoc();

    auto freeCountGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({}, i32Type), "__pic_free_count");
    Value freeCount = ob.create<memref::LoadOp>(loc, i32Type, freeCountGlobal, ValueRange{});
    Value hasFree = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, freeCount, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));

    Value newFreeCount = ob.create<LLVM::SubOp>(loc, i32Type, freeCount, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(1)));
    Value storeFreeCount = ob.create<LLVM::SelectOp>(loc, hasFree, newFreeCount, freeCount);
    ob.create<memref::StoreOp>(loc, storeFreeCount, freeCountGlobal, ValueRange{});
    Value safeNewCount = ob.create<LLVM::SelectOp>(loc, hasFree, newFreeCount, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0)));
    auto freeListGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i32Type), "__pic_free_list");
    Value freeIdx = ob.create<memref::LoadOp>(loc, i32Type, freeListGlobal, ValueRange{toIdx(ob, loc, safeNewCount)});

    auto alGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({}, i32Type), "__pic_allocator");
    Value bumpIdx = ob.create<memref::AtomicRMWOp>(loc, i32Type, arith::AtomicRMWKind::addi,
        ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(1)),
        alGlobal, ValueRange{});
    Value nIdx = ob.create<LLVM::SelectOp>(loc, hasFree, freeIdx, bumpIdx);

    auto netGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_net");
    auto slotFlagGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_slot_flag");
    auto refcountGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i32Type), "__pic_refcount");
    Value nIdx64 = safeZExt(ob, loc, i64Type, nIdx);
    Value base = ob.create<LLVM::ShlOp>(loc, i64Type, nIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2)));

    auto store = [&](int i, Value v) {
        Value off = ob.create<LLVM::AddOp>(loc, i64Type, base, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(i)));
        Value idx = toIdx(ob, loc, off);
        // A recycled node's slots may still hold links to other nodes; release them.
        emitSlotRefcountClear(ob, loc, idx, nIdx, netGlobal, slotFlagGlobal, refcountGlobal, i32Type);
        Value v32 = (v.getType() == i32Type) ? v : ob.create<LLVM::TruncOp>(loc, i32Type, v);
        ob.create<memref::StoreOp>(loc, v32, netGlobal, ValueRange{idx});
    };

    auto makePort = [&](int p) {
        return ob.create<LLVM::OrOp>(loc, i32Type, ob.create<LLVM::ShlOp>(loc, i32Type, nIdx, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(p)));
    };

    uint8_t typeVal = allocOp.getType();
    Value labelOrVal = allocOp.getLabelOrVal();

    Value typeValConst = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr((uint32_t)typeVal << 24));
    Value labelOrVal32 = ob.create<LLVM::TruncOp>(loc, i32Type, labelOrVal);
    Value maskConst = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0xFFFFFF));
    Value labelMasked = ob.create<LLVM::AndOp>(loc, labelOrVal32, maskConst);
    Value metaValueVal = ob.create<LLVM::OrOp>(loc, typeValConst, labelMasked);

    store(0, makePort(0)); store(1, makePort(1)); store(2, makePort(2)); store(3, metaValueVal);

    if (typeVal == NODE_DUP) {
        auto histGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i64Type), "__pic_history_net");
        Value hBase = ob.create<LLVM::ShlOp>(loc, i64Type, nIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1)));
        Value valWord0 = ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1ULL << 32));

        ob.create<memref::StoreOp>(loc, valWord0, histGlobal, ValueRange{toIdx(ob, loc, hBase)});

        Value hBasePlus1 = ob.create<LLVM::AddOp>(loc, hBase, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1)));

        Value valWord1 = ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2));
        ob.create<memref::StoreOp>(loc, valWord1, histGlobal, ValueRange{toIdx(ob, loc, hBasePlus1)});
    }

    return nIdx;
}

static void convertSetPortOp(OpBuilder &ob, pic::runtime::SetPortOp setOp, Value stateArg) {
    auto i32Type = ob.getI32Type();
    auto i64Type = ob.getI64Type();
    Location loc = setOp.getLoc();

    Value nIdx = setOp.getNodeIndex();
    int pIdx = setOp.getPortIndex();
    Value val = setOp.getPortValue();
    auto netGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_net");
    auto slotFlagGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_slot_flag");
    auto refcountGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i32Type), "__pic_refcount");
    Value nIdx64 = safeZExt(ob, loc, i64Type, nIdx);
    Value offset = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, nIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(pIdx)));
    Value idx = toIdx(ob, loc, offset);
    // A value write overwrites whatever was in this slot; release the old link if any.
    emitSlotRefcountClear(ob, loc, idx, nIdx, netGlobal, slotFlagGlobal, refcountGlobal, i32Type);
    ob.create<memref::StoreOp>(loc, val, netGlobal, ValueRange{idx});
}

static Value convertGetPortOp(OpBuilder &ob, pic::runtime::GetPortOp getOp, Value stateArg) {
    auto i32Type = ob.getI32Type();
    auto i64Type = ob.getI64Type();
    Location loc = getOp.getLoc();

    Value nIdx = getOp.getNodeIndex();
    int pIdx = getOp.getPortIndex();
    auto netGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_net");
    Value nIdx64 = safeZExt(ob, loc, i64Type, nIdx);
    Value offset = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, nIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(pIdx)));
    Value val32 = ob.create<memref::LoadOp>(loc, i32Type, netGlobal, ValueRange{toIdx(ob, loc, offset)});
    return val32;
}

static Value convertGetPortDynamicOp(OpBuilder &ob, pic::runtime::GetPortDynamicOp getOp, Value stateArg) {
    auto i32Type = ob.getI32Type();
    auto i64Type = ob.getI64Type();
    Location loc = getOp.getLoc();

    Value nIdx = getOp.getNodeIndex();
    Value pIdx = getOp.getPortIndex();
    auto netGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_net");
    Value nIdx64 = safeZExt(ob, loc, i64Type, nIdx);
    Value pIdx64 = safeZExt(ob, loc, i64Type, pIdx);
    Value offset = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, nIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2))), pIdx64);
    Value val32 = ob.create<memref::LoadOp>(loc, i32Type, netGlobal, ValueRange{toIdx(ob, loc, offset)});
    return val32;
}

static void convertLinkOp(OpBuilder &ob, pic::runtime::LinkOp linkOp, Value stateArg, func::FuncOp &f) {
    Value p1 = linkOp.getOperand(0);
    Value p2 = linkOp.getOperand(1);
    genLinkPorts(ob, linkOp.getLoc(), p1, p2, stateArg, f);
}

static void convertPushRedexOp(OpBuilder &ob, pic::runtime::PushRedexOp pushOp, Value stateArg) {
    auto i64Type = ob.getI64Type();
    auto loc = pushOp.getLoc();

    Value nA = pushOp.getNodeA();
    Value nB = pushOp.getNodeB();
    auto tailGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({}, i64Type), "__pic_queue_tail");
    Value curT = ob.create<memref::AtomicRMWOp>(loc, i64Type, arith::AtomicRMWKind::addi,
        ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1)),
        tailGlobal, ValueRange{});

    Block *curr = ob.getBlock();
    Block *doStore = curr->splitBlock(pushOp.getOperation());
    Block *cont = doStore->splitBlock(doStore->begin());

    ob.setInsertionPointToEnd(curr);
    ob.create<LLVM::BrOp>(loc, doStore);

    ob.setInsertionPointToStart(doStore);
    // Store redex pair (nodeA, nodeB) into queue buffer at the old tail index.
    // The queue stores full PORT ADDRESSES (matching what genNonBarrierLink
    // pushes and what convertPopRedexOp right-shifts by 2 to recover nodes).
    auto i32Type = ob.getI32Type();
    auto qGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({32000000}, i64Type), "__pic_queue");
    // The head/tail counters grow without bound but only the low bits index the
    // buffer (circular queue): an unbounded retry/redo volume can never read or
    // write past __pic_queue, so it degrades into harmless wrap-around instead of
    // an out-of-bounds access past the flat array.
    Value curTMod = ob.create<LLVM::URemOp>(loc, i64Type, curT, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(32000000)));
    Value nA2 = ob.create<LLVM::ShlOp>(loc, i32Type, safeZExt(ob, loc, i32Type, pushOp.getNodeA()), ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2)));
    Value nB2 = ob.create<LLVM::ShlOp>(loc, i32Type, safeZExt(ob, loc, i32Type, pushOp.getNodeB()), ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2)));
    Value r = ob.create<LLVM::OrOp>(loc, i64Type, safeZExt(ob, loc, i64Type, nA2), ob.create<LLVM::ShlOp>(loc, i64Type, safeZExt(ob, loc, i64Type, nB2), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(32))));
    ob.create<memref::StoreOp>(loc, r, qGlobal, ValueRange{toIdx(ob, loc, curTMod)});
    ob.create<LLVM::BrOp>(loc, cont);

    ob.setInsertionPointToStart(cont);
}

static std::array<Value, 3> convertPopRedexOp(OpBuilder &ob, pic::runtime::PopRedexOp popOp, Value stateArg, func::FuncOp &f) {
    auto i32Type = ob.getI32Type();
    auto i64Type = ob.getI64Type();
    auto i1Type = ob.getI1Type();
    Location loc = popOp.getLoc();

    auto headTy = MemRefType::get({}, i64Type);
    auto headGlobal = ob.create<memref::GetGlobalOp>(loc, headTy, "__pic_queue_head");
    auto tailGlobal = ob.create<memref::GetGlobalOp>(loc, headTy, "__pic_queue_tail");
    auto activeGlobal = ob.create<memref::GetGlobalOp>(loc, headTy, "__pic_active_count");
    auto lockGlobal = ob.create<memref::GetGlobalOp>(loc, headTy, "__pic_lock");
    auto qGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({32000000}, i64Type), "__pic_queue");

    Value oneVal = ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1));
    Value zeroVal = ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(0));

    Block *curr = ob.getBlock();
    Block *cont = curr->splitBlock(popOp.getOperation());

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

    Value finalValid = cont->addArgument(i1Type, loc);
    Value finalA = cont->addArgument(i32Type, loc);
    Value finalB = cont->addArgument(i32Type, loc);

    ob.setInsertionPointToEnd(curr);
    ob.create<LLVM::BrOp>(loc, spinStart);

    ob.setInsertionPointToStart(spinStart);
    Value prevLock = ob.create<memref::AtomicRMWOp>(loc, i64Type, arith::AtomicRMWKind::assign,
        oneVal, lockGlobal, ValueRange{});
    Value isLocked = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, prevLock, oneVal);
    ob.create<LLVM::CondBrOp>(loc, isLocked, spinStart, lockedCase);

    ob.setInsertionPointToStart(lockedCase);
    Value curH = ob.create<memref::LoadOp>(loc, i64Type, headGlobal, ValueRange{});
    Value curT = ob.create<memref::LoadOp>(loc, i64Type, tailGlobal, ValueRange{});
    Value hasElement = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, curH, curT);
    ob.create<LLVM::CondBrOp>(loc, hasElement, doPop, doWait);

    ob.setInsertionPointToStart(doPop);
    Value nextH = ob.create<LLVM::AddOp>(loc, i64Type, curH, oneVal);
    ob.create<memref::StoreOp>(loc, nextH, headGlobal, ValueRange{});
    ob.create<memref::StoreOp>(loc, zeroVal, lockGlobal, ValueRange{});
    ob.create<LLVM::BrOp>(loc, ValueRange{curH}, doLoad);

    ob.setInsertionPointToStart(doWait);
    ob.create<memref::StoreOp>(loc, zeroVal, lockGlobal, ValueRange{});
    Value negOneVal = ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(-1));
    ob.create<memref::AtomicRMWOp>(loc, i64Type, arith::AtomicRMWKind::addi,
        negOneVal, activeGlobal, ValueRange{});
    ob.create<LLVM::BrOp>(loc, waitStart);

    ob.setInsertionPointToStart(waitStart);
    Value act = ob.create<memref::LoadOp>(loc, i64Type, activeGlobal, ValueRange{});
    Value isZero = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, act, zeroVal);
    ob.create<LLVM::CondBrOp>(loc, isZero, terminate, checkQueue);

    ob.setInsertionPointToStart(terminate);
    Value falseVal = ob.create<LLVM::ConstantOp>(loc, i1Type, ob.getBoolAttr(false));
    Value zero32 = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(0));
    ob.create<LLVM::BrOp>(loc, ValueRange{falseVal, zero32, zero32}, cont);

    ob.setInsertionPointToStart(checkQueue);
    Value checkH = ob.create<memref::LoadOp>(loc, i64Type, headGlobal, ValueRange{});
    Value checkT = ob.create<memref::LoadOp>(loc, i64Type, tailGlobal, ValueRange{});
    Value checkHas = ob.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, checkH, checkT);
    ob.create<LLVM::CondBrOp>(loc, checkHas, wakeUp, keepWaiting);

    ob.setInsertionPointToStart(wakeUp);
    ob.create<memref::AtomicRMWOp>(loc, i64Type, arith::AtomicRMWKind::addi,
        oneVal, activeGlobal, ValueRange{});
    ob.create<LLVM::BrOp>(loc, spinStart);

    ob.setInsertionPointToStart(keepWaiting);
    ob.create<LLVM::BrOp>(loc, waitStart);

    Value popIdx = doLoad->addArgument(i64Type, loc);
    ob.setInsertionPointToStart(doLoad);
    // Circular buffer (matches the push side): popIdx grows without bound but the
    // low bits index __pic_queue, so a queue that grows past its flat size wraps
    // instead of reading out of bounds.
    Value popIdxMod = ob.create<LLVM::URemOp>(loc, i64Type, popIdx, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(32000000)));
    Value val = ob.create<memref::LoadOp>(loc, i64Type, qGlobal, ValueRange{toIdx(ob, loc, popIdxMod)});
    Value pA = ob.create<LLVM::TruncOp>(loc, i32Type, val);
    Value pB = ob.create<LLVM::TruncOp>(loc, i32Type, ob.create<LLVM::LShrOp>(loc, i64Type, val, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(32))));
    Value c2 = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2));
    Value nA = ob.create<LLVM::LShrOp>(loc, i32Type, pA, c2);
    Value nB = ob.create<LLVM::LShrOp>(loc, i32Type, pB, c2);
    Value trueVal = ob.create<LLVM::ConstantOp>(loc, i1Type, ob.getBoolAttr(true));
    ob.create<LLVM::BrOp>(loc, ValueRange{trueVal, nA, nB}, cont);

    ob.setInsertionPointToStart(cont);
    return {finalValid, finalA, finalB};
}

static Value convertGetHistoryOp(OpBuilder &ob, pic::runtime::GetHistoryOp histOp, Value stateArg) {
    auto i64Type = ob.getI64Type();
    Location loc = histOp.getLoc();

    Value nIdx = histOp.getNodeIndex();
    int wIdx = histOp.getWordIndex();
    auto histGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i64Type), "__pic_history_net");
    Value nIdx64 = safeZExt(ob, loc, i64Type, guardHistoryIndex(ob, loc, nIdx));
    Value offset = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, nIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1))), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(wIdx)));
    Value val = ob.create<memref::LoadOp>(loc, i64Type, histGlobal, ValueRange{toIdx(ob, loc, offset)});
    return val;
}

static void convertSetHistoryOp(OpBuilder &ob, pic::runtime::SetHistoryOp histOp, Value stateArg) {
    auto i64Type = ob.getI64Type();
    Location loc = histOp.getLoc();

    Value nIdx = histOp.getNodeIndex();
    int wIdx = histOp.getWordIndex();
    Value val = histOp.getWordValue();
    auto histGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i64Type), "__pic_history_net");
    Value nIdx64 = safeZExt(ob, loc, i64Type, guardHistoryIndex(ob, loc, nIdx));
    Value offset = ob.create<LLVM::AddOp>(loc, i64Type, ob.create<LLVM::ShlOp>(loc, i64Type, nIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1))), ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(wIdx)));
    ob.create<memref::StoreOp>(loc, val, histGlobal, ValueRange{toIdx(ob, loc, offset)});
}

static void convertUncomputeSweepOp(OpBuilder &ob, pic::runtime::UncomputeSweepOp sweepOp, Value stateArg, func::FuncOp &f) {
    auto i32Type = ob.getI32Type();
    auto i64Type = ob.getI64Type();
    Location loc = sweepOp.getLoc();

    Value boundaryId = sweepOp.getBoundaryId();
    auto alGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({}, i32Type), "__pic_allocator");
    Value rIdx = ob.create<memref::AtomicRMWOp>(loc, i32Type, arith::AtomicRMWKind::addi,
        ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(1)),
        alGlobal, ValueRange{});
    Value rIdx64 = safeZExt(ob, loc, i64Type, rIdx);
    Value base = ob.create<LLVM::ShlOp>(loc, i64Type, rIdx64, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(2)));
    auto netGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({64000000}, i32Type), "__pic_net");

    auto makePort = [&](Value idx, int p) {
        return ob.create<LLVM::OrOp>(loc, i32Type, ob.create<LLVM::ShlOp>(loc, i32Type, idx, ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(2))), ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr(p)));
    };

    Value rPort0 = makePort(rIdx, 0);
    ob.create<memref::StoreOp>(loc, rPort0, netGlobal, ValueRange{toIdx(ob, loc, base)});
    Value metaVal = ob.create<LLVM::ConstantOp>(loc, i32Type, ob.getI32IntegerAttr((2U << 30) | (8U << 24)));
    Value off3 = ob.create<LLVM::AddOp>(loc, i64Type, base, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(3)));
    ob.create<memref::StoreOp>(loc, metaVal, netGlobal, ValueRange{toIdx(ob, loc, off3)});

    genLinkPorts(ob, loc, rPort0, boundaryId, stateArg, f);
}

static void convertCheckpointBoundaryOp(OpBuilder &ob, pic::runtime::CheckpointBoundaryOp cpOp, Value stateArg) {
    auto i64Type = ob.getI64Type();
    Location loc = cpOp.getLoc();

    int boundaryId = cpOp.getBoundaryId();
    auto histGlobal = ob.create<memref::GetGlobalOp>(loc, MemRefType::get({16000000}, i64Type), "__pic_history_net");
    Value bId = ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(boundaryId));
    Value countVal = ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(cpOp.getNumOperands()));
    Value checkpointVal = ob.create<LLVM::ShlOp>(loc, i64Type, countVal, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(32)));
    checkpointVal = ob.create<LLVM::OrOp>(loc, i64Type, checkpointVal, ob.create<LLVM::ConstantOp>(loc, i64Type, ob.getI64IntegerAttr(1)));
    ob.create<memref::AtomicRMWOp>(loc, i64Type, arith::AtomicRMWKind::ori,
        checkpointVal, histGlobal, ValueRange{toIdx(ob, loc, bId)});
}

#endif // PIC_RUNTIME_TO_LLVM_CONVERSIONS_H