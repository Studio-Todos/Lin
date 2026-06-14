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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/IRMapping.h"
#include "PicReduceUtils.h"
#include <set>
#include <algorithm>
#include <sstream>
#include <optional>
#include <unordered_map>
#include <map>

using namespace mlir;

namespace {

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
            mlir::spirv::Capability::AtomicStorage,
            mlir::spirv::Capability::Float64},
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

    // 5) GPU-capable user ops (OP type nodes)
    struct GpuUserOpInfo {
        uint32_t hash;
        enum OpKind { Add, Sub, Mul, SDiv, AddF, SubF, MulF, DivF };
        OpKind kind;
        bool isFloat;
        unsigned bitWidth;
    };
    std::vector<GpuUserOpInfo> gpuUserOps;
    if (auto gpuOpsAttr = module->getAttrOfType<DictionaryAttr>("pic.gpu_user_ops")) {
        for (auto entry : gpuOpsAttr) {
            StringRef labelName = entry.getName();
            StringRef mlirText = entry.getValue().cast<StringAttr>().getValue();
            uint32_t hash = opcodeForLabel(labelName);
            auto payloadModule = parseSourceString<ModuleOp>(mlirText, ctx);
            if (!payloadModule) continue;
            auto tempFunc = payloadModule->lookupSymbol<func::FuncOp>("temp");
            if (!tempFunc || tempFunc.getBody().empty()) continue;
            std::optional<GpuUserOpInfo::OpKind> kind;
            bool isFloat = false;
            unsigned bitWidth = 32;
            tempFunc.walk([&](Operation *walkOp) {
                if (walkOp->hasTrait<OpTrait::IsTerminator>()) return;
                if (isa<func::FuncOp>(walkOp)) return;
                if (isa<arith::TruncIOp>(walkOp) || isa<arith::ExtUIOp>(walkOp) ||
                    isa<arith::BitcastOp>(walkOp)) return;
                if (isa<arith::ConstantOp>(walkOp)) return;
                Type resultType = walkOp->getResult(0).getType();
                if (resultType.isF32()) { isFloat = true; bitWidth = 32; }
                if (resultType.isF64()) { isFloat = true; bitWidth = 64; }
                if (resultType.isSignlessInteger(64)) bitWidth = 64;
                if (isa<arith::AddIOp>(walkOp)) { kind = GpuUserOpInfo::Add; } else
                if (isa<arith::SubIOp>(walkOp)) { kind = GpuUserOpInfo::Sub; } else
                if (isa<arith::MulIOp>(walkOp)) { kind = GpuUserOpInfo::Mul; } else
                if (isa<arith::DivSIOp>(walkOp)) { kind = GpuUserOpInfo::SDiv; } else
                if (isa<arith::AddFOp>(walkOp)) { kind = GpuUserOpInfo::AddF; isFloat = true; } else
                if (isa<arith::SubFOp>(walkOp)) { kind = GpuUserOpInfo::SubF; isFloat = true; } else
                if (isa<arith::MulFOp>(walkOp)) { kind = GpuUserOpInfo::MulF; isFloat = true; } else
                if (isa<arith::DivFOp>(walkOp)) { kind = GpuUserOpInfo::DivF; isFloat = true; }
            });
            if (kind) {
                gpuUserOps.push_back({hash, *kind, isFloat, bitWidth});
            }
        }
    }

    if (!gpuUserOps.empty()) {
        Value tOP = eb.create<mlir::spirv::ConstantOp>(loc, i32Type, eb.getI32IntegerAttr(5));
        Value isOpA = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeA, tOP);
        Value isOpB = eb.create<mlir::spirv::IEqualOp>(loc, eb.getI1Type(), typeB, tOP);
        Value isOpPair = eb.create<mlir::spirv::LogicalOrOp>(loc, eb.getI1Type(), isOpA, isOpB);

        mlir::spirv::SelectionOp::createIfThen(loc,
            eb.create<mlir::spirv::LogicalAndOp>(loc, eb.getI1Type(), notOob, isOpPair),
            [&](OpBuilder &b) {
              Value opNode = b.create<mlir::spirv::SelectOp>(loc, i32Type, isOpA, nodeA, nodeB);
              Value valNode = b.create<mlir::spirv::SelectOp>(loc, i32Type, isOpA, nodeB, nodeA);
              Value valMeta = b.create<mlir::spirv::SelectOp>(loc, i32Type, isOpA, metaB, metaA);
              Value labelField = b.create<mlir::spirv::BitwiseAndOp>(loc, i32Type, valMeta,
                  eb.create<mlir::spirv::ConstantOp>(loc, i32Type, b.getI32IntegerAttr(0xFFFFFF)));

              Value valBase = b.create<mlir::spirv::IMulOp>(loc, i32Type, valNode, c4);
              Value valP1 = makeSSBOLoad(b, netPtr,
                  b.create<mlir::spirv::IAddOp>(loc, i32Type, valBase, c1));
              Value valP2 = makeSSBOLoad(b, netPtr,
                  b.create<mlir::spirv::IAddOp>(loc, i32Type, valBase, c2));

              auto genOp = [&](GpuUserOpInfo &info) {
                  auto makeArg = [&](Value portVal, bool isFloat, unsigned bw) -> Value {
                      if (isFloat) {
                          if (bw == 64)
                              return b.create<mlir::spirv::BitcastOp>(loc, b.getF64Type(),
                                  b.create<mlir::spirv::SConvertOp>(loc, b.getIntegerType(64), portVal));
                          return b.create<mlir::spirv::BitcastOp>(loc, b.getF32Type(), portVal);
                      }
                      if (bw == 64)
                          return b.create<mlir::spirv::SConvertOp>(loc, b.getIntegerType(64), portVal);
                      return portVal;
                  };
                  auto makeResult = [&](Value result, bool isFloat, unsigned bw) -> Value {
                      if (isFloat) {
                          if (bw == 64)
                              return b.create<mlir::spirv::UConvertOp>(loc, i32Type,
                                  b.create<mlir::spirv::BitcastOp>(loc, b.getIntegerType(64), result));
                          return b.create<mlir::spirv::BitcastOp>(loc, i32Type, result);
                      }
                      if (bw == 64)
                          return b.create<mlir::spirv::UConvertOp>(loc, i32Type, result);
                      return result;
                  };

                  Value arg1 = makeArg(valP1, info.isFloat, info.bitWidth);
                  Value arg2 = makeArg(valP2, info.isFloat, info.bitWidth);
                  Type baseType = info.isFloat
                      ? Type(info.bitWidth == 64 ? b.getF64Type() : b.getF32Type())
                      : Type(info.bitWidth == 64 ? b.getIntegerType(64) : i32Type);
                  Value result;
                  switch (info.kind) {
                  case GpuUserOpInfo::Add:  result = b.create<mlir::spirv::IAddOp>(loc, baseType, arg1, arg2); break;
                  case GpuUserOpInfo::Sub:  result = b.create<mlir::spirv::ISubOp>(loc, baseType, arg1, arg2); break;
                  case GpuUserOpInfo::Mul:  result = b.create<mlir::spirv::IMulOp>(loc, baseType, arg1, arg2); break;
                  case GpuUserOpInfo::SDiv: result = b.create<mlir::spirv::SDivOp>(loc, baseType, arg1, arg2); break;
                  case GpuUserOpInfo::AddF: result = b.create<mlir::spirv::FAddOp>(loc, baseType, arg1, arg2); break;
                  case GpuUserOpInfo::SubF: result = b.create<mlir::spirv::FSubOp>(loc, baseType, arg1, arg2); break;
                  case GpuUserOpInfo::MulF: result = b.create<mlir::spirv::FMulOp>(loc, baseType, arg1, arg2); break;
                  case GpuUserOpInfo::DivF: result = b.create<mlir::spirv::FDivOp>(loc, baseType, arg1, arg2); break;
                  }
                  Value result32 = makeResult(result, info.isFloat, info.bitWidth);

                  Value opBase = b.create<mlir::spirv::IMulOp>(loc, i32Type, opNode, c4);
                  makeSSBOStore(b, netPtr,
                      b.create<mlir::spirv::IAddOp>(loc, i32Type, opBase, c1), result32);
              };

              for (auto &info : gpuUserOps) {
                  Value hashConst = b.create<mlir::spirv::ConstantOp>(loc, i32Type, b.getI32IntegerAttr(info.hash));
                  Value matches = b.create<mlir::spirv::IEqualOp>(loc, b.getI1Type(), labelField, hashConst);
                  mlir::spirv::SelectionOp::createIfThen(loc, matches, [&](OpBuilder &sb) {
                      genOp(info);
                  }, b);
              }
            }, eb);
    }

    eb.create<mlir::spirv::ReturnOp>(loc);

    // worker_thread is kept alive for the dispatch loop lowered by PicRuntimeToLLVMPass
#else
    llvm::outs() << "Info: SPIR-V dialect headers not available; PicRuntimeToSPIRVPass skipped.\n";
#endif
  }
};
} // namespace

std::unique_ptr<Pass> createPicRuntimeToSPIRVPass() { return std::make_unique<PicRuntimeToSPIRVPass>(); }
