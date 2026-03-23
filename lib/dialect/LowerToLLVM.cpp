#include "lin/LowerToLLVM.h"
#include "PicGraphDialect.h"
#include "PicReduceDialect.h"
#include "PicRuntimeDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

// We need to declare the C runtime functions in the LLVM module.
static void insertRuntimeDeclarations(ModuleOp module, OpBuilder &builder) {
    auto i32Type = builder.getI32Type();
    auto voidType = LLVM::LLVMVoidType::get(builder.getContext());

    // void inet_init(uint32_t capacity);
    auto initType = LLVM::LLVMFunctionType::get(voidType, {i32Type});
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "inet_init", initType);

    // uint32_t inet_alloc_node(uint8_t type);
    auto allocType = LLVM::LLVMFunctionType::get(i32Type, {i32Type});
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "inet_alloc_node", allocType);

    // void inet_set_value(uint32_t node_index, uint32_t value);
    auto setValType = LLVM::LLVMFunctionType::get(voidType, {i32Type, i32Type});
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "inet_set_value", setValType);

    // void inet_set_op_type(uint32_t node_index, uint8_t op_type);
    auto setOpType = LLVM::LLVMFunctionType::get(voidType, {i32Type, i32Type});
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "inet_set_op_type", setOpType);

    // void inet_link(uint32_t a, uint32_t b);
    auto linkType = LLVM::LLVMFunctionType::get(voidType, {i32Type, i32Type});
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "inet_link", linkType);

    // uint32_t inet_reduce();
    auto reduceType = LLVM::LLVMFunctionType::get(i32Type, {});
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "inet_reduce", reduceType);

    // void inet_print_net();
    auto printType = LLVM::LLVMFunctionType::get(voidType, {});
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "inet_print_net", printType);

    // void inet_free_memory();
    builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "inet_free_memory", printType);
}

// Helper to make a Port value (node_index << 2 | port_index)
static Value makePort(OpBuilder &builder, Location loc, Value nodeIndex, int portIndex) {
    auto i32Type = builder.getI32Type();
    Value shifted = builder.create<LLVM::ShlOp>(loc, i32Type, nodeIndex, builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(2)));
    Value port = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(portIndex));
    return builder.create<LLVM::OrOp>(loc, i32Type, shifted, port);
}

struct InetToLLVMLoweringPass : public PassWrapper<InetToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InetToLLVMLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    builder.setInsertionPointToStart(module.getBody());
    insertRuntimeDeclarations(module, builder);

    auto i32Type = builder.getI32Type();

    // We will convert all inet nodes directly into calls to inet_alloc_node etc in a main function wrapper

    // 1. Rename func.func to llvm.func, and clear arguments (it's a boot script now)
    SmallVector<func::FuncOp> funcsToConvert;
    module.walk([&](func::FuncOp funcOp) {
        funcsToConvert.push_back(funcOp);
    });

    for (auto funcOp : funcsToConvert) {
        builder.setInsertionPoint(funcOp);

        // We compile the inet instructions into a C main
        auto mainType = LLVM::LLVMFunctionType::get(i32Type, {});
        auto llvmFunc = builder.create<LLVM::LLVMFuncOp>(funcOp.getLoc(), "main", mainType);

        Block *block = llvmFunc.addEntryBlock();
        builder.setInsertionPointToStart(block);

        // Call inet_init(1024 * 1024)
        Value cap = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(1024 * 1024));
        builder.create<LLVM::CallOp>(funcOp.getLoc(), std::nullopt, "inet_init", ValueRange{cap});

        // Map mlir::Value ports to LLVM Port (i32) values
        DenseMap<Value, Value> valueToPort;

        // Traverse all operations inside the original function
        funcOp.walk([&](Operation *op) {
            Location loc = op->getLoc();
            if (op->getName().getStringRef() == "pic_graph.agent") {
                StringRef agentType = op->getAttrOfType<StringAttr>("agentType").getValue();
                StringRef label = op->getAttrOfType<StringAttr>("label").getValue();

                int nodeTypeEnum = 0; // NODE_ERA
                if (agentType == "gamma_plus") nodeTypeEnum = 1; // NODE_CON
                else if (agentType == "gamma_minus") nodeTypeEnum = 1; // NODE_CON for now
                else if (agentType == "delta") nodeTypeEnum = 2; // NODE_DUP
                else if (agentType == "omega") {
                    if (label == "num") nodeTypeEnum = 3; // NODE_NUM
                    else nodeTypeEnum = 4; // NODE_OP
                }

                Value typeVal = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(nodeTypeEnum));
                auto callOp = builder.create<LLVM::CallOp>(loc, TypeRange{i32Type}, "inet_alloc_node", ValueRange{typeVal});
                Value nodeIdx = callOp.getResult();

                if (nodeTypeEnum == 3) {
                    if (auto valAttr = op->getAttrOfType<IntegerAttr>("value")) {
                        int32_t val = valAttr.getInt();
                        Value numVal = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(val));
                        builder.create<LLVM::CallOp>(loc, std::nullopt, "inet_set_value", ValueRange{nodeIdx, numVal});
                    }
                } else if (nodeTypeEnum == 4) {
                    int32_t opCode = 0; // OP_ADD
                    if (label == "sub") opCode = 1;
                    else if (label == "lt") opCode = 4;
                    // Use OP_ADD as default or lookup dynamic opcode

                    Value opCodeVal = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(opCode));
                    builder.create<LLVM::CallOp>(loc, std::nullopt, "inet_set_op_type", ValueRange{nodeIdx, opCodeVal});
                }

                valueToPort[op->getResult(0)] = makePort(builder, loc, nodeIdx, 0);
                valueToPort[op->getResult(1)] = makePort(builder, loc, nodeIdx, 1);
                valueToPort[op->getResult(2)] = makePort(builder, loc, nodeIdx, 2);
            }
            else if (op->getName().getStringRef() == "pic_graph.link") {
                Value a = valueToPort[op->getOperand(0)];
                Value b = valueToPort[op->getOperand(1)];
                if (a && b) {
                    builder.create<LLVM::CallOp>(loc, std::nullopt, "inet_link", ValueRange{a, b});
                }
            }
        });

        // Finally, add inet_reduce, print, free, and return 0
        builder.create<LLVM::CallOp>(funcOp.getLoc(), TypeRange{i32Type}, "inet_reduce", std::nullopt);
        builder.create<LLVM::CallOp>(funcOp.getLoc(), std::nullopt, "inet_print_net", std::nullopt);
        builder.create<LLVM::CallOp>(funcOp.getLoc(), std::nullopt, "inet_free_memory", std::nullopt);

        Value zero = builder.create<LLVM::ConstantOp>(funcOp.getLoc(), i32Type, builder.getI32IntegerAttr(0));
        builder.create<LLVM::ReturnOp>(funcOp.getLoc(), ValueRange{zero});

        funcOp.erase();
    }
  }
};
}

std::unique_ptr<Pass> createInetToLLVMLoweringPass() {
  return std::make_unique<InetToLLVMLoweringPass>();
}
