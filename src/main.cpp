#include <iostream>
#include <memory>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"

#include "InetDialect.h"
#include "lin/LowerToLLVM.h"

extern "C" {
#include "lin/Parser.h"
#include "lin/Lowering.h"
}

using namespace mlir;

#include <fstream>
#include <sstream>

int main(int argc, char **argv) {
  if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <source_file.lin>\n";
      return 1;
  }

  std::ifstream file(argv[1]);
  if (!file.is_open()) {
      std::cerr << "Failed to open file: " << argv[1] << "\n";
      return 1;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string source_str = buffer.str();
  const char *source = source_str.c_str();

  std::cout << "Parsing Source:\n" << source << std::endl;
  AstNode *ast = parse(source);
  if (ast) {
      printAst(ast, 0);
  }

  std::cout << "\nInitializing Lin Compiler..." << std::endl;

  mlir::DialectRegistry registry;
  registry.insert<mlir::inet::InetDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  // Initialize LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::cout << "Inet dialect registered successfully.\n" << std::endl;

  if (ast) {
      // Lower AST to MLIR C types
      MlirContext cCtx = wrap(&context);
      MlirModule cModule = lowerAstToMlir(cCtx, ast);

      ModuleOp module = unwrap(cModule);

      std::cout << "Generated MLIR (before lowering):\n";
      module->print(llvm::outs());
      llvm::outs() << "\n";

      optimizeInteractionNetWithEGraphs(cModule);

      PassManager pm(&context);
      pm.addPass(createInetToLLVMLoweringPass());

      if (mlir::failed(pm.run(module))) {
          std::cerr << "Lowering pass failed.\n";
          return 1;
      }

      std::cout << "\nLowering pass successful. Generated LLVM IR:\n";
      module->print(llvm::outs());
      llvm::outs() << "\n";

      // JIT Execution
      auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);

      ExecutionEngineOptions engineOptions;
      engineOptions.transformer = optPipeline;

      auto maybeEngine = ExecutionEngine::create(module, engineOptions);
      if (!maybeEngine) {
          std::cerr << "Failed to construct an execution engine.\n";
          return 1;
      }

      auto &engine = maybeEngine.get();

      int32_t arg = 1;
      int32_t result = -1;

      std::cout << "\nExecuting JIT function fib_inet(" << arg << ")...\n";

      void *argsArray[2] = { &arg, &result };
      auto err = engine->invokePacked("fib_inet", argsArray);
      if (err) {
          std::cerr << "JIT execution failed.\n";
          return 1;
      }

      std::cout << "JIT execution successful. Result: " << result << std::endl;

      freeAst(ast);
  }

  std::cout << "Compilation complete.\n";
  return 0;
}
