#include <iostream>
#include <memory>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"

#include "PicGraphDialect.h"
#include "PicReduceDialect.h"
#include "PicRuntimeDialect.h"
#include "lin/LowerToLLVM.h"

extern "C" {
#include "lin/Parser.h"
#include "lin/Lowering.h"
}

using namespace mlir;

#include <fstream>
#include <sstream>

int main(int argc, char **argv) {
  if (argc < 3 || std::string(argv[1]) != "build") {
      std::cerr << "Usage: linc build <source_file.lin> [-o output_binary]\n";
      return 1;
  }

  std::string sourceFile = argv[2];
  std::string outputBinary = "linc_out";

  for (int i = 3; i < argc; ++i) {
      if (std::string(argv[i]) == "-o" && i + 1 < argc) {
          outputBinary = argv[i + 1];
          i++;
      }
  }

  std::ifstream file(sourceFile);
  if (!file.is_open()) {
      std::cerr << "Failed to open file: " << sourceFile << "\n";
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
  registry.insert<mlir::pic::graph::PicGraphDialect>();
  registry.insert<mlir::pic::reduce::PicReduceDialect>();
  registry.insert<mlir::pic::runtime::PicRuntimeDialect>();
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

  std::cout << "PIC dialects registered successfully.\n" << std::endl;

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

      mlir::LLVM::LLVMDialect *llvmDialect = context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

      llvm::LLVMContext llvmContext;
      auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
      if (!llvmModule) {
          std::cerr << "Failed to translate MLIR to LLVM IR.\n";
          return 1;
      }

      std::string error;
      auto target = llvm::TargetRegistry::lookupTarget("x86_64-unknown-linux-gnu", error);
      if (!target) {
          // If a specific target fails, try to get the default target triple
          auto defaultTriple = llvm::sys::getDefaultTargetTriple();
          target = llvm::TargetRegistry::lookupTarget(defaultTriple, error);
          if (!target) {
              std::cerr << "Failed to lookup target: " << error << "\n";
              return 1;
          }
      }

      llvm::TargetOptions opt;
      auto rm = std::optional<llvm::Reloc::Model>();
      auto targetMachine = target->createTargetMachine(llvm::sys::getDefaultTargetTriple(), "generic", "", opt, rm);

      llvmModule->setDataLayout(targetMachine->createDataLayout());
      llvmModule->setTargetTriple(llvm::sys::getDefaultTargetTriple());

      std::string objFile = outputBinary + ".o";
      std::error_code ec;
      llvm::raw_fd_ostream dest(objFile, ec, llvm::sys::fs::OF_None);
      if (ec) {
          std::cerr << "Could not open file: " << ec.message() << "\n";
          return 1;
      }

      llvm::legacy::PassManager pass;
      auto fileType = llvm::CodeGenFileType::ObjectFile;
      if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
          std::cerr << "TargetMachine can't emit a file of this type\n";
          return 1;
      }

      pass.run(*llvmModule);
      dest.flush();

      std::cout << "Successfully emitted object file to " << objFile << "\n";

      // Link the object file with runtime.c
      std::cout << "Linking into binary '" << outputBinary << "'...\n";

      // Need an absolute or project-relative path. Assuming the binary is run near the root.
      // A better way is to pass it via cmake definition or use an env var, but for MVP:
      std::string runtimePath = "/app/lib/runtime/runtime.c";
      std::string linkCmd = "gcc " + objFile + " " + runtimePath + " -o " + outputBinary;
      int res = system(linkCmd.c_str());
      if (res != 0) {
          std::cerr << "Linking failed.\n";
          return 1;
      }

      std::cout << "Successfully compiled and linked to '" << outputBinary << "'.\n";

      freeAst(ast);
  }

  std::cout << "Compilation complete.\n";
  return 0;
}
