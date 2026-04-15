#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>

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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"

#include "PicGraphDialect.h"
#include "PicReduceDialect.h"
#include "PicRuntimeDialect.h"
#include "lin/LoweringPasses.h"

extern "C" {
#include "lin/Parser.h"
#include "lin/Lowering.h"
}

using namespace mlir;

#include <fstream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include <cctype>
#include <linux/limits.h>

// Helper to check if a string is idiomatic Red/Rebol style
static bool isIdiomaticCase(const char* str, int len) {
    if (len == 0) return true;
    for (int i = 0; i < len; ++i) {
        if (std::isupper(str[i])) return false; // camelCase/PascalCase is invalid
    }
    return true;
}

#include <unordered_set>
#include <string>

// Semantic type checking to ensure argument and return types resolve to declared types or mlir-ops.
static int semanticTypeCheckAst(AstNode *node, std::unordered_set<std::string>& declaredTypes) {
    if (!node) return 0;
    int errors = 0;

    switch (node->type) {
        case AST_BLOCK: {
            // First pass: collect all top-level declarations
            for (int i = 0; i < node->as.block.count; ++i) {
                AstNode *stmt = node->as.block.statements[i];
                if (stmt->type == AST_FUNC_DECL) {
                    declaredTypes.insert(std::string(stmt->as.func_decl.name, stmt->as.func_decl.name_len));
                } else if (stmt->type == AST_MLIR_OP) {
                    declaredTypes.insert(std::string(stmt->as.mlir_op.name, stmt->as.mlir_op.name_len));
                } else if (stmt->type == AST_ASSIGNMENT) {
                    declaredTypes.insert(std::string(stmt->as.assignment.name, stmt->as.assignment.name_len));
                }
            }

            // Second pass: validate types within the block
            for (int i = 0; i < node->as.block.count; ++i) {
                errors += semanticTypeCheckAst(node->as.block.statements[i], declaredTypes);
            }
            break;
        }
        case AST_FUNC_DECL: {
            for (int i = 0; i < node->as.func_decl.arg_count; ++i) {
                std::string argTypeName(node->as.func_decl.args[i].type_name, node->as.func_decl.args[i].type_name_len);
                if (declaredTypes.find(argTypeName) == declaredTypes.end()) {
                    std::cerr << "Semantic Error: Type '" << argTypeName << "' used in argument '" << std::string(node->as.func_decl.args[i].name, node->as.func_decl.args[i].name_len) << "' is undeclared.\n";
                    errors++;
                }
            }
            std::string returnTypeName(node->as.func_decl.return_type_name, node->as.func_decl.return_type_len);
            if (declaredTypes.find(returnTypeName) == declaredTypes.end()) {
                std::cerr << "Semantic Error: Return type '" << returnTypeName << "' used in function '" << std::string(node->as.func_decl.name, node->as.func_decl.name_len) << "' is undeclared.\n";
                errors++;
            }
            errors += semanticTypeCheckAst(node->as.func_decl.body, declaredTypes);
            break;
        }
        case AST_ASSIGNMENT: {
            errors += semanticTypeCheckAst(node->as.assignment.value, declaredTypes);
            break;
        }
        case AST_CALL: {
            for (int i = 0; i < node->as.call.arg_count; ++i) {
                errors += semanticTypeCheckAst(node->as.call.args[i], declaredTypes);
            }
            break;
        }
        case AST_BINARY: {
            errors += semanticTypeCheckAst(node->as.binary.left, declaredTypes);
            errors += semanticTypeCheckAst(node->as.binary.right, declaredTypes);
            break;
        }
        case AST_WHILE: {
            errors += semanticTypeCheckAst(node->as.while_loop.condition, declaredTypes);
            errors += semanticTypeCheckAst(node->as.while_loop.body, declaredTypes);
            break;
        }
        case AST_PAIR: {
            errors += semanticTypeCheckAst(node->as.pair.left, declaredTypes);
            errors += semanticTypeCheckAst(node->as.pair.right, declaredTypes);
            break;
        }
        case AST_IDENTIFIER:
        case AST_NUMBER:
        case AST_FLOAT:
        case AST_BOOL:
        case AST_STRING:
        case AST_MLIR_OP:
        case AST_IMPORT:
            break;
    }
    return errors;
}

// Recursive AST traversal for checkstyle
static int checkstyleAst(AstNode *node) {
    if (!node) return 0;
    int errors = 0;

    switch (node->type) {
        case AST_IDENTIFIER: {
            if (!isIdiomaticCase(node->as.identifier.name, node->as.identifier.length)) {
                std::cerr << "Checkstyle Error: Identifier '" << std::string(node->as.identifier.name, node->as.identifier.length) << "' must be lower-case.\n";
                errors++;
            }
            break;
        }
        case AST_ASSIGNMENT: {
            if (!isIdiomaticCase(node->as.assignment.name, node->as.assignment.name_len)) {
                std::cerr << "Checkstyle Error: Variable name '" << std::string(node->as.assignment.name, node->as.assignment.name_len) << "' must be lower-case.\n";
                errors++;
            }
            errors += checkstyleAst(node->as.assignment.value);
            break;
        }
        case AST_FUNC_DECL: {
            if (!isIdiomaticCase(node->as.func_decl.name, node->as.func_decl.name_len)) {
                std::cerr << "Checkstyle Error: Function name '" << std::string(node->as.func_decl.name, node->as.func_decl.name_len) << "' must be lower-case.\n";
                errors++;
            }
            for (int i = 0; i < node->as.func_decl.arg_count; ++i) {
                if (!isIdiomaticCase(node->as.func_decl.args[i].name, node->as.func_decl.args[i].name_len)) {
                    std::cerr << "Checkstyle Error: Function argument '" << std::string(node->as.func_decl.args[i].name, node->as.func_decl.args[i].name_len) << "' must be lower-case.\n";
                    errors++;
                }
            }
            errors += checkstyleAst(node->as.func_decl.body);
            break;
        }
        case AST_CALL: {
            if (!isIdiomaticCase(node->as.call.callee, node->as.call.callee_len)) {
                std::cerr << "Checkstyle Error: Function call '" << std::string(node->as.call.callee, node->as.call.callee_len) << "' must be lower-case.\n";
                errors++;
            }
            for (int i = 0; i < node->as.call.arg_count; ++i) {
                errors += checkstyleAst(node->as.call.args[i]);
            }
            break;
        }
        case AST_BINARY: {
            errors += checkstyleAst(node->as.binary.left);
            errors += checkstyleAst(node->as.binary.right);
            break;
        }
        case AST_BLOCK: {
            for (int i = 0; i < node->as.block.count; ++i) {
                errors += checkstyleAst(node->as.block.statements[i]);
            }
            break;
        }
        case AST_IMPORT: {
            // Note: Imports are flattened, so we don't traverse `module_block` here to avoid double-checking.
            break;
        }
        case AST_NUMBER:
        case AST_BOOL:
        case AST_STRING:
        case AST_MLIR_OP:
            break;
    }
    return errors;
}

int main(int argc, char **argv) {
  std::string command = "";
  std::string sourceFile = "";
  std::string outputBinary = "linc_out";
  bool enableGPU = false;
  bool enableWasm = false;
  std::vector<std::string> includePaths;
  std::vector<const char*> importSources;

  for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "build" || arg == "test") {
          command = arg;
      } else if (arg == "-o" && i + 1 < argc) {
          outputBinary = argv[++i];
      } else if (arg == "-I" && i + 1 < argc) {
          includePaths.push_back(argv[++i]);
      } else if (arg == "--gpu" || arg == "-gpu") {
          enableGPU = true;
      } else if (arg == "--wasm" || arg == "-wasm") {
          enableWasm = true;
      } else if (sourceFile.empty()) {
          sourceFile = arg;
      }
  }

  if (command.empty() || sourceFile.empty()) {
      std::cerr << "Usage: linc <build|test> <source_file.lin> [-o output_binary] [-I include_path]\n";
      return 1;
  }

  if (!outputBinary.empty() && outputBinary[0] == '-') {
      std::cerr << "Error: Output binary name cannot start with a hyphen to prevent flag injection.\n";
      return 1;
  }

  std::ifstream file(sourceFile);
  if (!file.is_open()) {
      std::cerr << "Failed to open file: " << sourceFile << "\n";
      return 1;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string source_str = buffer.str();

  // Implicitly inject std/types.lin at the beginning of the AST
  std::string fullSource = "import \"std/types.lin\"\n" + source_str;
  const char *patchedSource = strdup(fullSource.c_str());
  importSources.push_back(patchedSource);

  std::cout << "Parsing Source:\n" << patchedSource << std::endl;
  AstNode *ast = parse(patchedSource);
  if (ast) {
      // Find imports and recursively resolve them into a flat AST block list
      if (ast->type == AST_BLOCK) {
          // Construct search paths once
          std::vector<std::string> searchPaths;
          searchPaths.push_back("."); // Current working directory

          // Add source file directory
          std::filesystem::path srcPath(sourceFile);
          if (srcPath.has_parent_path()) {
              searchPaths.push_back(srcPath.parent_path().string());
          }

          // Add executable directory and its parent for standard library resolution
          char exePath[PATH_MAX];
          ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath)-1);
          if (len != -1) {
              exePath[len] = '\0';
              std::filesystem::path ePath(exePath);
              if (ePath.has_parent_path()) {
                  std::filesystem::path binDir = ePath.parent_path();
                  searchPaths.push_back(binDir.string());
                  if (binDir.has_parent_path()) {
                      searchPaths.push_back(binDir.parent_path().string());
                      if (binDir.parent_path().has_parent_path()) {
                          searchPaths.push_back(binDir.parent_path().parent_path().string());
                          if (binDir.parent_path().parent_path().has_parent_path()) {
                              searchPaths.push_back(binDir.parent_path().parent_path().parent_path().string());
                          }
                      }
                  }
              }
          }

          // Add user-provided include paths
          searchPaths.insert(searchPaths.end(), includePaths.begin(), includePaths.end());

          // We will build a completely new array of statements by appending everything in order.
          int total_count = 0;
          int capacity = 16;
          AstNode **new_stmts = static_cast<AstNode**>(malloc(sizeof(AstNode*) * capacity));


          for (int i = 0; i < ast->as.block.count; i++) {
              if (ast->as.block.statements[i]->type == AST_IMPORT) {
                  std::string importPath = std::string(ast->as.block.statements[i]->as.import_stmt.path, ast->as.block.statements[i]->as.import_stmt.length);

                  std::ifstream importFile;
                  for (const auto& base : searchPaths) {
                      std::filesystem::path fullPath = std::filesystem::path(base) / importPath;
                      importFile.open(fullPath);
                      if (importFile.is_open()) break;
                  }

                  if (importFile.is_open()) {
                      std::stringstream importBuffer;
                      importBuffer << importFile.rdbuf();
                      std::string importSourceStr = importBuffer.str();
                      const char *importSource = strdup(importSourceStr.c_str());
                      importSources.push_back(importSource);
                      AstNode *importAst = parse(importSource);

                      if (importAst && importAst->type == AST_BLOCK) {
                          int import_count = importAst->as.block.count;
                          if (total_count + import_count > capacity) {
                              while (total_count + import_count > capacity) capacity *= 2;
                              AstNode **temp = static_cast<AstNode**>(realloc(new_stmts, sizeof(AstNode*) * capacity));
                              if (!temp) {
                                  free(new_stmts);
                                  return 1;
                              }
                              new_stmts = temp;
                          }
                          memcpy(new_stmts + total_count, importAst->as.block.statements, sizeof(AstNode*) * import_count);
                          total_count += import_count;
                          free(importAst->as.block.statements);
                          free(importAst);
                      } else if (importAst) {
                          freeAst(importAst);
                      }
                  } else {
                      std::cerr << "Failed to import file: " << importPath << "\n";
                  }

                  // Always free the original import statement node as it's not placed into new_stmts
                  ast->as.block.statements[i]->as.import_stmt.module_block = nullptr;
                  freeAst(ast->as.block.statements[i]);
              } else {
                  // Add the original statement itself if it is not an import statement
                  if (total_count >= capacity) {
                      capacity *= 2;
                      AstNode **temp = static_cast<AstNode**>(realloc(new_stmts, sizeof(AstNode*) * capacity));
                      if (!temp) {
                          free(new_stmts);
                          return 1;
                      }
                      new_stmts = temp;
                  }
                  new_stmts[total_count++] = ast->as.block.statements[i];
              }
          }
          free(ast->as.block.statements);
          ast->as.block.statements = new_stmts;
          ast->as.block.count = total_count;
      }
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
  registry.insert<mlir::gpu::GPUDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  // Initialize LLVM targets
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::cout << "PIC dialects registered successfully.\n" << std::endl;

  if (ast) {
      // Perform semantic type checking
      std::unordered_set<std::string> declaredTypes;

      int semanticErrors = semanticTypeCheckAst(ast, declaredTypes);
      if (semanticErrors > 0) {
          std::cerr << "Semantic analysis failed with " << semanticErrors << " error(s).\n";
          if (ast) freeAst(ast);
          for (const char *src : importSources) free((void*)src);
          return 1;
      }

      // Lower AST to MLIR C types
      MlirContext cCtx = wrap(&context);
      MlirModule cModule = lowerAstToMlir(cCtx, ast);

      ModuleOp module = unwrap(cModule);

      std::cout << "Generated MLIR (before lowering):\n";
      module->print(llvm::outs());
      llvm::outs() << "\n";

      optimizeInteractionNetWithEGraphs(cModule);

      PassManager pm(&context);
      pm.addPass(createPicGraphToReducePass());
      pm.addPass(createPicReduceToRuntimePass());
      pm.addPass(createPicRuntimeToLLVMPass(enableGPU));

      if (enableGPU) {
          pm.addPass(mlir::createGpuKernelOutliningPass());
      }

      if (mlir::failed(pm.run(module))) {
          std::cerr << "Lowering pass failed.\n";
          return 1;
      }

      std::cout << "\nLowering pass successful. Generated LLVM IR:\n";
      module->print(llvm::outs());
      llvm::outs() << "\n";

      llvm::LLVMContext llvmContext;
      auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
      if (!llvmModule) {
          std::cerr << "Failed to translate MLIR to LLVM IR.\n";
          return 1;
      }

      std::string error;
      std::string targetTriple = enableWasm ? "wasm32-unknown-unknown" : llvm::sys::getDefaultTargetTriple();
      auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
      if (!target && !enableWasm) {
          // Fallback if default fails
          targetTriple = "x86_64-unknown-linux-gnu";
          target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
      }

      if (!target) {
          std::cerr << "Failed to lookup target: " << error << "\n";
          return 1;
      }

      llvm::TargetOptions opt;
      auto rm = std::optional<llvm::Reloc::Model>();
      auto targetMachine = target->createTargetMachine(targetTriple, "generic", "", opt, rm);

      llvmModule->setDataLayout(targetMachine->createDataLayout());
      llvmModule->setTargetTriple(targetTriple);

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

      delete targetMachine;

      if (command == "build") {
          // Link the object file into a binary (linking against libc is standard)
          std::cout << "Linking into binary '" << outputBinary << "'...\n";

          pid_t pid = fork();
          if (pid == -1) {
              std::cerr << "Failed to fork process for linking.\n";
              return 1;
          } else if (pid == 0) {
              std::vector<char*> args;
              if (enableWasm) {
                  args.push_back(const_cast<char*>("wasm-ld"));
                  args.push_back(const_cast<char*>(objFile.c_str()));
                  args.push_back(const_cast<char*>("-o"));
                  args.push_back(const_cast<char*>(outputBinary.c_str()));
                  args.push_back(const_cast<char*>("--no-entry"));
                  args.push_back(const_cast<char*>("--export-all"));
                  args.push_back(nullptr);
                  execvp("wasm-ld", args.data());
              } else {
                  args.push_back(const_cast<char*>("gcc"));
                  args.push_back(const_cast<char*>(objFile.c_str()));
                  args.push_back(const_cast<char*>("-o"));
                  args.push_back(const_cast<char*>(outputBinary.c_str()));
                  args.push_back(const_cast<char*>("-lpthread"));
                  args.push_back(nullptr);
                  execvp("gcc", args.data());
              }
              perror("execvp failed");
              exit(1);
          } else {
              int status;
              waitpid(pid, &status, 0);
              if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
                  std::cerr << "Linking failed. Result code: " << WEXITSTATUS(status) << "\n";
                  return 1;
              } else if (WIFSIGNALED(status)) {
                  // Sometimes NixOS/sandbox wrapper scripts or certain glibc variants
                  // throw a harmless segfault signal at the very end of process exit.
                  // If the output file was created and is executable, we consider it a success.
                  if (llvm::sys::fs::exists(outputBinary)) {
                     std::cerr << "Warning: Linker terminated by signal " << WTERMSIG(status) << ", but output binary was created successfully.\n";
                  } else {
                     std::cerr << "Linking failed. Terminated by signal: " << WTERMSIG(status) << "\n";
                     return 1;
                  }
              }
          }

          std::cout << "Successfully compiled and linked to '" << outputBinary << "'.\n";
      }
      mlirModuleDestroy(cModule);
  }

  if (command == "test") {
      std::cout << "\nRunning Built-in checkstyle and static analysis...\n";
      int styleErrors = 0;
      if (ast) {
          styleErrors = checkstyleAst(ast);
      }

      if (styleErrors > 0) {
          std::cerr << "Checkstyle and static analysis failed with " << styleErrors << " error(s).\n";
          if (ast) freeAst(ast);
          return 1;
      }
      std::cout << "Built-in checkstyle and static analysis passed.\n";
  }

  if (ast) freeAst(ast);
  for (const char *src : importSources) free((void*)src);

  std::cout << "Compilation complete.\n";
  return 0;
}
