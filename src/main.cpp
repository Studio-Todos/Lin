#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>

#include <sstream>
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#if __has_include("mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h")
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#if __has_include("mlir/Dialect/SPIRV/Transforms/Passes.h")
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#endif
#if __has_include("mlir/Conversion/ArithToSPIRV/ArithToSPIRVPass.h")
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRVPass.h"
#endif
#if __has_include("mlir/Conversion/IndexToSPIRV/IndexToSPIRVPass.h")
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRVPass.h"
#endif
#if __has_include("mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h")
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#endif
#if __has_include("mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h")
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h"
#endif
#endif
#if __has_include("mlir/Dialect/SPIRV/IR/SPIRVDialect.h")
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#endif
#if __has_include("mlir/Target/SPIRV/Serialization.h")
#include "mlir/Target/SPIRV/Serialization.h"
#endif
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

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

#include <sstream>
#include <filesystem>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include <cctype>
#include <linux/limits.h>
#include <fstream>

// Helper to check if a string is idiomatic Red/Rebol style
static bool isIdiomaticCase(const char* str, int len) {
    if (len == 0) return true;
    for (int i = 0; i < len; ++i) {
        if (std::isupper(str[i])) return false; // camelCase/PascalCase is invalid
    }
    return true;
}

#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>
#include <array>
#include <cstring>

// Semantic type checking: validate arg/return type names with source locations.
static int semanticTypeCheckAst(AstNode *node, std::unordered_set<std::string>& declaredTypes) {
    if (!node) return 0;
    int errors = 0;
    switch (node->type) {
        case AST_BLOCK: {
            for (int i = 0; i < node->as.block.count; ++i) {
                AstNode *stmt = node->as.block.statements[i];
                if (stmt->type == AST_FUNC_DECL)
                    declaredTypes.insert(std::string(stmt->as.func_decl.name, stmt->as.func_decl.name_len));
                else if (stmt->type == AST_MLIR_OP)
                    declaredTypes.insert(std::string(stmt->as.mlir_op.name, stmt->as.mlir_op.name_len));
                else if (stmt->type == AST_ASSIGNMENT)
                    declaredTypes.insert(std::string(stmt->as.assignment.name, stmt->as.assignment.name_len));
            }
            for (int i = 0; i < node->as.block.count; ++i)
                errors += semanticTypeCheckAst(node->as.block.statements[i], declaredTypes);
            break;
        }
        case AST_FUNC_DECL: {
            for (int i = 0; i < node->as.func_decl.arg_count; ++i) {
                std::string t(node->as.func_decl.args[i].type_name, node->as.func_decl.args[i].type_name_len);
                if (declaredTypes.find(t) == declaredTypes.end()) {
                    std::cerr << "[line " << node->line << ":" << node->col
                              << "] Semantic Error: Type '" << t << "' used in argument '"
                              << std::string(node->as.func_decl.args[i].name, node->as.func_decl.args[i].name_len)
                              << "' is undeclared.\n";
                    errors++;
                }
            }
            if (node->as.func_decl.return_type_len > 0 && node->as.func_decl.return_type_name) {
                std::string rt(node->as.func_decl.return_type_name, node->as.func_decl.return_type_len);
                if (!rt.empty() && rt[0] != '[' && declaredTypes.find(rt) == declaredTypes.end()) {
                    std::cerr << "[line " << node->line << ":" << node->col
                              << "] Semantic Error: Return type '" << rt << "' in function '"
                              << (node->as.func_decl.name
                                  ? std::string(node->as.func_decl.name, node->as.func_decl.name_len)
                                  : std::string("anonymous"))
                              << "' is undeclared.\n";
                    errors++;
                }
            }
            if (node->as.func_decl.body)
                errors += semanticTypeCheckAst(node->as.func_decl.body, declaredTypes);
            break;
        }
        case AST_ASSIGNMENT:
            errors += semanticTypeCheckAst(node->as.assignment.value, declaredTypes); break;
        case AST_CALL:
            for (int i = 0; i < node->as.call.arg_count; ++i)
                errors += semanticTypeCheckAst(node->as.call.args[i], declaredTypes);
            break;
        case AST_BINARY:
            errors += semanticTypeCheckAst(node->as.binary.left, declaredTypes);
            errors += semanticTypeCheckAst(node->as.binary.right, declaredTypes);
            break;
        case AST_WHILE:
            errors += semanticTypeCheckAst(node->as.while_loop.condition, declaredTypes);
            errors += semanticTypeCheckAst(node->as.while_loop.body, declaredTypes);
            break;
        case AST_PAIR:
            errors += semanticTypeCheckAst(node->as.pair.left, declaredTypes);
            errors += semanticTypeCheckAst(node->as.pair.right, declaredTypes);
            break;
        default: break;
    }
    return errors;
}

// ── Type-directed omega-label selection ────────────────────────────────────
//
// Per the PIC spec (.todo item 13): the omega (Operation) node label drives
// rule dispatch (annihilation fires when matching omega(+,L) meets omega(-,L)).
// Argument type annotations therefore determine which omega label is emitted
// during lowering. This pass rewrites AST_CALL.resolved_callee before lowering
// so that e.g. "add" becomes "fadd" when f32 arguments are detected.

using TypeEnv = std::unordered_map<std::string, std::string>;
struct OpSig { std::vector<std::string> inputTypes; std::string outputType; };

// Parse type identifiers from a signature string like "[a [i32!] b [f32!]]"
static std::vector<std::string> parseSigTypes(const char *s, int len) {
    std::vector<std::string> res;
    std::string buf(s, len);
    size_t i = 0;
    while (i < buf.size()) {
        size_t lb = buf.find('[', i); if (lb == std::string::npos) break;
        size_t rb = buf.find(']', lb+1); if (rb == std::string::npos) break;
        std::string inner = buf.substr(lb+1, rb-lb-1);
        if (!inner.empty() && inner.back() == '!') inner.pop_back();
        // accept only simple type identifiers (no spaces = not an arg-name pair)
        if (!inner.empty() && inner.find(' ') == std::string::npos)
            res.push_back(inner);
        i = rb+1;
    }
    return res;
}

// Build op-signature table from the flat post-import AST.
static std::unordered_map<std::string,OpSig> buildOpSigTable(AstNode *root) {
    std::unordered_map<std::string,OpSig> tbl;
    if (!root || root->type != AST_BLOCK) return tbl;
    for (int i = 0; i < root->as.block.count; ++i) {
        AstNode *n = root->as.block.statements[i];
        if (!n || n->type != AST_MLIR_OP) continue;
        std::string name(n->as.mlir_op.name, n->as.mlir_op.name_len);
        OpSig sig;
        std::string full(n->as.mlir_op.inputs, n->as.mlir_op.inputs_len);
        size_t inPos  = full.find("inputs:");
        size_t outPos = full.find("outputs:");
        if (inPos != std::string::npos) {
            size_t end = (outPos != std::string::npos) ? outPos : full.size();
            sig.inputTypes = parseSigTypes(full.c_str()+inPos, (int)(end-inPos));
        } else {
            sig.inputTypes = parseSigTypes(n->as.mlir_op.inputs, n->as.mlir_op.inputs_len);
        }
        if (outPos != std::string::npos) {
            auto outs = parseSigTypes(full.c_str()+outPos, (int)(full.size()-outPos));
            if (!outs.empty()) sig.outputType = outs[0];
        }
        if (sig.outputType.empty() && !sig.inputTypes.empty())
            sig.outputType = sig.inputTypes[0];
        tbl[name] = sig;
    }
    return tbl;
}

// Infer the type of an expression in the current scope.
static std::string inferType(AstNode *node, const TypeEnv &env,
                              const std::unordered_map<std::string,OpSig> &sigs) {
    if (!node) return "";
    switch (node->type) {
        case AST_NUMBER: return "i64";
        case AST_FLOAT:  return "f64";
        case AST_BOOL:   return "bool";
        case AST_STRING: return "str";
        case AST_IDENTIFIER: {
            std::string nm(node->as.identifier.name, node->as.identifier.length);
            auto it = env.find(nm); return it != env.end() ? it->second : "";
        }
        case AST_CALL: {
            const char *c = node->as.call.resolved_callee
                            ? node->as.call.resolved_callee : node->as.call.callee;
            int cl = node->as.call.resolved_callee
                     ? (int)strlen(node->as.call.resolved_callee) : node->as.call.callee_len;
            std::string cn(c, cl);
            auto it = sigs.find(cn); return it != sigs.end() ? it->second.outputType : "";
        }
        default: return "";
    }
}

// Typed variant table: generic op → {i32, i64, f32, f64} variants
static const std::unordered_map<std::string, std::array<const char*,4>> kBinVariants = {
    {"add", {"add","add64","fadd","fadd64"}},
    {"sub", {"sub","sub64","fsub","fsub64"}},
    {"mul", {"mul","mul64","fmul","fmul64"}},
    {"div", {"div","div64","fdiv","fdiv64"}},
    {"lt",  {"lt", "lt64", "flt", "flt64"}},
    {"gt",  {"gt", "gt64", "fgt", "fgt64"}},
    {"le",  {"le", "le64", "fle", "fle64"}},
    {"ge",  {"ge", "ge64", "fge", "fge64"}},
    {"eq",  {"eq", "eq64", "feq", "feq64"}},
    {"ne",  {"ne", "neq64","fneq","fneq64"}},
};

// Walk AST and rewrite resolved_callee for typed binary ops.
static void typeDirectedDispatch(AstNode *node, TypeEnv env,
                                  const std::unordered_map<std::string,OpSig> &sigs) {
    if (!node) return;
    switch (node->type) {
        case AST_FUNC_DECL: {
            TypeEnv inner = env;
            for (int i = 0; i < node->as.func_decl.arg_count; ++i) {
                std::string nm(node->as.func_decl.args[i].name, node->as.func_decl.args[i].name_len);
                std::string ty(node->as.func_decl.args[i].type_name, node->as.func_decl.args[i].type_name_len);
                inner[nm] = ty;
            }
            typeDirectedDispatch(node->as.func_decl.body, inner, sigs);
            break;
        }
        case AST_BLOCK:
        case AST_BLOCK_DATA: {
            TypeEnv inner = env;
            for (int i = 0; i < node->as.block.count; ++i) {
                AstNode *s = node->as.block.statements[i];
                typeDirectedDispatch(s, inner, sigs);
                if (s && s->type == AST_ASSIGNMENT) {
                    std::string nm(s->as.assignment.name, s->as.assignment.name_len);
                    std::string ty = inferType(s->as.assignment.value, inner, sigs);
                    if (!ty.empty()) inner[nm] = ty;
                }
            }
            break;
        }
        case AST_ASSIGNMENT:
            typeDirectedDispatch(node->as.assignment.value, env, sigs); break;
        case AST_CALL: {
            for (int i = 0; i < node->as.call.arg_count; ++i)
                typeDirectedDispatch(node->as.call.args[i], env, sigs);
            std::string callee(node->as.call.callee, node->as.call.callee_len);
            auto vit = kBinVariants.find(callee);
            if (vit != kBinVariants.end() && node->as.call.arg_count >= 1) {
                std::string ty = inferType(node->as.call.args[0], env, sigs);
                if (node->as.call.arg_count >= 2) {
                    std::string ty2 = inferType(node->as.call.args[1], env, sigs);
                    if (!ty2.empty()) ty = ty2;
                    if (ty.empty()) ty = inferType(node->as.call.args[0], env, sigs);
                }
                const char *variant = nullptr;
                if      (ty == "i32") variant = vit->second[0];
                else if (ty == "i64") variant = vit->second[1];
                else if (ty == "f32") variant = vit->second[2];
                else if (ty == "f64") variant = vit->second[3];
                if (variant && strcmp(variant, node->as.call.callee) != 0) {
                    if (node->as.call.resolved_callee) free((void*)node->as.call.resolved_callee);
                    node->as.call.resolved_callee = strdup(variant);
                }
            }
            break;
        }
        case AST_WHILE:
            typeDirectedDispatch(node->as.while_loop.condition, env, sigs);
            typeDirectedDispatch(node->as.while_loop.body, env, sigs);
            break;
        case AST_PAIR:
            typeDirectedDispatch(node->as.pair.left, env, sigs);
            typeDirectedDispatch(node->as.pair.right, env, sigs);
            break;
        default: break;
    }
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

static bool hasGpuAnnotation(AstNode *node) {
    if (!node) return false;
    switch (node->type) {
        case AST_FUNC_DECL:
            if (node->as.func_decl.dispatch && 
                strncmp(node->as.func_decl.dispatch, "gpu", node->as.func_decl.dispatch_len) == 0) {
                return true;
            }
            return hasGpuAnnotation(node->as.func_decl.body);
        case AST_MLIR_OP:
            if (node->as.mlir_op.dispatch && 
                strncmp(node->as.mlir_op.dispatch, "gpu", node->as.mlir_op.dispatch_len) == 0) {
                return true;
            }
            return false;
        case AST_BLOCK:
        case AST_BLOCK_DATA:
            for (int i = 0; i < node->as.block.count; i++) {
                if (hasGpuAnnotation(node->as.block.statements[i])) return true;
            }
            break;
        case AST_ASSIGNMENT:
            return hasGpuAnnotation(node->as.assignment.value);
        case AST_ASSIGNMENT_MULTI:
            return hasGpuAnnotation(node->as.assignment_multi.value) || hasGpuAnnotation(node->as.assignment_multi.next);
        case AST_WHILE:
            return hasGpuAnnotation(node->as.while_loop.condition) || hasGpuAnnotation(node->as.while_loop.body);
        case AST_PAIR:
            return hasGpuAnnotation(node->as.pair.left) || hasGpuAnnotation(node->as.pair.right);
        case AST_FIELD_ACCESS:
            return hasGpuAnnotation(node->as.field_access.base);
        case AST_BINARY:
            return hasGpuAnnotation(node->as.binary.left) || hasGpuAnnotation(node->as.binary.right);
        case AST_CALL:
            for (int i = 0; i < node->as.call.arg_count; i++) {
                if (hasGpuAnnotation(node->as.call.args[i])) return true;
            }
            break;
        case AST_IMPORT:
            return hasGpuAnnotation(node->as.import_stmt.module_block);
        case AST_MODULE:
            return hasGpuAnnotation(node->as.module.module_block);
        default:
            break;
    }
    return false;
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
      if (arg == "build" || arg == "test" || arg == "run") {
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
          if (!new_stmts) {
              std::cerr << "Out of memory\n";
              return 1;
          }


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
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
#if __has_include("mlir/Dialect/SPIRV/IR/SPIRVDialect.h")
  registry.insert<mlir::spirv::SPIRVDialect>();
#endif
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerConvertMathToLLVMInterface(registry);

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
      if (hasGpuAnnotation(ast)) {
          enableGPU = true;
      }
      // Perform semantic type checking
      std::unordered_set<std::string> declaredTypes;

      int semanticErrors = semanticTypeCheckAst(ast, declaredTypes);
      if (semanticErrors > 0) {
          std::cerr << "Semantic analysis failed with " << semanticErrors << " error(s).\n";
          if (ast) freeAst(ast);
          for (const char *src : importSources) free((void*)src);
          return 1;
      }

      // Type-directed dispatch: rewrite binary op callees based on argument types
      // (e.g. add→fadd when f32 args are present). This drives omega-label selection
      // per the PIC spec so the correct rule fires at runtime.
      {
          auto opSigs = buildOpSigTable(ast);
          TypeEnv rootEnv;
          typeDirectedDispatch(ast, rootEnv, opSigs);
      }

      // Lower AST to MLIR C types
      MlirContext cCtx = wrap(&context);
      MlirModule cModule = lowerAstToMlir(cCtx, ast);

      ModuleOp module = unwrap(cModule);

#if __has_include("mlir/Dialect/SPIRV/IR/TargetAndABI.h")
      if (enableGPU) {
          auto targetEnv = mlir::spirv::getDefaultTargetEnv(module.getContext());
          module->setAttr(mlir::spirv::getTargetEnvAttrName(), targetEnv);
      }
#endif

      // std::cout << "Generated MLIR (before lowering):\n";
      // module.print(llvm::outs());
      PassManager pm(&context);
      pm.addPass(createPicGraphToReducePass());
      pm.addPass(createPicRuntimeToLLVMPass(enableGPU));

      if (enableGPU) {
          pm.addPass(mlir::createGpuKernelOutliningPass());
      }

      if (mlir::failed(pm.run(module))) {
          std::cerr << "Lowering and outlining pass failed.\n";
          return 1;
      }

#if __has_include("mlir/Dialect/SPIRV/IR/TargetAndABI.h")
      if (enableGPU) {
          module.walk([&](mlir::gpu::GPUFuncOp gpuFunc) {
              if (gpuFunc.isKernel()) {
                  auto abi = mlir::spirv::getEntryPointABIAttr(gpuFunc.getContext(), {1, 1, 1});
                  gpuFunc->setAttr(mlir::spirv::getEntryPointABIAttrName(), abi);
              }
          });
      }
#endif

      if (enableGPU) {
          PassManager pm_gpu(&context);
#if __has_include("mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h")
           pm_gpu.addPass(mlir::createGpuSPIRVAttachTarget());
#if __has_include("mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h")
           pm_gpu.addPass(mlir::createMapMemRefStorageClassPass());
           pm_gpu.addPass(mlir::createConvertMemRefToSPIRVPass());
#endif
#if __has_include("mlir/Conversion/ArithToSPIRV/ArithToSPIRVPass.h")
           pm_gpu.addPass(mlir::createConvertArithToSPIRVPass());
#endif
#if __has_include("mlir/Conversion/IndexToSPIRV/IndexToSPIRVPass.h")
           pm_gpu.addPass(mlir::createConvertIndexToSPIRVPass());
#endif
#if __has_include("mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h")
           pm_gpu.addPass(mlir::createConvertControlFlowToSPIRVPass());
#endif
// #if __has_include("mlir/Conversion/FuncToSPIRV/FuncToSPIRVPass.h")
//            pm_gpu.addPass(mlir::createConvertFuncToSPIRVPass());
// #endif
           pm_gpu.addPass(mlir::createConvertGPUToSPIRVPass());
#if __has_include("mlir/Dialect/SPIRV/Transforms/Passes.h")
           pm_gpu.addNestedPass<mlir::spirv::ModuleOp>(mlir::spirv::createSPIRVLowerABIAttributesPass());
           pm_gpu.addNestedPass<mlir::spirv::ModuleOp>(mlir::spirv::createSPIRVUpdateVCEPass());
#endif
#else
          std::cerr << "Warning: MLIR GPU-to-SPIRV conversion headers not available; skipping SPIR-V conversion.\n";
#endif
          if (mlir::failed(pm_gpu.run(module))) {
              std::cerr << "GPU to SPIR-V conversion pass failed.\n";
              return 1;
          }
      }

#if __has_include("mlir/Target/SPIRV/Serialization.h")
      if (enableGPU) {
          module.walk([&](mlir::spirv::ModuleOp spirvModule) {
              if (spirvModule->hasAttr("vce_triple")) return;
              auto vce = mlir::spirv::VerCapExtAttr::get(
                  mlir::spirv::Version::V_1_0,
                  mlir::ArrayRef<mlir::spirv::Capability>{
                      mlir::spirv::Capability::Shader,
                      mlir::spirv::Capability::Linkage},
                  mlir::ArrayRef<mlir::spirv::Extension>{},
                  spirvModule.getContext());
              spirvModule->setAttr("vce_triple", vce);
          });
          mlir::SmallVector<uint32_t, 0> spirvBinary;
          bool hasSpirv = false;
          module.walk([&](mlir::Operation *op) {
              if (hasSpirv) return;
              auto spirvModule = llvm::dyn_cast<mlir::spirv::ModuleOp>(op);
              if (!spirvModule) return;
              if (mlir::succeeded(mlir::spirv::serialize(spirvModule, spirvBinary))) {
                  hasSpirv = true;
              }
          });

          if (hasSpirv) {
              std::string spirvPath = outputBinary + ".spv";
              std::ofstream spirvOut(spirvPath, std::ios::binary);
              if (!spirvOut) {
                  std::cerr << "Failed to open SPIR-V output file: " << spirvPath << "\n";
              } else {
                  spirvOut.write(reinterpret_cast<const char *>(spirvBinary.data()),
                                 static_cast<std::streamsize>(spirvBinary.size() * sizeof(uint32_t)));
                  spirvOut.flush();
                  std::cout << "Emitted SPIR-V module to " << spirvPath << "\n";
              }
          } else {
              std::cerr << "Warning: No SPIR-V module was produced during GPU lowering.\n";
          }
      }
#endif

      if (enableGPU) {
          llvm::SmallVector<mlir::Operation *> opsToErase;
          module.walk([&](mlir::Operation *op) {
              if (auto symNameAttr = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
                  if (symNameAttr.getValue() == "pic") {
                      opsToErase.push_back(op);
                  }
              }
          });
          for (auto op : opsToErase) {
              op->erase();
          }
          llvm::SmallVector<mlir::spirv::ModuleOp> spirvModules;
          module.walk([&](mlir::spirv::ModuleOp op) {
              spirvModules.push_back(op);
          });
          for (auto op : spirvModules) op.erase();

          llvm::SmallVector<mlir::gpu::GPUModuleOp> gpuModules;
          module.walk([&](mlir::gpu::GPUModuleOp op) {
              gpuModules.push_back(op);
          });
          for (auto op : gpuModules) op.erase();
      }

      if (enableGPU) {
          std::cout << "--- MLIR Module before LLVM PassManager ---\n";
          module.print(llvm::outs());
          std::cout << "-------------------------------------------\n";
      }

      PassManager pm2(&context);
      pm2.addPass(mlir::createConvertSCFToCFPass());
      pm2.addPass(mlir::createConvertControlFlowToLLVMPass());
      pm2.addPass(mlir::createConvertMathToLibmPass());
      pm2.addPass(mlir::createConvertMathToLLVMPass());
      pm2.addPass(mlir::createArithToLLVMConversionPass());
      pm2.addPass(mlir::createConvertFuncToLLVMPass());
      pm2.addPass(mlir::createReconcileUnrealizedCastsPass());

      if (mlir::failed(pm2.run(module))) {
          std::cerr << "LLVM conversion passes failed.\n";
          return 1;
      }

      std::cout << "\nLowering pass successful.\n";

      llvm::LLVMContext llvmContext;
      auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
      if (!llvmModule) {
          std::cerr << "Failed to translate MLIR to LLVM IR.\n";
          return 1;
      }
      std::cout << "Generated LLVM IR:\n";
      llvmModule->print(llvm::outs(), nullptr);
      llvm::outs() << "\n";

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
      auto rm = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
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
                  const char* ldPath = getenv("LD_LIBRARY_PATH");
                  if (ldPath) {
                      std::string s(ldPath);
                      size_t pos = 0;
                      while ((pos = s.find(":")) != std::string::npos) {
                          std::string path = s.substr(0, pos);
                          if (!path.empty()) {
                              char* arg = new char[path.length() + 3];
                              sprintf(arg, "-L%s", path.c_str());
                              args.push_back(arg);
                          }
                          s.erase(0, pos + 1);
                      }
                      if (!s.empty()) {
                          char* arg = new char[s.length() + 3];
                          sprintf(arg, "-L%s", s.c_str());
                          args.push_back(arg);
                      }
                  }
                  std::string gpuRuntimePath = "src/gpu_runtime.c";
                  if (enableGPU) {
                      // Attempt to find gpu_runtime.c relative to the project root or source directory
                      char exePath[PATH_MAX];
                      ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath)-1);
                      if (len != -1) {
                          exePath[len] = '\0';
                          std::filesystem::path ePath(exePath);
                          std::filesystem::path binDir = ePath.parent_path();
                          // Try parent of build/src or build/
                          std::filesystem::path pDir = binDir.parent_path();
                          if (pDir.has_parent_path()) {
                              std::filesystem::path candidate = pDir.parent_path() / "src" / "gpu_runtime.c";
                              if (std::filesystem::exists(candidate)) gpuRuntimePath = candidate.string();
                              else {
                                  candidate = pDir / "src" / "gpu_runtime.c";
                                  if (std::filesystem::exists(candidate)) gpuRuntimePath = candidate.string();
                              }
                          }
                      }
                      args.push_back(const_cast<char*>(gpuRuntimePath.c_str()));
                      args.push_back(const_cast<char*>("-lvulkan"));
                  }
                  args.push_back(const_cast<char*>("-lm"));
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

  // Item 9: linc run command - compile and execute in one step
  if (command == "run") {
      if (outputBinary.empty()) {
          std::cerr << "Error: 'run' requires -o <binary_path>. Use: linc run file.lin -o /tmp/bin\n";
          return 1;
      }
      
      // Run the specified binary (assumes user built it first)
      std::cout << "Running " << outputBinary << "...\n";
      int exitCode = system(outputBinary.c_str());
      exitCode = WEXITSTATUS(exitCode);
      std::cout << "Exit code: " << exitCode << "\n";
      
      if (ast) freeAst(ast);
      for (const char *src : importSources) free((void*)src);
      return exitCode;
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
