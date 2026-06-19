#include "lin/Semantic.h"
#include "../lib/dialect/PicReduceUtils.h"
#include <iostream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <array>
#include <cctype>
#include <cstring>

// Helper to check if a string is idiomatic Red/Rebol style
static bool isIdiomaticCase(const char* str, int len) {
    if (len == 0) return true;
    for (int i = 0; i < len; ++i) {
        if (std::isupper(str[i])) return false; // camelCase/PascalCase is invalid
    }
    return true;
}

// Semantic type checking: validate arg/return type names with source locations.
int semanticTypeCheckAst(AstNode *node, std::unordered_set<std::string>& declaredTypes) {
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
static std::unordered_map<std::string, OpSig> buildOpSigTable(AstNode *root) {
    std::unordered_map<std::string, OpSig> tbl;
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
                              const std::unordered_map<std::string, OpSig> &sigs) {
    if (!node) return "";
    switch (node->type) {
        case AST_NUMBER: return kDefaultType;
        case AST_FLOAT:  return kF64Type;
        case AST_BOOL:   return kBoolType;
        case AST_STRING: return kStrType;
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
static const std::unordered_map<std::string, std::array<const char*, 4>> kBinVariants = {
    {"add", {"add","add64","fadd","fadd64"}},
    {"sub", {"sub","sub64","fsub","fsub64"}},
    {"mul", {"mul","mul64","fmul","fmul64"}},
    {"div", {"div","div64","fdiv","fdiv64"}},
    {"lt",  {"lt", "slt64", "flt", "flt64"}},
    {"gt",  {"gt", "sgt64", "fgt", "fgt64"}},
    {"le",  {"le", "sle64", "fle", "fle64"}},
    {"ge",  {"ge", "sge64", "fge", "fge64"}},
    {"eq",  {"eq", "eq64", "feq", "feq64"}},
    {"ne",  {"ne", "ne64","fneq","fneq64"}},
};

// Typed variant table: generic 1-arg op → arg type → specific variant
static const std::unordered_map<std::string, std::unordered_map<std::string, const char*>> kUnaryVariants = {
    {"print", {
        {"i32",  "print_i32"},
        {kDefaultType, "print_i64"},
        {"f32",  "print_f32"},
        {kF64Type,  "print_f64"},
        {kStrType,  "print_str"},
        {kBoolType, "print_i32"},
    }},
};

// Walk AST and rewrite resolved_callee for typed binary ops.
static void typeDirectedDispatch(AstNode *node, TypeEnv env,
                                  const std::unordered_map<std::string, OpSig> &sigs) {
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
                int ti = typeIndex(ty);
                if (ti >= 0) variant = vit->second[ti];
                if (variant && strcmp(variant, node->as.call.callee) != 0) {
                    if (node->as.call.resolved_callee) free((void*)node->as.call.resolved_callee);
                    node->as.call.resolved_callee = strdup(variant);
                }
            }
            auto uit = kUnaryVariants.find(callee);
            if (uit != kUnaryVariants.end() && node->as.call.arg_count >= 1) {
                int tyIdx = (node->as.call.arg_count == 2) ? 1 : 0;
                std::string ty = inferType(node->as.call.args[tyIdx], env, sigs);
                auto tyIt = uit->second.find(ty);
                if (tyIt != uit->second.end()) {
                    const char *variant = tyIt->second;
                    if (strcmp(variant, node->as.call.callee) != 0) {
                        if (node->as.call.resolved_callee) free((void*)node->as.call.resolved_callee);
                        node->as.call.resolved_callee = strdup(variant);
                    }
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

void performTypeDirectedDispatch(AstNode *ast) {
    auto opSigs = buildOpSigTable(ast);
    TypeEnv rootEnv;
    typeDirectedDispatch(ast, rootEnv, opSigs);
}

// Recursive AST traversal for checkstyle
int checkstyleAst(AstNode *node) {
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
        default:
            break;
    }
    return errors;
}

bool hasGpuAnnotation(AstNode *node) {
    if (!node) return false;
    switch (node->type) {
        case AST_FUNC_DECL:
            if (node->as.func_decl.dispatch && 
                strncmp(node->as.func_decl.dispatch, kTargetGpu, node->as.func_decl.dispatch_len) == 0) {
                return true;
            }
            return hasGpuAnnotation(node->as.func_decl.body);
        case AST_MLIR_OP:
            if (node->as.mlir_op.dispatch && 
                strncmp(node->as.mlir_op.dispatch, kTargetGpu, node->as.mlir_op.dispatch_len) == 0) {
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
