#include "Internal.h"
#include <iostream>
#include <cstring>
#include <array>

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
//
// Per the PIC spec (.todo item 13): the omega (Operation) node label drives
// rule dispatch (annihilation fires when matching omega(+,L) meets omega(-,L)).
// Argument type annotations therefore determine which omega label is emitted
// during lowering. This pass rewrites AST_CALL.resolved_callee before lowering
// so that e.g. "add" becomes "fadd" when f32 arguments are detected.
using TypeEnv = std::unordered_map<std::string, std::string>;
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
std::unordered_map<std::string,OpSig> buildOpSigTable(AstNode *root) {
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
void typeDirectedDispatch(AstNode *node, TypeEnv env,
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
