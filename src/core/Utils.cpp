#include "Internal.h"
#include <iostream>
#include <cstring>

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
    }
    return errors;
}

bool hasGpuAnnotation(AstNode *node) {
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
