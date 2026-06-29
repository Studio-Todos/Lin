#include "lin/Ast.h"
#include <stdio.h>
#include <stdlib.h>

void freeAst(AstNode *node) {
    if (!node) return;
    if (node->type == AST_BINARY) {
        freeAst(node->as.binary.left);
        freeAst(node->as.binary.right);
    } else if (node->type == AST_ASSIGNMENT) {
        freeAst(node->as.assignment.value);
    } else if (node->type == AST_CALL) {
        if (node->as.call.resolved_callee) free((void*)node->as.call.resolved_callee);
        for (int i=0; i<node->as.call.arg_count; i++) freeAst(node->as.call.args[i]);
        free(node->as.call.args);
    } else if (node->type == AST_STRING) {
        free((void*)node->as.string.value);
    } else if (node->type == AST_WHILE) {
        freeAst(node->as.while_loop.condition);
        freeAst(node->as.while_loop.body);
    } else if (node->type == AST_PAIR) {
        freeAst(node->as.pair.left);
        freeAst(node->as.pair.right);
    } else if (node->type == AST_FIELD_ACCESS) {
        freeAst(node->as.field_access.base);
    } else if (node->type == AST_BLOCK || node->type == AST_BLOCK_DATA || node->type == AST_BLOCK_LITERAL) {
        for (int i=0; i<node->as.block.count; i++) freeAst(node->as.block.statements[i]);
        free(node->as.block.statements);
    } else if (node->type == AST_FUNC_DECL) {
        freeAst(node->as.func_decl.body);
        free(node->as.func_decl.args);
    } else if (node->type == AST_IMPORT) {
        // do nothing for imported AST root nodes to avoid double free
    }
    free(node);
}

void printAst(AstNode *node, int depth) {
    if (!node) return;
    for (int i=0; i<depth; i++) printf("  ");

    switch (node->type) {
        case AST_NUMBER: printf("Number(%ld)\n", (long)node->as.number.value); break;
        case AST_FLOAT: printf("Float(%f)\n", node->as.f_number.value); break;
        case AST_STRING: printf("String(%.*s)\n", node->as.string.length, node->as.string.value); break;
        case AST_BOOL: printf("Bool(%s)\n", node->as.boolean.value ? "true" : "false"); break;
        case AST_IDENTIFIER: printf("Ident(%.*s)\n", node->as.identifier.length, node->as.identifier.name); break;
        case AST_BINARY:
            printf("Binary(%d)\n", node->as.binary.op);
            printAst(node->as.binary.left, depth + 1);
            printAst(node->as.binary.right, depth + 1);
            break;
        case AST_CALL:
            printf("Call(%.*s)\n", node->as.call.callee_len, node->as.call.callee);
            for (int i=0; i<node->as.call.arg_count; i++) printAst(node->as.call.args[i], depth + 1);
            break;
        case AST_WHILE:
            printf("While\n");
            printAst(node->as.while_loop.condition, depth + 1);
            printAst(node->as.while_loop.body, depth + 1);
            break;
        case AST_PAIR:
            printf("Pair\n");
            printAst(node->as.pair.left, depth + 1);
            printAst(node->as.pair.right, depth + 1);
            break;
        case AST_FIELD_ACCESS:
            printf("FieldAccess(.%d)\n", node->as.field_access.field_index);
            printAst(node->as.field_access.base, depth + 1);
            break;
        case AST_BLOCK:
        case AST_BLOCK_DATA:
        case AST_BLOCK_LITERAL:
            printf("Block\n");
            for (int i=0; i<node->as.block.count; i++) printAst(node->as.block.statements[i], depth + 1);
            break;
        case AST_FUNC_DECL:
            if (node->as.func_decl.dispatch) {
                printf("FuncDecl(%.*s, dispatch: @%.*s, args: [", node->as.func_decl.name_len, node->as.func_decl.name, node->as.func_decl.dispatch_len, node->as.func_decl.dispatch);
            } else {
                printf("FuncDecl(%.*s, args: [", node->as.func_decl.name_len, node->as.func_decl.name);
            }
            for (int j = 0; j < node->as.func_decl.arg_count; j++) {
                printf("%.*s%s", node->as.func_decl.args[j].name_len, node->as.func_decl.args[j].name,
                       (j < node->as.func_decl.arg_count - 1) ? ", " : "");
            }
            printf("])\n");
            printAst(node->as.func_decl.body, depth + 1);
            break;
        case AST_MLIR_OP:
            if (node->as.mlir_op.dispatch) {
                printf("MlirOp(%.*s, dispatch: @%.*s)\n", node->as.mlir_op.name_len, node->as.mlir_op.name, node->as.mlir_op.dispatch_len, node->as.mlir_op.dispatch);
            } else {
                printf("MlirOp(%.*s)\n", node->as.mlir_op.name_len, node->as.mlir_op.name);
            }
            if (node->as.mlir_op.inverse_payload) {
                printf("  inverse: %.*s\n", node->as.mlir_op.inverse_len, node->as.mlir_op.inverse_payload);
            }
            break;
        case AST_IMPORT:
            printf("Import(%.*s)\n", node->as.import_stmt.length, node->as.import_stmt.path);
            if (node->as.import_stmt.module_block) {
                printAst(node->as.import_stmt.module_block, depth + 1);
            }
            break;
        case AST_ASSIGNMENT:
            printf("Assignment(%.*s)\n", node->as.assignment.name_len, node->as.assignment.name);
            printAst(node->as.assignment.value, depth + 1);
            break;
    }
}
