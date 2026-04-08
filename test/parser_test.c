#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "lin/Parser.h"

void test_number() {
    const char *source = "42";
    AstNode *ast = parse(source);
    assert(ast != NULL);
    assert(ast->type == AST_BLOCK);
    assert(ast->as.block.count == 1);
    assert(ast->as.block.statements[0]->type == AST_NUMBER);
    assert(ast->as.block.statements[0]->as.number.value == 42);
    freeAst(ast);
    printf("test_number passed\n");
}

void test_string() {
    const char *source = "\"hello\"";
    AstNode *ast = parse(source);
    assert(ast != NULL);
    assert(ast->type == AST_BLOCK);
    assert(ast->as.block.count == 1);
    assert(ast->as.block.statements[0]->type == AST_STRING);
    assert(ast->as.block.statements[0]->as.string.length == 5);
    assert(strncmp(ast->as.block.statements[0]->as.string.value, "hello", 5) == 0);
    freeAst(ast);
    printf("test_string passed\n");
}

void test_identifier() {
    const char *source = "my-var";
    AstNode *ast = parse(source);
    assert(ast != NULL);
    assert(ast->type == AST_BLOCK);
    assert(ast->as.block.count == 1);
    assert(ast->as.block.statements[0]->type == AST_IDENTIFIER);
    assert(ast->as.block.statements[0]->as.identifier.length == 6);
    assert(strncmp(ast->as.block.statements[0]->as.identifier.name, "my-var", 6) == 0);
    freeAst(ast);
    printf("test_identifier passed\n");
}

void test_assignment() {
    const char *source = "x: 10";
    AstNode *ast = parse(source);
    assert(ast != NULL);
    assert(ast->type == AST_BLOCK);
    assert(ast->as.block.count == 1);
    assert(ast->as.block.statements[0]->type == AST_ASSIGNMENT);
    assert(ast->as.block.statements[0]->as.assignment.name_len == 1);
    assert(ast->as.block.statements[0]->as.assignment.name[0] == 'x');
    assert(ast->as.block.statements[0]->as.assignment.value->type == AST_NUMBER);
    assert(ast->as.block.statements[0]->as.assignment.value->as.number.value == 10);
    freeAst(ast);
    printf("test_assignment passed\n");
}

void test_call() {
    const char *source = "(print \"hi\")";
    AstNode *ast = parse(source);
    assert(ast != NULL);
    assert(ast->type == AST_BLOCK);
    assert(ast->as.block.count == 1);
    assert(ast->as.block.statements[0]->type == AST_CALL);
    assert(strncmp(ast->as.block.statements[0]->as.call.callee, "print", 5) == 0);
    assert(ast->as.block.statements[0]->as.call.arg_count == 1);
    assert(ast->as.block.statements[0]->as.call.args[0]->type == AST_STRING);
    freeAst(ast);
    printf("test_call passed\n");
}

void test_either() {
    const char *source = "either 1 [ 2 ] [ 3 ]";
    AstNode *ast = parse(source);
    assert(ast != NULL);
    assert(ast->type == AST_BLOCK);
    assert(ast->as.block.count == 1);
    assert(ast->as.block.statements[0]->type == AST_CALL);
    assert(strncmp(ast->as.block.statements[0]->as.call.callee, "either", 6) == 0);
    assert(ast->as.block.statements[0]->as.call.arg_count == 3);
    assert(ast->as.block.statements[0]->as.call.args[0]->type == AST_NUMBER);
    assert(ast->as.block.statements[0]->as.call.args[1]->type == AST_BLOCK);
    assert(ast->as.block.statements[0]->as.call.args[2]->type == AST_BLOCK);
    freeAst(ast);
    printf("test_either passed\n");
}

void test_func_decl() {
    const char *source = "f: func [x [i32!] return: [i32!]] [ x ]";
    AstNode *ast = parse(source);
    assert(ast != NULL);
    assert(ast->type == AST_BLOCK);
    assert(ast->as.block.count == 1);
    assert(ast->as.block.statements[0]->type == AST_FUNC_DECL);
    assert(strncmp(ast->as.block.statements[0]->as.func_decl.name, "f", 1) == 0);
    assert(ast->as.block.statements[0]->as.func_decl.arg_count == 1);
    assert(strncmp(ast->as.block.statements[0]->as.func_decl.args[0].name, "x", 1) == 0);
    assert(ast->as.block.statements[0]->as.func_decl.body->type == AST_BLOCK);
    freeAst(ast);
    printf("test_func_decl passed\n");
}

void test_mlir_op() {
    const char *source = "my-op: mlir-op [ %0 ] { %1 = \"test.op\"(%0) : (i32) -> i32 }";
    AstNode *ast = parse(source);
    assert(ast != NULL);
    assert(ast->type == AST_BLOCK);
    assert(ast->as.block.count == 1);
    assert(ast->as.block.statements[0]->type == AST_MLIR_OP);
    assert(strncmp(ast->as.block.statements[0]->as.mlir_op.name, "my-op", 5) == 0);
    assert(ast->as.block.statements[0]->as.mlir_op.payload_len > 0);
    freeAst(ast);
    printf("test_mlir_op passed\n");
}

void test_import() {
    const char *source = "import \"std/math.lin\"";
    AstNode *ast = parse(source);
    assert(ast != NULL);
    assert(ast->type == AST_BLOCK);
    assert(ast->as.block.count == 1);
    assert(ast->as.block.statements[0]->type == AST_IMPORT);
    assert(ast->as.block.statements[0]->as.import_stmt.length == 12);
    assert(strncmp(ast->as.block.statements[0]->as.import_stmt.path, "std/math.lin", 12) == 0);
    freeAst(ast);
    printf("test_import passed\n");
}

int main() {
    test_number();
    test_string();
    test_identifier();
    test_assignment();
    test_call();
    test_either();
    test_func_decl();
    test_mlir_op();
    test_import();
    printf("All parser tests passed!\n");
    return 0;
}
