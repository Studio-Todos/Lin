#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lin/Parser.h"
#include "lin/Ast.h"

void test_print_simple() {
    printf("--- Simple Types ---\n");
    const char *source = "42 3.14 \"hello\" true my-var";
    AstNode *ast = parse(source);
    if (ast) {
        printAst(ast, 0);
        freeAst(ast);
    }
}

void test_print_complex() {
    printf("--- Complex Types ---\n");
    const char *source =
        "x: 10\n"
        "(print x)\n"
        "while true [ (nop) ]\n"
        "either true [ 1 ] [ 2 ]\n"
        "f: func [a [i32!] return: [i32!]] [ a ]\n"
        "import \"std/math.lin\"\n"
        "my-op: mlir-op [ %0 ] { %1 = \"test.op\"(%0) : (i32) -> i32 }";
    AstNode *ast = parse(source);
    if (ast) {
        printAst(ast, 0);
        freeAst(ast);
    }
}

void test_print_call_expr() {
    printf("--- Call Expression (Manual) ---\n");
    AstNode *arg1 = calloc(1, sizeof(AstNode));
    arg1->type = AST_NUMBER;
    arg1->as.number.value = 1;

    AstNode *arg2 = calloc(1, sizeof(AstNode));
    arg2->type = AST_NUMBER;
    arg2->as.number.value = 2;

    AstNode *call = calloc(1, sizeof(AstNode));
    call->type = AST_CALL;
    call->as.call.callee = "add";
    call->as.call.callee_len = 3;
    call->as.call.arg_count = 2;
    call->as.call.args = malloc(sizeof(AstNode*) * 2);
    call->as.call.args[0] = arg1;
    call->as.call.args[1] = arg2;

    printAst(call, 0);
    freeAst(call);
}

void test_print_field_access() {
    printf("--- Field Access ---\n");
    const char *source = "obj.0";
    AstNode *ast = parse(source);
    if (ast) {
        printAst(ast, 0);
        freeAst(ast);
    }
}

int main() {
    test_print_simple();
    test_print_complex();
    test_print_call_expr();
    test_print_field_access();
    return 0;
}
