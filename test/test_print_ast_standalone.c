#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lin/Parser.h"

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

void test_print_binary_manual() {
    printf("--- Binary Type (Manual) ---\n");
    // AST_BINARY is not produced by the parser currently (it produces AST_CALL),
    // so we construct it manually to test the printAst case.
    AstNode *left = calloc(1, sizeof(AstNode));
    left->type = AST_NUMBER;
    left->as.number.value = 1;

    AstNode *right = calloc(1, sizeof(AstNode));
    right->type = AST_NUMBER;
    right->as.number.value = 2;

    AstNode *bin = calloc(1, sizeof(AstNode));
    bin->type = AST_BINARY;
    bin->as.binary.left = left;
    bin->as.binary.right = right;
    bin->as.binary.op = TOKEN_PLUS;

    printAst(bin, 0);

    freeAst(bin);
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
    test_print_binary_manual();
    test_print_field_access();
    return 0;
}
