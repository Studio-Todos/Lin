#include <stdio.h>
#include <stdlib.h>
#include "lin/Parser.h"
int main() {
    AstNode *ast = parse("while [ x < 10 ] [ x: x + 1 ]");
    if (!ast) return 1;
    freeAst(ast);
    printf("Parsing while successful\n");
    return 0;
}
