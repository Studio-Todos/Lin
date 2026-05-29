#ifndef LINALANG_AST_H
#define LINALANG_AST_H

#include "lin/Parser.h"

#ifdef __cplusplus
extern "C" {
#endif

void freeAst(AstNode *node);
void printAst(AstNode *node, int depth);

#ifdef __cplusplus
}
#endif

#endif // LINALANG_AST_H
