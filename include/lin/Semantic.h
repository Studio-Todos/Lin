#ifndef LINALANG_SEMANTIC_H
#define LINALANG_SEMANTIC_H

#include "lin/Parser.h"
#include <unordered_set>
#include <string>

// Run all checkstyle and static analysis. Returns error count (0 on success).
// source: the full source text (for diagnostic source-line display), can be NULL.
int checkstyleAst(AstNode *node, const char *source);

// Checks if function or sub-statements have GPU annotation.
bool hasGpuAnnotation(AstNode *node);

// Run semantic type checking. Returns error count.
// source: the full source text (for diagnostic source-line display), can be NULL.
int semanticTypeCheckAst(AstNode *node, std::unordered_set<std::string>& declaredTypes, const char *source);

// Type-directed omega-label selection/dispatch.
void performTypeDirectedDispatch(AstNode *ast);

#endif // LINALANG_SEMANTIC_H