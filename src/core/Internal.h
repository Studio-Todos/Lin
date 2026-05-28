#pragma once

#include "lin/Parser.h"
#include <unordered_set>
#include <unordered_map>
#include <string>

using TypeEnv = std::unordered_map<std::string, std::string>;

int semanticTypeCheckAst(AstNode *node, std::unordered_set<std::string>& declaredTypes);
#include <vector>
struct OpSig { std::vector<std::string> inputTypes; std::string outputType; };

void typeDirectedDispatch(AstNode *node, TypeEnv env, const std::unordered_map<std::string, OpSig> &sigs);
int checkstyleAst(AstNode *node);
bool hasGpuAnnotation(AstNode *node);

std::unordered_map<std::string,OpSig> buildOpSigTable(AstNode *root);
