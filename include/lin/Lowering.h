#ifndef LINALANG_LOWERING_H
#define LINALANG_LOWERING_H

#include "mlir-c/IR.h"
#include "lin/Parser.h"

MlirModule lowerAstToMlir(MlirContext ctx, AstNode *ast);

// Future E-Graph optimization placeholder hook
void optimizeInteractionNetWithEGraphs(MlirModule module);

#endif // LINALANG_LOWERING_H
