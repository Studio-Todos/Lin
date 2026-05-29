#ifndef LINALANG_LOWERING_H
#define LINALANG_LOWERING_H

#include "mlir-c/IR.h"
#include "lin/Parser.h"

#ifdef __cplusplus
extern "C" {
#endif

MlirModule lowerAstToMlir(MlirContext ctx, AstNode *ast);

#ifdef __cplusplus
}
#endif

#endif // LINALANG_LOWERING_H
