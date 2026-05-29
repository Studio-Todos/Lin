#ifndef LINALANG_EGRAPHOPTIMIZER_H
#define LINALANG_EGRAPHOPTIMIZER_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

void optimizeInteractionNetWithEGraphs(MlirModule module);

#ifdef __cplusplus
}
#endif

#endif // LINALANG_EGRAPHOPTIMIZER_H
