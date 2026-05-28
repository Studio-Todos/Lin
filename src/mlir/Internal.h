#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "lin/Parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;
    int name_len;
    MlirValue value;
} EnvVar;

typedef struct {
    EnvVar *vars;
    int count;
    int capacity;
} Environment;

void env_init(Environment *env);
void env_free(Environment *env, MlirContext ctx, MlirBlock block, MlirLocation loc);
void env_add(Environment *env, const char *name, int name_len, MlirValue value);
MlirValue env_get(Environment *env, const char *name, int name_len);
void env_set(Environment *env, const char *name, int name_len, MlirValue value);
MlirValue env_fetch(MlirContext ctx, MlirBlock block, MlirLocation loc, Environment *env, const char *name, int name_len);

MlirType getPicPortType(MlirContext ctx);

typedef struct {
    char (*names)[256];
    int count;
    int capacity;
} FreeVars;

void linkValues(MlirBlock block, MlirLocation loc, MlirValue v1, MlirValue v2);
void createOmega(MlirContext ctx, MlirBlock block, MlirLocation loc, const char *labelName, const char *polarity, MlirValue *p0, MlirValue *p1, MlirValue *p2);
MlirValue bundleClosure(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue omegaP0, MlirValue capBundle);
MlirValue createPair(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue left, MlirValue right);
MlirBlock createFunctionBlock(MlirContext ctx, MlirLocation loc, MlirBlock moduleBody, const char *prefixedName);
void registerFunction(MlirContext ctx, MlirLocation loc, MlirBlock moduleBody, const char *funcName, const char *prefixedName);
void addFreeVar(FreeVars *fv, const char *name, int len);
void findFreeVars(AstNode *node, FreeVars *fv, const char **bound, int bound_count);
int count_var_usage(AstNode *node, const char *name, int name_len);
MlirValue createEra(MlirContext ctx, MlirBlock block, MlirLocation loc);
void linkToEra(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue v);

void optimizeInteractionNetWithEGraphs(MlirModule module);

MlirValue createOmegaP0(MlirContext ctx, MlirBlock block, MlirLocation loc, const char *labelName, const char *polarity);

#ifdef __cplusplus
}
#endif
