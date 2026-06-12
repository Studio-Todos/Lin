#include "lin/Lowering.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static void sanitizeMlirName(char *name) {
    for (char *p = name; *p; p++) {
        if (*p == '-') *p = '_';
        if (*p == '?') *p = '_';
    }
}

#define MAX_MLIR_OPS 256

static const char *mlirOpNames[MAX_MLIR_OPS];
static int mlirOpCount = 0;

static void addMlirOpName(const char *name, int len) {
    if (mlirOpCount < MAX_MLIR_OPS) {
        char *s = malloc(len + 1);
        strncpy(s, name, len);
        s[len] = '\0';
        mlirOpNames[mlirOpCount++] = s;
    }
}

static int isMlirOpName(const char *name, int len) {
    for (int i = 0; i < mlirOpCount; i++) {
        if ((int)strlen(mlirOpNames[i]) == len && strncmp(mlirOpNames[i], name, len) == 0)
            return 1;
    }
    return 0;
}

static MlirValue createEra(MlirContext ctx, MlirBlock block, MlirLocation loc);
static void linkToEra(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue v);

#ifdef ENABLE_DEBUG_LOGS
#define LOG_REDEX(...) printf(__VA_ARGS__)
#define LOG_STDERR(...) fprintf(stderr, __VA_ARGS__)
#else
#define LOG_REDEX(...) ((void)0)
#define LOG_STDERR(...) ((void)0)
#endif

static MlirType getPicPortType(MlirContext ctx) {
    return mlirTypeParseGet(ctx, mlirStringRefCreateFromCString("!pic_graph.port"));
}

static void linkValues(MlirBlock block, MlirLocation loc, MlirValue v1, MlirValue v2) {
    MlirOperationState linkState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
    MlirValue linkOps[] = {v1, v2};
    mlirOperationStateAddOperands(&linkState, 2, linkOps);
    mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkState));
}

static MlirValue createOmegaP0(MlirContext ctx, MlirBlock block, MlirLocation loc, const char *labelName, const char *polarity) {
    MlirType portType = getPicPortType(ctx);
    MlirType agentTypes[] = {portType, portType, portType};

    MlirOperationState baseState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    MlirAttribute agTypeAttr  = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
    MlirNamedAttribute agTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), agTypeAttr);
    MlirAttribute fnPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(polarity));
    MlirNamedAttribute fnPolNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), fnPolAttr);
    MlirAttribute fnLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(labelName));
    MlirNamedAttribute fnLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), fnLabelAttr);
    MlirNamedAttribute agentAttrs[3];
    agentAttrs[0] = agTypeNamed;
    agentAttrs[1] = fnPolNamedAttr;
    agentAttrs[2] = fnLabelNamedAttr;
    mlirOperationStateAddAttributes(&baseState, 3, agentAttrs);
    mlirOperationStateAddResults(&baseState, 3, agentTypes);
    MlirOperation baseOp = mlirOperationCreate(&baseState);
    mlirBlockAppendOwnedOperation(block, baseOp);

    if (strcmp(polarity, "+") == 0) {
        linkToEra(ctx, block, loc, mlirOperationGetResult(baseOp, 1));
        linkToEra(ctx, block, loc, mlirOperationGetResult(baseOp, 2));
    }

    return mlirOperationGetResult(baseOp, 0);
}

static void createOmega(MlirContext ctx, MlirBlock block, MlirLocation loc, const char *labelName, const char *polarity, MlirValue *p0, MlirValue *p1, MlirValue *p2) {
    MlirType portType = getPicPortType(ctx);
    MlirType agentTypes[] = {portType, portType, portType};

    MlirOperationState baseState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    MlirAttribute agTypeAttr  = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
    MlirNamedAttribute agTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), agTypeAttr);
    MlirAttribute fnPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(polarity));
    MlirNamedAttribute fnPolNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), fnPolAttr);
    MlirAttribute fnLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(labelName));
    MlirNamedAttribute fnLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), fnLabelAttr);
    MlirNamedAttribute agentAttrs[3];
    agentAttrs[0] = agTypeNamed;
    agentAttrs[1] = fnPolNamedAttr;
    agentAttrs[2] = fnLabelNamedAttr;
    mlirOperationStateAddAttributes(&baseState, 3, agentAttrs);
    mlirOperationStateAddResults(&baseState, 3, agentTypes);
    MlirOperation baseOp = mlirOperationCreate(&baseState);
    mlirBlockAppendOwnedOperation(block, baseOp);
    *p0 = mlirOperationGetResult(baseOp, 0);
    *p1 = mlirOperationGetResult(baseOp, 1);
    *p2 = mlirOperationGetResult(baseOp, 2);
}


static MlirValue bundleClosure(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue omegaP0, MlirValue capBundle) {
    MlirType portType = getPicPortType(ctx);
    MlirType agentTypes[] = {portType, portType, portType};

    MlirAttribute pairType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
    MlirNamedAttribute pairTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), pairType);
    MlirAttribute plusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
    MlirNamedAttribute plusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), plusPol);
    MlirAttribute labelPair = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
    MlirNamedAttribute labelPairAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelPair);

    MlirOperationState closureState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    MlirNamedAttribute closureAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
    mlirOperationStateAddAttributes(&closureState, 3, closureAttrs);
    mlirOperationStateAddResults(&closureState, 3, agentTypes);
    MlirOperation closureOp = mlirOperationCreate(&closureState);
    mlirBlockAppendOwnedOperation(block, closureOp);

    MlirValue clP0 = mlirOperationGetResult(closureOp, 0);
    MlirValue clP1 = mlirOperationGetResult(closureOp, 1);
    MlirValue clP2 = mlirOperationGetResult(closureOp, 2);

    linkValues(block, loc, clP1, omegaP0);
    linkValues(block, loc, clP2, capBundle);

    return clP0;
}

static MlirValue createPair(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue left, MlirValue right) {
    MlirType portType = getPicPortType(ctx);
    MlirType agentTypes[] = {portType, portType, portType};

    MlirOperationState pairState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("delta"));
    MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
    MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
    MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
    MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
    MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);
    MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
    mlirOperationStateAddAttributes(&pairState, 3, attrs);
    mlirOperationStateAddResults(&pairState, 3, agentTypes);

    MlirOperation pairOp = mlirOperationCreate(&pairState);
    mlirBlockAppendOwnedOperation(block, pairOp);

    MlirValue p0 = mlirOperationGetResult(pairOp, 0);
    MlirValue p1 = mlirOperationGetResult(pairOp, 1);
    MlirValue p2 = mlirOperationGetResult(pairOp, 2);

    linkValues(block, loc, p1, left);
    linkValues(block, loc, p2, right);

    return p0;
}


static MlirBlock createFunctionBlock(MlirContext ctx, MlirLocation loc, MlirBlock moduleBody, const char *prefixedName) {
    MlirType portType = getPicPortType(ctx);
    MlirType funcArgTypes[] = {portType, portType, portType};
    MlirLocation funcArgLocs[] = {loc, loc, loc};
    MlirType retTypes[] = {portType};
    MlirType funcType = mlirFunctionTypeGet(ctx, 3, funcArgTypes, 1, retTypes);

    MlirOperationState funcState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);
    MlirAttribute fnNameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(prefixedName));
    MlirNamedAttribute fnNameNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")), fnNameAttr);
    MlirAttribute fnTypeAttr = mlirTypeAttrGet(funcType);
    MlirNamedAttribute fnTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")), fnTypeAttr);
    MlirNamedAttribute funcAttrs[] = {fnNameNamed, fnTypeNamed};
    mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);

    MlirRegion innerRegion = mlirRegionCreate();
    MlirBlock innerBlock = mlirBlockCreate(3, funcArgTypes, funcArgLocs);
    mlirRegionAppendOwnedBlock(innerRegion, innerBlock);
    mlirOperationStateAddOwnedRegions(&funcState, 1, &innerRegion);

    MlirOperation funcOp = mlirOperationCreate(&funcState);
    if (!mlirBlockIsNull(moduleBody)) {
        mlirBlockAppendOwnedOperation(moduleBody, funcOp);
    }
    return innerBlock;
}

static void registerFunction(MlirContext ctx, MlirLocation loc, MlirBlock moduleBody, const char *funcName, const char *prefixedName) {
    MlirOperationState regState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.registry"), loc);
    MlirAttribute opNameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(funcName));
    MlirNamedAttribute opNameNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("op_name")), opNameAttr);
    
    char payloadBuf[2048];
    snprintf(payloadBuf, sizeof(payloadBuf), "  %%res = func.call @%s(%%env, %%arg, %%state) : (i64, i64, i64) -> i64\n", prefixedName);
    MlirAttribute payloadAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(payloadBuf));
    MlirNamedAttribute payloadNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("payload")), payloadAttr);
    
    MlirAttribute argNamesAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("[env][arg][state]"));
    MlirNamedAttribute argNamesNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("arg_names")), argNamesAttr);
    
    MlirNamedAttribute regAttrs[] = {opNameNamed, payloadNamed, argNamesNamed};
    mlirOperationStateAddAttributes(&regState, 3, regAttrs);
    if (!mlirBlockIsNull(moduleBody)) {
        mlirBlockAppendOwnedOperation(moduleBody, mlirOperationCreate(&regState));
    }
}


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

static void env_init(Environment *env) {
    env->capacity = 16;
    env->count = 0;
    env->vars = (EnvVar*)malloc(sizeof(EnvVar) * env->capacity);
    if (!env->vars) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
}

static void env_free(Environment *env, MlirContext ctx, MlirBlock block, MlirLocation loc) {
    // Before freeing the environment, any variable that is still bound (meaning it wasn't consumed completely)
    // must be connected to an Eraser (epsilon) node to satisfy the linearity property.
    if (!mlirBlockIsNull(block)) {
        for (int i = 0; i < env->count; i++) {
            if (!mlirValueIsNull(env->vars[i].value)) {
                MlirOperationState eraState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

                MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
                MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
                MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
                MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
                MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
                MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

                MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
                mlirOperationStateAddAttributes(&eraState, 3, attrs);

                MlirType portType = getPicPortType(ctx);
                MlirType eraTypes[] = {portType, portType, portType};
                mlirOperationStateAddResults(&eraState, 3, eraTypes);

                MlirOperation eraOp = mlirOperationCreate(&eraState);
                mlirBlockAppendOwnedOperation(block, eraOp);

                MlirValue eraP0 = mlirOperationGetResult(eraOp, 0);

                MlirOperationState linkState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue linkOps[] = {eraP0, env->vars[i].value};
                mlirOperationStateAddOperands(&linkState, 2, linkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkState));
            }
        }
    }

    free(env->vars);
}

static void env_add(Environment *env, const char *name, int name_len, MlirValue value) {
    if (env->count >= env->capacity) {
        env->capacity *= 2;
        EnvVar *tmp = (EnvVar*)realloc(env->vars, sizeof(EnvVar) * env->capacity);
        if (!tmp) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
        env->vars = tmp;
    }
    env->vars[env->count].name = name;
    env->vars[env->count].name_len = name_len;
    env->vars[env->count].value = value;
    env->count++;
}

static MlirValue env_get(Environment *env, const char *name, int name_len) {
    // Go backwards to get the most recent binding
    for (int i = env->count - 1; i >= 0; i--) {
        if (env->vars[i].name_len == name_len && strncmp(env->vars[i].name, name, name_len) == 0) {
            return env->vars[i].value;
        }
    }
    MlirValue nullVal = {NULL};
    return nullVal;
}

static void env_set(Environment *env, const char *name, int name_len, MlirValue value);

static MlirValue env_fetch(MlirContext ctx, MlirBlock block, MlirLocation loc, Environment *env, const char *name, int name_len) {
    MlirValue val = env_get(env, name, name_len);
    if (mlirValueIsNull(val)) return val;

    MlirType portType = getPicPortType(ctx);
    MlirOperationState dupState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("delta"));
    MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
    MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
    MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
    MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("dup"));
    MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);
    MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
    mlirOperationStateAddAttributes(&dupState, 3, attrs);
    MlirType dupTypes[] = {portType, portType, portType};
    mlirOperationStateAddResults(&dupState, 3, dupTypes);
    MlirOperation dupOp = mlirOperationCreate(&dupState);
    mlirBlockAppendOwnedOperation(block, dupOp);

    MlirValue dupP0 = mlirOperationGetResult(dupOp, 0); // principal
    MlirValue dupP1 = mlirOperationGetResult(dupOp, 1); // aux l
    MlirValue dupP2 = mlirOperationGetResult(dupOp, 2); // aux r

    MlirOperationState linkState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
    MlirValue linkOps[] = {dupP0, val};
    mlirOperationStateAddOperands(&linkState, 2, linkOps);
    mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkState));

    env_set(env, name, name_len, dupP2);
    return dupP1;
}

typedef struct {
    char (*names)[256];
    int count;
    int capacity;
} FreeVars;

static void addFreeVar(FreeVars *fv, const char *name, int len) {
    if (len > 255) len = 255;
    for (int i = 0; i < fv->count; i++) {
        if ((int)strlen(fv->names[i]) == len && strncmp(fv->names[i], name, len) == 0) return;
    }
    if (fv->count >= fv->capacity) {
        fv->capacity = fv->capacity < 8 ? 8 : fv->capacity * 2;
        fv->names = realloc(fv->names, sizeof(*fv->names) * fv->capacity);
    }
    strncpy(fv->names[fv->count], name, len);
    fv->names[fv->count][len] = '\0';
    fv->count++;
}

static void findFreeVars(AstNode *node, FreeVars *fv, const char **bound, int bound_count) {
    if (!node) return;
    switch (node->type) {
        case AST_IDENTIFIER: {
            bool is_bound = false;
            for (int i = 0; i < bound_count; i++) {
                if (strlen(bound[i]) == (size_t)node->as.identifier.length &&
                    strncmp(node->as.identifier.name, bound[i], node->as.identifier.length) == 0) {
                    is_bound = true;
                    break;
                }
            }
            if (!is_bound) addFreeVar(fv, node->as.identifier.name, node->as.identifier.length);
            break;
        }
        case AST_ASSIGNMENT: {
            findFreeVars(node->as.assignment.value, fv, bound, bound_count);
            break;
        }
        case AST_BINARY: {
            findFreeVars(node->as.binary.left, fv, bound, bound_count);
            findFreeVars(node->as.binary.right, fv, bound, bound_count);
            break;
        }
        case AST_CALL: {
            // Callee might be a variable
            bool is_bound = false;
            for (int i = 0; i < bound_count; i++) {
                if (strlen(bound[i]) == (size_t)node->as.call.callee_len &&
                    strncmp(node->as.call.callee, bound[i], node->as.call.callee_len) == 0) {
                    is_bound = true;
                    break;
                }
            }
            // Item 4c: do NOT special-case any callee as a builtin here.
            // If a name like "add" or "either" is not in scope it will fail at use time,
            // which is correct per the spec's explicit-import philosophy.
            if (!is_bound) addFreeVar(fv, node->as.call.callee, node->as.call.callee_len);
            
            for (int i = 0; i < node->as.call.arg_count; i++) {
                findFreeVars(node->as.call.args[i], fv, bound, bound_count);
            }
            break;
        }
        case AST_BLOCK:
        case AST_BLOCK_DATA: {
            // For a block, we need to handle local assignments as bound variables
            int local_capacity = node->as.block.count;
            const char **local_bound = malloc(sizeof(char*) * (bound_count + local_capacity));
            memcpy(local_bound, bound, sizeof(char*) * bound_count);
            int local_count = bound_count;
            
            for (int i = 0; i < node->as.block.count; i++) {
                AstNode *stmt = node->as.block.statements[i];
                if (stmt->type == AST_ASSIGNMENT) {
                    // Pre-scan won't work perfectly for sequential assignments but it's OK for now
                    findFreeVars(stmt->as.assignment.value, fv, local_bound, local_count);
                    
                    // Duplicate check before adding to local_bound
                    bool already_bound = false;
                    for (int j = 0; j < local_count; j++) {
                        if (strlen(local_bound[j]) == (size_t)stmt->as.assignment.name_len &&
                            strncmp(local_bound[j], stmt->as.assignment.name, stmt->as.assignment.name_len) == 0) {
                            already_bound = true;
                            break;
                        }
                    }
                    if (!already_bound) {
                        char *name = malloc(stmt->as.assignment.name_len + 1);
                        memcpy(name, stmt->as.assignment.name, stmt->as.assignment.name_len);
                        name[stmt->as.assignment.name_len] = '\0';
                        local_bound[local_count++] = name;
                    }
                } else if (stmt->type == AST_FUNC_DECL && stmt->as.func_decl.name_len > 0) {
                     char *name = malloc(stmt->as.func_decl.name_len + 1);
                     memcpy(name, stmt->as.func_decl.name, stmt->as.func_decl.name_len);
                     name[stmt->as.func_decl.name_len] = '\0';
                     local_bound[local_count++] = name;
                     findFreeVars(stmt, fv, local_bound, local_count);
                } else {
                    findFreeVars(stmt, fv, local_bound, local_count);
                }
            }
            // Cleanup local names
            for (int i = bound_count; i < local_count; i++) free((void*)local_bound[i]);
            free(local_bound);
            break;
        }
        case AST_FUNC_DECL: {
            int new_bound_count = bound_count + node->as.func_decl.arg_count;
            const char **new_bound = malloc(sizeof(char*) * new_bound_count);
            memcpy(new_bound, bound, sizeof(char*) * bound_count);
            for (int i = 0; i < node->as.func_decl.arg_count; i++) {
                new_bound[bound_count + i] = node->as.func_decl.args[i].name;
            }
            findFreeVars(node->as.func_decl.body, fv, new_bound, new_bound_count);
            free(new_bound);
            break;
        }
        case AST_WHILE: {
            findFreeVars(node->as.while_loop.condition, fv, bound, bound_count);
            findFreeVars(node->as.while_loop.body, fv, bound, bound_count);
            break;
        }
        case AST_PAIR: {
            findFreeVars(node->as.pair.left, fv, bound, bound_count);
            findFreeVars(node->as.pair.right, fv, bound, bound_count);
            break;
        }
        default: break;
    }
}

static void env_set(Environment *env, const char *name, int name_len, MlirValue value) {
    // Update the most recent binding if we duplicate
    for (int i = env->count - 1; i >= 0; i--) {
        if (env->vars[i].name_len == name_len && strncmp(env->vars[i].name, name, name_len) == 0) {
            env->vars[i].value = value;
            return;
        }
    }
    // If not found, shouldn't happen for valid use, but add it just in case
    env_add(env, name, name_len, value);
}

// Counts occurrences of a variable in the AST to determine Dup/Era insertion
static int count_var_usage(AstNode *node, const char *name, int name_len) {
    if (!node) return 0;

    if (node->type == AST_IDENTIFIER) {
        if (node->as.identifier.length == name_len && strncmp(node->as.identifier.name, name, name_len) == 0) {
            return 1;
        }
        return 0;
    }

    if (node->type == AST_BINARY) {
        return count_var_usage(node->as.binary.left, name, name_len) +
               count_var_usage(node->as.binary.right, name, name_len);
    }

    if (node->type == AST_BLOCK) {
        int c = 0;
        for (int i = 0; i < node->as.block.count; i++) {
            c += count_var_usage(node->as.block.statements[i], name, name_len);
        }
        return c;
    }

    if (node->type == AST_ASSIGNMENT) {
        // If assigned *to* this variable, it shadows it. We stop counting below this scope?
        // Actually, this is just simple counting for the expression RHS
        return count_var_usage(node->as.assignment.value, name, name_len);
    }

    if (node->type == AST_CALL) {
        int c = 0;
        // Don't count mlir-op names as variable references — they're registered ops,
        // not user-defined functions or variables.
        if (!isMlirOpName(node->as.call.callee, node->as.call.callee_len)) {
            if (node->as.call.callee_len == name_len && strncmp(node->as.call.callee, name, name_len) == 0) {
                c++;
            }
        }
        // NOTE: resolved_callee is intentionally NOT checked here.
        // It is set by type-directed dispatch (e.g. print(string) -> print_str)
        // and the resolved name is an mlir-op, not a variable reference.
        // Checking it would cause mlir-op names to be spuriously captured
        // as free variables, leading to incorrect closure call dispatch
        // instead of the built-in omega agent path.
        for (int i = 0; i < node->as.call.arg_count; i++) {
            c += count_var_usage(node->as.call.args[i], name, name_len);
        }
        return c;
    }

    if (node->type == AST_WHILE) {
        return count_var_usage(node->as.while_loop.condition, name, name_len) +
               count_var_usage(node->as.while_loop.body, name, name_len);
    }

    if (node->type == AST_IMPORT || node->type == AST_FUNC_DECL || node->type == AST_MLIR_OP || node->type == AST_STRING) {
        return 0;
    }

    return 0;
}

static MlirValue createEra(MlirContext ctx, MlirBlock block, MlirLocation loc) {
    MlirOperationState eraState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    MlirAttribute eraTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
    MlirNamedAttribute eraTypeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), eraTypeAttr);
    MlirAttribute eraPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
    MlirNamedAttribute eraPolNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), eraPolAttr);
    MlirAttribute eraLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
    MlirNamedAttribute eraLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), eraLabelAttr);
    MlirNamedAttribute eraAttrs[] = {eraTypeNamedAttr, eraPolNamedAttr, eraLabelNamedAttr};
    mlirOperationStateAddAttributes(&eraState, 3, eraAttrs);
    MlirType portType = getPicPortType(ctx);
    MlirType eraTypes[] = {portType, portType, portType};
    mlirOperationStateAddResults(&eraState, 3, eraTypes);
    MlirOperation eraOp = mlirOperationCreate(&eraState);
    mlirBlockAppendOwnedOperation(block, eraOp);
    return mlirOperationGetResult(eraOp, 0);
}

static void linkToEra(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue v) {
    linkValues(block, loc, v, createEra(ctx, block, loc));
}


static MlirValue lowerExpression(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env, bool is_top_level) {
    if (!expr) {
        MlirValue nullVal = {NULL};
        return nullVal;
    }

    MlirOperation parentOp = mlirBlockGetParentOperation(block);
    MlirOperation moduleOp = parentOp;
    while (!mlirOperationIsNull(moduleOp)) {
        MlirStringRef opName = mlirIdentifierStr(mlirOperationGetName(moduleOp));
        if (strncmp(opName.data, "builtin.module", 14) == 0) break;
        moduleOp = mlirOperationGetParentOperation(moduleOp);
    }

    MlirBlock moduleBody = {NULL};
    if (!mlirOperationIsNull(moduleOp)) {
        moduleBody = mlirModuleGetBody(mlirModuleFromOperation(moduleOp));
    }

    if (expr->type == AST_NUMBER || expr->type == AST_BOOL) {
        MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(expr->type == AST_NUMBER ? "i32" : "bool"));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

        int64_t val = (expr->type == AST_NUMBER) ? expr->as.number.value : (expr->as.boolean.value ? 1 : 0);
        MlirAttribute valAttr = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 64), val);
        MlirNamedAttribute valNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("value")), valAttr);

        MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr, valNamedAttr};
        mlirOperationStateAddAttributes(&state, 4, attrs);

        MlirType portType = getPicPortType(ctx);
        MlirType types[] = {portType, portType, portType};
        mlirOperationStateAddResults(&state, 3, types);

        MlirOperation op = mlirOperationCreate(&state);
        mlirBlockAppendOwnedOperation(block, op);
        return mlirOperationGetResult(op, 0); // principal port
    }


    if (expr->type == AST_FLOAT) {
        MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("f64"));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

        union { double f; int64_t i; } cast;
        cast.f = expr->as.f_number.value;

        MlirAttribute valAttr = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 64), cast.i);
        MlirNamedAttribute valNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("value")), valAttr);

        MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr, valNamedAttr};
        mlirOperationStateAddAttributes(&state, 4, attrs);

        MlirType portType = getPicPortType(ctx);
        MlirType types[] = {portType, portType, portType};
        mlirOperationStateAddResults(&state, 3, types);

        MlirOperation op = mlirOperationCreate(&state);
        mlirBlockAppendOwnedOperation(block, op);
        return mlirOperationGetResult(op, 0); // principal port
    }

    if (expr->type == AST_STRING) {
        // Encode a string as an op that will be translated to a global pointer
        MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("str"));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

        // the value will be the string payload
        MlirAttribute valAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.string.value, expr->as.string.length));
        MlirNamedAttribute valNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("str_val")), valAttr);

        MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr, valNamedAttr};
        mlirOperationStateAddAttributes(&state, 4, attrs);

        MlirType portType = getPicPortType(ctx);
        MlirType types[] = {portType, portType, portType};
        mlirOperationStateAddResults(&state, 3, types);

        MlirOperation op = mlirOperationCreate(&state);
        mlirBlockAppendOwnedOperation(block, op);
        return mlirOperationGetResult(op, 0); // principal port
    }

    if (expr->type == AST_IDENTIFIER) {
        MlirValue val = env_fetch(ctx, block, loc, env, expr->as.identifier.name, expr->as.identifier.length);
        if (mlirValueIsNull(val)) {
            fprintf(stderr, "Unbound variable: %.*s\n", expr->as.identifier.length, expr->as.identifier.name);
        }
        return val;
    }

    if (expr->type == AST_MLIR_OP) {
        addMlirOpName(expr->as.mlir_op.name, expr->as.mlir_op.name_len);

        MlirOperationState regState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.registry"), loc);

        MlirAttribute nameAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.name, expr->as.mlir_op.name_len));
        MlirNamedAttribute nameNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("op_name")), nameAttr);

        MlirAttribute payloadAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.mlir_payload, expr->as.mlir_op.payload_len));
        MlirNamedAttribute payloadNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("payload")), payloadAttr);

        // Pass the cleaned-up inputs as [%name1][%name2]...
        char cleanNames[2048] = "";
        const char *p = expr->as.mlir_op.inputs;
        int len = expr->as.mlir_op.inputs_len;
        bool foundInputs = false;
        for (int i = 0; i < len; i++) {
            if (p[i] == '[') {
                i++;
                while (i < len && isspace((unsigned char)p[i])) i++;
                if (i + 7 <= len && strncmp(p + i, "inputs:", 7) == 0) {
                    i += 7;
                    while (i < len && p[i] != '[') i++;
                    if (i < len && p[i] == '[') {
                        i++; // Enter [arg [type] ...]
                        while (i < len && p[i] != ']') {
                            while (i < len && isspace((unsigned char)p[i])) i++;
                            if (i >= len || p[i] == ']') break;

                            const char *nameStart = p + i;
                            while (i < len && !isspace((unsigned char)p[i]) && p[i] != '[' && p[i] != ']') i++;
                            int nameLen = (int)(p + i - nameStart);

                            if (nameLen > 0) {
                                strcat(cleanNames, "[%");
                                strncat(cleanNames, nameStart, nameLen);
                                
                                while (i < len && isspace((unsigned char)p[i])) i++;
                                if (i < len && p[i] == '[') {
                                    i++;
                                    const char *typeStart = p + i;
                                    while (i < len && p[i] != '!' && p[i] != ']' && !isspace((unsigned char)p[i])) {
                                        i++;
                                    }
                                    int typeLen = (int)(p + i - typeStart);
                                    if (typeLen > 0) {
                                        strcat(cleanNames, "_");
                                        strncat(cleanNames, typeStart, typeLen);
                                    }
                                    
                                    int depth = 1;
                                    while (i < len && depth > 0) {
                                        if (p[i] == '[') depth++;
                                        else if (p[i] == ']') depth--;
                                        i++;
                                    }
                                }
                                strcat(cleanNames, "]");
                            }
                        }
                    }
                    break;
                }
            }
        }
        MlirAttribute namesAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(cleanNames));
        MlirNamedAttribute namesNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("arg_names")), namesAttr);

        int regAttrCount = 3;
        MlirNamedAttribute attrs[4];
        attrs[0] = nameNamedAttr;
        attrs[1] = payloadNamedAttr;
        attrs[2] = namesNamedAttr;
        if (expr->as.mlir_op.dispatch) {
            MlirAttribute dispatchAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.dispatch, expr->as.mlir_op.dispatch_len));
            attrs[3] = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("dispatch")), dispatchAttr);
            regAttrCount = 4;
        }
        mlirOperationStateAddAttributes(&regState, regAttrCount, attrs);

        MlirOperation regOp = mlirOperationCreate(&regState);
        mlirBlockAppendOwnedOperation(block, regOp);

        MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.name, expr->as.mlir_op.name_len));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

        int agentAttrCount = 3;
        MlirNamedAttribute agentAttrs[4];
        agentAttrs[0] = typeNamedAttr;
        agentAttrs[1] = polNamedAttr;
        agentAttrs[2] = labelNamedAttr;
        if (expr->as.mlir_op.dispatch) {
            MlirAttribute dispatchAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.dispatch, expr->as.mlir_op.dispatch_len));
            agentAttrs[3] = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("dispatch")), dispatchAttr);
            agentAttrCount = 4;
        }
        mlirOperationStateAddAttributes(&state, agentAttrCount, agentAttrs);

        MlirType portType = getPicPortType(ctx);
        MlirType types[] = {portType, portType, portType};
        mlirOperationStateAddResults(&state, 3, types);

        MlirOperation op = mlirOperationCreate(&state);
        mlirBlockAppendOwnedOperation(block, op);

        linkToEra(ctx, block, loc, mlirOperationGetResult(op, 1));
        linkToEra(ctx, block, loc, mlirOperationGetResult(op, 2));
        
        MlirValue result = mlirOperationGetResult(op, 0);
        return result;
    }

    if (expr->type == AST_ASSIGNMENT) {
        MlirValue rhs = lowerExpression(ctx, block, loc, expr->as.assignment.value, env, false);
        env_add(env, expr->as.assignment.name, expr->as.assignment.name_len, rhs);
        return rhs;
    }

    if (expr->type == AST_PAIR) {
        MlirValue left = lowerExpression(ctx, block, loc, expr->as.pair.left, env, false);
        MlirValue right = lowerExpression(ctx, block, loc, expr->as.pair.right, env, false);

        MlirOperationState pairState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("delta"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

        MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
        mlirOperationStateAddAttributes(&pairState, 3, attrs);

        MlirType portType = getPicPortType(ctx);
        MlirType types[] = {portType, portType, portType};
        mlirOperationStateAddResults(&pairState, 3, types);

        MlirOperation pairOp = mlirOperationCreate(&pairState);
        mlirBlockAppendOwnedOperation(block, pairOp);

        MlirValue p0 = mlirOperationGetResult(pairOp, 0);
        MlirValue p1 = mlirOperationGetResult(pairOp, 1);
        MlirValue p2 = mlirOperationGetResult(pairOp, 2);

        MlirOperationState linkLeftState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue leftOps[] = {p1, left};
        mlirOperationStateAddOperands(&linkLeftState, 2, leftOps);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkLeftState));

        MlirOperationState linkRightState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue rightOps[] = {p2, right};
        mlirOperationStateAddOperands(&linkRightState, 2, rightOps);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkRightState));

        return p0;
    }

    if (expr->type == AST_FIELD_ACCESS) {
        MlirValue base_val = lowerExpression(ctx, block, loc, expr->as.field_access.base, env, false);
        
        MlirOperationState selState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        
        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("delta"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        
        char labelStr[16];
        snprintf(labelStr, sizeof(labelStr), "proj_%d", expr->as.field_access.field_index);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(labelStr));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);
        
        MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
        mlirOperationStateAddAttributes(&selState, 3, attrs);
        
        MlirType portType = getPicPortType(ctx);
        MlirType types[] = {portType, portType, portType};
        mlirOperationStateAddResults(&selState, 3, types);
        
        MlirOperation selOp = mlirOperationCreate(&selState);
        mlirBlockAppendOwnedOperation(block, selOp);
        
        MlirValue p0 = mlirOperationGetResult(selOp, 0);
        MlirValue p1 = mlirOperationGetResult(selOp, 1);
        MlirValue p2 = mlirOperationGetResult(selOp, 2);
        
        MlirOperationState linkState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue linkOps[] = {p1, base_val};
        mlirOperationStateAddOperands(&linkState, 2, linkOps);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkState));
        
        // Item 4d: p2 is unused when only one field is accessed; attach an eraser so no
        // port is left dangling (a dangling principal port causes the reduction to stall).
        linkToEra(ctx, block, loc, p2);
        
        return p0;
    }

    if (expr->type == AST_BLOCK) {
        MlirValue lastVal = {NULL};
        for (int i = 0; i < expr->as.block.count; i++) {
            AstNode *stmt = expr->as.block.statements[i];
            // Skip import and mlir_op since they have no runtime code
            if (!stmt) continue;
            if (stmt->type == AST_IMPORT) continue;
            lastVal = lowerExpression(ctx, block, loc, stmt, env, false);
        }

        if (mlirValueIsNull(lastVal)) {
            MlirOperationState fallbackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
            MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
            MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
            MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);
            MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
            mlirOperationStateAddAttributes(&fallbackState, 3, attrs);
            MlirType fbPortType = getPicPortType(ctx);
            MlirType types[] = {fbPortType, fbPortType, fbPortType};
            mlirOperationStateAddResults(&fallbackState, 3, types);
            MlirOperation fbOp = mlirOperationCreate(&fallbackState);
            mlirBlockAppendOwnedOperation(block, fbOp);
            lastVal = mlirOperationGetResult(fbOp, 0);
        }

        return lastVal;
    }

    if (expr->type == AST_WHILE) {
        // Find all active variables in env
        int active_count = 0;
        for (int i = 0; i < env->count; i++) {
            if (!mlirValueIsNull(env->vars[i].value)) {
                active_count++;
            }
        }
        const char **active_names = (const char **)malloc(sizeof(char*) * active_count);
        int *active_lens = (int *)malloc(sizeof(int) * active_count);
        MlirValue *active_vals = (MlirValue *)malloc(sizeof(MlirValue) * active_count);
        int idx = 0;
        for (int i = 0; i < env->count; i++) {
            if (!mlirValueIsNull(env->vars[i].value)) {
                active_names[idx] = env->vars[i].name;
                active_lens[idx] = env->vars[i].name_len;
                active_vals[idx] = env->vars[i].value;
                env->vars[i].value = (MlirValue){NULL}; // Consume
                idx++;
            }
        }

        MlirType portType = getPicPortType(ctx);
        MlirType agentTypes[] = {portType, portType, portType};

        MlirAttribute pairType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
        MlirNamedAttribute pairTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), pairType);
        MlirAttribute plusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
        MlirNamedAttribute plusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), plusPol);
        MlirAttribute minusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
        MlirNamedAttribute minusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), minusPol);
        MlirAttribute labelPair = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
        MlirNamedAttribute labelPairAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelPair);

        // Pack init_bundle in parent block
        MlirValue currentBundle = createEra(ctx, block, loc);
        for (int i = active_count - 1; i >= 0; i--) {
            MlirOperationState packState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirNamedAttribute packAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
            mlirOperationStateAddAttributes(&packState, 3, packAttrs);
            mlirOperationStateAddResults(&packState, 3, agentTypes);
            MlirOperation packOp = mlirOperationCreate(&packState);
            mlirBlockAppendOwnedOperation(block, packOp);

            MlirValue p0 = mlirOperationGetResult(packOp, 0);
            MlirValue p1 = mlirOperationGetResult(packOp, 1);
            MlirValue p2 = mlirOperationGetResult(packOp, 2);

            linkValues(block, loc, p1, active_vals[i]);
            linkValues(block, loc, p2, currentBundle);
            currentBundle = p0;
        }
        MlirValue init_bundle = currentBundle;

        static int loop_counter = 0;
        char macro_func_name[128];
        char body_func_name[128];
        char exit_func_name[128];
        snprintf(macro_func_name, sizeof(macro_func_name), "_loop_macro_%d", loop_counter);
        snprintf(body_func_name, sizeof(body_func_name), "_loop_body_%d", loop_counter);
        snprintf(exit_func_name, sizeof(exit_func_name), "_loop_exit_%d", loop_counter);
        loop_counter++;

        char prefixedMacroName[256];
        char prefixedBodyName[256];
        char prefixedExitName[256];
        sprintf(prefixedMacroName, "lin_%s", macro_func_name);
        sprintf(prefixedBodyName, "lin_%s", body_func_name);
        sprintf(prefixedExitName, "lin_%s", exit_func_name);

        // 1. Generate _loop_exit_X
        {
            MlirBlock exitBlock = createFunctionBlock(ctx, loc, moduleBody, prefixedExitName);
            registerFunction(ctx, loc, moduleBody, exit_func_name, prefixedExitName);

            MlirValue exitRawBundle = mlirBlockGetArgument(exitBlock, 0);
            MlirValue exitResultPort = mlirBlockGetArgument(exitBlock, 1);

            MlirOperationState exitUnpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirNamedAttribute exitUnpackAttrs[] = {pairTypeAttr, minusPolAttr, labelPairAttr};
            mlirOperationStateAddAttributes(&exitUnpackState, 3, exitUnpackAttrs);
            mlirOperationStateAddResults(&exitUnpackState, 3, agentTypes);
            MlirOperation exitUnpackOp = mlirOperationCreate(&exitUnpackState);
            mlirBlockAppendOwnedOperation(exitBlock, exitUnpackOp);
            linkValues(exitBlock, loc, mlirOperationGetResult(exitUnpackOp, 0), exitRawBundle);

            MlirValue exitEnvBundle = mlirOperationGetResult(exitUnpackOp, 1);
            MlirValue exitMainArg = mlirOperationGetResult(exitUnpackOp, 2);

            linkToEra(ctx, exitBlock, loc, exitEnvBundle);
            linkValues(exitBlock, loc, exitMainArg, exitResultPort);

            MlirOperationState exitRetState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
            MlirType exitI64Type = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString("i64"));
            MlirOperationState exitCastState = mlirOperationStateGet(mlirStringRefCreateFromCString("builtin.unrealized_conversion_cast"), loc);
            mlirOperationStateAddOperands(&exitCastState, 1, &exitResultPort);
            mlirOperationStateAddResults(&exitCastState, 1, &exitI64Type);
            MlirOperation exitCast = mlirOperationCreate(&exitCastState);
            mlirBlockAppendOwnedOperation(exitBlock, exitCast);
            MlirValue exitRetVal = mlirOperationGetResult(exitCast, 0);
            mlirOperationStateAddOperands(&exitRetState, 1, &exitRetVal);
            mlirBlockAppendOwnedOperation(exitBlock, mlirOperationCreate(&exitRetState));
        }

        // 2. Generate _loop_body_X
        {
            MlirBlock bodyBlock = createFunctionBlock(ctx, loc, moduleBody, prefixedBodyName);
            registerFunction(ctx, loc, moduleBody, body_func_name, prefixedBodyName);

            MlirValue bodyRawBundle = mlirBlockGetArgument(bodyBlock, 0);
            MlirValue bodyResultPort = mlirBlockGetArgument(bodyBlock, 1);

            MlirOperationState bodyUnpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirNamedAttribute bodyUnpackAttrs[] = {pairTypeAttr, minusPolAttr, labelPairAttr};
            mlirOperationStateAddAttributes(&bodyUnpackState, 3, bodyUnpackAttrs);
            mlirOperationStateAddResults(&bodyUnpackState, 3, agentTypes);
            MlirOperation bodyUnpackOp = mlirOperationCreate(&bodyUnpackState);
            mlirBlockAppendOwnedOperation(bodyBlock, bodyUnpackOp);
            linkValues(bodyBlock, loc, mlirOperationGetResult(bodyUnpackOp, 0), bodyRawBundle);

            MlirValue bodyEnvBundle = mlirOperationGetResult(bodyUnpackOp, 1);
            MlirValue bodyMainArg = mlirOperationGetResult(bodyUnpackOp, 2);

            MlirOperationState capUnpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            mlirOperationStateAddAttributes(&capUnpackState, 3, bodyUnpackAttrs);
            mlirOperationStateAddResults(&capUnpackState, 3, agentTypes);
            MlirOperation capUnpackOp = mlirOperationCreate(&capUnpackState);
            mlirBlockAppendOwnedOperation(bodyBlock, capUnpackOp);
            linkValues(bodyBlock, loc, mlirOperationGetResult(capUnpackOp, 0), bodyEnvBundle);

            MlirValue loop_macro_closure = mlirOperationGetResult(capUnpackOp, 1);
            MlirValue capRemaining = mlirOperationGetResult(capUnpackOp, 2);
            linkToEra(ctx, bodyBlock, loc, capRemaining);

            Environment bodyEnv;
            env_init(&bodyEnv);
            MlirValue bodyCurrentBundle = bodyMainArg;
            for (int i = 0; i < active_count; i++) {
                MlirOperationState unpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                mlirOperationStateAddAttributes(&unpackState, 3, bodyUnpackAttrs);
                mlirOperationStateAddResults(&unpackState, 3, agentTypes);
                MlirOperation unpackOp = mlirOperationCreate(&unpackState);
                mlirBlockAppendOwnedOperation(bodyBlock, unpackOp);

                MlirValue p0 = mlirOperationGetResult(unpackOp, 0);
                MlirValue p1 = mlirOperationGetResult(unpackOp, 1);
                MlirValue p2 = mlirOperationGetResult(unpackOp, 2);

                linkValues(bodyBlock, loc, p0, bodyCurrentBundle);
                env_add(&bodyEnv, active_names[i], active_lens[i], p1);
                bodyCurrentBundle = p2;
            }
            linkToEra(ctx, bodyBlock, loc, bodyCurrentBundle);

            MlirValue body_res = lowerExpression(ctx, bodyBlock, loc, expr->as.while_loop.body, &bodyEnv, true);

            MlirValue *next_vals = malloc(sizeof(MlirValue) * active_count);
            for (int i = 0; i < active_count; i++) {
                MlirValue val = {NULL};
                for (int j = bodyEnv.count - 1; j >= 0; j--) {
                    if (bodyEnv.vars[j].name_len == active_lens[i] &&
                        strncmp(bodyEnv.vars[j].name, active_names[i], active_lens[i]) == 0) {
                        val = bodyEnv.vars[j].value;
                        bodyEnv.vars[j].value = (MlirValue){NULL}; // Consume
                        break;
                    }
                }
                if (mlirValueIsNull(val)) {
                    val = createEra(ctx, bodyBlock, loc);
                }
                next_vals[i] = val;
            }
            linkToEra(ctx, bodyBlock, loc, body_res);
            env_free(&bodyEnv, ctx, bodyBlock, loc);

            MlirValue next_bundle = createEra(ctx, bodyBlock, loc);
            for (int i = active_count - 1; i >= 0; i--) {
                MlirOperationState packState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirNamedAttribute packAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
                mlirOperationStateAddAttributes(&packState, 3, packAttrs);
                mlirOperationStateAddResults(&packState, 3, agentTypes);
                MlirOperation packOp = mlirOperationCreate(&packState);
                mlirBlockAppendOwnedOperation(bodyBlock, packOp);

                MlirValue p0 = mlirOperationGetResult(packOp, 0);
                MlirValue p1 = mlirOperationGetResult(packOp, 1);
                MlirValue p2 = mlirOperationGetResult(packOp, 2);

                linkValues(bodyBlock, loc, p1, next_vals[i]);
                linkValues(bodyBlock, loc, p2, next_bundle);
                next_bundle = p0;
            }
            free(next_vals);

            MlirOperationState macroUnpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            mlirOperationStateAddAttributes(&macroUnpackState, 3, bodyUnpackAttrs);
            mlirOperationStateAddResults(&macroUnpackState, 3, agentTypes);
            MlirOperation macroUnpackOp = mlirOperationCreate(&macroUnpackState);
            mlirBlockAppendOwnedOperation(bodyBlock, macroUnpackOp);
            linkValues(bodyBlock, loc, mlirOperationGetResult(macroUnpackOp, 0), loop_macro_closure);

            MlirValue macroF = mlirOperationGetResult(macroUnpackOp, 1);
            MlirValue macroClEnv = mlirOperationGetResult(macroUnpackOp, 2);

            MlirValue appP0, appP1, appP2;
            createOmega(ctx, bodyBlock, loc, "call", "-", &appP0, &appP1, &appP2);
            linkValues(bodyBlock, loc, appP0, macroF);

            MlirOperationState callPackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirNamedAttribute callPackAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
            mlirOperationStateAddAttributes(&callPackState, 3, callPackAttrs);
            mlirOperationStateAddResults(&callPackState, 3, agentTypes);
            MlirOperation callPackOp = mlirOperationCreate(&callPackState);
            mlirBlockAppendOwnedOperation(bodyBlock, callPackOp);

            MlirValue cpP0 = mlirOperationGetResult(callPackOp, 0);
            MlirValue cpP1 = mlirOperationGetResult(callPackOp, 1);
            MlirValue cpP2 = mlirOperationGetResult(callPackOp, 2);

            linkValues(bodyBlock, loc, cpP1, macroClEnv);
            linkValues(bodyBlock, loc, cpP2, next_bundle);
            linkValues(bodyBlock, loc, cpP0, appP1);
            linkValues(bodyBlock, loc, appP2, bodyResultPort);

            MlirOperationState bodyRetState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
            MlirType bodyI64Type = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString("i64"));
            MlirOperationState bodyCastState = mlirOperationStateGet(mlirStringRefCreateFromCString("builtin.unrealized_conversion_cast"), loc);
            mlirOperationStateAddOperands(&bodyCastState, 1, &bodyResultPort);
            mlirOperationStateAddResults(&bodyCastState, 1, &bodyI64Type);
            MlirOperation bodyCast = mlirOperationCreate(&bodyCastState);
            mlirBlockAppendOwnedOperation(bodyBlock, bodyCast);
            MlirValue bodyRetVal = mlirOperationGetResult(bodyCast, 0);
            mlirOperationStateAddOperands(&bodyRetState, 1, &bodyRetVal);
            mlirBlockAppendOwnedOperation(bodyBlock, mlirOperationCreate(&bodyRetState));
        }

        // 3. Generate _loop_macro_X
        {
            MlirBlock macroBlock = createFunctionBlock(ctx, loc, moduleBody, prefixedMacroName);
            registerFunction(ctx, loc, moduleBody, macro_func_name, prefixedMacroName);

            MlirValue macroRawBundle = mlirBlockGetArgument(macroBlock, 0);
            MlirValue macroResultPort = mlirBlockGetArgument(macroBlock, 1);

            MlirOperationState macroUnpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirNamedAttribute macroUnpackAttrs[] = {pairTypeAttr, minusPolAttr, labelPairAttr};
            mlirOperationStateAddAttributes(&macroUnpackState, 3, macroUnpackAttrs);
            mlirOperationStateAddResults(&macroUnpackState, 3, agentTypes);
            MlirOperation macroUnpackOp = mlirOperationCreate(&macroUnpackState);
            mlirBlockAppendOwnedOperation(macroBlock, macroUnpackOp);
            linkValues(macroBlock, loc, mlirOperationGetResult(macroUnpackOp, 0), macroRawBundle);

            MlirValue macroEnvBundle = mlirOperationGetResult(macroUnpackOp, 1);
            MlirValue macroMainArg = mlirOperationGetResult(macroUnpackOp, 2);
            linkToEra(ctx, macroBlock, loc, macroEnvBundle);

            Environment macroEnv;
            env_init(&macroEnv);
            MlirValue macroCurrentBundle = macroMainArg;
            for (int i = 0; i < active_count; i++) {
                MlirOperationState unpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                mlirOperationStateAddAttributes(&unpackState, 3, macroUnpackAttrs);
                mlirOperationStateAddResults(&unpackState, 3, agentTypes);
                MlirOperation unpackOp = mlirOperationCreate(&unpackState);
                mlirBlockAppendOwnedOperation(macroBlock, unpackOp);

                MlirValue p0 = mlirOperationGetResult(unpackOp, 0);
                MlirValue p1 = mlirOperationGetResult(unpackOp, 1);
                MlirValue p2 = mlirOperationGetResult(unpackOp, 2);

                linkValues(macroBlock, loc, p0, macroCurrentBundle);
                env_add(&macroEnv, active_names[i], active_lens[i], p1);
                macroCurrentBundle = p2;
            }
            linkToEra(ctx, macroBlock, loc, macroCurrentBundle);

            MlirValue cond = lowerExpression(ctx, macroBlock, loc, expr->as.while_loop.condition, &macroEnv, true);

            MlirValue *v_trues = malloc(sizeof(MlirValue) * active_count);
            MlirValue *v_falses = malloc(sizeof(MlirValue) * active_count);
            for (int i = 0; i < active_count; i++) {
                MlirValue v = {NULL};
                for (int j = macroEnv.count - 1; j >= 0; j--) {
                    if (macroEnv.vars[j].name_len == active_lens[i] &&
                        strncmp(macroEnv.vars[j].name, active_names[i], active_lens[i]) == 0) {
                        v = macroEnv.vars[j].value;
                        macroEnv.vars[j].value = (MlirValue){NULL}; // Consume
                        break;
                    }
                }
                if (mlirValueIsNull(v)) {
                    v = createEra(ctx, macroBlock, loc);
                }
                MlirOperationState dupState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirAttribute deltaType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("delta"));
                MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), deltaType);
                MlirAttribute starPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
                MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), starPol);
                MlirAttribute dupLabel = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("dup"));
                MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), dupLabel);
                MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
                mlirOperationStateAddAttributes(&dupState, 3, attrs);
                mlirOperationStateAddResults(&dupState, 3, agentTypes);
                MlirOperation dupOp = mlirOperationCreate(&dupState);
                mlirBlockAppendOwnedOperation(macroBlock, dupOp);

                MlirValue dupP0 = mlirOperationGetResult(dupOp, 0);
                MlirValue dupP1 = mlirOperationGetResult(dupOp, 1);
                MlirValue dupP2 = mlirOperationGetResult(dupOp, 2);

                linkValues(macroBlock, loc, dupP0, v);
                v_trues[i] = dupP1;
                v_falses[i] = dupP2;
            }
            env_free(&macroEnv, ctx, macroBlock, loc);

            MlirValue false_cap_bundle = createEra(ctx, macroBlock, loc);
            for (int i = active_count - 1; i >= 0; i--) {
                MlirOperationState packState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirNamedAttribute packAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
                mlirOperationStateAddAttributes(&packState, 3, packAttrs);
                mlirOperationStateAddResults(&packState, 3, agentTypes);
                MlirOperation packOp = mlirOperationCreate(&packState);
                mlirBlockAppendOwnedOperation(macroBlock, packOp);

                MlirValue p0 = mlirOperationGetResult(packOp, 0);
                MlirValue p1 = mlirOperationGetResult(packOp, 1);
                MlirValue p2 = mlirOperationGetResult(packOp, 2);

                linkValues(macroBlock, loc, p1, v_falses[i]);
                linkValues(macroBlock, loc, p2, false_cap_bundle);
                false_cap_bundle = p0;
            }
            MlirValue exit_omega = createOmegaP0(ctx, macroBlock, loc, exit_func_name, "+");
            MlirValue false_branch = bundleClosure(ctx, macroBlock, loc, exit_omega, false_cap_bundle);

            MlirValue macro_omega = createOmegaP0(ctx, macroBlock, loc, macro_func_name, "+");
            MlirValue macro_dummy_era = createEra(ctx, macroBlock, loc);
            MlirValue loop_macro_closure_inner = bundleClosure(ctx, macroBlock, loc, macro_omega, macro_dummy_era);

            MlirValue true_cap_bundle = createEra(ctx, macroBlock, loc);
            for (int i = active_count - 1; i >= 0; i--) {
                MlirOperationState packState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirNamedAttribute packAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
                mlirOperationStateAddAttributes(&packState, 3, packAttrs);
                mlirOperationStateAddResults(&packState, 3, agentTypes);
                MlirOperation packOp = mlirOperationCreate(&packState);
                mlirBlockAppendOwnedOperation(macroBlock, packOp);

                MlirValue p0 = mlirOperationGetResult(packOp, 0);
                MlirValue p1 = mlirOperationGetResult(packOp, 1);
                MlirValue p2 = mlirOperationGetResult(packOp, 2);

                linkValues(macroBlock, loc, p1, v_trues[i]);
                linkValues(macroBlock, loc, p2, true_cap_bundle);
                true_cap_bundle = p0;
            }
            MlirOperationState macroPackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirNamedAttribute macroPackAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
            mlirOperationStateAddAttributes(&macroPackState, 3, macroPackAttrs);
            mlirOperationStateAddResults(&macroPackState, 3, agentTypes);
            MlirOperation macroPackOp = mlirOperationCreate(&macroPackState);
            mlirBlockAppendOwnedOperation(macroBlock, macroPackOp);

            MlirValue mp0 = mlirOperationGetResult(macroPackOp, 0);
            MlirValue mp1 = mlirOperationGetResult(macroPackOp, 1);
            MlirValue mp2 = mlirOperationGetResult(macroPackOp, 2);

            linkValues(macroBlock, loc, mp1, loop_macro_closure_inner);
            linkValues(macroBlock, loc, mp2, true_cap_bundle);
            true_cap_bundle = mp0;

            MlirValue body_omega = createOmegaP0(ctx, macroBlock, loc, body_func_name, "+");
            MlirValue true_branch = bundleClosure(ctx, macroBlock, loc, body_omega, true_cap_bundle);

            MlirValue branches_pair = createPair(ctx, macroBlock, loc, true_branch, false_branch);

            MlirValue eitherP0, eitherP1, eitherP2;
            createOmega(ctx, macroBlock, loc, "either", "-", &eitherP0, &eitherP1, &eitherP2);

            linkValues(macroBlock, loc, eitherP1, cond);
            linkValues(macroBlock, loc, eitherP2, branches_pair);
            linkValues(macroBlock, loc, eitherP0, macroResultPort);

            MlirOperationState macroRetState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
            MlirType macroI64Type = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString("i64"));
            MlirOperationState macroCastState = mlirOperationStateGet(mlirStringRefCreateFromCString("builtin.unrealized_conversion_cast"), loc);
            mlirOperationStateAddOperands(&macroCastState, 1, &macroResultPort);
            mlirOperationStateAddResults(&macroCastState, 1, &macroI64Type);
            MlirOperation macroCast = mlirOperationCreate(&macroCastState);
            mlirBlockAppendOwnedOperation(macroBlock, macroCast);
            MlirValue macroRetVal = mlirOperationGetResult(macroCast, 0);
            mlirOperationStateAddOperands(&macroRetState, 1, &macroRetVal);
            mlirBlockAppendOwnedOperation(macroBlock, mlirOperationCreate(&macroRetState));

            free(v_trues);
            free(v_falses);
        }

        // 4. Call _loop_macro_X in parent block
        MlirValue parent_macro_omega = createOmegaP0(ctx, block, loc, macro_func_name, "+");
        MlirValue parent_macro_dummy_era = createEra(ctx, block, loc);
        MlirValue macro_closure = bundleClosure(ctx, block, loc, parent_macro_omega, parent_macro_dummy_era);

        MlirOperationState macroUnpackStateParent = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirNamedAttribute macroUnpackAttrsParent[] = {pairTypeAttr, minusPolAttr, labelPairAttr};
        mlirOperationStateAddAttributes(&macroUnpackStateParent, 3, macroUnpackAttrsParent);
        mlirOperationStateAddResults(&macroUnpackStateParent, 3, agentTypes);
        MlirOperation macroUnpackOpParent = mlirOperationCreate(&macroUnpackStateParent);
        mlirBlockAppendOwnedOperation(block, macroUnpackOpParent);
        linkValues(block, loc, mlirOperationGetResult(macroUnpackOpParent, 0), macro_closure);

        MlirValue macroFParent = mlirOperationGetResult(macroUnpackOpParent, 1);
        MlirValue macroClEnvParent = mlirOperationGetResult(macroUnpackOpParent, 2);

        MlirValue appP0Parent, appP1Parent, appP2Parent;
        createOmega(ctx, block, loc, "call", "-", &appP0Parent, &appP1Parent, &appP2Parent);
        linkValues(block, loc, appP0Parent, macroFParent);

        MlirOperationState callPackStateParent = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirNamedAttribute callPackAttrsParent[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
        mlirOperationStateAddAttributes(&callPackStateParent, 3, callPackAttrsParent);
        mlirOperationStateAddResults(&callPackStateParent, 3, agentTypes);
        MlirOperation callPackOpParent = mlirOperationCreate(&callPackStateParent);
        mlirBlockAppendOwnedOperation(block, callPackOpParent);

        MlirValue cpP0Parent = mlirOperationGetResult(callPackOpParent, 0);
        MlirValue cpP1Parent = mlirOperationGetResult(callPackOpParent, 1);
        MlirValue cpP2Parent = mlirOperationGetResult(callPackOpParent, 2);

        linkValues(block, loc, cpP1Parent, macroClEnvParent);
        linkValues(block, loc, cpP2Parent, init_bundle);
        linkValues(block, loc, cpP0Parent, appP1Parent);

        // Unpack parentCurrentBundle (which is appP2Parent)
        MlirValue parentCurrentBundle = appP2Parent;
        for (int i = 0; i < active_count; i++) {
            MlirOperationState unpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            mlirOperationStateAddAttributes(&unpackState, 3, macroUnpackAttrsParent);
            mlirOperationStateAddResults(&unpackState, 3, agentTypes);
            MlirOperation unpackOp = mlirOperationCreate(&unpackState);
            mlirBlockAppendOwnedOperation(block, unpackOp);

            MlirValue p0 = mlirOperationGetResult(unpackOp, 0);
            MlirValue p1 = mlirOperationGetResult(unpackOp, 1);
            MlirValue p2 = mlirOperationGetResult(unpackOp, 2);

            linkValues(block, loc, p0, parentCurrentBundle);
            env_add(env, active_names[i], active_lens[i], p1);
            parentCurrentBundle = p2;
        }
        linkToEra(ctx, block, loc, parentCurrentBundle);

        free(active_names);
        free(active_lens);
        free(active_vals);

        return createEra(ctx, block, loc);
    }

    if (expr->type == AST_FUNC_DECL) {
        // NEW CLOSURE LOGIC START
        static int anon_counter = 0;
        char funcNameStr[256];
        if (expr->as.func_decl.name_len > 0) {
            snprintf(funcNameStr, sizeof(funcNameStr), "%.*s", expr->as.func_decl.name_len, expr->as.func_decl.name);
        } else {
            snprintf(funcNameStr, sizeof(funcNameStr), "anon_fn_%d", anon_counter++);
        }
        sanitizeMlirName(funcNameStr);

        // Skip type declaration functions to avoid duplicate symbol errors.
        // A type declaration function is a function with no arguments, a return type, and an empty body.
        if (expr->as.func_decl.name_len > 0 &&
            expr->as.func_decl.arg_count == 0 &&
            expr->as.func_decl.return_type_len > 0 &&
            (!expr->as.func_decl.body ||
             (expr->as.func_decl.body->type == AST_BLOCK && expr->as.func_decl.body->as.block.count == 0))) {
            MlirValue nullVal = {NULL};
            return nullVal;
        }

        int arg_count = expr->as.func_decl.arg_count;
        
        // Phase 1: Detect captures
        FreeVars fv = {NULL, 0, 0};
        const char **bound_args = malloc(sizeof(char*) * arg_count);
        for (int i = 0; i < arg_count; i++) bound_args[i] = expr->as.func_decl.args[i].name;
        findFreeVars(expr->as.func_decl.body, &fv, bound_args, arg_count);
        free(bound_args);

        LOG_REDEX("DEBUG_CAPTURES: Function %.*s (len %d), free vars count: %d\n", expr->as.func_decl.name_len, expr->as.func_decl.name, expr->as.func_decl.name_len, fv.count);
        for (int i = 0; i < fv.count; i++) {
            MlirValue testFetch = env_get(env, fv.names[i], strlen(fv.names[i]));
            LOG_REDEX("  - %s (env_getIsNull: %d)\n", fv.names[i], mlirValueIsNull(testFetch));
        }

        int total_args = fv.count + arg_count;
        MlirType portType = getPicPortType(ctx);

        // Phase 2: Create the underlying function
        // All user-defined functions take THREE arguments: [captures_bundle, argument, runtime_state]

        MlirType funcArgTypes[] = {portType, portType, portType};
        MlirLocation funcArgLocs[] = {loc, loc, loc};
        MlirType retTypes[] = {portType};
        MlirType funcType = mlirFunctionTypeGet(ctx, 3, funcArgTypes, 1, retTypes);

        MlirOperationState funcState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);
        char prefixedName[512];
        sprintf(prefixedName, "lin_%s", funcNameStr);
        MlirAttribute fnNameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(prefixedName));
        MlirNamedAttribute fnNameNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")), fnNameAttr);
        MlirAttribute fnTypeAttr = mlirTypeAttrGet(funcType);
        MlirNamedAttribute fnTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")), fnTypeAttr);
        MlirNamedAttribute funcAttrs[] = {fnNameNamed, fnTypeNamed};
        mlirOperationStateAddAttributes(&funcState, 2, funcAttrs);

        MlirRegion innerRegion = mlirRegionCreate();
        MlirBlock innerBlock = mlirBlockCreate(3, funcArgTypes, funcArgLocs);
        mlirRegionAppendOwnedBlock(innerRegion, innerBlock);
        mlirOperationStateAddOwnedRegions(&funcState, 1, &innerRegion);

        MlirOperation funcOp = mlirOperationCreate(&funcState);
        if (!mlirBlockIsNull(moduleBody)) {
            mlirBlockAppendOwnedOperation(moduleBody, funcOp);
        }

        // Lower body
        Environment innerEnv;
        env_init(&innerEnv);
        
        MlirValue rawBundle = mlirBlockGetArgument(innerBlock, 0);
        MlirValue rawResultPort = mlirBlockGetArgument(innerBlock, 1);

        MlirValue inputBundle = rawBundle;
        MlirValue resultPort = rawResultPort;

        MlirAttribute pairType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
        MlirNamedAttribute pairTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), pairType);
        MlirAttribute minusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
        MlirNamedAttribute minusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), minusPol);
        MlirAttribute plusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
        MlirNamedAttribute plusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), plusPol);
        MlirAttribute labelPair = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
        MlirNamedAttribute labelPairAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelPair);
        MlirType agentTypes[] = {portType, portType, portType};

        // Unpack inputBundle into (env_bundle, main_arg)
        MlirOperationState unpackMainState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirNamedAttribute unpackMainAttrs[] = {pairTypeAttr, minusPolAttr, labelPairAttr};
        mlirOperationStateAddAttributes(&unpackMainState, 3, unpackMainAttrs);
        mlirOperationStateAddResults(&unpackMainState, 3, agentTypes);
        MlirOperation unpackMainOp = mlirOperationCreate(&unpackMainState);
        mlirBlockAppendOwnedOperation(innerBlock, unpackMainOp);

        MlirValue umP0 = mlirOperationGetResult(unpackMainOp, 0);
        MlirValue envBundle = mlirOperationGetResult(unpackMainOp, 1);
        MlirValue mainArg = mlirOperationGetResult(unpackMainOp, 2);

        MlirOperationState linkMainState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue linkMainOps[] = {umP0, inputBundle};
        mlirOperationStateAddOperands(&linkMainState, 2, linkMainOps);
        mlirBlockAppendOwnedOperation(innerBlock, mlirOperationCreate(&linkMainState));

        // Unpack captures from envBundle
        MlirValue currentBundle = envBundle;
        for (int i = 0; i < fv.count; i++) {
            MlirOperationState unpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirNamedAttribute unpackAttrs[] = {pairTypeAttr, minusPolAttr, labelPairAttr};
            mlirOperationStateAddAttributes(&unpackState, 3, unpackAttrs);
            mlirOperationStateAddResults(&unpackState, 3, agentTypes);
            MlirOperation unpackOp = mlirOperationCreate(&unpackState);
            mlirBlockAppendOwnedOperation(innerBlock, unpackOp);

            MlirValue p0 = mlirOperationGetResult(unpackOp, 0);
            MlirValue p1 = mlirOperationGetResult(unpackOp, 1);
            MlirValue p2 = mlirOperationGetResult(unpackOp, 2);

            MlirOperationState linkState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue linkOps[] = {p0, currentBundle};
            mlirOperationStateAddOperands(&linkState, 2, linkOps);
            mlirBlockAppendOwnedOperation(innerBlock, mlirOperationCreate(&linkState));

            env_add(&innerEnv, fv.names[i], strlen(fv.names[i]), p1);
            currentBundle = p2;
        }
        // Era the remaining bundle port (the end-of-bundle ERA)
        MlirOperationState eraInnerState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirAttribute eraType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
        MlirNamedAttribute eraTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), eraType);
        MlirAttribute starPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
        MlirNamedAttribute starPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), starPol);
        MlirAttribute eraLabel = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
        MlirNamedAttribute eraLabelAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), eraLabel);
        MlirNamedAttribute eraAttrs[] = {eraTypeAttr, starPolAttr, eraLabelAttr};
        mlirOperationStateAddAttributes(&eraInnerState, 3, eraAttrs);
        mlirOperationStateAddResults(&eraInnerState, 3, agentTypes);
        MlirOperation eraInnerOp = mlirOperationCreate(&eraInnerState);
        mlirBlockAppendOwnedOperation(innerBlock, eraInnerOp);
        MlirOperationState linkEraState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue eraOps[] = {mlirOperationGetResult(eraInnerOp, 0), currentBundle};
        mlirOperationStateAddOperands(&linkEraState, 2, eraOps);
        mlirBlockAppendOwnedOperation(innerBlock, mlirOperationCreate(&linkEraState));

        // Add main argument to env
        if (arg_count > 0) {
            // Multi-arg support: unpack the pair chain to bind N arguments
            MlirValue unpackChainArg = mainArg;
            for (int ai = 0; ai < arg_count; ai++) {
                MlirOperationState unpackMainState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirNamedAttribute unpackMainAttrs[] = {pairTypeAttr, minusPolAttr, labelPairAttr};
                mlirOperationStateAddAttributes(&unpackMainState, 3, unpackMainAttrs);
                mlirOperationStateAddResults(&unpackMainState, 3, agentTypes);
                MlirOperation unpackOp = mlirOperationCreate(&unpackMainState);
                mlirBlockAppendOwnedOperation(innerBlock, unpackOp);
                MlirValue uP0 = mlirOperationGetResult(unpackOp, 0);
                MlirValue uP1 = mlirOperationGetResult(unpackOp, 1);
                MlirValue uP2 = mlirOperationGetResult(unpackOp, 2);
                MlirOperationState linkState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue linkOps[] = {uP0, unpackChainArg};
                mlirOperationStateAddOperands(&linkState, 2, linkOps);
                mlirBlockAppendOwnedOperation(innerBlock, mlirOperationCreate(&linkState));
                env_add(&innerEnv, expr->as.func_decl.args[ai].name, expr->as.func_decl.args[ai].name_len, uP1);
                unpackChainArg = uP2;
            }
            // Era the end-of-chain marker
            {
                MlirOperationState eraArgState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                mlirOperationStateAddAttributes(&eraArgState, 3, eraAttrs);
                mlirOperationStateAddResults(&eraArgState, 3, agentTypes);
                MlirOperation eraArgOp = mlirOperationCreate(&eraArgState);
                mlirBlockAppendOwnedOperation(innerBlock, eraArgOp);
                MlirOperationState linkEraArgState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue eraArgOps[] = {mlirOperationGetResult(eraArgOp, 0), unpackChainArg};
                mlirOperationStateAddOperands(&linkEraArgState, 2, eraArgOps);
                mlirBlockAppendOwnedOperation(innerBlock, mlirOperationCreate(&linkEraArgState));
            }
        } else {
            MlirOperationState eraArgState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            mlirOperationStateAddAttributes(&eraArgState, 3, eraAttrs);
            mlirOperationStateAddResults(&eraArgState, 3, agentTypes);
            MlirOperation eraArgOp = mlirOperationCreate(&eraArgState);
            mlirBlockAppendOwnedOperation(innerBlock, eraArgOp);
            MlirOperationState linkEraArgState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue eraArgOps[] = {mlirOperationGetResult(eraArgOp, 0), mainArg};
            mlirOperationStateAddOperands(&linkEraArgState, 2, eraArgOps);
            mlirBlockAppendOwnedOperation(innerBlock, mlirOperationCreate(&linkEraArgState));
        }

        MlirValue bodyResult = lowerExpression(ctx, innerBlock, loc, expr->as.func_decl.body, &innerEnv, true);
        env_free(&innerEnv, ctx, innerBlock, loc);

        MlirOperationState linkBodyState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue bodyLinkOps[] = {bodyResult, resultPort};
        mlirOperationStateAddOperands(&linkBodyState, 2, bodyLinkOps);
        mlirBlockAppendOwnedOperation(innerBlock, mlirOperationCreate(&linkBodyState));

        MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        // Cast resultPort to i64 for return signature
        MlirType i64Type = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString("i64"));
        MlirOperationState castRetState = mlirOperationStateGet(mlirStringRefCreateFromCString("builtin.unrealized_conversion_cast"), loc);
        mlirOperationStateAddOperands(&castRetState, 1, &resultPort);
        mlirOperationStateAddResults(&castRetState, 1, &i64Type);
        MlirOperation castRet = mlirOperationCreate(&castRetState);
        mlirBlockAppendOwnedOperation(innerBlock, castRet);
        MlirValue retVal = mlirOperationGetResult(castRet, 0);

        mlirOperationStateAddOperands(&retState, 1, &retVal);
        mlirBlockAppendOwnedOperation(innerBlock, mlirOperationCreate(&retState));

        // Phase 2.5: Register payload for the dispatcher
        // This allows omega agents with this label to fire rule_fire_op
        char payloadBuf[2048];
        snprintf(payloadBuf, sizeof(payloadBuf), "  %%res = func.call @lin_%s(%%arg0, %%arg1, %%state) : (i64, i64, i64) -> i64\n", funcNameStr);
        MlirOperationState regState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.registry"), loc);
        MlirAttribute opNameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(funcNameStr));
        MlirNamedAttribute opNameNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("op_name")), opNameAttr);
        MlirAttribute payloadAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(payloadBuf));
        MlirNamedAttribute payloadNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("payload")), payloadAttr);
        MlirAttribute argNamesAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("[%arg0][%arg1][%state]"));
        MlirNamedAttribute argNamesNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("arg_names")), argNamesAttr);
        int regAttrCount = 3;
        MlirNamedAttribute regAttrs[4];
        regAttrs[0] = opNameNamed;
        regAttrs[1] = payloadNamed;
        regAttrs[2] = argNamesNamed;
        if (expr->as.func_decl.dispatch) {
            MlirAttribute dispatchAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.func_decl.dispatch, expr->as.func_decl.dispatch_len));
            regAttrs[3] = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("dispatch")), dispatchAttr);
            regAttrCount = 4;
        }
        mlirOperationStateAddAttributes(&regState, regAttrCount, regAttrs);
        if (!mlirBlockIsNull(moduleBody)) {
            mlirBlockAppendOwnedOperation(moduleBody, mlirOperationCreate(&regState));
        }

        // Phase 3: Create closure pair (omega+, bundle)
        MlirOperationState baseState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirAttribute agTypeAttr  = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute agTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), agTypeAttr);
        MlirAttribute fnLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(funcNameStr));
        MlirNamedAttribute fnLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), fnLabelAttr);
        int agentAttrCount = 3;
        MlirNamedAttribute agentAttrs[4];
        agentAttrs[0] = agTypeNamed;
        agentAttrs[1] = plusPolAttr;
        agentAttrs[2] = fnLabelNamedAttr;
        if (expr->as.func_decl.dispatch) {
            MlirAttribute dispatchAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.func_decl.dispatch, expr->as.func_decl.dispatch_len));
            agentAttrs[3] = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("dispatch")), dispatchAttr);
            agentAttrCount = 4;
        }
        mlirOperationStateAddAttributes(&baseState, agentAttrCount, agentAttrs);
        mlirOperationStateAddResults(&baseState, 3, agentTypes);
        MlirOperation baseOp = mlirOperationCreate(&baseState);
        mlirBlockAppendOwnedOperation(block, baseOp);
        linkToEra(ctx, block, loc, mlirOperationGetResult(baseOp, 1));
        linkToEra(ctx, block, loc, mlirOperationGetResult(baseOp, 2));
        MlirValue omegaP0 = mlirOperationGetResult(baseOp, 0);

        MlirOperationState eraOuterState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        mlirOperationStateAddAttributes(&eraOuterState, 3, eraAttrs);
        mlirOperationStateAddResults(&eraOuterState, 3, agentTypes);
        MlirOperation eraOuterOp = mlirOperationCreate(&eraOuterState);
        mlirBlockAppendOwnedOperation(block, eraOuterOp);
        MlirValue currentOuterBundle = mlirOperationGetResult(eraOuterOp, 0);

        for (int i = fv.count - 1; i >= 0; i--) {
            MlirValue capVal = env_fetch(ctx, block, loc, env, fv.names[i], strlen(fv.names[i]));
            if (mlirValueIsNull(capVal)) {
                MlirOperationState eraCapState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                mlirOperationStateAddAttributes(&eraCapState, 3, eraAttrs);
                mlirOperationStateAddResults(&eraCapState, 3, agentTypes);
                MlirOperation eraCapOp = mlirOperationCreate(&eraCapState);
                mlirBlockAppendOwnedOperation(block, eraCapOp);
                capVal = mlirOperationGetResult(eraCapOp, 0);
            }
            MlirOperationState packState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirNamedAttribute packAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
            mlirOperationStateAddAttributes(&packState, 3, packAttrs);
            mlirOperationStateAddResults(&packState, 3, agentTypes);
            MlirOperation packOp = mlirOperationCreate(&packState);
            mlirBlockAppendOwnedOperation(block, packOp);

            MlirValue p0 = mlirOperationGetResult(packOp, 0);
            MlirValue p1 = mlirOperationGetResult(packOp, 1);
            MlirValue p2 = mlirOperationGetResult(packOp, 2);

            MlirOperationState linkCapState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue capLinkOps[] = {p1, capVal};
            mlirOperationStateAddOperands(&linkCapState, 2, capLinkOps);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkCapState));

            MlirOperationState linkNextState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue nextLinkOps[] = {p2, currentOuterBundle};
            mlirOperationStateAddOperands(&linkNextState, 2, nextLinkOps);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkNextState));

            currentOuterBundle = p0;
        }

        MlirOperationState closureState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirNamedAttribute closureAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
        mlirOperationStateAddAttributes(&closureState, 3, closureAttrs);
        mlirOperationStateAddResults(&closureState, 3, agentTypes);
        MlirOperation closureOp = mlirOperationCreate(&closureState);
        mlirBlockAppendOwnedOperation(block, closureOp);

        MlirValue closureP0 = mlirOperationGetResult(closureOp, 0);
        MlirValue closureP1 = mlirOperationGetResult(closureOp, 1);
        MlirValue closureP2 = mlirOperationGetResult(closureOp, 2);

        MlirOperationState linkFState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue fLinkOps[] = {closureP1, omegaP0};
        mlirOperationStateAddOperands(&linkFState, 2, fLinkOps);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkFState));

        MlirOperationState linkEnvState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue envLinkOps[] = {closureP2, currentOuterBundle};
        mlirOperationStateAddOperands(&linkEnvState, 2, envLinkOps);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkEnvState));

        MlirValue currentVal = closureP0;

        if (expr->as.func_decl.name_len > 0) {
            env_add(env, expr->as.func_decl.name, expr->as.func_decl.name_len, currentVal);
        }

        if (fv.names) free(fv.names);
        return currentVal;
    }

    if (expr->type == AST_CALL) {

        // Use resolved_callee (type-directed rewrite) if set, else fall back to callee.
        // resolved_callee is set by the type checker for binary ops (e.g. add→fadd for f32).
        const char *effectiveCallee = expr->as.call.resolved_callee
                                      ? expr->as.call.resolved_callee
                                      : expr->as.call.callee;
        int effectiveCalleeLen = expr->as.call.resolved_callee
                                 ? (int)strlen(expr->as.call.resolved_callee)
                                 : expr->as.call.callee_len;

        // Built-in "pair" operator: (pair a b) creates a delta-(pair) node
        if (effectiveCalleeLen == 4 && memcmp(effectiveCallee, "pair", 4) == 0 && expr->as.call.arg_count == 2) {
            MlirValue left = lowerExpression(ctx, block, loc, expr->as.call.args[0], env, false);
            MlirValue right = lowerExpression(ctx, block, loc, expr->as.call.args[1], env, false);
            MlirOperationState pairState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirAttribute pTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("delta"));
            MlirNamedAttribute pTypeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), pTypeAttr);
            MlirAttribute pPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
            MlirNamedAttribute pPolNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), pPolAttr);
            MlirAttribute pLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
            MlirNamedAttribute pLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), pLabelAttr);
            MlirNamedAttribute pAttrs[] = {pTypeNamedAttr, pPolNamedAttr, pLabelNamedAttr};
            mlirOperationStateAddAttributes(&pairState, 3, pAttrs);
            MlirType portType = getPicPortType(ctx);
            MlirType pTypes[] = {portType, portType, portType};
            mlirOperationStateAddResults(&pairState, 3, pTypes);
            MlirOperation pairOp = mlirOperationCreate(&pairState);
            mlirBlockAppendOwnedOperation(block, pairOp);
            MlirValue p0 = mlirOperationGetResult(pairOp, 0);
            MlirValue p1 = mlirOperationGetResult(pairOp, 1);
            MlirValue p2 = mlirOperationGetResult(pairOp, 2);
            MlirOperationState lls = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue lOps[] = {p1, left};
            mlirOperationStateAddOperands(&lls, 2, lOps);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&lls));
            MlirOperationState lrs = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue rOps[] = {p2, right};
            mlirOperationStateAddOperands(&lrs, 2, rOps);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&lrs));
            return p0;
        }

        // Check if callee is a user-defined function in the environment
        // Skip the env lookup if the resolved callee comes from type-directed dispatch
        // OR if the callee is a known mlir-op name (registered in std/io.lin or similar).
        // Mlir-ops store omega+ agent ports in the env, not function closures.
        // Treating them as closures creates a malformed omega-(call) agent.
        MlirValue currentVal = {NULL};
        if (!expr->as.call.resolved_callee && !isMlirOpName(effectiveCallee, effectiveCalleeLen)) {
            currentVal = env_fetch(ctx, block, loc, env, effectiveCallee, effectiveCalleeLen);
            if (mlirValueIsNull(currentVal))
                currentVal = env_fetch(ctx, block, loc, env, expr->as.call.callee, expr->as.call.callee_len);
        }

        if (!mlirValueIsNull(currentVal)) {
            MlirType portType = getPicPortType(ctx);
            MlirAttribute agTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
            MlirNamedAttribute agTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), agTypeAttr);
            MlirAttribute minusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
            MlirNamedAttribute minusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), minusPol);
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.call.callee, expr->as.call.callee_len));
            MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);
            MlirType agentTypes[] = {portType, portType, portType};

            // Multi-arg support: pack ALL args into a pair chain before calling
            // This replaces the currying loop which only worked for single-arg functions
            MlirValue argsPack;
            
            // Define pair attributes for packing args
            MlirAttribute pairType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
            MlirNamedAttribute pairTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), pairType);
            MlirAttribute plusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
            MlirNamedAttribute plusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), plusPol);
            MlirAttribute labelPair = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
            MlirNamedAttribute labelPairAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelPair);
            
            if (expr->as.call.arg_count == 0) {
                MlirOperationState eraState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirAttribute eraType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
                MlirNamedAttribute eraTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), eraType);
                MlirAttribute starPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
                MlirNamedAttribute starPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), starPol);
                MlirAttribute eraLabel = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
                MlirNamedAttribute eraLabelAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), eraLabel);
                MlirNamedAttribute eraAttrs[] = {eraTypeAttr, starPolAttr, eraLabelAttr};
                mlirOperationStateAddAttributes(&eraState, 3, eraAttrs);
                mlirOperationStateAddResults(&eraState, 3, agentTypes);
                MlirOperation eraOp = mlirOperationCreate(&eraState);
                mlirBlockAppendOwnedOperation(block, eraOp);
                argsPack = mlirOperationGetResult(eraOp, 0);
            } else {
                // Build pair chain: pack all args into nested pairs
                // Last arg is linked to era to terminate the chain
                MlirOperationState eraState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirAttribute eraType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
                MlirNamedAttribute eraTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), eraType);
                MlirAttribute starPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
                MlirNamedAttribute starPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), starPol);
                MlirAttribute eraLabel = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
                MlirNamedAttribute eraLabelAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), eraLabel);
                MlirNamedAttribute eraAttrs[] = {eraTypeAttr, starPolAttr, eraLabelAttr};
                mlirOperationStateAddAttributes(&eraState, 3, eraAttrs);
                mlirOperationStateAddResults(&eraState, 3, agentTypes);
                MlirOperation eraOp = mlirOperationCreate(&eraState);
                mlirBlockAppendOwnedOperation(block, eraOp);
                MlirValue chainEnd = mlirOperationGetResult(eraOp, 1);
                
                argsPack = chainEnd;
                for (int ai = expr->as.call.arg_count - 1; ai >= 0; ai--) {
                    MlirValue argVal = lowerExpression(ctx, block, loc, expr->as.call.args[ai], env, false);
                    MlirOperationState packState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                    MlirNamedAttribute packAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
                    mlirOperationStateAddAttributes(&packState, 3, packAttrs);
                    mlirOperationStateAddResults(&packState, 3, agentTypes);
                    MlirOperation packOp = mlirOperationCreate(&packState);
                    mlirBlockAppendOwnedOperation(block, packOp);
                    MlirValue p0 = mlirOperationGetResult(packOp, 0);
                    MlirValue p1 = mlirOperationGetResult(packOp, 1);
                    MlirValue p2 = mlirOperationGetResult(packOp, 2);
                    MlirOperationState link1State = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                    MlirValue l1Ops[] = {p1, argVal};
                    mlirOperationStateAddOperands(&link1State, 2, l1Ops);
                    mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&link1State));
                    MlirOperationState link2State = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                    MlirValue l2Ops[] = {p2, argsPack};
                    mlirOperationStateAddOperands(&link2State, 2, l2Ops);
                    mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&link2State));
                    argsPack = p0;
                }
            }

            // Unpack closure currentVal into (f, env_bundle) — single call
            MlirOperationState unpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirNamedAttribute unpackAttrs[] = {pairTypeAttr, minusPolAttr, labelPairAttr};
            mlirOperationStateAddAttributes(&unpackState, 3, unpackAttrs);
            mlirOperationStateAddResults(&unpackState, 3, agentTypes);
            MlirOperation unpackOp = mlirOperationCreate(&unpackState);
            mlirBlockAppendOwnedOperation(block, unpackOp);

            MlirValue uP0 = mlirOperationGetResult(unpackOp, 0); // callee (the closure pair)
            MlirValue uP1 = mlirOperationGetResult(unpackOp, 1); // f
            MlirValue uP2 = mlirOperationGetResult(unpackOp, 2); // env_bundle

            MlirOperationState linkUState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue linkUOps[] = {uP0, currentVal};
            mlirOperationStateAddOperands(&linkUState, 2, linkUOps);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkUState));

            // Now bundle (env_bundle, argsPack) into a pair for the call
            // Use omega- for the call trigger
            MlirOperationState appState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirAttribute omegaType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
                MlirNamedAttribute omegaTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), omegaType);
                MlirAttribute callMinusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
                MlirNamedAttribute callMinusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), callMinusPol);
                // Use the label from the callee if it's an identifier, otherwise generic "call"
                MlirAttribute callLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("call"));
                MlirNamedAttribute callLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), callLabelAttr);
                MlirNamedAttribute appAttrs[] = {omegaTypeAttr, callMinusPolAttr, callLabelNamedAttr};
                mlirOperationStateAddAttributes(&appState, 3, appAttrs);
                mlirOperationStateAddResults(&appState, 3, agentTypes);
                MlirOperation appOp = mlirOperationCreate(&appState);
                mlirBlockAppendOwnedOperation(block, appOp);

                MlirValue appP0 = mlirOperationGetResult(appOp, 0); // callee (omega+)
                MlirValue appP1 = mlirOperationGetResult(appOp, 1); // pair(env, args)
                MlirValue appP2 = mlirOperationGetResult(appOp, 2); // result

                // Link appP0 ↔ uP1 (the function pointer)
                MlirOperationState linkFState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue fLinkOps[] = {appP0, uP1};
                mlirOperationStateAddOperands(&linkFState, 2, fLinkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkFState));

                // Create the pair(env, args)
                MlirOperationState packState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirNamedAttribute packAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};
                mlirOperationStateAddAttributes(&packState, 3, packAttrs);
                mlirOperationStateAddResults(&packState, 3, agentTypes);
                MlirOperation packOp = mlirOperationCreate(&packState);
                mlirBlockAppendOwnedOperation(block, packOp);

                MlirValue pP0 = mlirOperationGetResult(packOp, 0); // the pair
                MlirValue pP1 = mlirOperationGetResult(packOp, 1); // env
                MlirValue pP2 = mlirOperationGetResult(packOp, 2); // args

                // Link pP1 ↔ uP2
                MlirOperationState linkEnvState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue envLinkOps[] = {pP1, uP2};
                mlirOperationStateAddOperands(&linkEnvState, 2, envLinkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkEnvState));

                // Link pP2 ↔ argsPack (all args packed as nested pair)
                MlirOperationState linkArgState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue argLinkOps[] = {pP2, argsPack};
                mlirOperationStateAddOperands(&linkArgState, 2, argLinkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkArgState));

                // Link appP1 ↔ pP0
                MlirOperationState linkCallState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue callLinkOps[] = {appP1, pP0};
                mlirOperationStateAddOperands(&linkCallState, 2, callLinkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkCallState));

                currentVal = appP2;

            return currentVal;
        }

        // Built-in / unknown callee: lower as omega agent
        // For 2-arg calls (e.g. print_i32 state value):
        //   omega-(op_label) p0 ↔ value_literal (active pair: literal + op)
        //   omega- p1 ↔ state_literal
        //   omega- p2 = era
        // The binary dispatch reads:
        //   val0 = opNode p1 followed value → state (mapped to %arg0)
        //   val1 = valNode (paired) value → value (mapped to %arg1)
        if (expr->as.call.arg_count == 2) {
            MlirValue left = lowerExpression(ctx, block, loc, expr->as.call.args[0], env, false);
            MlirValue right = lowerExpression(ctx, block, loc, expr->as.call.args[1], env, false);

            MlirType portType = getPicPortType(ctx);
            MlirType agentTypes[] = {portType, portType, portType};

            // Create omega-(effLabel) — the call site
            MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
            MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
            MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(effectiveCallee, effectiveCalleeLen));
            MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);
            MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
            mlirOperationStateAddAttributes(&state, 3, attrs);
            mlirOperationStateAddResults(&state, 3, agentTypes);
            MlirOperation op = mlirOperationCreate(&state);
            mlirBlockAppendOwnedOperation(block, op);

            MlirValue result = mlirOperationGetResult(op, 0);
            MlirValue p1 = mlirOperationGetResult(op, 1);
            MlirValue p2 = mlirOperationGetResult(op, 2);

            // Link omega- p1 ↔ left (state arg) — binary dispatch follows this
            linkValues(block, loc, p1, left);
            // Link omega- p2 = era
            linkToEra(ctx, block, loc, p2);
            // Link omega- p0 ↔ right p0 (value arg forms active pair with literal)
            linkValues(block, loc, result, right);

            return result;
        } else if (expr->as.call.arg_count == 1) {
            MlirValue arg = lowerExpression(ctx, block, loc, expr->as.call.args[0], env, false);

            // Synthesize a 0 literal for the state to form a proper literal+op active pair
            MlirType portType = getPicPortType(ctx);
            MlirType agentTypes[] = {portType, portType, portType};

            MlirOperationState litState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirAttribute litTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
            MlirNamedAttribute litTypeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), litTypeAttr);
            MlirAttribute litPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
            MlirNamedAttribute litPolNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), litPolAttr);
            MlirAttribute litLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("i32"));
            MlirNamedAttribute litLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), litLabelAttr);
            MlirAttribute litValAttr = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 64), 0);
            MlirNamedAttribute litValNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("value")), litValAttr);
            MlirNamedAttribute litAttrs[] = {litTypeNamedAttr, litPolNamedAttr, litLabelNamedAttr, litValNamedAttr};
            mlirOperationStateAddAttributes(&litState, 4, litAttrs);
            mlirOperationStateAddResults(&litState, 3, agentTypes);
            MlirOperation litOp = mlirOperationCreate(&litState);
            mlirBlockAppendOwnedOperation(block, litOp);
            MlirValue stateLiteral = mlirOperationGetResult(litOp, 0);

            // Create omega-(effLabel) — the call site
            MlirOperationState opState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
            MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
            MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(effectiveCallee, effectiveCalleeLen));
            MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);
            MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
            mlirOperationStateAddAttributes(&opState, 3, attrs);
            mlirOperationStateAddResults(&opState, 3, agentTypes);
            MlirOperation op = mlirOperationCreate(&opState);
            mlirBlockAppendOwnedOperation(block, op);

            MlirValue result = mlirOperationGetResult(op, 0);
            MlirValue p1 = mlirOperationGetResult(op, 1);
            MlirValue p2 = mlirOperationGetResult(op, 2);

            // Link omega- p1 ↔ state literal (binary dispatch follows this for %arg0/state)
            linkValues(block, loc, p1, stateLiteral);
            // Link omega- p2 = era
            linkToEra(ctx, block, loc, p2);
            // Link omega- p0 ↔ arg p0 (value arg forms active pair)
            linkValues(block, loc, result, arg);

            return result;
        }

        // For general calls, wait! If arg_count is neither 1 nor 2, what happens?
        // Fallback to ERA
        MlirOperationState fallbackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

        MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
        mlirOperationStateAddAttributes(&fallbackState, 3, attrs);

        MlirType fbPortType = getPicPortType(ctx);
        MlirType types[] = {fbPortType, fbPortType, fbPortType};
        mlirOperationStateAddResults(&fallbackState, 3, types);
        MlirOperation fbOp = mlirOperationCreate(&fallbackState);
        mlirBlockAppendOwnedOperation(block, fbOp);
        return mlirOperationGetResult(fbOp, 0);
    }

    MlirOperationState fallbackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

    MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
    MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
    MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
    MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
    MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
    MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

    MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
    mlirOperationStateAddAttributes(&fallbackState, 3, attrs);

    MlirType fbPortType = getPicPortType(ctx);
    MlirType types[] = {fbPortType, fbPortType, fbPortType};
    mlirOperationStateAddResults(&fallbackState, 3, types);
    MlirOperation fbOp = mlirOperationCreate(&fallbackState);
    mlirBlockAppendOwnedOperation(block, fbOp);
    return mlirOperationGetResult(fbOp, 0);
}

MlirModule lowerAstToMlir(MlirContext ctx, AstNode *ast) {
    MlirLocation loc = mlirLocationUnknownGet(ctx);
    MlirModule module = mlirModuleCreateEmpty(loc);
    MlirBlock moduleBody = mlirModuleGetBody(module);

    Environment env;
    env_init(&env);
    if (mlirOpCount == 0) {
        addMlirOpName("pair", 4);
    }

    MlirBlock block = {NULL};
    if (ast->type == AST_FUNC_DECL) {
        char funcNameStr[256];
        snprintf(funcNameStr, sizeof(funcNameStr), "%.*s", ast->as.func_decl.name_len, ast->as.func_decl.name);
        sanitizeMlirName(funcNameStr);
        bool isMain = (strcmp(funcNameStr, "main") == 0);

        MlirOperationState funcState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);
        if (isMain) {
            snprintf(funcNameStr, sizeof(funcNameStr), "main_inet_entry");
        }
        MlirAttribute nameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(funcNameStr));
        MlirNamedAttribute nameNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")), nameAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &nameNamedAttr);

        MlirType portType = getPicPortType(ctx);
        if (isMain) {
            MlirType argTypes[] = {portType};
            MlirType retTypes[] = {portType};
            MlirType funcType = mlirFunctionTypeGet(ctx, 1, argTypes, 1, retTypes);
            MlirAttribute typeAttr = mlirTypeAttrGet(funcType);
            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")), typeAttr);
            mlirOperationStateAddAttributes(&funcState, 1, &typeNamedAttr);

            MlirRegion region = mlirRegionCreate();
            block = mlirBlockCreate(1, argTypes, &loc);
            mlirRegionAppendOwnedBlock(region, block);
            mlirOperationStateAddOwnedRegions(&funcState, 1, &region);

            MlirOperation funcOp = mlirOperationCreate(&funcState);
            mlirBlockAppendOwnedOperation(moduleBody, funcOp);

            int lin_arg_count = ast->as.func_decl.arg_count;
            for (int i = 0; i < lin_arg_count && i < 2; i++) {
                MlirValue eraVal = createEra(ctx, block, loc);
                env_add(&env, ast->as.func_decl.args[i].name, ast->as.func_decl.args[i].name_len, eraVal);
            }

            MlirValue result = lowerExpression(ctx, block, loc, ast->as.func_decl.body, &env, true);
            env_free(&env, ctx, block, loc);

            MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
            mlirOperationStateAddOperands(&retState, 1, &result);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&retState));
        } else {
            MlirType i64Type = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString("i64"));
            MlirType *types = (MlirType *)malloc(sizeof(MlirType) * 3);
            MlirLocation *locs = (MlirLocation *)malloc(sizeof(MlirLocation) * 3);
            if (!types || !locs) {
                fprintf(stderr, "Out of memory\n");
                exit(1);
            }
            types[0] = i64Type; locs[0] = loc;
            types[1] = i64Type; locs[1] = loc;
            types[2] = i64Type; locs[2] = loc;

            MlirType retTypes[] = {i64Type};
            MlirType funcType = mlirFunctionTypeGet(ctx, 3, types, 1, retTypes);
            LOG_STDERR("Lowering Func: %s with 1 result\n", funcNameStr);
            MlirAttribute typeAttr = mlirTypeAttrGet(funcType);
            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")), typeAttr);
            mlirOperationStateAddAttributes(&funcState, 1, &typeNamedAttr);

            MlirRegion region = mlirRegionCreate();
            block = mlirBlockCreate(3, types, locs);
            mlirRegionAppendOwnedBlock(region, block);
            mlirOperationStateAddOwnedRegions(&funcState, 1, &region);

            MlirOperation funcOp = mlirOperationCreate(&funcState);
            mlirBlockAppendOwnedOperation(moduleBody, funcOp);

            int lin_arg_count = ast->as.func_decl.arg_count;
            for (int i = 0; i < lin_arg_count && i < 2; i++) {
                MlirValue argValue = mlirBlockGetArgument(block, i);
                env_add(&env, ast->as.func_decl.args[i].name, ast->as.func_decl.args[i].name_len, argValue);
            }
            
            // Handle unused ports (link to ERA)
            MlirAttribute eraTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
            MlirNamedAttribute eraTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), eraTypeAttr);
            MlirAttribute starPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
            MlirNamedAttribute starPolNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), starPolAttr);
            MlirAttribute eraLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
            MlirNamedAttribute eraLabelNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), eraLabelAttr);
            MlirNamedAttribute eraAttrs[] = {eraTypeNamed, starPolNamed, eraLabelNamed};

            for (int i = lin_arg_count; i < 2; i++) {
                MlirOperationState eraState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                mlirOperationStateAddAttributes(&eraState, 3, eraAttrs);
                MlirType agentTypes[] = {portType, portType, portType};
                mlirOperationStateAddResults(&eraState, 3, agentTypes);
                MlirOperation eraOp = mlirOperationCreate(&eraState);
                mlirBlockAppendOwnedOperation(block, eraOp);
                
                MlirOperationState linkState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue linkOps[] = {mlirOperationGetResult(eraOp, 0), mlirBlockGetArgument(block, i)};
                mlirOperationStateAddOperands(&linkState, 2, linkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkState));
            }

            MlirValue result = lowerExpression(ctx, block, loc, ast->as.func_decl.body, &env, true);
            env_free(&env, ctx, block, loc);

            MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
            mlirOperationStateAddOperands(&retState, 1, &result);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&retState));

            free(types);
            free(locs);
        }
    } else if (ast->type == AST_BLOCK) {
        // Anonymous top-level block — wraps everything in main_inet_entry.
        // We use 0 arguments so that user scripts don't need a phantom arg.
        MlirType portType = getPicPortType(ctx);
        MlirType argTypes[] = {portType};
        MlirType retTypes[] = {portType};
        MlirType funcType = mlirFunctionTypeGet(ctx, 1, argTypes, 1, retTypes);

        MlirOperationState funcState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);

        MlirAttribute nameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("main_inet_entry"));
        MlirNamedAttribute nameNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")), nameAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &nameNamedAttr);

        MlirAttribute typeAttr = mlirTypeAttrGet(funcType);
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")), typeAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &typeNamedAttr);

        MlirRegion region = mlirRegionCreate();
        block = mlirBlockCreate(1, argTypes, &loc);
        MlirValue mainArg = mlirBlockGetArgument(block, 0);
        mlirRegionAppendOwnedBlock(region, block);
        mlirOperationStateAddOwnedRegions(&funcState, 1, &region);

        MlirOperation funcOp = mlirOperationCreate(&funcState);
        mlirBlockAppendOwnedOperation(moduleBody, funcOp);

        // lowerExpression(AST_BLOCK) iterates all statements, handles nested func decls,
        // and returns the last expression's port value.
        MlirValue result = lowerExpression(ctx, block, loc, ast, &env, true);

        // Erase any remaining live variables before the return
        env_free(&env, ctx, block, loc);

        MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        mlirOperationStateAddOperands(&retState, 1, &result);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&retState));
    } else {
        env_free(&env, ctx, block, loc);
    }

    return module;
}


