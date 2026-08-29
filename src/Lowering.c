#include "lin/Lowering.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Process C-style escape sequences in a string literal.
// Returns a malloc'd buffer (caller frees) or NULL if no escapes present.
static char *processEscapes(const char *src, int len, int *outLen) {
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) return NULL;
    int w = 0;
    for (int i = 0; i < len; i++) {
        char c = src[i];
        if (c == '\\' && i + 1 < len) {
            char e = src[++i];
            switch (e) {
                case 'n': buf[w++] = '\n'; break;
                case 't': buf[w++] = '\t'; break;
                case 'r': buf[w++] = '\r'; break;
                case '0': buf[w++] = '\0'; break;
                case '\\': buf[w++] = '\\'; break;
                case '"': buf[w++] = '"'; break;
                case '\'': buf[w++] = '\''; break;
                default: buf[w++] = '\\'; buf[w++] = e; break;
            }
        } else {
            buf[w++] = c;
        }
    }
    buf[w] = '\0';
    *outLen = w;
    return buf;
}

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

static void freeMlirOpNames(void) {
    for (int i = 0; i < mlirOpCount; i++) {
        free((void*)mlirOpNames[i]);
        mlirOpNames[i] = NULL;
    }
    mlirOpCount = 0;
}

static int isMlirOpName(const char *name, int len) {
    for (int i = 0; i < mlirOpCount; i++) {
        if ((int)strlen(mlirOpNames[i]) == len && strncmp(mlirOpNames[i], name, len) == 0)
            return 1;
    }
    return 0;
}

// True if this op INGESTS f32 operands (so float literals should be lowered to
// single-precision bits). f{op}32 math/comparison ops, print_f32, f32_to_* casts.
// f64_to_f32 has an f64 input so is excluded.
static int isF32IngestOp(const char *name, int len) {
    if (len == 9 && strncmp(name, "print_f32", 9) == 0) return 1;
    if (len == 10 && (strncmp(name, "f32_to_i32", 10) == 0 || strncmp(name, "f32_to_f64", 10) == 0)) return 1;
    if (len >= 6 && name[0] == 'f' && name[len-2] == '3' && name[len-1] == '2')
        return strncmp(name, "f64_to", 6) != 0;
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
    bool had_error;
    int f32_ctx;
} Environment;

static void env_init(Environment *env) {
    env->capacity = 16;
    env->count = 0;
    env->had_error = false;
    env->f32_ctx = 0;
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
                MlirValue eraP0 = createEra(ctx, block, loc);

                linkValues(block, loc, eraP0, env->vars[i].value);
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

static MlirValue lowerExpression(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env, bool is_top_level);

// Name of the function whose body is currently being lowered, if any. Used to
// detect a self-recursive call so it can be given a FRESH per-call-site omega
// (instead of reusing the shared self-closure omega), avoiding principal
// aliasing when two frames of the same closure are live simultaneously.
static const char *g_currentFuncName = NULL;
static int g_currentFuncNameLen = 0;

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

    linkValues(block, loc, dupP0, val);
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
        char (*tmp)[256] = realloc(fv->names, sizeof(*fv->names) * fv->capacity);
        if (!tmp) {
            fprintf(stderr, "Out of memory\n");
            exit(1);
        }
        fv->names = tmp;
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
            // EXCEPTION: language keywntor/control-flow call forms (either/pair/copy)
            // are NOT variables — they must never be captured as free vars, or the
            // closure would bind garbage to their name and force the omega-call path.
            bool isControlCallee =
                (node->as.call.callee_len == 6 && memcmp(node->as.call.callee, "either", 6) == 0) ||
                (node->as.call.callee_len == 4 && memcmp(node->as.call.callee, "pair", 4) == 0) ||
                (node->as.call.callee_len == 4 && memcmp(node->as.call.callee, "copy", 4) == 0);
            if (!is_bound && !isControlCallee)
                addFreeVar(fv, node->as.call.callee, node->as.call.callee_len);
            
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


static MlirValue makeOmegaLiteral(MlirContext ctx, MlirBlock block, MlirLocation loc,
                                   const char *label, int64_t val, bool isString,
                                   const char *strVal, int strLen) {
    MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
    MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
    MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
    MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
    MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(label));
    MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

    if (isString) {
        int processedLen = 0;
        char *processed = processEscapes(strVal, strLen, &processedLen);
        const char *finalStr = processed ? processed : strVal;
        int finalLen = processed ? processedLen : strLen;
        MlirAttribute valAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(finalStr, finalLen));
        MlirNamedAttribute valNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("str_val")), valAttr);
        MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr, valNamedAttr};
        mlirOperationStateAddAttributes(&state, 4, attrs);
        free(processed);
    } else {
        MlirAttribute valAttr = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 64), val);
        MlirNamedAttribute valNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("value")), valAttr);
        MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr, valNamedAttr};
        mlirOperationStateAddAttributes(&state, 4, attrs);
    }

    MlirType portType = getPicPortType(ctx);
    MlirType types[] = {portType, portType, portType};
    mlirOperationStateAddResults(&state, 3, types);

    MlirOperation op = mlirOperationCreate(&state);
    mlirBlockAppendOwnedOperation(block, op);
    return mlirOperationGetResult(op, 0);
}



static MlirBlock findModuleBody(MlirBlock block) {
    if (mlirBlockIsNull(block)) {
        MlirBlock nullBlock = {NULL};
        return nullBlock;
    }
    MlirOperation parentOp = mlirBlockGetParentOperation(block);
    MlirOperation moduleOp = parentOp;
    while (!mlirOperationIsNull(moduleOp)) {
        MlirStringRef opName = mlirIdentifierStr(mlirOperationGetName(moduleOp));
        if (strncmp(opName.data, "builtin.module", 14) == 0) break;
        moduleOp = mlirOperationGetParentOperation(moduleOp);
    }
    if (!mlirOperationIsNull(moduleOp)) {
        return mlirModuleGetBody(mlirModuleFromOperation(moduleOp));
    }
    MlirBlock nullBlock = {NULL};
    return nullBlock;
}

static MlirValue lowerAssignmentExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {

    MlirValue rhs = lowerExpression(ctx, block, loc, expr->as.assignment.value, env, false);
    env_add(env, expr->as.assignment.name, expr->as.assignment.name_len, rhs);
    return rhs;
}

static MlirValue lowerPairExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {

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

    linkValues(block, loc, p1, left);
    linkValues(block, loc, p2, right);
    return p0;
}

static MlirValue lowerFieldAccessExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {

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
    
    linkValues(block, loc, p1, base_val);// Item 4d: p2 is unused when only one field is accessed; attach an eraser so no
    // port is left dangling (a dangling principal port causes the reduction to stall).
    linkToEra(ctx, block, loc, p2);
    
    return p0;
}

static MlirValue lowerBlockExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {

    MlirValue lastVal = {NULL};
    for (int i = 0; i < expr->as.block.count; i++) {
        AstNode *stmt = expr->as.block.statements[i];
        // Skip import and mlir_op since they have no runtime code
        if (!stmt) continue;
        if (stmt->type == AST_IMPORT) continue;
        lastVal = lowerExpression(ctx, block, loc, stmt, env, false);
    }

    if (mlirValueIsNull(lastVal)) {
        lastVal = createEra(ctx, block, loc);
    }

    return lastVal;
}

static MlirValue lowerBlockDataExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {
    // Create pair chain from block elements: pair(e0, pair(e1, ... pair(eN, era)))
    // Each pair is a gamma+ agent with label "pair"
    MlirType portType = getPicPortType(ctx);
    MlirType agentTypes[] = {portType, portType, portType};

    MlirAttribute pairTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
    MlirNamedAttribute pType = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), pairTypeAttr);
    MlirAttribute plusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
    MlirNamedAttribute pPol = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), plusPol);
    MlirAttribute labelPair = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
    MlirNamedAttribute pLabel = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelPair);
    MlirNamedAttribute pAttrs[] = {pType, pPol, pLabel};

    MlirValue chain = createEra(ctx, block, loc);
    for (int i = expr->as.block.count - 1; i >= 0; i--) {
        AstNode *stmt = expr->as.block.statements[i];
        if (!stmt || stmt->type == AST_IMPORT) continue;
        MlirValue elem = lowerExpression(ctx, block, loc, stmt, env, false);

        MlirOperationState pairState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        mlirOperationStateAddAttributes(&pairState, 3, pAttrs);
        mlirOperationStateAddResults(&pairState, 3, agentTypes);
        MlirOperation pairOp = mlirOperationCreate(&pairState);
        mlirBlockAppendOwnedOperation(block, pairOp);

        MlirValue p0 = mlirOperationGetResult(pairOp, 0);
        MlirValue p1 = mlirOperationGetResult(pairOp, 1);
        MlirValue p2 = mlirOperationGetResult(pairOp, 2);
        linkValues(block, loc, p1, elem);
        linkValues(block, loc, p2, chain);
        chain = p0;
    }
    return chain;
}

static MlirValue lowerWhileExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env, MlirBlock moduleBody) {

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
    snprintf(prefixedMacroName, sizeof(prefixedMacroName), "lin_%s", macro_func_name);
    snprintf(prefixedBodyName, sizeof(prefixedBodyName), "lin_%s", body_func_name);
    snprintf(prefixedExitName, sizeof(prefixedExitName), "lin_%s", exit_func_name);

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

        linkToEra(ctx, exitBlock, loc, exitMainArg);
        // The exit closure's env IS the final state bundle (the v_falses
        // chain). Return it as the loop result so the parent unpacks the finals.
        linkValues(exitBlock, loc, exitEnvBundle, exitResultPort);

        MlirOperationState exitRetState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        mlirOperationStateAddOperands(&exitRetState, 1, &exitResultPort);
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
        // The body's live vars are delivered via the closure env (the v_trues
        // chain captured by the macro's true branch), NOT via the main arg.
        linkToEra(ctx, bodyBlock, loc, bodyMainArg);

        Environment bodyEnv;
        env_init(&bodyEnv);
        MlirValue bodyCurrentBundle = capRemaining;
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
        mlirOperationStateAddOperands(&bodyRetState, 1, &bodyResultPort);
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

        // Correct either port mapping (same as lowerStatementEither):
        // p1=cond(principal), p0=branches(pair), p2=result.
        MlirValue eitherP0, eitherP1, eitherP2;
        createOmega(ctx, macroBlock, loc, "either", "-", &eitherP0, &eitherP1, &eitherP2);

        linkValues(macroBlock, loc, eitherP1, cond);
        linkValues(macroBlock, loc, eitherP0, branches_pair);
        MlirValue eitherResult = eitherP2;

        // Call-after-select: unpack the chosen closure (body or exit) and invoke
        // it via omega-(call). pack(envOfClosure, era) — the body/exit closures
        // receive their live vars through the closure env, not the main arg.
        MlirOperationState macCu = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        mlirOperationStateAddAttributes(&macCu, 3, macroUnpackAttrs);
        mlirOperationStateAddResults(&macCu, 3, agentTypes);
        MlirOperation macCuOp = mlirOperationCreate(&macCu);
        mlirBlockAppendOwnedOperation(macroBlock, macCuOp);
        linkValues(macroBlock, loc, mlirOperationGetResult(macCuOp, 0), eitherResult);
        MlirValue macFVal = mlirOperationGetResult(macCuOp, 1);
        MlirValue macEnv = mlirOperationGetResult(macCuOp, 2);

        MlirValue macAP0, macAP1, macAP2;
        createOmega(ctx, macroBlock, loc, "call", "-", &macAP0, &macAP1, &macAP2);
        linkValues(macroBlock, loc, macAP0, macFVal);

        MlirOperationState macCP = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        mlirOperationStateAddAttributes(&macCP, 3, macroPackAttrs);
        mlirOperationStateAddResults(&macCP, 3, agentTypes);
        MlirOperation macCPOp = mlirOperationCreate(&macCP);
        mlirBlockAppendOwnedOperation(macroBlock, macCPOp);
        linkValues(macroBlock, loc, mlirOperationGetResult(macCPOp, 1), macEnv);
        linkValues(macroBlock, loc, mlirOperationGetResult(macCPOp, 2), createEra(ctx, macroBlock, loc));
        linkValues(macroBlock, loc, macAP1, mlirOperationGetResult(macCPOp, 0));
        linkValues(macroBlock, loc, macAP2, macroResultPort);

        MlirOperationState macroRetState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        mlirOperationStateAddOperands(&macroRetState, 1, &macroResultPort);
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

static MlirValue lowerFuncDeclExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env, MlirBlock moduleBody) {

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
    snprintf(prefixedName, sizeof(prefixedName), "lin_%s", funcNameStr);
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

    // Create the function's shared entry omega BEFORE lowering the body, so that a
    // self-reference inside the body can bind to a closure over this SAME omega.
    // Keeping it acyclic (bundle = era) lets recursive calls follow the normal
    // closure-call path instead of capturing a cyclic closure (which the eager
    // reducer cannot duplicate).
    MlirValue selfOmegaP0 = {NULL};
    if (expr->as.func_decl.name_len > 0) {
        MlirAttribute symTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute symTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), symTypeAttr);
        MlirAttribute symPlusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
        MlirNamedAttribute symPlusPolNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), symPlusPol);
        MlirAttribute symLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(funcNameStr));
        MlirNamedAttribute symLabelNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), symLabelAttr);
        MlirOperationState selfState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirNamedAttribute selfAttrs[] = {symTypeNamed, symPlusPolNamed, symLabelNamed};
        mlirOperationStateAddAttributes(&selfState, 3, selfAttrs);
        MlirType selfPorts[] = {portType, portType, portType};
        mlirOperationStateAddResults(&selfState, 3, selfPorts);
        MlirOperation selfOp = mlirOperationCreate(&selfState);
        mlirBlockAppendOwnedOperation(block, selfOp);
        linkToEra(ctx, block, loc, mlirOperationGetResult(selfOp, 1));
        linkToEra(ctx, block, loc, mlirOperationGetResult(selfOp, 2));
        selfOmegaP0 = mlirOperationGetResult(selfOp, 0);
    }

    // Build a NON-cyclic self-closure (omega + era bundle) in the outer block. It is
    // used in place of ERA when the closure packs a capture equal to its own name, so
    // a recursive body can resolve and call itself through the normal closure-call path.
    MlirValue selfClosure = {NULL};
    if (!mlirValueIsNull(selfOmegaP0)) {
        MlirValue selfDummyBundle = createEra(ctx, block, loc);
        selfClosure = bundleClosure(ctx, block, loc, selfOmegaP0, selfDummyBundle);
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

    linkValues(innerBlock, loc, umP0, inputBundle);// Unpack captures from envBundle
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

        linkValues(innerBlock, loc, p0, currentBundle);
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
    linkValues(innerBlock, loc, mlirOperationGetResult(eraInnerOp, 0), currentBundle);// Add main argument to env
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
            linkValues(innerBlock, loc, uP0, unpackChainArg);
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
            linkValues(innerBlock, loc, mlirOperationGetResult(eraArgOp, 0), unpackChainArg);
        }
    } else {
        MlirOperationState eraArgState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        mlirOperationStateAddAttributes(&eraArgState, 3, eraAttrs);
        mlirOperationStateAddResults(&eraArgState, 3, agentTypes);
        MlirOperation eraArgOp = mlirOperationCreate(&eraArgState);
        mlirBlockAppendOwnedOperation(innerBlock, eraArgOp);
        linkValues(innerBlock, loc, mlirOperationGetResult(eraArgOp, 0), mainArg);
        }

    // If the body's last statement is a bare identifier, return the stored value's
    // principal directly instead of an env_fetch dup aux. A single-consumption return
    // must hand the caller a principal port so its op can fire a redex against it.
    // Lower all statements except the trailing identifier, then use the bound value.
    MlirValue bodyResult = {NULL};
    bool tailIsIdentifier = false;
    const char *savedFuncName = g_currentFuncName;
    int savedFuncNameLen = g_currentFuncNameLen;
    if (expr->as.func_decl.name_len > 0) {
        g_currentFuncName = expr->as.func_decl.name;
        g_currentFuncNameLen = expr->as.func_decl.name_len;
    }
    if (expr->as.func_decl.body && expr->as.func_decl.body->type == AST_BLOCK &&
        expr->as.func_decl.body->as.block.count > 0) {
        AstNode *tail = expr->as.func_decl.body->as.block.statements[expr->as.func_decl.body->as.block.count - 1];
        if (tail && tail->type == AST_IDENTIFIER) {
            for (int bi = 0; bi < expr->as.func_decl.body->as.block.count - 1; bi++) {
                AstNode *bs = expr->as.func_decl.body->as.block.statements[bi];
                if (!bs || bs->type == AST_IMPORT) continue;
                lowerExpression(ctx, innerBlock, loc, bs, &innerEnv, false);
            }
            bodyResult = env_get(&innerEnv, tail->as.identifier.name, tail->as.identifier.length);
            tailIsIdentifier = true;
        }
    }
    if (!tailIsIdentifier) {
        bodyResult = lowerExpression(ctx, innerBlock, loc, expr->as.func_decl.body, &innerEnv, true);
    }
    g_currentFuncName = savedFuncName;
    g_currentFuncNameLen = savedFuncNameLen;
    env_free(&innerEnv, ctx, innerBlock, loc);

    linkValues(innerBlock, loc, bodyResult, resultPort);
    MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
    mlirOperationStateAddOperands(&retState, 1, &resultPort);
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
    // The function's externally-visible closure gets its OWN freshly-created omega
    // (baseOp), distinct from the selfClosure's omega. This avoids aliasing the same
    // omega node between the external closure and the recursive self-closure, which
    // corrupted the net when two recursive frames were live simultaneously.
    MlirValue omegaP0 = mlirOperationGetResult(baseOp, 0);

    MlirOperationState eraOuterState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    mlirOperationStateAddAttributes(&eraOuterState, 3, eraAttrs);
    mlirOperationStateAddResults(&eraOuterState, 3, agentTypes);
    MlirOperation eraOuterOp = mlirOperationCreate(&eraOuterState);
    mlirBlockAppendOwnedOperation(block, eraOuterOp);
    MlirValue currentOuterBundle = mlirOperationGetResult(eraOuterOp, 0);

    for (int i = fv.count - 1; i >= 0; i--) {
        MlirValue capVal = env_fetch(ctx, block, loc, env, fv.names[i], strlen(fv.names[i]));
        // A self-reference (the function's own name) is not in the enclosing env yet;
        // substitute the shared non-cyclic self-closure instead of an unbound era.
        bool isSelfCapture = (expr->as.func_decl.name_len > 0) &&
                             (strlen(fv.names[i]) == (size_t)expr->as.func_decl.name_len) &&
                             (memcmp(fv.names[i], expr->as.func_decl.name, expr->as.func_decl.name_len) == 0);
        if (mlirValueIsNull(capVal) && isSelfCapture && !mlirValueIsNull(selfClosure)) {
            capVal = selfClosure;
        } else if (mlirValueIsNull(capVal)) {
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

        linkValues(block, loc, p1, capVal);
        linkValues(block, loc, p2, currentOuterBundle);
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

    linkValues(block, loc, closureP1, omegaP0);
    linkValues(block, loc, closureP2, currentOuterBundle);
    MlirValue currentVal = closureP0;

    if (expr->as.func_decl.name_len > 0) {
        env_add(env, expr->as.func_decl.name, expr->as.func_decl.name_len, currentVal);
    }

    if (fv.names) free(fv.names);
    return currentVal;
}

// Build ONE deferred branch closure for a statement-either. Registers a func
// that unpacks the captured active-env bundle, runs the branch block (side
// effects like prints), and returns the new "state" value; returns a closure
// (gamma+ pair) over the branch func's omega and the shared capture bundle.
static MlirValue buildEitherBranchClosure(MlirContext ctx, MlirBlock block, MlirLocation loc,
                                          MlirBlock moduleBody, const char *funcName,
                                          const char *prefixedName, AstNode *brBlock,
                                          const char **active_names, int *active_lens,
                                          int active_count, MlirValue cap_bundle,
                                          bool valueMode) {
    MlirType portType = getPicPortType(ctx);
    MlirType agentTypes[] = {portType, portType, portType};

    MlirAttribute pairType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
    MlirNamedAttribute pairTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), pairType);
    MlirAttribute minusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
    MlirNamedAttribute minusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), minusPol);
    MlirAttribute labelPair = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
    MlirNamedAttribute labelPairAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelPair);
    MlirNamedAttribute unpackAttrs[] = {pairTypeAttr, minusPolAttr, labelPairAttr};

    MlirBlock bBlock = createFunctionBlock(ctx, loc, moduleBody, prefixedName);
    registerFunction(ctx, loc, moduleBody, funcName, prefixedName);

    MlirValue bRawBundle = mlirBlockGetArgument(bBlock, 0);
    MlirValue bResultPort = mlirBlockGetArgument(bBlock, 1);

    MlirOperationState bu = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    mlirOperationStateAddAttributes(&bu, 3, unpackAttrs);
    mlirOperationStateAddResults(&bu, 3, agentTypes);
    MlirOperation buOp = mlirOperationCreate(&bu);
    mlirBlockAppendOwnedOperation(bBlock, buOp);
    linkValues(bBlock, loc, mlirOperationGetResult(buOp, 0), bRawBundle);

    Environment bEnv;
    env_init(&bEnv);
    // arg0 = pack(gamma+) wrapping cap_bundle; buOp (gamma-) peels the pack,
    // so buOp.p1 IS cap_bundle (the per-var gamma+ chain). Unpack that chain's
    // p2-chain directly; do NOT add an extra unpack level (that would try to
    // unpack a value node as a pair and stall).
    MlirValue bCur2 = mlirOperationGetResult(buOp, 1);
    for (int i = 0; i < active_count; i++) {
        MlirOperationState uu = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        mlirOperationStateAddAttributes(&uu, 3, unpackAttrs);
        mlirOperationStateAddResults(&uu, 3, agentTypes);
        MlirOperation uuOp = mlirOperationCreate(&uu);
        mlirBlockAppendOwnedOperation(bBlock, uuOp);
        linkValues(bBlock, loc, mlirOperationGetResult(uuOp, 0), bCur2);
        env_add(&bEnv, active_names[i], active_lens[i], mlirOperationGetResult(uuOp, 1));
        bCur2 = mlirOperationGetResult(uuOp, 2);
    }
    linkToEra(ctx, bBlock, loc, bCur2);

    MlirValue brRes = lowerExpression(ctx, bBlock, loc, brBlock, &bEnv, true);
    MlirValue stVal = {NULL};
    if (valueMode) {
        // Value branch: the block's single expression value IS the result.
        stVal = brRes;
    } else {
        for (int j = bEnv.count - 1; j >= 0; j--) {
            if (bEnv.vars[j].name_len == 5 && strncmp(bEnv.vars[j].name, "state", 5) == 0) {
                stVal = bEnv.vars[j].value;
                bEnv.vars[j].value = (MlirValue){NULL};
                break;
            }
        }
    }
    if (mlirValueIsNull(stVal)) stVal = createEra(ctx, bBlock, loc);
    linkValues(bBlock, loc, stVal, bResultPort);
    env_free(&bEnv, ctx, bBlock, loc);

    MlirOperationState ret = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
    mlirOperationStateAddOperands(&ret, 1, &bResultPort);
    mlirBlockAppendOwnedOperation(bBlock, mlirOperationCreate(&ret));

    MlirValue brOmega = createOmegaP0(ctx, block, loc, funcName, "+");
    return bundleClosure(ctx, block, loc, brOmega, cap_bundle);
}

// Statement-either: lower the two branch blocks as DEFERRED closures so that,
// after selectCase links the either's result to the chosen closure, a
// closure-call invokes that branch (running its side effects) and yields the
// new state value. This is the same primitive `while` needs.
// When valueMode is set, each branch block is a single VALUE expression and
// the closure returns that value (instead of threading a "state" variable);
// the resulting closure-call result carries the selected branch's value.
static MlirValue lowerStatementEither(MlirContext ctx, MlirBlock block, MlirLocation loc,
                                      AstNode *expr, Environment *env, MlirBlock moduleBody,
                                      bool valueMode) {
    MlirType portType = getPicPortType(ctx);
    MlirType agentTypes[] = {portType, portType, portType};

    MlirAttribute pairType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
    MlirNamedAttribute pairTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), pairType);
    MlirAttribute plusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
    MlirNamedAttribute plusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), plusPol);
    MlirAttribute labelPair = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
    MlirNamedAttribute labelPairAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelPair);
    MlirNamedAttribute packAttrs[] = {pairTypeAttr, plusPolAttr, labelPairAttr};

    // Capture active env vars (consume from env) into cap_bundle.
    int active_count = 0;
    for (int i = 0; i < env->count; i++) {
        if (!mlirValueIsNull(env->vars[i].value)) active_count++;
    }
    const char **active_names = (const char **)malloc(sizeof(char *) * (active_count ? active_count : 1));
    int *active_lens = (int *)malloc(sizeof(int) * (active_count ? active_count : 1));
    MlirValue *active_vals = (MlirValue *)malloc(sizeof(MlirValue) * (active_count ? active_count : 1));
    int vidx = 0;
    for (int i = 0; i < env->count; i++) {
        if (!mlirValueIsNull(env->vars[i].value)) {
            active_names[vidx] = env->vars[i].name;
            active_lens[vidx] = env->vars[i].name_len;
            // Capture a DUP of each live value so the branch consumes a private
            // copy while `env` keeps the original for subsequent statements.
            // A direct move would destructively rewire the shared cell that a
            // trailing statement later reads (OOB at runtime).
            active_vals[vidx] = env_fetch(ctx, block, loc, env, active_names[vidx], active_lens[vidx]);
            vidx++;
        }
    }

    MlirValue cap_bundle = createEra(ctx, block, loc);
    for (int i = active_count - 1; i >= 0; i--) {
        MlirOperationState p = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        mlirOperationStateAddAttributes(&p, 3, packAttrs);
        mlirOperationStateAddResults(&p, 3, agentTypes);
        MlirOperation pOp = mlirOperationCreate(&p);
        mlirBlockAppendOwnedOperation(block, pOp);
        linkValues(block, loc, mlirOperationGetResult(pOp, 1), active_vals[i]);
        linkValues(block, loc, mlirOperationGetResult(pOp, 2), cap_bundle);
        cap_bundle = mlirOperationGetResult(pOp, 0);
    }

    static int either_counter = 0;
    char t_name[128], f_name[128], p_t[160], p_f[160];
    snprintf(t_name, sizeof(t_name), "_either_t_%d", either_counter);
    snprintf(f_name, sizeof(f_name), "_either_f_%d", either_counter);
    either_counter++;
    snprintf(p_t, sizeof(p_t), "lin_%s", t_name);
    snprintf(p_f, sizeof(p_f), "lin_%s", f_name);

    AstNode *thenB = expr->as.call.args[1]->as.pair.left;
    AstNode *elseB = expr->as.call.args[1]->as.pair.right;

    MlirValue thenClosure = buildEitherBranchClosure(ctx, block, loc, moduleBody, t_name, p_t, thenB, active_names, active_lens, active_count, cap_bundle, valueMode);
    MlirValue elseClosure = buildEitherBranchClosure(ctx, block, loc, moduleBody, f_name, p_f, elseB, active_names, active_lens, active_count, cap_bundle, valueMode);
    MlirValue branches_pair = createPair(ctx, block, loc, thenClosure, elseClosure);

    MlirValue cond = lowerExpression(ctx, block, loc, expr->as.call.args[0], env, false);

    // Correct either port mapping: p1=cond(principal), p0=branches(reduced port2=pair), p2=result(reduced port1).
    MlirValue eP0, eP1, eP2;
    createOmega(ctx, block, loc, "either", "-", &eP0, &eP1, &eP2);
    linkValues(block, loc, eP1, cond);
    linkValues(block, loc, eP0, branches_pair);
    MlirValue eitherResult = eP2; // selectCase links this to the chosen closure

    // Call-after-select: unpack the chosen closure and invoke it via omega-(call).
    MlirAttribute minusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
    MlirNamedAttribute minusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), minusPol);
    MlirNamedAttribute unpackAttrs[] = {pairTypeAttr, minusPolAttr, labelPairAttr};

    MlirOperationState cu = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    mlirOperationStateAddAttributes(&cu, 3, unpackAttrs);
    mlirOperationStateAddResults(&cu, 3, agentTypes);
    MlirOperation cuOp = mlirOperationCreate(&cu);
    mlirBlockAppendOwnedOperation(block, cuOp);
    linkValues(block, loc, mlirOperationGetResult(cuOp, 0), eitherResult);
    MlirValue fVal = mlirOperationGetResult(cuOp, 1);
    MlirValue envOfClosure = mlirOperationGetResult(cuOp, 2);

    MlirValue aP0, aP1, aP2;
    createOmega(ctx, block, loc, "call", "-", &aP0, &aP1, &aP2);
    linkValues(block, loc, aP0, fVal);

    MlirOperationState ap = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
    mlirOperationStateAddAttributes(&ap, 3, packAttrs);
    mlirOperationStateAddResults(&ap, 3, agentTypes);
    MlirOperation apOp = mlirOperationCreate(&ap);
    mlirBlockAppendOwnedOperation(block, apOp);
    linkValues(block, loc, mlirOperationGetResult(apOp, 1), envOfClosure);
    linkValues(block, loc, mlirOperationGetResult(apOp, 2), createEra(ctx, block, loc));
    linkValues(block, loc, aP1, mlirOperationGetResult(apOp, 0));

    free(active_names);
    free(active_lens);
    free(active_vals);
    return aP2;
}


static MlirValue lowerCallExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {

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
        linkValues(block, loc, p1, left);
        linkValues(block, loc, p2, right);
        return p0;
    }

    // Handle (copy [...]) - create pair chain from block elements
    if (effectiveCalleeLen == 4 && memcmp(effectiveCallee, "copy", 4) == 0 && expr->as.call.arg_count >= 1) {
        AstNode *firstArg = expr->as.call.args[0];
        if (firstArg->type == AST_BLOCK || firstArg->type == AST_BLOCK_DATA) {
            return lowerBlockDataExpr(ctx, block, loc, firstArg, env);
        }
    }

    // Statement-either / value-either: (either cond [then][else]) where the
    // branch blocks contain either side-effecting statements (assignments/calls)
    // or NON-LITERAL single expressions (identifiers, nested calls). These must
    // be lowered as DEFERRED closures so only the chosen branch runs (the eager
    // lowerPairExpr path would execute both branches and crash selectCase).
    // Pure-literal value branches ([1][0]) stay on the fast value-either path.
    if (effectiveCalleeLen == 6 && memcmp(effectiveCallee, "either", 6) == 0 && expr->as.call.arg_count == 2) {
        AstNode *branchPair = expr->as.call.args[1];
        bool branchIsStmt = false;
        bool branchIsValue = false;
        if (branchPair && branchPair->type == AST_PAIR) {
            AstNode *branches[2] = {branchPair->as.pair.left, branchPair->as.pair.right};
            for (int bi = 0; bi < 2; bi++) {
                AstNode *b = branches[bi];
                if (b && b->type == AST_BLOCK && b->as.block.count > 0) {
                    AstNode *s0 = b->as.block.statements[0];
                    if (b->as.block.count != 1 || (s0 && s0->type == AST_ASSIGNMENT)) {
                        branchIsStmt = true;
                    } else if (s0 && s0->type != AST_NUMBER && s0->type != AST_BOOL && s0->type != AST_FLOAT) {
                        branchIsValue = true;
                    }
                }
            }
        }
        if (branchIsStmt || branchIsValue) {
            MlirBlock moduleBody = findModuleBody(block);
            return lowerStatementEither(ctx, block, loc, expr, env, moduleBody, branchIsValue && !branchIsStmt);
        }
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
    // A self-recursive call: allocate a FRESH per-call omega for this call site
    // instead of reusing the single shared closure-omega node. Each runtime call
    // re-instantiates the body (via lin_<name>) and re-allocates this fresh node,
    // so two simultaneously-live frames of the same closure never alias one omega
    // principal (which previously caused an OOB in the reducer).
    bool isSelfCall = (g_currentFuncNameLen > 0) &&
                      (effectiveCalleeLen == g_currentFuncNameLen) &&
                      (memcmp(effectiveCallee, g_currentFuncName, g_currentFuncNameLen) == 0);
    MlirValue selfFreshOmega = {NULL};
    if (isSelfCall) {
        // Label the fresh omega with the function's registered name (a proper
        // NUL-terminated string). effectiveCallee is a length-prefixed slice
        // into the source and would otherwise be read past its end by the
        // NUL-terminated label builders, corrupting the dispatch label.
        char selfName[256];
        snprintf(selfName, sizeof(selfName), "%.*s", (int)g_currentFuncNameLen, g_currentFuncName);
        selfFreshOmega = createOmegaP0(ctx, block, loc, selfName, "+");
    }
    if (!mlirValueIsNull(currentVal)) {        MlirType portType = getPicPortType(ctx);
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
                linkValues(block, loc, p1, argVal);
                linkValues(block, loc, p2, argsPack);
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

        linkValues(block, loc, uP0, currentVal);// Now bundle (env_bundle, argsPack) into a pair for the call
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
            if (!mlirValueIsNull(selfFreshOmega)) {
                linkValues(block, loc, appP0, selfFreshOmega);
                linkToEra(ctx, block, loc, uP1);
            } else {
                linkValues(block, loc, appP0, uP1);
            }// Create the pair(env, args)
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
            linkValues(block, loc, pP1, uP2);// Link pP2 ↔ argsPack (all args packed as nested pair)
            linkValues(block, loc, pP2, argsPack);// Link appP1 ↔ pP0
            linkValues(block, loc, appP1, pP0);
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
        int saved_f32 = env->f32_ctx;
        env->f32_ctx = isF32IngestOp(effectiveCallee, effectiveCalleeLen);
        MlirValue left = lowerExpression(ctx, block, loc, expr->as.call.args[0], env, false);
        MlirValue right = lowerExpression(ctx, block, loc, expr->as.call.args[1], env, false);
        env->f32_ctx = saved_f32;

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
        // Link omega- p0 ↔ right p0 (value arg forms active pair with literal)
        linkValues(block, loc, result, right);
        // Return p2 as the result port — p0 is the active pair partner (linked to value arg).
        // Returning p2 avoids double-linking p0 when nested calls link the result to a consumer.
        // p2 is unlinked here; the runtime defaults it to era. If a consumer links to p2,
        // the LinkOp overwrites the era default, and after the FireOp the result node links to p2's target.

        return p2;
    } else if (expr->as.call.arg_count == 1) {
        int saved_f32 = env->f32_ctx;
        env->f32_ctx = isF32IngestOp(effectiveCallee, effectiveCalleeLen);
        MlirValue arg = lowerExpression(ctx, block, loc, expr->as.call.args[0], env, false);
        env->f32_ctx = saved_f32;

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
        // Link omega- p0 ↔ arg p0 (value arg forms active pair)
        linkValues(block, loc, result, arg);
        // Return p2 as the result port (avoids double-linking p0 for nested calls)

        return p2;
    }

    return createEra(ctx, block, loc);
}



static MlirValue lowerLiteralExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr) {
    if (expr->type == AST_NUMBER || expr->type == AST_BOOL) {
        int64_t val = (expr->type == AST_NUMBER) ? expr->as.number.value : (expr->as.boolean.value ? 1 : 0);
        const char *label = "i32";
        if (expr->type == AST_NUMBER) {
            if (val < -2147483648LL || val > 2147483647LL) {
                label = "i64";
            }
        }
        return makeOmegaLiteral(ctx, block, loc, label, val, false, NULL, 0);
    }
    if (expr->type == AST_FLOAT) {
        union { double f; int64_t i; } cast;
        cast.f = expr->as.f_number.value;
        return makeOmegaLiteral(ctx, block, loc, "f64", cast.i, false, NULL, 0);
    }
    if (expr->type == AST_STRING) {
        return makeOmegaLiteral(ctx, block, loc, "str", 0, true, expr->as.string.value, expr->as.string.length);
    }
    MlirValue nullVal = {NULL};
    return nullVal;
}

static MlirValue lowerIdentifierExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {
    MlirValue val = env_fetch(ctx, block, loc, env, expr->as.identifier.name, expr->as.identifier.length);
    if (mlirValueIsNull(val)) {
        fprintf(stderr, "Unbound variable: %.*s\n", expr->as.identifier.length, expr->as.identifier.name);
        env->had_error = true;
    }
    return val;
}

static MlirValue lowerMlirOpExpr(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {

    addMlirOpName(expr->as.mlir_op.name, expr->as.mlir_op.name_len);

    MlirOperationState regState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.registry"), loc);

    MlirAttribute nameAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.name, expr->as.mlir_op.name_len));
    MlirNamedAttribute nameNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("op_name")), nameAttr);

    MlirAttribute payloadAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.mlir_payload, expr->as.mlir_op.payload_len));
    MlirNamedAttribute payloadNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("payload")), payloadAttr);

    // Pass the cleaned-up inputs as [%name1][%name2]...
char cleanNames[2048] = "";
    size_t cleanRem = sizeof(cleanNames) - 1;
    const char *p = expr->as.mlir_op.inputs;
    int len = expr->as.mlir_op.inputs_len;
    for (int i = 0; i < len; i++) {
        if (p[i] == '[') {
            i++;
            while (i < len && isspace((unsigned char)p[i])) i++;
            if (i + 7 <= len && strncmp(p + i, "inputs:", 7) == 0) {
                i += 7;
                while (i < len && p[i] != '[') i++;
                if (i < len && p[i] == '[') {
                    i++;
                    while (i < len && p[i] != ']') {
                        while (i < len && isspace((unsigned char)p[i])) i++;
                        if (i >= len || p[i] == ']') break;

                        const char *nameStart = p + i;
                        while (i < len && !isspace((unsigned char)p[i]) && p[i] != '[' && p[i] != ']') i++;
                        int nameLen = (int)(p + i - nameStart);

                        if (nameLen > 0 && cleanRem > 10) {
                            strcat(cleanNames, "[%");
                            strncat(cleanNames, nameStart, nameLen);
                            cleanRem = sizeof(cleanNames) - strlen(cleanNames) - 1;

                            while (i < len && isspace((unsigned char)p[i])) i++;
                            if (i < len && p[i] == '[') {
                                i++;
                                const char *typeStart = p + i;
                                while (i < len && p[i] != '!' && p[i] != ']' && !isspace((unsigned char)p[i])) {
                                    i++;
                                }
                                int typeLen = (int)(p + i - typeStart);
                                if (typeLen > 0 && cleanRem > 10) {
                                    strcat(cleanNames, "_");
                                    strncat(cleanNames, typeStart, typeLen);
                                    cleanRem = sizeof(cleanNames) - strlen(cleanNames) - 1;
                                }

                                int depth = 1;
                                while (i < len && depth > 0) {
                                    if (p[i] == '[') depth++;
                                    else if (p[i] == ']') depth--;
                                    i++;
                                }
                            }
                            if (cleanRem > 1) strcat(cleanNames, "]");
                            cleanRem = sizeof(cleanNames) - strlen(cleanNames) - 1;
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
    MlirNamedAttribute attrs[5];
    attrs[0] = nameNamedAttr;
    attrs[1] = payloadNamedAttr;
    attrs[2] = namesNamedAttr;
    if (expr->as.mlir_op.dispatch) {
        MlirAttribute dispatchAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.dispatch, expr->as.mlir_op.dispatch_len));
        attrs[3] = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("dispatch")), dispatchAttr);
        regAttrCount = 4;
    }
    if (expr->as.mlir_op.inverse_payload) {
        MlirAttribute invPayloadAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.inverse_payload, expr->as.mlir_op.inverse_len));
        attrs[regAttrCount] = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("inverse_payload")), invPayloadAttr);
        regAttrCount++;
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


static MlirValue lowerExpression(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env, bool is_top_level) {
if (!expr) {
        MlirValue nullVal = {NULL};
        return nullVal;
    }

    MlirBlock moduleBody = findModuleBody(block);

if (expr->type == AST_NUMBER || expr->type == AST_BOOL || expr->type == AST_STRING) {
        return lowerLiteralExpr(ctx, block, loc, expr);
    }

    if (expr->type == AST_FLOAT) {
        if (env->f32_ctx) {
            union { float f; uint32_t u; } cast;
            cast.f = (float)expr->as.f_number.value;
            return makeOmegaLiteral(ctx, block, loc, "f32", (int64_t)cast.u, false, NULL, 0);
        }
        return lowerLiteralExpr(ctx, block, loc, expr);
    }

    if (expr->type == AST_IDENTIFIER) {
        return lowerIdentifierExpr(ctx, block, loc, expr, env);
    }

    if (expr->type == AST_MLIR_OP) {
        return lowerMlirOpExpr(ctx, block, loc, expr, env);
    }

    if (expr->type == AST_ASSIGNMENT) {
        return lowerAssignmentExpr(ctx, block, loc, expr, env);
    }

    if (expr->type == AST_PAIR) {
        return lowerPairExpr(ctx, block, loc, expr, env);
    }

    if (expr->type == AST_FIELD_ACCESS) {
        return lowerFieldAccessExpr(ctx, block, loc, expr, env);
    }

    if (expr->type == AST_BLOCK) {
        return lowerBlockExpr(ctx, block, loc, expr, env);
    }

    if (expr->type == AST_BLOCK_DATA) {
        return lowerBlockDataExpr(ctx, block, loc, expr, env);
    }

    if (expr->type == AST_WHILE) {
        return lowerWhileExpr(ctx, block, loc, expr, env, moduleBody);
    }

    if (expr->type == AST_FUNC_DECL) {
        return lowerFuncDeclExpr(ctx, block, loc, expr, env, moduleBody);
    }

    if (expr->type == AST_CALL) {
        return lowerCallExpr(ctx, block, loc, expr, env);
    }

    return createEra(ctx, block, loc);
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
            for (int i = 0; i < lin_arg_count && i < 3; i++) {
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
            for (int i = 0; i < lin_arg_count && i < 3; i++) {
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

            for (int i = lin_arg_count; i < 3; i++) {
                MlirOperationState eraState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                mlirOperationStateAddAttributes(&eraState, 3, eraAttrs);
                MlirType agentTypes[] = {portType, portType, portType};
                mlirOperationStateAddResults(&eraState, 3, agentTypes);
                MlirOperation eraOp = mlirOperationCreate(&eraState);
                mlirBlockAppendOwnedOperation(block, eraOp);
                
                linkValues(block, loc, mlirOperationGetResult(eraOp, 0), mlirBlockGetArgument(block, i));
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

        // Detect a top-level `main` function: its body is the program entry point
        // and must be executed inline. Other func decls (helper functions) are
        // lowered normally so their registrations and closures exist.
        AstNode *mainDecl = NULL;
        for (int i = 0; i < ast->as.block.count; i++) {
            AstNode *stmt = ast->as.block.statements[i];
            if (stmt && stmt->type == AST_FUNC_DECL &&
                stmt->as.func_decl.name_len == 4 &&
                strncmp(stmt->as.func_decl.name, "main", 4) == 0) {
                mainDecl = stmt;
                break;
            }
        }

        MlirValue result = {NULL};
        if (mainDecl) {
            for (int i = 0; i < ast->as.block.count; i++) {
                AstNode *stmt = ast->as.block.statements[i];
                if (!stmt || stmt->type == AST_IMPORT) continue;
                if (stmt == mainDecl) {
                    result = lowerExpression(ctx, block, loc, mainDecl->as.func_decl.body, &env, true);
                } else {
                    result = lowerExpression(ctx, block, loc, stmt, &env, false);
                }
            }
        } else {
            result = lowerExpression(ctx, block, loc, ast, &env, true);
        }
        if (mlirValueIsNull(result)) {
            result = createEra(ctx, block, loc);
        }

        // Erase any remaining live variables before the return
        env_free(&env, ctx, block, loc);

        MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        mlirOperationStateAddOperands(&retState, 1, &result);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&retState));
    } else {
        env_free(&env, ctx, block, loc);
    }

    // Collect type declarations (func decls with 0 args, return type, empty body)
    // and add them as a module attribute for use by dialect passes.
    char typeNamesBuf[1024] = {0};
    int typeNamesLen = 0;
    {
        // Walk AST recursively for type declarations
        AstNode *stack[256];
        int sp = 0;
        stack[sp++] = ast;
        while (sp > 0) {
            AstNode *n = stack[--sp];
            if (!n) continue;
            if (n->type == AST_FUNC_DECL &&
                n->as.func_decl.arg_count == 0 &&
                n->as.func_decl.return_type_len > 0 &&
                n->as.func_decl.name_len > 0 &&
                (!n->as.func_decl.body ||
                 (n->as.func_decl.body->type == AST_BLOCK && n->as.func_decl.body->as.block.count == 0))) {
                if (typeNamesLen > 0) typeNamesBuf[typeNamesLen++] = ',';
                memcpy(typeNamesBuf + typeNamesLen, n->as.func_decl.name, n->as.func_decl.name_len);
                typeNamesLen += n->as.func_decl.name_len;
            } else if (n->type == AST_BLOCK || n->type == AST_BLOCK_DATA) {
                for (int i = 0; i < n->as.block.count && sp < 256; i++)
                    stack[sp++] = n->as.block.statements[i];
            }
        }
    }
    if (typeNamesLen > 0) {
        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(typeNamesBuf, typeNamesLen));
        mlirOperationSetAttributeByName(mlirModuleGetOperation(module), mlirStringRefCreateFromCString("lin.type_names"), typeAttr);
    }

    freeMlirOpNames();
    return module;
}


