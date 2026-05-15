#include "lin/Lowering.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static MlirType getPicPortType(MlirContext ctx) {
    return mlirTypeParseGet(ctx, mlirStringRefCreateFromCString("!pic_graph.port"));
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
            // Built-in operations are not free variables
            bool is_builtin = (node->as.call.callee_len == 3 && strncmp(node->as.call.callee, "add", 3) == 0) ||
                             (node->as.call.callee_len == 6 && strncmp(node->as.call.callee, "either", 6) == 0);
            if (!is_bound && !is_builtin) addFreeVar(fv, node->as.call.callee, node->as.call.callee_len);
            
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

static MlirValue lowerExpression(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {
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
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("num"));
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
                                    if (i + 3 <= len && strncmp(p + i, "f32", 3) == 0) strcat(cleanNames, "_f32");
                                    else if (i + 3 <= len && strncmp(p + i, "f64", 3) == 0) strcat(cleanNames, "_f64");
                                    else if (i + 3 <= len && strncmp(p + i, "i64", 3) == 0) strcat(cleanNames, "_i64");
                                    else if (i + 3 <= len && strncmp(p + i, "i32", 3) == 0) strcat(cleanNames, "_i32");
                                    else if (i + 2 <= len && strncmp(p + i, "i1", 2) == 0) strcat(cleanNames, "_i1");
                                    
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

        MlirNamedAttribute attrs[] = {nameNamedAttr, payloadNamedAttr, namesNamedAttr};
        mlirOperationStateAddAttributes(&regState, 3, attrs);

        MlirOperation regOp = mlirOperationCreate(&regState);
        mlirBlockAppendOwnedOperation(block, regOp);

        MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.mlir_op.name, expr->as.mlir_op.name_len));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

        MlirNamedAttribute agentAttrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
        mlirOperationStateAddAttributes(&state, 3, agentAttrs);

        MlirType portType = getPicPortType(ctx);
        MlirType types[] = {portType, portType, portType};
        mlirOperationStateAddResults(&state, 3, types);

        MlirOperation op = mlirOperationCreate(&state);
        mlirBlockAppendOwnedOperation(block, op);
        
        MlirValue result = mlirOperationGetResult(op, 0);
        if (expr->as.mlir_op.name_len > 0) {
            env_add(env, expr->as.mlir_op.name, expr->as.mlir_op.name_len, result);
        }
        return result;
    }

    if (expr->type == AST_ASSIGNMENT) {
        MlirValue rhs = lowerExpression(ctx, block, loc, expr->as.assignment.value, env);
        env_add(env, expr->as.assignment.name, expr->as.assignment.name_len, rhs);
        return rhs;
    }

    if (expr->type == AST_PAIR) {
        MlirValue left = lowerExpression(ctx, block, loc, expr->as.pair.left, env);
        MlirValue right = lowerExpression(ctx, block, loc, expr->as.pair.right, env);

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
        MlirValue base_val = lowerExpression(ctx, block, loc, expr->as.field_access.base, env);
        
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
        
        return p0;
    }

    if (expr->type == AST_BLOCK) {
        MlirValue lastVal = {NULL};
        for (int i = 0; i < expr->as.block.count; i++) {
            AstNode *stmt = expr->as.block.statements[i];
            // Skip import and mlir_op since they have no runtime code
            if (!stmt) continue;
            if (stmt->type == AST_IMPORT) continue;
            lastVal = lowerExpression(ctx, block, loc, stmt, env);
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
        // Lower to recursion.
        // We dynamically generate an MLIR function that represents the loop.
        static int loop_counter = 0;
        char loop_func_name[64];
        snprintf(loop_func_name, sizeof(loop_func_name), "_loop_macro_%d", loop_counter++);

        MlirOperationState funcState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);
        MlirAttribute nameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(loop_func_name));
        MlirNamedAttribute nameNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")), nameAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &nameNamedAttr);

        // Function arguments are the current environment variables.
        int arg_count = env->count;
        MlirType portType = getPicPortType(ctx);
        MlirType *types = NULL;
        MlirLocation *locs = NULL;
        if (arg_count > 0) {
            types = (MlirType *)malloc(sizeof(MlirType) * arg_count);
            locs = (MlirLocation *)malloc(sizeof(MlirLocation) * arg_count);
            if (!types || !locs) {
                fprintf(stderr, "Out of memory\n");
                exit(1);
            }
            for (int i = 0; i < arg_count; i++) {
                types[i] = portType;
                locs[i] = loc;
            }
        }
        MlirType retTypes[] = {portType};
        MlirType funcType = mlirFunctionTypeGet(ctx, arg_count, types, 1, retTypes);
        MlirAttribute typeAttr = mlirTypeAttrGet(funcType);
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")), typeAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &typeNamedAttr);

        MlirRegion region = mlirRegionCreate();
        MlirBlock loopBlock = mlirBlockCreate(arg_count, types, locs);
        mlirRegionAppendOwnedBlock(region, loopBlock);
        mlirOperationStateAddOwnedRegions(&funcState, 1, &region);

        MlirOperation funcOp = mlirOperationCreate(&funcState);
        if (!mlirBlockIsNull(moduleBody)) {
            mlirBlockAppendOwnedOperation(moduleBody, funcOp);
        }

        // Inside the loop function
        Environment loopEnv;
        env_init(&loopEnv);
        for (int i = 0; i < arg_count; i++) {
            MlirValue argValue = mlirBlockGetArgument(loopBlock, i);
            env_add(&loopEnv, env->vars[i].name, env->vars[i].name_len, argValue);
        }

        // Evaluate condition
        MlirValue cond = lowerExpression(ctx, loopBlock, loc, expr->as.while_loop.condition, &loopEnv);

        // We use an either branch to route based on condition.
        // But instead of generating an AST_EITHER, we'll construct the true branch manually,
        // which evaluates the body and then recursively calls the loop function.
        MlirValue body_res = lowerExpression(ctx, loopBlock, loc, expr->as.while_loop.body, &loopEnv);

        // True branch: evaluate body, then recursively call the loop function.
        MlirOperationState recState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirAttribute recTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute recTypeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), recTypeAttr);
        MlirAttribute recPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
        MlirNamedAttribute recPolNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), recPolAttr);
        MlirAttribute recLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(loop_func_name));
        MlirNamedAttribute recLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), recLabelAttr);
        MlirNamedAttribute recAttrs[] = {recTypeNamedAttr, recPolNamedAttr, recLabelNamedAttr};
        mlirOperationStateAddAttributes(&recState, 3, recAttrs);
        MlirType recTypes[3] = {portType, portType, portType};
        mlirOperationStateAddResults(&recState, 3, recTypes);
        MlirOperation recOp = mlirOperationCreate(&recState);
        mlirBlockAppendOwnedOperation(loopBlock, recOp);

        MlirValue recP0 = mlirOperationGetResult(recOp, 0); // principal result
        MlirValue recP1 = mlirOperationGetResult(recOp, 1); // arg 1

        // Link Dummy Arg for MVP
        MlirOperationState dummyLink = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue dummyLinkOps[] = {recP1, body_res};
        mlirOperationStateAddOperands(&dummyLink, 2, dummyLinkOps);
        mlirBlockAppendOwnedOperation(loopBlock, mlirOperationCreate(&dummyLink));

        // Loop branch node
        MlirOperationState eitherState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirAttribute eitherTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute eitherTypeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), eitherTypeAttr);
        MlirAttribute eitherPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
        MlirNamedAttribute eitherPolNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), eitherPolAttr);
        MlirAttribute eitherLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("branch"));
        MlirNamedAttribute eitherLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), eitherLabelAttr);
        MlirNamedAttribute eitherAttrs[] = {eitherTypeNamedAttr, eitherPolNamedAttr, eitherLabelNamedAttr};
        mlirOperationStateAddAttributes(&eitherState, 3, eitherAttrs);
        MlirType eitherTypes[] = {portType, portType, portType};
        mlirOperationStateAddResults(&eitherState, 3, eitherTypes);
        MlirOperation eitherOp = mlirOperationCreate(&eitherState);
        mlirBlockAppendOwnedOperation(loopBlock, eitherOp);

        MlirValue eitherP0 = mlirOperationGetResult(eitherOp, 0); // cond
        MlirValue eitherP1 = mlirOperationGetResult(eitherOp, 1); // true
        MlirValue eitherP2 = mlirOperationGetResult(eitherOp, 2); // false

        // Link Condition
        MlirOperationState linkCond = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue linkCondOps[] = {eitherP0, cond};
        mlirOperationStateAddOperands(&linkCond, 2, linkCondOps);
        mlirBlockAppendOwnedOperation(loopBlock, mlirOperationCreate(&linkCond));

        // Link True Branch to recursion
        MlirOperationState linkTrue = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue linkTrueOps[] = {eitherP1, recP0};
        mlirOperationStateAddOperands(&linkTrue, 2, linkTrueOps);
        mlirBlockAppendOwnedOperation(loopBlock, mlirOperationCreate(&linkTrue));

        // Link False Branch to an epsilon/null state
        MlirOperationState eraState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirAttribute eraTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("epsilon"));
        MlirNamedAttribute eraTypeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), eraTypeAttr);
        MlirAttribute eraPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
        MlirNamedAttribute eraPolNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), eraPolAttr);
        MlirAttribute eraLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("era"));
        MlirNamedAttribute eraLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), eraLabelAttr);
        MlirNamedAttribute eraAttrs[] = {eraTypeNamedAttr, eraPolNamedAttr, eraLabelNamedAttr};
        mlirOperationStateAddAttributes(&eraState, 3, eraAttrs);
        MlirType eraTypes[] = {portType, portType, portType};
        mlirOperationStateAddResults(&eraState, 3, eraTypes);
        MlirOperation eraOp = mlirOperationCreate(&eraState);
        mlirBlockAppendOwnedOperation(loopBlock, eraOp);

        MlirValue eraP0 = mlirOperationGetResult(eraOp, 0);

        MlirOperationState linkFalse = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirValue linkFalseOps[] = {eitherP2, eraP0};
        mlirOperationStateAddOperands(&linkFalse, 2, linkFalseOps);
        mlirBlockAppendOwnedOperation(loopBlock, mlirOperationCreate(&linkFalse));

        // env_free MUST precede func.return
        env_free(&loopEnv, ctx, loopBlock, loc);

        // Return the condition from the loop function
        MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        mlirOperationStateAddOperands(&retState, 1, &cond);
        mlirBlockAppendOwnedOperation(loopBlock, mlirOperationCreate(&retState));

        if (types) free(types);
        if (locs) free(locs);

        // Now, emit the AST_CALL to the loop function in the current block
        MlirOperationState callState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirAttribute callTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute callTypeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), callTypeAttr);
        MlirAttribute callPolAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
        MlirNamedAttribute callPolNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), callPolAttr);
        MlirAttribute callLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(loop_func_name));
        MlirNamedAttribute callLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), callLabelAttr);
        MlirNamedAttribute callAttrs[] = {callTypeNamedAttr, callPolNamedAttr, callLabelNamedAttr};
        mlirOperationStateAddAttributes(&callState, 3, callAttrs);
        MlirType callTypes[] = {portType, portType, portType};
        mlirOperationStateAddResults(&callState, 3, callTypes);
        MlirOperation callOp = mlirOperationCreate(&callState);
        mlirBlockAppendOwnedOperation(block, callOp);

        MlirValue callResult = mlirOperationGetResult(callOp, 0);
        MlirValue callP1 = mlirOperationGetResult(callOp, 1);

        // Link initial state (just a dummy variable since MVP omegas take 2 auxiliary max)
        MlirOperationState callLink = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
        MlirOperationState callEraState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        mlirOperationStateAddAttributes(&callEraState, 3, eraAttrs);
        mlirOperationStateAddResults(&callEraState, 3, eraTypes);
        MlirOperation callEraOp = mlirOperationCreate(&callEraState);
        mlirBlockAppendOwnedOperation(block, callEraOp);

        MlirValue callEraP0 = mlirOperationGetResult(callEraOp, 0);

        MlirValue callLinkOps[] = {callP1, callEraP0};
        mlirOperationStateAddOperands(&callLink, 2, callLinkOps);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&callLink));

        return callResult;
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

        // Skip type declaration functions to avoid duplicate symbol errors
        if (expr->as.func_decl.name_len > 0 && (
            strcmp(funcNameStr, "i1") == 0 || strcmp(funcNameStr, "i8") == 0 ||
            strcmp(funcNameStr, "i16") == 0 || strcmp(funcNameStr, "i32") == 0 ||
            strcmp(funcNameStr, "i64") == 0 || strcmp(funcNameStr, "f32") == 0 ||
            strcmp(funcNameStr, "f64") == 0 || strcmp(funcNameStr, "bool") == 0 ||
            strcmp(funcNameStr, "str") == 0)) {
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
            env_add(&innerEnv, expr->as.func_decl.args[0].name, expr->as.func_decl.args[0].name_len, mainArg);
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

        MlirValue bodyResult = lowerExpression(ctx, innerBlock, loc, expr->as.func_decl.body, &innerEnv);
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
        MlirNamedAttribute regAttrs[] = {opNameNamed, payloadNamed, argNamesNamed};
        mlirOperationStateAddAttributes(&regState, 3, regAttrs);
        if (!mlirBlockIsNull(moduleBody)) {
            mlirBlockAppendOwnedOperation(moduleBody, mlirOperationCreate(&regState));
        }

        // Phase 3: Create closure pair (omega+, bundle)
        MlirOperationState baseState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
        MlirAttribute agTypeAttr  = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute agTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), agTypeAttr);
        MlirAttribute fnLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(funcNameStr));
        MlirNamedAttribute fnLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), fnLabelAttr);
        MlirNamedAttribute agentAttrs[] = {agTypeNamed, plusPolAttr, fnLabelNamedAttr};
        mlirOperationStateAddAttributes(&baseState, 3, agentAttrs);
        mlirOperationStateAddResults(&baseState, 3, agentTypes);
        MlirOperation baseOp = mlirOperationCreate(&baseState);
        mlirBlockAppendOwnedOperation(block, baseOp);
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

        // Check if callee is a user-defined function in the environment
        MlirValue currentVal = env_fetch(ctx, block, loc, env, effectiveCallee, effectiveCalleeLen);
        // If not found with effective callee, also try original callee (for closures stored by original name)
        if (mlirValueIsNull(currentVal) && expr->as.call.resolved_callee)
            currentVal = env_fetch(ctx, block, loc, env, expr->as.call.callee, expr->as.call.callee_len);

        if (!mlirValueIsNull(currentVal)) {
            MlirType portType = getPicPortType(ctx);
            MlirAttribute agTypeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
            MlirNamedAttribute agTypeNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), agTypeAttr);
            MlirAttribute minusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
            MlirNamedAttribute minusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), minusPol);
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.call.callee, expr->as.call.callee_len));
            MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);
            MlirType agentTypes[] = {portType, portType, portType};

            // Apply each argument one by one (currying)
            for (int i = 0; i < expr->as.call.arg_count; i++) {
                MlirValue argVal = lowerExpression(ctx, block, loc, expr->as.call.args[i], env);

                // For closure calling:
                // Unpack closure currentVal into (f, env_bundle)
                MlirOperationState unpackState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirAttribute pairType = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("gamma"));
                MlirNamedAttribute pairTypeAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), pairType);
                MlirAttribute minusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
                MlirNamedAttribute minusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), minusPol);
                MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("pair"));
                MlirNamedAttribute labelAttrAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);
                MlirNamedAttribute unpackAttrs[] = {pairTypeAttr, minusPolAttr, labelAttrAttr};
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

                // Now bundle (env_bundle, argVal) into a new pair for the next level
                // Wait! No, the function takes exactly (env, arg).
                // So we just call f(uP2, argVal).
                
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
                MlirValue appP1 = mlirOperationGetResult(appOp, 1); // pair(env, arg)
                MlirValue appP2 = mlirOperationGetResult(appOp, 2); // result

                // Link appP0 ↔ uP1 (the function pointer)
                MlirOperationState linkFState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue fLinkOps[] = {appP0, uP1};
                mlirOperationStateAddOperands(&linkFState, 2, fLinkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkFState));

                // Create the pair(env, argVal)
                MlirOperationState packState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                MlirAttribute plusPol = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
                MlirNamedAttribute plusPolAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), plusPol);
                MlirNamedAttribute packAttrs[] = {pairTypeAttr, plusPolAttr, labelAttrAttr};
                mlirOperationStateAddAttributes(&packState, 3, packAttrs);
                mlirOperationStateAddResults(&packState, 3, agentTypes);
                MlirOperation packOp = mlirOperationCreate(&packState);
                mlirBlockAppendOwnedOperation(block, packOp);

                MlirValue pP0 = mlirOperationGetResult(packOp, 0); // the pair
                MlirValue pP1 = mlirOperationGetResult(packOp, 1); // env
                MlirValue pP2 = mlirOperationGetResult(packOp, 2); // arg

                // Link pP1 ↔ uP2
                MlirOperationState linkEnvState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue envLinkOps[] = {pP1, uP2};
                mlirOperationStateAddOperands(&linkEnvState, 2, envLinkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkEnvState));

                // Link pP2 ↔ argVal
                MlirOperationState linkArgState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue argLinkOps[] = {pP2, argVal};
                mlirOperationStateAddOperands(&linkArgState, 2, argLinkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkArgState));

                // Link appP1 ↔ pP0
                MlirOperationState linkCallState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
                MlirValue callLinkOps[] = {appP1, pP0};
                mlirOperationStateAddOperands(&linkCallState, 2, callLinkOps);
                mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkCallState));

                currentVal = appP2;
            }

            return currentVal;
        }

        // Built-in / unknown callee: lower as omega agent (existing behaviour)
        if (expr->as.call.arg_count == 2) {
            MlirValue left = lowerExpression(ctx, block, loc, expr->as.call.args[0], env);
            MlirValue right = lowerExpression(ctx, block, loc, expr->as.call.args[1], env);

            MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

            MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
            MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-"));
            MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);

            // Use resolved_callee (type-directed) as the omega label when set.
            // This is the key mechanism: omega(-, label) annihilates with omega(+, label)
            // at runtime, so the label must match the registered mlir-op payload name.
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(effectiveCallee, effectiveCalleeLen));
            MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

            MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
            mlirOperationStateAddAttributes(&state, 3, attrs);

            MlirType portType = getPicPortType(ctx);
            MlirType types[] = {portType, portType, portType};
            mlirOperationStateAddResults(&state, 3, types);

            MlirOperation op = mlirOperationCreate(&state);
            mlirBlockAppendOwnedOperation(block, op);

            MlirValue result = mlirOperationGetResult(op, 0);
            MlirValue p1 = mlirOperationGetResult(op, 1);
            MlirValue p2 = mlirOperationGetResult(op, 2);

            MlirOperationState linkLeftState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue leftOps[] = {p1, left};
            mlirOperationStateAddOperands(&linkLeftState, 2, leftOps);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkLeftState));

            MlirOperationState linkRightState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue rightOps[] = {p2, right};
            mlirOperationStateAddOperands(&linkRightState, 2, rightOps);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkRightState));

            return result;
        } else if (expr->as.call.arg_count == 1) {
            MlirValue arg = lowerExpression(ctx, block, loc, expr->as.call.args[0], env);

            MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

            MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
            MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-")); // operations consume, so -
            MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);

            // Use the effective callee (type-directed) as the omega label
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(effectiveCallee, effectiveCalleeLen));
            MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

            MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
            mlirOperationStateAddAttributes(&state, 3, attrs);

            MlirType portType = getPicPortType(ctx);
            MlirType types[] = {portType, portType, portType};
            mlirOperationStateAddResults(&state, 3, types);

            MlirOperation op = mlirOperationCreate(&state);
            mlirBlockAppendOwnedOperation(block, op);

            MlirValue result = mlirOperationGetResult(op, 0); // result is principle
            MlirValue p1 = mlirOperationGetResult(op, 1); // lhs
            MlirValue p2 = mlirOperationGetResult(op, 2); // Unused for unary ops, handled by ERA later in reduce pass if needed

            (void)p2;

            MlirOperationState linkLeftState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue leftOps[] = {p1, arg};
            mlirOperationStateAddOperands(&linkLeftState, 2, leftOps);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkLeftState));

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

    MlirBlock block = {NULL};
    if (ast->type == AST_FUNC_DECL) {
        MlirOperationState funcState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);

        // We use the function name from the AST
        char funcNameStr[256];
        snprintf(funcNameStr, sizeof(funcNameStr), "%.*s", ast->as.func_decl.name_len, ast->as.func_decl.name);
        if (strcmp(funcNameStr, "main") == 0) {
            snprintf(funcNameStr, sizeof(funcNameStr), "main_inet_entry");
        }
        MlirAttribute nameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(funcNameStr));
        MlirNamedAttribute nameNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")), nameAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &nameNamedAttr);

        MlirType portType = getPicPortType(ctx);
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
        fprintf(stderr, "Lowering Func: %s with 1 result\n", funcNameStr);
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

        MlirValue result = lowerExpression(ctx, block, loc, ast->as.func_decl.body, &env);
        env_free(&env, ctx, block, loc);

        MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        mlirOperationStateAddOperands(&retState, 1, &result);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&retState));

        free(types);
        free(locs);
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
        MlirValue result = lowerExpression(ctx, block, loc, ast, &env);

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

void optimizeInteractionNetWithEGraphs(MlirModule module) {
    // E-Graph optimizer for PIC graph (Native C Implementation)
    // Rule 4 - Congruence: Inside boundaries, identical subgraphs collapse as they normalize.

    MlirBlock body = mlirModuleGetBody(module);
    if (mlirBlockIsNull(body)) return;

    MlirOperation funcOp = mlirBlockGetFirstOperation(body);
    if (mlirOperationIsNull(funcOp)) return;

    // We expect funcOp to be `func.func` with a region and block
    MlirStringRef funcName = mlirIdentifierStr(mlirOperationGetName(funcOp));
    if (strncmp(funcName.data, "func.func", funcName.length) != 0) return;

    MlirRegion funcRegion = mlirOperationGetRegion(funcOp, 0);
    if (mlirRegionIsNull(funcRegion)) return;

    MlirBlock funcBlock = mlirRegionGetFirstBlock(funcRegion);
    if (mlirBlockIsNull(funcBlock)) return;

    MlirOperation op = mlirBlockGetFirstOperation(funcBlock);

    struct ConstHash {
        int64_t value;
        MlirOperation op;
    };
    struct ConstHash hashes[1024];
    int hashCount = 0;

    // Allocation-safe approach: First iterate and collect ops to rewrite/delete
    struct RewriteAction {
        MlirOperation dupOp;
        MlirOperation currOp;
        MlirOperation existingOp;
        MlirValue dupP0, dupP1, dupP2;
        MlirValue existingP0, currP0;
        MlirLocation loc;
    };

    struct RewriteAction rewrites[1024];
    int rewriteCount = 0;

    while (!mlirOperationIsNull(op)) {
        MlirOperation currOp = op;
        op = mlirOperationGetNextInBlock(op); // Capture next safely for reading

        MlirIdentifier nameId = mlirOperationGetName(currOp);
        MlirStringRef nameRef = mlirIdentifierStr(nameId);

        if (nameRef.length == 15 && strncmp(nameRef.data, "pic_graph.agent", 15) == 0) {
            MlirAttribute labelAttr = mlirOperationGetAttributeByName(currOp, mlirStringRefCreateFromCString("label"));

            if (!mlirAttributeIsNull(labelAttr) && mlirAttributeIsAString(labelAttr)) {
                MlirStringRef labelStr = mlirStringAttrGetValue(labelAttr);

                if (labelStr.length == 3 && strncmp(labelStr.data, "num", 3) == 0) {
                    MlirAttribute valAttr = mlirOperationGetAttributeByName(currOp, mlirStringRefCreateFromCString("value"));
                    if (!mlirAttributeIsNull(valAttr) && mlirAttributeIsAInteger(valAttr)) {
                        int64_t val = mlirIntegerAttrGetValueInt(valAttr);

                        int found = 0;
                        MlirOperation existingOp = {NULL};
                        for (int i = 0; i < hashCount; i++) {
                            if (hashes[i].value == val) {
                                found = 1;
                                existingOp = hashes[i].op;
                                break;
                            }
                        }

                        if (!found && hashCount < 1024) {
                            hashes[hashCount].value = val;
                            hashes[hashCount].op = currOp;
                            hashCount++;
                        } else if (found && rewriteCount < 1024) {
                            // Stage the rewrite
                            MlirLocation loc = mlirOperationGetLocation(currOp);
                            MlirContext ctx = mlirModuleGetContext(module);

                            MlirOperationState dupState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
                            MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("delta"));
                            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
                            MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
                            MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
                            MlirAttribute dupLabelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("dup"));
                            MlirNamedAttribute dupLabelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), dupLabelAttr);

                            MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, dupLabelNamedAttr};
                            mlirOperationStateAddAttributes(&dupState, 3, attrs);

                            MlirType portType = getPicPortType(ctx);
                            MlirType dupTypes[] = {portType, portType, portType};
                            mlirOperationStateAddResults(&dupState, 3, dupTypes);

                            MlirOperation dupOp = mlirOperationCreate(&dupState);

                            rewrites[rewriteCount].dupOp = dupOp;
                            rewrites[rewriteCount].currOp = currOp;
                            rewrites[rewriteCount].existingOp = existingOp;
                            rewrites[rewriteCount].dupP0 = mlirOperationGetResult(dupOp, 0);
                            rewrites[rewriteCount].dupP1 = mlirOperationGetResult(dupOp, 1);
                            rewrites[rewriteCount].dupP2 = mlirOperationGetResult(dupOp, 2);
                            rewrites[rewriteCount].existingP0 = mlirOperationGetResult(existingOp, 0);
                            rewrites[rewriteCount].currP0 = mlirOperationGetResult(currOp, 0);
                            rewrites[rewriteCount].loc = loc;

                            rewriteCount++;
                        }
                    }
                }
            }
        }
    }

    // Now perform the rewrites safely
    for (int i = 0; i < rewriteCount; i++) {
        struct RewriteAction action = rewrites[i];

        mlirBlockInsertOwnedOperationAfter(funcBlock, action.existingOp, action.dupOp);

        MlirOperation linkOp1 = {NULL};
        MlirOperation linkOp2 = {NULL};

        MlirOperation searchOp = mlirBlockGetFirstOperation(funcBlock);
        while (!mlirOperationIsNull(searchOp)) {
            MlirIdentifier lnkId = mlirOperationGetName(searchOp);
            MlirStringRef lnkRef = mlirIdentifierStr(lnkId);
            if (lnkRef.length == 14 && strncmp(lnkRef.data, "pic_graph.link", 14) == 0) {
                if (mlirValueEqual(mlirOperationGetOperand(searchOp, 0), action.existingP0) ||
                    mlirValueEqual(mlirOperationGetOperand(searchOp, 1), action.existingP0)) {
                    linkOp1 = searchOp;
                }
                if (mlirValueEqual(mlirOperationGetOperand(searchOp, 0), action.currP0) ||
                    mlirValueEqual(mlirOperationGetOperand(searchOp, 1), action.currP0)) {
                    linkOp2 = searchOp;
                }
            }
            searchOp = mlirOperationGetNextInBlock(searchOp);
        }

        if (!mlirOperationIsNull(linkOp1) && !mlirOperationIsNull(linkOp2)) {
            MlirOperation opsToDestroy[4];
            int destroyCount = 0;
            if (mlirValueEqual(mlirOperationGetOperand(linkOp1, 0), action.existingP0)) {
                MlirValue otherPort = mlirOperationGetOperand(linkOp1, 1);
                MlirOperationState newLink1 = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), action.loc);
                MlirValue ops1[] = {action.dupP1, otherPort};
                mlirOperationStateAddOperands(&newLink1, 2, ops1);
                mlirBlockInsertOwnedOperationAfter(funcBlock, linkOp1, mlirOperationCreate(&newLink1));
            } else {
                MlirValue otherPort = mlirOperationGetOperand(linkOp1, 0);
                MlirOperationState newLink1 = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), action.loc);
                MlirValue ops1[] = {otherPort, action.dupP1};
                mlirOperationStateAddOperands(&newLink1, 2, ops1);
                mlirBlockInsertOwnedOperationAfter(funcBlock, linkOp1, mlirOperationCreate(&newLink1));
            }
            opsToDestroy[destroyCount++] = linkOp1;

            if (mlirValueEqual(mlirOperationGetOperand(linkOp2, 0), action.currP0)) {
                MlirValue otherPort = mlirOperationGetOperand(linkOp2, 1);
                MlirOperationState newLink2 = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), action.loc);
                MlirValue ops2[] = {action.dupP2, otherPort};
                mlirOperationStateAddOperands(&newLink2, 2, ops2);
                mlirBlockInsertOwnedOperationAfter(funcBlock, linkOp2, mlirOperationCreate(&newLink2));
            } else {
                MlirValue otherPort = mlirOperationGetOperand(linkOp2, 0);
                MlirOperationState newLink2 = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), action.loc);
                MlirValue ops2[] = {otherPort, action.dupP2};
                mlirOperationStateAddOperands(&newLink2, 2, ops2);
                mlirBlockInsertOwnedOperationAfter(funcBlock, linkOp2, mlirOperationCreate(&newLink2));
            }
            opsToDestroy[destroyCount++] = linkOp2;

            MlirOperationState newLink3 = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), action.loc);
            MlirValue ops3[] = {action.existingP0, action.dupP0};
            mlirOperationStateAddOperands(&newLink3, 2, ops3);
            mlirBlockInsertOwnedOperationAfter(funcBlock, action.dupOp, mlirOperationCreate(&newLink3));

            opsToDestroy[destroyCount++] = action.currOp;

            // Destroy out-of-band to prevent invalidating our `searchOp` traversal
            // Call mlirOperationRemoveFromParent to fix MLIR doubly-linked list corruption
            for (int j = 0; j < destroyCount; j++) {
                mlirOperationRemoveFromParent(opsToDestroy[j]);
                mlirOperationDestroy(opsToDestroy[j]);
            }
        }
    }
}
