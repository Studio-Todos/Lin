#include "Internal.h"

void env_init(Environment *env) {
    env->capacity = 16;
    env->count = 0;
    env->vars = (EnvVar*)malloc(sizeof(EnvVar) * env->capacity);
    if (!env->vars) {
        fprintf(stderr, "Out of memory\n");
        exit(1);
    }
}

void env_free(Environment *env, MlirContext ctx, MlirBlock block, MlirLocation loc) {
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
void env_add(Environment *env, const char *name, int name_len, MlirValue value) {
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
MlirValue env_get(Environment *env, const char *name, int name_len) {
    // Go backwards to get the most recent binding
    for (int i = env->count - 1; i >= 0; i--) {
        if (env->vars[i].name_len == name_len && strncmp(env->vars[i].name, name, name_len) == 0) {
            return env->vars[i].value;
        }
    }
    MlirValue nullVal = {NULL};
    return nullVal;
}
void env_set(Environment *env, const char *name, int name_len, MlirValue value) {
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

MlirValue env_fetch(MlirContext ctx, MlirBlock block, MlirLocation loc, Environment *env, const char *name, int name_len) {
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
