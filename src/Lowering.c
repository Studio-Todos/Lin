#include "lin/Lowering.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
}

static void env_free(Environment *env) {
    free(env->vars);
}

static void env_add(Environment *env, const char *name, int name_len, MlirValue value) {
    if (env->count >= env->capacity) {
        env->capacity *= 2;
        env->vars = (EnvVar*)realloc(env->vars, sizeof(EnvVar) * env->capacity);
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

    if (node->type == AST_EITHER) {
        return count_var_usage(node->as.either.condition, name, name_len) +
               count_var_usage(node->as.either.true_branch, name, name_len) +
               count_var_usage(node->as.either.false_branch, name, name_len);
    }

    return 0;
}

static MlirValue lowerExpression(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, Environment *env) {
    if (!expr) {
        MlirValue nullVal = {NULL};
        return nullVal;
    }

    if (expr->type == AST_NUMBER) {
        MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("+"));
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("num"));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

        MlirAttribute valAttr = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 32), expr->as.number.value);
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

    if (expr->type == AST_IDENTIFIER) {
        MlirValue val = env_get(env, expr->as.identifier.name, expr->as.identifier.length);
        if (mlirValueIsNull(val)) {
            fprintf(stderr, "Unbound variable: %.*s\n", expr->as.identifier.length, expr->as.identifier.name);
            // Fallback to Era to prevent crash
        } else {
            // For MVP, if it's used multiple times we need a dup.
            // But if we do static analysis, we should've ALREADY dup'd it when it was bound!
            // Actually, the easiest static analysis is:
            // When we fetch the variable, we check how many times it's used *after* this.
            // If more than 0, we insert a dup right now, return p1, and put p2 back in the env!
            // Wait, we don't need to re-scan. We just check if it's the last use.
            // Better yet, just DUP it on fetch, and the caller handles it.
            // Let's implement dynamic DUP on fetch:

            // To be purely interaction net, a port can only be used once.
            // If we fetch it, we consume it. If the environment still needs it for later, we must DUP it.
            // But how do we know if it's needed later? We can just conservatively DUP it, and if it's not used,
            // we'll leave an unlinked port (which we can clean up with an ERA at the end of the block).
            // Let's use conservative DUP for now:

            MlirOperationState dupState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);
            MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("delta"));
            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
            MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("*"));
            MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("dup"));
            MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

            MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
            mlirOperationStateAddAttributes(&dupState, 3, attrs);

            MlirType portType = getPicPortType(ctx);
            MlirType dupTypes[] = {portType, portType, portType};
            mlirOperationStateAddResults(&dupState, 3, dupTypes);
            MlirOperation dupOp = mlirOperationCreate(&dupState);
            mlirBlockAppendOwnedOperation(block, dupOp);

            MlirValue dupP0 = mlirOperationGetResult(dupOp, 0); // principal
            MlirValue dupP1 = mlirOperationGetResult(dupOp, 1); // aux l
            MlirValue dupP2 = mlirOperationGetResult(dupOp, 2); // aux r

            // Link the old value to dup's principal port (p0)
            MlirOperationState linkState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
            MlirValue linkOps[] = {dupP0, val};
            mlirOperationStateAddOperands(&linkState, 2, linkOps);
            mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkState));

            // Update env to hold p2
            env_set(env, expr->as.identifier.name, expr->as.identifier.length, dupP2);

            // Return p1 for this usage
            return dupP1;
        }
    }

    if (expr->type == AST_ASSIGNMENT) {
        MlirValue rhs = lowerExpression(ctx, block, loc, expr->as.assignment.value, env);
        env_add(env, expr->as.assignment.name, expr->as.assignment.name_len, rhs);
        // An assignment itself doesn't have a value in Rebol/Lin for now, but we return the value to chain if needed
        return rhs;
    }

    if (expr->type == AST_BLOCK) {
        MlirValue lastVal = {NULL};
        for (int i = 0; i < expr->as.block.count; i++) {
            lastVal = lowerExpression(ctx, block, loc, expr->as.block.statements[i], env);
        }
        return lastVal;
    }

    if (expr->type == AST_EITHER) {
        // For MVP we just use a PIC agent named "if" and cheat during LLVM lowering
        MlirValue cond = lowerExpression(ctx, block, loc, expr->as.either.condition, env);
        MlirValue t_branch = lowerExpression(ctx, block, loc, expr->as.either.true_branch, env);
        MlirValue f_branch = lowerExpression(ctx, block, loc, expr->as.either.false_branch, env);

        (void)t_branch;
        (void)f_branch;

        MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

        MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
        MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-")); // operations consume, so -
        MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);
        MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("scf.if"));
        MlirNamedAttribute labelNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("label")), labelAttr);

        MlirNamedAttribute attrs[] = {typeNamedAttr, polNamedAttr, labelNamedAttr};
        mlirOperationStateAddAttributes(&state, 3, attrs);

        MlirType portType = getPicPortType(ctx);
        MlirType types[] = {portType, portType, portType};
        mlirOperationStateAddResults(&state, 3, types);
        MlirOperation op = mlirOperationCreate(&state);
        mlirBlockAppendOwnedOperation(block, op);

        // Return condition for MVP
        return cond;
    }

    if (expr->type == AST_CALL) {
        // For MVP prefix function calls, we map them directly to omega nodes with the callee name
        if (expr->as.call.arg_count == 2) {
            MlirValue left = lowerExpression(ctx, block, loc, expr->as.call.args[0], env);
            MlirValue right = lowerExpression(ctx, block, loc, expr->as.call.args[1], env);

            MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.agent"), loc);

            MlirAttribute typeAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("omega"));
            MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("agentType")), typeAttr);
            MlirAttribute polAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("-")); // operations consume, so -
            MlirNamedAttribute polNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("polarity")), polAttr);

            // Use the callee name directly as the label
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.call.callee, expr->as.call.callee_len));
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
            MlirValue p2 = mlirOperationGetResult(op, 2); // rhs

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

            // Use the callee name directly as the label
            MlirAttribute labelAttr = mlirStringAttrGet(ctx, mlirStringRefCreate(expr->as.call.callee, expr->as.call.callee_len));
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

    if (ast->type == AST_FUNC_DECL) {
        MlirOperationState funcState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);

        // We use the function name from the AST
        MlirAttribute nameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("main_inet"));
        MlirNamedAttribute nameNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")), nameAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &nameNamedAttr);

        MlirType portType = getPicPortType(ctx);
        int arg_count = ast->as.func_decl.arg_count;
        MlirType *types = (MlirType *)malloc(sizeof(MlirType) * arg_count);
        MlirLocation *locs = (MlirLocation *)malloc(sizeof(MlirLocation) * arg_count);
        for (int i = 0; i < arg_count; i++) {
            types[i] = portType;
            locs[i] = loc;
        }
        MlirType retTypes[] = {portType};
        MlirType funcType = mlirFunctionTypeGet(ctx, arg_count, types, 1, retTypes);
        MlirAttribute typeAttr = mlirTypeAttrGet(funcType);
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")), typeAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &typeNamedAttr);

        MlirRegion region = mlirRegionCreate();
        MlirBlock block = mlirBlockCreate(arg_count, types, locs);
        mlirRegionAppendOwnedBlock(region, block);
        mlirOperationStateAddOwnedRegions(&funcState, 1, &region);

        MlirOperation funcOp = mlirOperationCreate(&funcState);
        mlirBlockAppendOwnedOperation(moduleBody, funcOp);

        for (int i = 0; i < arg_count; i++) {
            MlirValue argValue = mlirBlockGetArgument(block, i);
            env_add(&env, ast->as.func_decl.args[i].name, ast->as.func_decl.args[i].name_len, argValue);
        }

        MlirValue result = lowerExpression(ctx, block, loc, ast->as.func_decl.body, &env);

        MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        mlirOperationStateAddOperands(&retState, 1, &result);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&retState));

        free(types);
        free(locs);
    } else if (ast->type == AST_BLOCK) {
        // Just evaluating an anonymous block (like test_parser.lin)
        MlirOperationState funcState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);

        MlirAttribute nameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("main_inet"));
        MlirNamedAttribute nameNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")), nameAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &nameNamedAttr);

        MlirType portType = getPicPortType(ctx);
        MlirType types[] = {portType};
        MlirType retTypes[] = {portType};
        MlirType funcType = mlirFunctionTypeGet(ctx, 1, types, 1, retTypes);
        MlirAttribute typeAttr = mlirTypeAttrGet(funcType);
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")), typeAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &typeNamedAttr);

        MlirRegion region = mlirRegionCreate();
        MlirBlock block = mlirBlockCreate(1, types, &loc);
        mlirRegionAppendOwnedBlock(region, block);
        mlirOperationStateAddOwnedRegions(&funcState, 1, &region);

        MlirOperation funcOp = mlirOperationCreate(&funcState);
        mlirBlockAppendOwnedOperation(moduleBody, funcOp);

        MlirValue result = lowerExpression(ctx, block, loc, ast, &env);

        MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        mlirOperationStateAddOperands(&retState, 1, &result);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&retState));
    }

    env_free(&env);

    return module;
}

void optimizeInteractionNetWithEGraphs(MlirModule module) {
    printf("[E-Graph Optimizer Placeholder]: Skipping for MVP.\n");
}
