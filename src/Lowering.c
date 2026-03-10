#include "lin/Lowering.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/IR.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static MlirType getInetPortType(MlirContext ctx) {
    return mlirTypeParseGet(ctx, mlirStringRefCreateFromCString("!inet.port"));
}

static MlirValue lowerExpression(MlirContext ctx, MlirBlock block, MlirLocation loc, AstNode *expr, MlirValue *varMap, int numVars) {
    if (expr->type == AST_NUMBER) {
        MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("inet.num"), loc);
        MlirAttribute attr = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 32), expr->as.number.value);
        MlirNamedAttribute namedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("value")), attr);
        mlirOperationStateAddAttributes(&state, 1, &namedAttr);

        MlirType portType = getInetPortType(ctx);
        mlirOperationStateAddResults(&state, 1, &portType);

        MlirOperation op = mlirOperationCreate(&state);
        mlirBlockAppendOwnedOperation(block, op);
        return mlirOperationGetResult(op, 0);
    }

    if (expr->type == AST_IDENTIFIER) {
        if (numVars > 0) return varMap[0];
    }

    if (expr->type == AST_BINARY) {
        MlirValue left = lowerExpression(ctx, block, loc, expr->as.binary.left, varMap, numVars);
        MlirValue right = lowerExpression(ctx, block, loc, expr->as.binary.right, varMap, numVars);

        MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("inet.op"), loc);

        const char *opName = "unknown";
        if (expr->as.binary.op == TOKEN_PLUS) opName = "add";
        else if (expr->as.binary.op == TOKEN_MINUS) opName = "sub";
        else if (expr->as.binary.op == TOKEN_LESS) opName = "lt";

        MlirAttribute attr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(opName));
        MlirNamedAttribute namedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("opName")), attr);
        mlirOperationStateAddAttributes(&state, 1, &namedAttr);

        MlirType portType = getInetPortType(ctx);
        MlirType types[] = {portType, portType, portType};
        mlirOperationStateAddResults(&state, 3, types);

        MlirOperation op = mlirOperationCreate(&state);
        mlirBlockAppendOwnedOperation(block, op);

        MlirValue result = mlirOperationGetResult(op, 0);
        MlirValue p1 = mlirOperationGetResult(op, 1);
        MlirValue p2 = mlirOperationGetResult(op, 2);

        MlirOperationState linkLeftState = mlirOperationStateGet(mlirStringRefCreateFromCString("inet.link"), loc);
        MlirValue leftOps[] = {p1, left};
        mlirOperationStateAddOperands(&linkLeftState, 2, leftOps);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkLeftState));

        MlirOperationState linkRightState = mlirOperationStateGet(mlirStringRefCreateFromCString("inet.link"), loc);
        MlirValue rightOps[] = {p2, right};
        mlirOperationStateAddOperands(&linkRightState, 2, rightOps);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkRightState));

        return result;
    }

    MlirOperationState fallbackState = mlirOperationStateGet(mlirStringRefCreateFromCString("inet.era"), loc);
    MlirType fbPortType = getInetPortType(ctx);
    mlirOperationStateAddResults(&fallbackState, 1, &fbPortType);
    MlirOperation fbOp = mlirOperationCreate(&fallbackState);
    mlirBlockAppendOwnedOperation(block, fbOp);
    return mlirOperationGetResult(fbOp, 0);
}

MlirModule lowerAstToMlir(MlirContext ctx, AstNode *ast) {
    MlirLocation loc = mlirLocationUnknownGet(ctx);
    MlirModule module = mlirModuleCreateEmpty(loc);
    MlirBlock moduleBody = mlirModuleGetBody(module);

    if (ast->type == AST_FUNC_DECL) {
        MlirOperationState funcState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.func"), loc);

        MlirAttribute nameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("fib_inet"));
        MlirNamedAttribute nameNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("sym_name")), nameAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &nameNamedAttr);

        MlirType portType = getInetPortType(ctx);
        MlirType types[] = {portType};
        MlirType funcType = mlirFunctionTypeGet(ctx, 1, types, 1, types);
        MlirAttribute typeAttr = mlirTypeAttrGet(funcType);
        MlirNamedAttribute typeNamedAttr = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("function_type")), typeAttr);
        mlirOperationStateAddAttributes(&funcState, 1, &typeNamedAttr);

        MlirRegion region = mlirRegionCreate();
        MlirBlock block = mlirBlockCreate(1, types, &loc);
        mlirRegionAppendOwnedBlock(region, block);
        mlirOperationStateAddOwnedRegions(&funcState, 1, &region);

        MlirOperation funcOp = mlirOperationCreate(&funcState);
        mlirBlockAppendOwnedOperation(moduleBody, funcOp);

        MlirValue argValue = mlirBlockGetArgument(block, 0);

        MlirValue result = lowerExpression(ctx, block, loc, ast->as.func_decl.body->as.block.statements[0]->as.either.condition, &argValue, 1);

        MlirOperationState retState = mlirOperationStateGet(mlirStringRefCreateFromCString("func.return"), loc);
        mlirOperationStateAddOperands(&retState, 1, &result);
        mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&retState));
    }

    return module;
}

void optimizeInteractionNetWithEGraphs(MlirModule module) {
    printf("[E-Graph Optimizer Placeholder]: Skipping for MVP.\n");
}
