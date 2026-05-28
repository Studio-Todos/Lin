#include "Internal.h"

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
