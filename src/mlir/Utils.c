#include "Internal.h"

MlirType getPicPortType(MlirContext ctx) {
    return mlirTypeParseGet(ctx, mlirStringRefCreateFromCString("!pic_graph.port"));
}

void linkValues(MlirBlock block, MlirLocation loc, MlirValue v1, MlirValue v2) {
    MlirOperationState linkState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.link"), loc);
    MlirValue linkOps[] = {v1, v2};
    mlirOperationStateAddOperands(&linkState, 2, linkOps);
    mlirBlockAppendOwnedOperation(block, mlirOperationCreate(&linkState));
}

MlirValue createOmegaP0(MlirContext ctx, MlirBlock block, MlirLocation loc, const char *labelName, const char *polarity) {
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
    return mlirOperationGetResult(baseOp, 0);
}

void createOmega(MlirContext ctx, MlirBlock block, MlirLocation loc, const char *labelName, const char *polarity, MlirValue *p0, MlirValue *p1, MlirValue *p2) {
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


MlirValue bundleClosure(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue omegaP0, MlirValue capBundle) {
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

MlirValue createPair(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue left, MlirValue right) {
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


MlirBlock createFunctionBlock(MlirContext ctx, MlirLocation loc, MlirBlock moduleBody, const char *prefixedName) {
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

void registerFunction(MlirContext ctx, MlirLocation loc, MlirBlock moduleBody, const char *funcName, const char *prefixedName) {
    MlirOperationState regState = mlirOperationStateGet(mlirStringRefCreateFromCString("pic_graph.registry"), loc);
    MlirAttribute opNameAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(funcName));
    MlirNamedAttribute opNameNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("op_name")), opNameAttr);
    MlirAttribute payloadAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString(prefixedName));
    MlirNamedAttribute payloadNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("payload")), payloadAttr);
    MlirAttribute argNamesAttr = mlirStringAttrGet(ctx, mlirStringRefCreateFromCString("[env][arg]"));
    MlirNamedAttribute argNamesNamed = mlirNamedAttributeGet(mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("arg_names")), argNamesAttr);
    MlirNamedAttribute regAttrs[] = {opNameNamed, payloadNamed, argNamesNamed};
    mlirOperationStateAddAttributes(&regState, 3, regAttrs);
    if (!mlirBlockIsNull(moduleBody)) {
        mlirBlockAppendOwnedOperation(moduleBody, mlirOperationCreate(&regState));
    }
}

void addFreeVar(FreeVars *fv, const char *name, int len) {
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

void findFreeVars(AstNode *node, FreeVars *fv, const char **bound, int bound_count) {
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
int count_var_usage(AstNode *node, const char *name, int name_len) {
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
        if (node->as.call.callee_len == name_len && strncmp(node->as.call.callee, name, name_len) == 0) {
            c++;
        } else if (node->as.call.resolved_callee && strlen(node->as.call.resolved_callee) == (size_t)name_len && strncmp(node->as.call.resolved_callee, name, name_len) == 0) {
            c++;
        }
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

MlirValue createEra(MlirContext ctx, MlirBlock block, MlirLocation loc) {
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

void linkToEra(MlirContext ctx, MlirBlock block, MlirLocation loc, MlirValue v) {
    linkValues(block, loc, v, createEra(ctx, block, loc));
}
