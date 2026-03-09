#include <stdio.h>
#include <stdlib.h>

#include "mlir-c/IR.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/RegisterEverything.h"
#include "lin/InetDialectCAPI.h"

int main(int argc, char **argv) {
  printf("Initializing Lin Compiler...\n");

  MlirContext ctx = mlirContextCreate();

  // Load all core MLIR dialects
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);

  // Register our custom Inet dialect
  mlirContextRegisterInetDialect(ctx);
  mlirContextLoadAllAvailableDialects(ctx);

  printf("Inet dialect registered successfully.\n");

  // Create a simple module
  MlirLocation loc = mlirLocationUnknownGet(ctx);
  MlirModule module = mlirModuleCreateEmpty(loc);

  // Dump the module
  printf("Dumping empty module:\n");
  MlirOperation moduleOp = mlirModuleGetOperation(module);
  mlirOperationDump(moduleOp);

  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);

  printf("Compilation complete.\n");
  return 0;
}
