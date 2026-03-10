#include "lin/LowerToLLVMCAPI.h"
#include "lin/LowerToLLVM.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Wrap.h"

MlirPass mlirCreateInetToLLVMLoweringPass() {
    return wrap(createInetToLLVMLoweringPass().release());
}
