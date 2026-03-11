//===- PicRuntimeDialect.td - PIC Runtime dialect definition -*- tablegen -*-===//
#ifndef PIC_RUNTIME_DIALECT
#define PIC_RUNTIME_DIALECT

include "mlir/IR/OpBase.td"

def PicRuntime_Dialect : Dialect {
  let name = "pic_runtime";
  let summary = "Polarized Interaction Combinator Runtime Dialect";
  let description = [{
    Lowers into target-specific runtimes (CPU/GPU).
  }];
  let cppNamespace = "::mlir::pic::runtime";
  let useDefaultTypePrinterParser = 0;
}

class PicRuntime_Op<string mnemonic, list<Trait> traits = []> :
    Op<PicRuntime_Dialect, mnemonic, traits>;

def PicRuntime_CpuDispatchOp : PicRuntime_Op<"cpu_dispatch"> {}
def PicRuntime_GpuBatchOp : PicRuntime_Op<"gpu_batch"> {}
def PicRuntime_SyncPointOp : PicRuntime_Op<"sync_point"> {}

#endif // PIC_RUNTIME_DIALECT
