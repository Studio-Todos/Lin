//===- PicRuntimeDialect.td - PIC Runtime dialect definition -*- tablegen -*-===//
#ifndef PIC_RUNTIME_DIALECT
#define PIC_RUNTIME_DIALECT

include "mlir/IR/OpBase.td"

def PicRuntime_Dialect : Dialect {
  let name = "pic_runtime";
  let summary = "Polarized Interaction Combinator Runtime Dialect";
  let description = [{
    Lowers into target-specific runtimes (CPU/GPU) using a flat array memory model.
  }];
  let cppNamespace = "::mlir::pic::runtime";
  let useDefaultTypePrinterParser = 0;
}

class PicRuntime_Op<string mnemonic, list<Trait> traits = []> :
    Op<PicRuntime_Dialect, mnemonic, traits>;

// Types
def PicRuntime_PortType : DialectType<PicRuntime_Dialect, CPred<"$_self.isa<::mlir::pic::runtime::PortType>()">, "pic_runtime.port type">;

// Operations for array-based graph manipulation
def PicRuntime_AllocNodeOp : PicRuntime_Op<"alloc_node"> {
  let summary = "Allocates a node in the 128-bit atomic array";
  let arguments = (ins I8Attr:$type, I64:$label_or_val, OptionalAttr<I1Attr>:$inside_boundary);
  let results = (outs I32:$index);
}

def PicRuntime_SetPortOp : PicRuntime_Op<"set_port"> {
  let summary = "Sets a port on a node";
  let arguments = (ins I32:$node_index, I8Attr:$port_index, I64:$port_value);
}

def PicRuntime_GetPortOp : PicRuntime_Op<"get_port"> {
  let summary = "Gets a port on a node";
  let arguments = (ins I32:$node_index, I8Attr:$port_index);
  let results = (outs I64:$port_value);
}

def PicRuntime_GetPortDynamicOp : PicRuntime_Op<"get_port_dynamic"> {
  let summary = "Gets a port on a node dynamically";
  let arguments = (ins I32:$node_index, I8:$port_index);
  let results = (outs I64:$port_value);
}

def PicRuntime_LinkOp : PicRuntime_Op<"link"> {
  let summary = "Links two ports together, potentially creating a redex";
  let arguments = (ins I64:$port_a, I64:$port_b);
}

def PicRuntime_PushRedexOp : PicRuntime_Op<"push_redex"> {
  let summary = "Pushes an active pair to the thread-local or global redex queue";
  let arguments = (ins I32:$node_a, I32:$node_b);
}

def PicRuntime_PopRedexOp : PicRuntime_Op<"pop_redex"> {
  let summary = "Pops an active pair from the redex queue";
  let results = (outs I1:$valid, I32:$node_a, I32:$node_b);
}

// Runtime routing and synchronization
def PicRuntime_CpuDispatchOp : PicRuntime_Op<"cpu_dispatch"> {
  let summary = "Dispatches a redex to CPU execution pool";
  let arguments = (ins I32:$node_a, I32:$node_b);
}

def PicRuntime_GpuBatchOp : PicRuntime_Op<"gpu_batch"> {
  let summary = "Batches ops for execution on GPU";
  let arguments = (ins I32:$node_a, I32:$node_b, I32Attr:$op_label);
}

def PicRuntime_SyncPointOp : PicRuntime_Op<"sync_point"> {
  let summary = "A synchronization point between CPU and GPU tasks";
}

def PicRuntime_UncomputeSweepOp : PicRuntime_Op<"uncompute_sweep"> {
  let summary = "Reads boundary history log and drives R-signal propagation in reverse";
  let arguments = (ins I32:$boundary_id);
}

def PicRuntime_GetHistoryOp : PicRuntime_Op<"get_history"> {
  let summary = "Gets a history log word";
  let arguments = (ins I32:$node_index, I8Attr:$word_index);
  let results = (outs I64:$word_value);
}

def PicRuntime_SetHistoryOp : PicRuntime_Op<"set_history"> {
  let summary = "Sets a history log word";
  let arguments = (ins I32:$node_index, I8Attr:$word_index, I64:$word_value);
}

#endif // PIC_RUNTIME_DIALECT

