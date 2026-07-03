//===- PicReduceDialect.td - PIC Reduce dialect definition ---*- tablegen -*-===//
#ifndef PIC_REDUCE_DIALECT
#define PIC_REDUCE_DIALECT

include "mlir/IR/OpBase.td"

def PicReduce_Dialect : Dialect {
  let name = "pic_reduce";
  let summary = "Polarized Interaction Combinator Reduction Dialect";
  let description = [{
    Represents scheduling-aware reduction operations.
    These ops are markers for active redexes in the runtime reduction engine.
  }];
  let cppNamespace = "::mlir::pic::reduce";
  let useDefaultTypePrinterParser = 0;
}

class PicReduce_Op<string mnemonic, list<Trait> traits = []> :
    Op<PicReduce_Dialect, mnemonic, traits>;

// Annihilate: γ⁺ ~ γ⁻ → splice aux wires (Rule 1)
// Simple marker op - no attributes needed
def PicReduce_AnnihilateOp : PicReduce_Op<"annihilate"> {
  let summary = "Annihilation rule for γ⁺ ⋈ γ⁻";
  let arguments = (ins I32:$node_a, I32:$node_b, I64:$state, OptionalAttr<StrAttr>:$dispatch_flag);
}

// Duplicate: δ → γ⁺ ⋈ γ⁻ (Rule 2)
def PicReduce_DuplicateOp : PicReduce_Op<"duplicate"> {
  let summary = "Duplication rule for δ";
  let arguments = (ins I32:$node_a, I32:$node_b, I64:$state, OptionalAttr<StrAttr>:$dispatch_flag);
}

// Erase: ε ⋈ X → ε (Rule 3)
def PicReduce_EraseOp : PicReduce_Op<"erase"> {
  let summary = "Erasure rule for ε ⋈ X";
  let arguments = (ins I32:$node_a, I32:$node_b, I64:$state, OptionalAttr<StrAttr>:$dispatch_flag);
}

// Fire-op: ω(~) with inputs → evaluate (Rule congruence)
def PicReduce_FireOpOp : PicReduce_Op<"fire_op"> {
  let summary = "Operation firing rule for ω agents";
  let arguments = (ins I32:$node_a, I32:$node_b, I64:$state, OptionalAttr<StrAttr>:$dispatch_flag);
}

def PicReduce_LogBondOp : PicReduce_Op<"log_bond"> {
  let summary = "Deposits an L-bond on spliced wires";
  let arguments = (ins I32Attr:$node_a_idx, I32Attr:$node_b_idx);
}

def PicReduce_ReverseVectorOp : PicReduce_Op<"reverse_vector"> {
  let summary = "Reverse-Vector propagation marker";
  let arguments = (ins I32:$node_a, I32:$node_b, I64:$state, OptionalAttr<StrAttr>:$dispatch_flag);
}

#endif // PIC_REDUCE_DIALECT
