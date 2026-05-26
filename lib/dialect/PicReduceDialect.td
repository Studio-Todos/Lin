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
}

// Duplicate: δ → γ⁺ ⋈ γ⁻ (Rule 2)
def PicReduce_DuplicateOp : PicReduce_Op<"duplicate"> {
  let summary = "Duplication rule for δ";
}

// Erase: ε ⋈ X → ε (Rule 3)
def PicReduce_EraseOp : PicReduce_Op<"erase"> {
  let summary = "Erasure rule for ε ⋈ X";
}

// Fire-op: ω(~) with inputs → evaluate (Rule congruence)
def PicReduce_FireOpOp : PicReduce_Op<"fire_op"> {
  let summary = "Operation firing rule for ω agents";
}

// Merge boundary: [A] ⋈ [B] → [A ∪ B] (Rule 4)
def PicReduce_MergeBoundaryOp : PicReduce_Op<"merge_boundary"> {
  let summary = "Boundary merge rule";
}

def PicReduce_LogBondOp : PicReduce_Op<"log_bond"> {
  let summary = "Deposits an L-bond on spliced wires";
  let arguments = (ins I32Attr:$node_a_idx, I32Attr:$node_b_idx);
}

def PicReduce_ReverseVectorOp : PicReduce_Op<"reverse_vector"> {
  let summary = "Reverse-Vector propagation marker";
}

def PicReduce_UncomputeOp : PicReduce_Op<"uncompute"> {
  let summary = "Inverts a specific node pair using its log payload";
}

def PicReduce_LogStateOp : PicReduce_Op<"log_state"> {
  let summary = "Pushes a structural delta token to the boundary history log";
}

#endif // PIC_REDUCE_DIALECT
