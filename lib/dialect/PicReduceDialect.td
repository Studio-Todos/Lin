//===- PicReduceDialect.td - PIC Reduce dialect definition ---*- tablegen -*-===//
#ifndef PIC_REDUCE_DIALECT
#define PIC_REDUCE_DIALECT

include "mlir/IR/OpBase.td"

def PicReduce_Dialect : Dialect {
  let name = "pic_reduce";
  let summary = "Polarized Interaction Combinator Reduction Dialect";
  let description = [{
    Represents scheduling-aware reduction operations.
  }];
  let cppNamespace = "::mlir::pic::reduce";
  let useDefaultTypePrinterParser = 0;
}

class PicReduce_Op<string mnemonic, list<Trait> traits = []> :
    Op<PicReduce_Dialect, mnemonic, traits>;

// Reductions
def PicReduce_AnnihilateOp : PicReduce_Op<"annihilate"> {}
def PicReduce_DuplicateOp : PicReduce_Op<"duplicate"> {}
def PicReduce_EraseOp : PicReduce_Op<"erase"> {}
def PicReduce_FireOpOp : PicReduce_Op<"fire_op"> {}
def PicReduce_MergeBoundaryOp : PicReduce_Op<"merge_boundary"> {}

#endif // PIC_REDUCE_DIALECT
