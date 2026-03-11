//===- PicGraphDialect.td - PIC Graph dialect definition -----*- tablegen -*-===//
#ifndef PIC_GRAPH_DIALECT
#define PIC_GRAPH_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def PicGraph_Dialect : Dialect {
  let name = "pic_graph";
  let summary = "Polarized Interaction Combinator Graph Dialect";
  let description = [{
    A pure combinator graph representation.
  }];
  let cppNamespace = "::mlir::pic::graph";
  let useDefaultTypePrinterParser = 1;
}

def PicGraph_PortType : DialectType<PicGraph_Dialect, CPred<"$_self.isa<::mlir::pic::graph::PortType>()">, "pic_graph.port type">;

class PicGraph_Op<string mnemonic, list<Trait> traits = []> :
    Op<PicGraph_Dialect, mnemonic, traits>;

def PicGraph_AgentOp : PicGraph_Op<"agent"> {
  let summary = "Agent node (node with polarity)";
  let description = [{
    An agent node in the PIC graph. Polarity can be positive (+), negative (-), or neutral (*).
    Type can be Constructor (\gamma^+), Destructor (\gamma^-), Duplicator (\delta), Eraser (\epsilon), or Operation (\omega).
  }];
  let arguments = (ins Builtin_StringAttr:$agentType, Builtin_StringAttr:$polarity, Builtin_StringAttr:$label, OptionalAttr<I32Attr>:$value);
  let results = (outs PicGraph_PortType:$p0, PicGraph_PortType:$p1, PicGraph_PortType:$p2);
}

def PicGraph_LinkOp : PicGraph_Op<"link"> {
  let summary = "Links two ports";
  let arguments = (ins PicGraph_PortType:$a, PicGraph_PortType:$b);
}

// A boundary is a region in the MLIR sense
def PicGraph_BoundaryOp : PicGraph_Op<"boundary"> {
  let summary = "Boundary region";
  let arguments = (ins Variadic<PicGraph_PortType>:$inputs);
  let results = (outs Variadic<PicGraph_PortType>:$outputs);
  let regions = (region AnyRegion:$body);
}

#endif // PIC_GRAPH_DIALECT
