//===- InetDialect.td - Inet dialect definition -----------*- tablegen -*-===//
//
// This is the definition file for the Interaction Net (Inet) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef INET_DIALECT
#define INET_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"

def Inet_Dialect : Dialect {
  let name = "inet";
  let summary = "A dialect for Interaction Nets.";
  let description = [{
    The Inet dialect provides a representation for Interaction Nets.
    It serves as the IR for the Lin programming language lowering and
    optimization process before lowering to native machine code.
  }];
  let cppNamespace = "::mlir::inet";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Inet Types.
//===----------------------------------------------------------------------===//
def Inet_PortType : DialectType<Inet_Dialect, CPred<"$_self.isa<::mlir::inet::PortType>()">, "inet.port type">;

//===----------------------------------------------------------------------===//
// Base inet operation definition.
//===----------------------------------------------------------------------===//

class Inet_Op<string mnemonic, list<Trait> traits = []> :
    Op<Inet_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Inet Operations.
//===----------------------------------------------------------------------===//

def Inet_ConOp : Inet_Op<"con"> {
  let summary = "Constructor node (Con)";
  let description = [{
    A Constructor node with a principal port and two auxiliary ports.
  }];
  let results = (outs Inet_PortType:$p0, Inet_PortType:$p1, Inet_PortType:$p2);
}

def Inet_DupOp : Inet_Op<"dup"> {
  let summary = "Duplicator node (Dup)";
  let description = [{
    A Duplicator node. Like Con, but for copying/sharing.
  }];
  let results = (outs Inet_PortType:$p0, Inet_PortType:$p1, Inet_PortType:$p2);
}

def Inet_EraOp : Inet_Op<"era"> {
  let summary = "Eraser node (Era)";
  let description = [{
    An Eraser node. Has a single principal port, representing dropped values.
  }];
  let results = (outs Inet_PortType:$p0);
}

def Inet_NumOp : Inet_Op<"num"> {
  let summary = "Number node (Num)";
  let description = [{
    An unboxed 32-bit integer primitive value.
  }];
  let arguments = (ins I32Attr:$value);
  let results = (outs Inet_PortType:$p0);
}

def Inet_OpNodeOp : Inet_Op<"op"> {
  let summary = "General Operation node (Op)";
  let description = [{
    A general operation node, like Add, Sub, Eq, etc.
  }];
  let arguments = (ins Builtin_StringAttr:$opName);
  let results = (outs Inet_PortType:$p0, Inet_PortType:$p1, Inet_PortType:$p2);
}

def Inet_ExtOpNodeOp : Inet_Op<"ext_op"> {
  let summary = "External MLIR Operation node";
  let description = [{
    An extensible operation node that can encapsulate any standard MLIR operation.
    This prevents us from having to manually define a custom inet op for every
    possible arithmetic or library function.
  }];
  let arguments = (ins Builtin_StringAttr:$mlirOpName);
  // Ext nodes act like standard Op nodes: p0=result/evaluate, p1=arg1, p2=arg2
  let results = (outs Inet_PortType:$p0, Inet_PortType:$p1, Inet_PortType:$p2);
}

def Inet_LinkOp : Inet_Op<"link"> {
  let summary = "Links two ports together";
  let description = [{
    Creates a wire between two interaction net ports.
  }];
  let arguments = (ins Inet_PortType:$a, Inet_PortType:$b);
}

#endif // INET_DIALECT
