//===- InetDialect.td - Inet dialect definition -----------*- tablegen -*-===//
//
// This is the definition file for the Interaction Net (Inet) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef INET_DIALECT
#define INET_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Inet dialect definition.
//===----------------------------------------------------------------------===//

def Inet_Dialect : Dialect {
  let name = "inet";
  let summary = "A dialect for Interaction Nets.";
  let description = [{
    The Inet dialect provides a representation for Interaction Nets.
    It serves as the IR for the Lin programming language lowering and
    optimization process before lowering to native machine code.
  }];
  let cppNamespace = "::mlir::inet";
}

//===----------------------------------------------------------------------===//
// Base inet operation definition.
//===----------------------------------------------------------------------===//

class Inet_Op<string mnemonic, list<Trait> traits = []> :
    Op<Inet_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Inet Operations.
//===----------------------------------------------------------------------===//

def Inet_NodeOp : Inet_Op<"node"> {
  let summary = "Defines an interaction net node.";
  let description = [{
    The `inet.node` operation defines a basic interaction net node.
  }];

  let arguments = (ins AnyType:$kind);
  let results = (outs AnyType:$out);
}

#endif // INET_DIALECT
