//===- InetDialect.cpp - Inet dialect implementation ----------------------===//
//
// This is the implementation file for the Inet dialect.
//
//===----------------------------------------------------------------------===//

#include "InetDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::inet;

// Include the auto-generated dialect implementation.
#include "InetDialect.cpp.inc"

#define GET_OP_CLASSES
#include "InetDialectOps.cpp.inc"

void InetDialect::initialize() {
  addTypes<PortType>();
  addOperations<
#define GET_OP_LIST
#include "InetDialectOps.cpp.inc"
      >();
}

Type InetDialect::parseType(DialectAsmParser &parser) const {
  if (succeeded(parser.parseOptionalKeyword("port")))
    return PortType::get(getContext());
  parser.emitError(parser.getNameLoc(), "unknown inet type");
  return Type();
}

void InetDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (llvm::isa<PortType>(type))
    printer << "port";
  else
    printer << "<unknown inet type>";
}
