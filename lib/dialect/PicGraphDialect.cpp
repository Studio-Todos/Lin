//===- PicGraphDialect.cpp - PIC Graph dialect implementation ----------===//
#include "PicGraphDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::pic::graph;

#include "PicGraphDialect.cpp.inc"

#define GET_OP_CLASSES
#include "PicGraphDialectOps.cpp.inc"

void PicGraphDialect::initialize() {
  addTypes<PortType>();
  addOperations<
#define GET_OP_LIST
#include "PicGraphDialectOps.cpp.inc"
      >();
}

Type PicGraphDialect::parseType(DialectAsmParser &parser) const {
  if (succeeded(parser.parseOptionalKeyword("port")))
    return PortType::get(getContext());
  parser.emitError(parser.getNameLoc(), "unknown pic_graph type");
  return Type();
}

void PicGraphDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (llvm::isa<PortType>(type))
    printer << "port";
  else
    printer << "<unknown pic_graph type>";
}
