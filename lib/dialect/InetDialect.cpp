//===- InetDialect.cpp - Inet dialect implementation ----------------------===//
//
// This is the implementation file for the Inet dialect.
//
//===----------------------------------------------------------------------===//

#include "InetDialect.h"

using namespace mlir;
using namespace mlir::inet;

// Include the auto-generated dialect implementation.
#include "InetDialect.cpp.inc"

void InetDialect::initialize() {
  // We'll register the operations defined in our .td file once we compile them.
}
