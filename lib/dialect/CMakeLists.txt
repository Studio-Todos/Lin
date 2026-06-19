set(MLIR_TABLEGEN_EXE "/nix/store/mh0is1h5921i38ah1lkv2hgik69imp7f-mlir-18.1.8/bin/mlir-tblgen")

# PIC Graph
set(LLVM_TARGET_DEFINITIONS PicGraphDialect.td)
mlir_tablegen(PicGraphDialect.h.inc -gen-dialect-decls -dialect=pic_graph)
mlir_tablegen(PicGraphDialect.cpp.inc -gen-dialect-defs -dialect=pic_graph)
mlir_tablegen(PicGraphDialectOps.h.inc -gen-op-decls)
mlir_tablegen(PicGraphDialectOps.cpp.inc -gen-op-defs)

# PIC Reduce
set(LLVM_TARGET_DEFINITIONS PicReduceDialect.td)
mlir_tablegen(PicReduceDialect.h.inc -gen-dialect-decls -dialect=pic_reduce)
mlir_tablegen(PicReduceDialect.cpp.inc -gen-dialect-defs -dialect=pic_reduce)
mlir_tablegen(PicReduceDialectOps.h.inc -gen-op-decls)
mlir_tablegen(PicReduceDialectOps.cpp.inc -gen-op-defs)

# PIC Runtime
set(LLVM_TARGET_DEFINITIONS PicRuntimeDialect.td)
mlir_tablegen(PicRuntimeDialect.h.inc -gen-dialect-decls -dialect=pic_runtime)
mlir_tablegen(PicRuntimeDialect.cpp.inc -gen-dialect-defs -dialect=pic_runtime)
mlir_tablegen(PicRuntimeDialectOps.h.inc -gen-op-decls)
mlir_tablegen(PicRuntimeDialectOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(PicDialectsIncGen)

add_mlir_dialect_library(PicDialects
  PARTIAL_SOURCES_INTENDED
  PicGraphDialect.cpp
  PicReduceDialect.cpp
  PicRuntimeDialect.cpp
  PicDialectsCAPI.cpp
  Pass_PicGraphToReduce.cpp
  Pass_PicGraphVerify.cpp
  Pass_PicReduceToRuntime.cpp
  Pass_PicReduceLowering.cpp
  Pass_PicRuntimeToLLVM.cpp
  Pass_PicRuntimeToSPIRV.cpp


  DEPENDS
  PicDialectsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
  MLIRPass
  MLIRLLVMDialect
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRArithToLLVM
  MLIRArithToLLVM
  MLIRGPUDialect
  MLIRSPIRVDialect
  MLIRSPIRVSerialization
)

target_include_directories(PicDialects PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/include
)
