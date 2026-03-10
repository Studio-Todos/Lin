set(LLVM_TARGET_DEFINITIONS InetDialect.td)

set(MLIR_TABLEGEN_EXE "/nix/store/mh0is1h5921i38ah1lkv2hgik69imp7f-mlir-18.1.8/bin/mlir-tblgen")

mlir_tablegen(InetDialect.h.inc -gen-dialect-decls -dialect=inet)
mlir_tablegen(InetDialect.cpp.inc -gen-dialect-defs -dialect=inet)
mlir_tablegen(InetDialectOps.h.inc -gen-op-decls)
mlir_tablegen(InetDialectOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(InetDialectIncGen)

add_mlir_dialect_library(InetDialect
  PARTIAL_SOURCES_INTENDED
  InetDialect.cpp
  InetDialectCAPI.cpp
  LowerToLLVM.cpp
  LowerToLLVMCAPI.cpp

  DEPENDS
  InetDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
  MLIRPass
  MLIRLLVMDialect
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRArithToLLVM
)

target_include_directories(InetDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/include
)
