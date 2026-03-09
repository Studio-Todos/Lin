set(LLVM_TARGET_DEFINITIONS InetDialect.td)

mlir_tablegen(InetDialect.h.inc -gen-dialect-decls)
mlir_tablegen(InetDialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(InetDialectIncGen)

add_mlir_dialect_library(InetDialect
  InetDialect.cpp
  InetDialectCAPI.cpp

  DEPENDS
  InetDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
)

target_include_directories(InetDialect PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/include
)
