add_executable(lin main.cpp Parser.c Lowering.c)

target_link_libraries(lin PRIVATE
  InetDialect
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRExecutionEngine
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngineUtils
  MLIRCAPIIR
  MLIRCAPIRegisterEverything
)

target_include_directories(lin PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
