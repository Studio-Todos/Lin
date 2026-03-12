add_executable(linc main.cpp Parser.c Lowering.c)

target_link_libraries(linc PRIVATE
  PicDialects
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

target_include_directories(linc PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
