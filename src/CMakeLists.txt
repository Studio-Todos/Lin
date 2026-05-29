add_executable(linc main.cpp Parser.c Lowering.c gpu_runtime.c Ast.c Semantic.cpp EGraphOptimizer.c)

find_package(Vulkan REQUIRED)

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
  MLIRGPUDialect
  MLIRGPUTransforms
  Vulkan::Vulkan
)

target_include_directories(linc PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
