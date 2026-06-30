add_executable(linc main.cpp Parser.c Lowering.c gpu_runtime.c Ast.c Diagnostic.c Semantic.cpp EGraphOptimizer.c)

find_package(Vulkan QUIET)

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
)

if(Vulkan_FOUND)
  target_link_libraries(linc PRIVATE Vulkan::Vulkan)
  target_compile_definitions(linc PRIVATE HAVE_VULKAN)
endif()

target_include_directories(linc PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
