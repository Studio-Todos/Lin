add_executable(linc core/main.cpp core/Semantic.cpp core/Utils.cpp parser/Lexer.c parser/Parser.c mlir/Environment.c mlir/Utils.c mlir/Optimize.c mlir/Lowering.c runtime/gpu_runtime.c)

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
