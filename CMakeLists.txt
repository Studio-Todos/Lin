cmake_minimum_required(VERSION 3.20.0)

project(Lin C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found MLIR ${MLIR_VERSION} at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_program(MLIR_TABLEGEN_EXE mlir-tblgen REQUIRED)
set(MLIR_TABLEGEN_EXE ${MLIR_TABLEGEN_EXE} CACHE STRING "Path to mlir-tblgen" FORCE)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(LLVM_TARGETS_TO_BUILD "X86" CACHE STRING "Targets to build")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/dialect)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(lib/dialect)
add_subdirectory(src)
add_subdirectory(test)
