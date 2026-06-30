cmake_minimum_required(VERSION 3.20.0)

project(Lin C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc")

# Workaround for nixos ncurses bug
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

find_package(MLIR REQUIRED CONFIG)

# MLIRConfig.cmake sets MLIR_TABLEGEN_EXE to bare "mlir-tblgen" which tablegen()
# treats as a missing build target (not an executable path). Re-find the full path.
unset(MLIR_TABLEGEN_EXE)
find_program(MLIR_TABLEGEN_EXE mlir-tblgen REQUIRED)

message(STATUS "Found MLIR ${MLIR_VERSION} at ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_program(MLIR_TABLEGEN_EXE mlir-tblgen REQUIRED NO_CACHE)
# Ensure absolute path so tablegen() uses it as an executable, not a build target
get_filename_component(MLIR_TABLEGEN_EXE "${MLIR_TABLEGEN_EXE}" ABSOLUTE)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

set(LLVM_TARGETS_TO_BUILD "X86;WebAssembly" CACHE STRING "Targets to build")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/dialect)

add_definitions(${LLVM_DEFINITIONS})

option(ENABLE_DEBUG_LOGS "Enable compiler debug logs and IR dumps" OFF)
if (ENABLE_DEBUG_LOGS)
    add_compile_definitions(ENABLE_DEBUG_LOGS)
endif()

add_subdirectory(lib/dialect)
add_subdirectory(src)
add_subdirectory(test)
