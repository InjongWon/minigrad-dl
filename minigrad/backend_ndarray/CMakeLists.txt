# IDE / clangd: emits compile_commands.json with pybind11 + Python -I flags
# so #include <pybind11/numpy.h> resolves (and Python.h pulls in correctly).
#
# From this directory:
#   mkdir -p build && cd build && cmake ..
cmake_minimum_required(VERSION 3.15)
project(minigrad_ndarray_backend_ide CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
  OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _pybind_inc_rc
)

if(NOT _pybind_inc_rc EQUAL 0)
  message(FATAL_ERROR "pybind11 not found for this Python (${Python3_EXECUTABLE}). "
          "Install: pip install pybind11")
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _py_inc_rc
)

if(NOT _py_inc_rc EQUAL 0 OR NOT PYTHON_INCLUDE_DIR)
  message(FATAL_ERROR "Could not get Python include path for ${Python3_EXECUTABLE}.")
endif()

message(STATUS "pybind11 include: ${PYBIND11_INCLUDE_DIR}")
message(STATUS "Python include: ${PYTHON_INCLUDE_DIR}")

set(_BACKEND_INCLUDES "${PYBIND11_INCLUDE_DIR}" "${PYTHON_INCLUDE_DIR}")

add_library(ndarray_backend_cpu OBJECT ndarray_backend_cpu.cc)
target_compile_features(ndarray_backend_cpu PRIVATE cxx_std_17)
target_include_directories(ndarray_backend_cpu PRIVATE ${_BACKEND_INCLUDES})

# Parse .cu as C++ here so tooling works without a CUDA toolkit installed.
add_library(ndarray_backend_cuda OBJECT ndarray_backend_cuda.cu)
set_source_files_properties(ndarray_backend_cuda.cu PROPERTIES LANGUAGE CXX)
target_compile_features(ndarray_backend_cuda PRIVATE cxx_std_17)
target_include_directories(ndarray_backend_cuda PRIVATE ${_BACKEND_INCLUDES})
