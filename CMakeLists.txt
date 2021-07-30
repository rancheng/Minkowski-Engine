cmake_minimum_required(VERSION 3.11)
project(minkowski)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# find_package(OpenBLAS REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(BLA_DEFINITIONS)
# include directories
set(MINK_INCLUDE_DIRS "")
# library directories
set(MINK_LIB_DIRS "")
include(FindPythonInterp)
message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")

execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE WITH_ABI
)
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE PYTHON_HEADER_DIR
)
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "from torch.utils.cpp_extension import include_paths; [print(p, end=';') for p in include_paths()]"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE PYTORCH_INCLUDES
)
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "from torch.utils.cpp_extension import library_paths; [print(p, end=';') for p in library_paths()]"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE PYTORCH_LIBRARIES
)
execute_process(
        COMMAND bash "-c" "uname -s"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE UNAME
)

message(STATUS "WITH_ABI: ${WITH_ABI}")
message(STATUS "PYTHON_HEADER_DIR: ${PYTHON_HEADER_DIR}")
message(STATUS "PYTORCH_INCLUDES: ${PYTORCH_INCLUDES}")
message(STATUS "PYTORCH_LIBRARIES: ${PYTORCH_LIBRARIES}")
message(STATUS "UNAME: ${UNAME}")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${WITH_ABI})
list(APPEND MINK_INCLUDE_DIRS ${PYTHON_HEADER_DIR})
list(APPEND MINK_INCLUDE_DIRS ${PYTORCH_INCLUDES})
list(APPEND MINK_LIB_DIRS ${PYTORCH_LIBRARIES})

message(STATUS "MINK_INCLUDE_DIRS: ${MINK_INCLUDE_DIRS}")
message(STATUS "MINK_LIB_DIRS: ${MINK_LIB_DIRS}")

# platform
set(LINUX 0)
set(OSX 0)
if ("${UNAME}" STREQUAL "Linux")
    message("== Compiling under ${UNAME} platform ==")
    set(LINUX 1)
    message(STATUS "LINUX: ${LINUX}")
elseif ("${UNAME}" STREQUAL "Darwin")
    message("== Compiling under ${UNAME} platform ==")
    set(OSX 1)
    set(CXX "/usr/local/opt/llvm/bin/clang")
    execute_process(
            COMMAND bash "-c" "sw_vers -productVersion | cut -f 1 -d ."
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE OSX_MAJOR_VERSION
    )
    execute_process(
            COMMAND bash "-c" "sw_vers -productVersion | cut -f 2 -d ."
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_STRIP_TRAILING_WHITESPACE
            OUTPUT_VARIABLE OSX_MINOR_VERSION
    )
    list(APPEND MINK_INCLUDE_DIRS "/usr/local/opt/llvm/include")
    list(APPEND MINK_LIB_DIRS "/usr/local/opt/llvm/lib")
endif ()

if (NOT "${CPU_ONLY}" STREQUAL "1")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=minkowski")
    execute_process(
            COMMAND ${PYTHON_EXECUTABLE} -c "from torch.utils.cpp_extension import _find_cuda_home; print(_find_cuda_home())"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            OUTPUT_VARIABLE CUDA_DIR
    )
    message(STATUS "CUDA_DIR: ${CUDA_DIR}")
    list(APPEND MINK_INCLUDE_DIRS "${CUDA_DIR}/include")
    list(APPEND MINK_LIB_DIRS "${CUDA_DIR}/lib64")
endif ()

# user defined options for MKL

find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if (OpenBLAS_FOUND)
    # We should agree on BLA/BLAS variables
    set(BLA_INCLUDE_DIR ${OpenBLAS_INCLUDE_DIRS})
    set(BLA_LIBRARIES ${OpenBLAS_LIBRARIES})
    list(APPEND MINK_INCLUDE_DIRS BLA_INCLUDE_DIR)
    list(APPEND MINK_LIB_DIRS BLA_LIBRARIES)
    set(BLA_DEFINITIONS USE_OPENBLAS)
endif ()

message(STATUS "BLAS Libraries: ${BLA_LIBRARIES}")

find_package(Python COMPONENTS Interpreter Development)

message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_SITELIB ${Python_SITELIB}")
message(STATUS "PyTorch_SITELIB ${Python_SITELIB}/torch/share/cmake")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

list(APPEND CMAKE_PREFIX_PATH ${Python_SITELIB}/torch/share/cmake/Torch)

# message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
find_package(Torch REQUIRED)
if (TORCH_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${TORCH_CXX_FLAGS}")
    list(APPEND MINK_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
    list(APPEND MINK_LIB_DIRS ${TORCH_LIBRARIES})
endif ()
#message(STATUS "TORCH_CXX_FLAGS ${TORCH_CXX_FLAGS}")

find_package(MKL QUIET)

if (MKL_FOUND)
    set(BLA_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    set(BLA_DEFINITIONS USE_MKL)
else()
    list(REMOVE_ITEM MINK_INCLUDE_DIRS BLA_INCLUDE_DIR)
    list(REMOVE_ITEM MINK_LIB_DIRS BLA_LIBRARIES)
endif()


list(REMOVE_DUPLICATES MINK_INCLUDE_DIRS)
list(REMOVE_DUPLICATES MINK_LIB_DIRS)

message(STATUS "MINK_INCLUDE_DIRS: ${MINK_INCLUDE_DIRS}")
message(STATUS "MINK_LIB_DIRS: ${MINK_LIB_DIRS}")

#list(APPEND CMAKE_MODULE_PATH ${Python_SITELIB}/torch/share/cmake)
#list(APPEND CMAKE_MODULE_PATH ${Python_SITELIB}/torch/share/cmake/Torch)
#
##find_package(Torch REQUIRED)
#find_package(OpenMP REQUIRED)
##if (OPENMP_FOUND)
##    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
##    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
##    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
##endif()
##SET(GCC_COMPILE_FLAGS "-fwrapv -std=c++14")
#
#
#set(CMAKE_CXX_FLAGS "-MMD -MP -ffast-math -funsafe-math-optimizations -fno-math-errno -DBATCH_FIRST=1 -fopenmp -fPIC\
#    -fwrapv -std=c++14 -DDEBUG -g -O0\
#    -I${Python_INCLUDE_DIRS}\
#    -I${Python_INCLUDE_DIRS}/..\
#    -I${Python_SITELIB}/torch/include\
#    -I${Python_SITELIB}/torch/include/torch/csrc/api/include\
#    -I${Python_SITELIB}/torch/include/TH\
#    -I${Python_SITELIB}/torch/include/THC\
#    -I./\
#    -I/usr/local/cuda-10.2/include\
#    -DTORCH_API_INCLUDE_EXTENSION_H\
#    -DTORCH_EXTENSION_NAME=minkowski\
#    -D_GLIBCXX_USE_CXX11_ABI=0\
#    -Wall\
#    -Wcomment\
#    -Wno-sign-compare\
#    -Wno-deprecated-declarations")
#
#set(CMAKE_CUDA_FLAGS "-g -G -std=c++14 -ccbin=g++ -Xcompiler -fPIC -DDEBUG -g -O0\
#    -I${Python_INCLUDE_DIRS}\
#    -I${Python_INCLUDE_DIRS}/..\
#    -I${Python_SITELIB}/torch/include\
#    -I${Python_SITELIB}/torch/include/torch/csrc/api/include\
#    -I${Python_SITELIB}/torch/include/TH\
#    -I${Python_SITELIB}/torch/include/THC\
#    -I./\
#    -I/usr/local/cuda-10.2/include\
#    -DTORCH_API_INCLUDE_EXTENSION_H\
#    -DTORCH_EXTENSION_NAME=minkowski\
#    -D_GLIBCXX_USE_CXX11_ABI=0")
#
#
#message("######### ${Python_INCLUDE_DIRS} ############")
#message("######### ${Python_LIBRARY_DIRS} ############")
#message("######### ${Python_SITELIB} ############")
#
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#add_subdirectory(pybind11)
#
#set(minkowski_SOURCE_FILES
#        ${PROJECT_SOURCE_DIR}/src/3rdparty/robin_hood.h
#        ${PROJECT_SOURCE_DIR}/src/primitives/small_vector.hpp
#        ${PROJECT_SOURCE_DIR}/src/broadcast.cpp
#        ${PROJECT_SOURCE_DIR}/src/broadcast.cu
#        ${PROJECT_SOURCE_DIR}/src/broadcast.cuh
#        ${PROJECT_SOURCE_DIR}/src/broadcast.hpp
#        ${PROJECT_SOURCE_DIR}/src/common.hpp
#        ${PROJECT_SOURCE_DIR}/src/convolution.cpp
#        ${PROJECT_SOURCE_DIR}/src/convolution.cu
#        ${PROJECT_SOURCE_DIR}/src/convolution.cuh
#        ${PROJECT_SOURCE_DIR}/src/convolution.hpp
#        ${PROJECT_SOURCE_DIR}/src/convolution_transpose.cpp
#        ${PROJECT_SOURCE_DIR}/src/coords_key.cpp
#        ${PROJECT_SOURCE_DIR}/src/coords_manager.cpp
#        ${PROJECT_SOURCE_DIR}/src/coords_manager.cu
#        ${PROJECT_SOURCE_DIR}/src/coords_manager.hpp
#        ${PROJECT_SOURCE_DIR}/src/coordsmap.cpp
#        ${PROJECT_SOURCE_DIR}/src/coordsmap.hpp
#        ${PROJECT_SOURCE_DIR}/src/gpu.cu
#        ${PROJECT_SOURCE_DIR}/src/gpu.cuh
#        ${PROJECT_SOURCE_DIR}/src/gpu_memory_manager.hpp
#        ${PROJECT_SOURCE_DIR}/src/math_functions.cpp
#        ${PROJECT_SOURCE_DIR}/src/math_functions.cu
#        ${PROJECT_SOURCE_DIR}/src/math_functions.hpp
#        ${PROJECT_SOURCE_DIR}/src/mkl_alternate.hpp
#        ${PROJECT_SOURCE_DIR}/src/pooling_avg.cpp
#        ${PROJECT_SOURCE_DIR}/src/pooling_avg.cu
#        ${PROJECT_SOURCE_DIR}/src/pooling_avg.cuh
#        ${PROJECT_SOURCE_DIR}/src/pooling_avg.hpp
#        ${PROJECT_SOURCE_DIR}/src/pooling_global_avg.cpp
#        ${PROJECT_SOURCE_DIR}/src/pooling_global_max.cpp
#        ${PROJECT_SOURCE_DIR}/src/pooling_max.cpp
#        ${PROJECT_SOURCE_DIR}/src/pooling_max.cu
#        ${PROJECT_SOURCE_DIR}/src/pooling_max.hpp
#        ${PROJECT_SOURCE_DIR}/src/pooling_transpose.cpp
#        ${PROJECT_SOURCE_DIR}/src/pruning.cpp
#        ${PROJECT_SOURCE_DIR}/src/pruning.cu
#        ${PROJECT_SOURCE_DIR}/src/pruning.cuh
#        ${PROJECT_SOURCE_DIR}/src/pruning.hpp
#        ${PROJECT_SOURCE_DIR}/src/quantization.cpp
#        ${PROJECT_SOURCE_DIR}/src/region.cpp
#        ${PROJECT_SOURCE_DIR}/src/types.hpp
#        ${PROJECT_SOURCE_DIR}/src/union.cpp
#        ${PROJECT_SOURCE_DIR}/src/union.cu
#        ${PROJECT_SOURCE_DIR}/src/union.cuh
#        ${PROJECT_SOURCE_DIR}/src/union.hpp
#        ${PROJECT_SOURCE_DIR}/src/utils.hpp
#        )
#
#message("#######Project source file:  ${TORCH_INCLUDE_DIRS} ########")
#
#include_directories(
#        ${PROJECT_SOURCE_DIR}
#        ${Python_SITELIB}/torch/include
#        ${Python_SITELIB}/torch/include/torch/csrc/api/include
#        ${PROJECT_SOURCE_DIR}/pybind
#        ${OpenBLAS_INCLUDE_DIR}
#)
#
#
#pybind11_add_module(minkowski ${PROJECT_SOURCE_DIR}/pybind/minkowski.cpp ${minkowski_SOURCE_FILES})
#set_target_properties(minkowski PROPERTIES LINK_FLAGS "-pthread -fPIC -Wall -Wcomment -Wno-sign-compare\
#-Wno-deprecated-declarations -Wl,-rpath= -Wl,--no-as-needed -Wl,--sysroot=/")
#set(CMAKE_SHARED_LINKER_FLAGS "-L${Python_SITELIB}/torch/lib\
#    -L/usr/local/cuda-10.2/lib64 -lstdc++ -lc10 -lcaffe2 -ltorch -ltorch_python -l_C -lcudart -lcublas\
#    -lcusparse -lcaffe2_gpu -lc10_cuda -lopenblas")
