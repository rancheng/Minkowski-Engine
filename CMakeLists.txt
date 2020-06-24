cmake_minimum_required(VERSION 3.10)
project(MinkowskiEngineBackend LANGUAGES CXX CUDA)
set(PYBIND11_CPP_STANDARD -std=c++14)
set(DEBUG TRUE)
set(LIBTORCH "/usr/local/libtorch")
set(PYTHON "python")
set(CXX "g++")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenBLAS REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

EXECUTE_PROCESS ( COMMAND ${PYTHON} -c "from torch.utils.cpp_extension import _find_cuda_home; print(_find_cuda_home())" OUTPUT_VARIABLE CUDA_HOME)

EXECUTE_PROCESS ( COMMAND ${PYTHON} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())" OUTPUT_VARIABLE PYTHON_HEADER_DIR)
EXECUTE_PROCESS ( COMMAND ${PYTHON} -c "from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]" OUTPUT_VARIABLE PYTORCH_INCLUDES)
EXECUTE_PROCESS ( COMMAND ${PYTHON} -c "from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]" OUTPUT_VARIABLE PYTORCH_LIBRARIES)
# remove whitespaces and newline in the output path of python script
string(REGEX REPLACE "(\r?\n)+" "" CUDA_HOME ${CUDA_HOME})
string(REGEX REPLACE "(\r?\n)+" "" PYTHON_HEADER_DIR ${PYTHON_HEADER_DIR})
#string(REGEX REPLACE "(\r?\n)+" "" PYTORCH_INCLUDES ${PYTORCH_INCLUDES})
string(REGEX REPLACE "(\r?\n)+" "" PYTORCH_LIBRARIES ${PYTORCH_LIBRARIES})
#message("#######PYTORCH_INCLUDES:  ${PYTORCH_INCLUDES} ########")
#message("#######PYTORCH_LIBRARIES:  ${PYTORCH_LIBRARIES} ########")
get_filename_component(PYTHON_HEADER_DIR_PARENT_DIR ${PYTHON_HEADER_DIR} DIRECTORY)
string(REPLACE "\n" ";" PYTORCH_INCLUDES ${PYTORCH_INCLUDES})
set(PYTORCH_INCLUDES_I "")
foreach(X IN LISTS PYTORCH_INCLUDES)
    if( NOT X STREQUAL "" )
        list(APPEND PYTORCH_INCLUDES_I "-I${X}")
    endif()
endforeach()
set(PYTORCH_INCLUDES_I_CXX "")
foreach(X IN LISTS PYTORCH_INCLUDES)
    if( NOT X STREQUAL "" )
        list(APPEND PYTORCH_INCLUDES_I_CXX "-I${X}")
    endif()
endforeach()
string( REPLACE ";" " " PYTORCH_INCLUDES_I_CXX "${PYTORCH_INCLUDES_I_CXX}" )
#message("#######CUDA_HOME:  ${CUDA_HOME} ########")
#message("#######PYTORCH_INCLUDES_I_CXX:  ${PYTORCH_INCLUDES_I_CXX} ########")
include_directories(
        ${PYTHON_HEADER_DIR}
        ${PYTHON_HEADER_DIR_PARENT_DIR}
        ${PYTORCH_INCLUDES}
        )

# Determine ABI support
EXECUTE_PROCESS ( COMMAND ${PYTHON} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" OUTPUT_VARIABLE WITH_ABI)

# Determine platform
EXECUTE_PROCESS ( COMMAND bash -c "uname -s" OUTPUT_VARIABLE UNAME)

if("${UNAME}" STREQUAL "Linux")
    message("Compiling under:  ${UNAME}")
    set(LINUX TRUE)
elseif("${UNAME}" STREQUAL "Darwin")
    message("Compiling under:  ${UNAME}")
    set(OSX TRUE)
    EXECUTE_PROCESS ( COMMAND bash -c "sw_vers -productVersion | cut -f 1 -d ." OUTPUT_VARIABLE OSX_MAJOR_VERSION)
    EXECUTE_PROCESS ( COMMAND bash -c "sw_vers -productVersion | cut -f 2 -d ." OUTPUT_VARIABLE OSX_MINOR_VERSION)
    set(CXX /usr/local/opt/llvm/bin/clang)
    # brew install llvm libomp
    include_directories(/usr/local/opt/llvm/include)
    link_directories(/usr/local/opt/llvm/lib)
endif()

if(NOT CPU_ONLY)
    message("Building with CUDA support")
    # CUDA ROOT DIR that contains bin/ lib64/ and include/
    # CUDA_HOME := /usr/local/cuda
    SET(NVCC ${CUDA_HOME}/bin/nvcc)
    set(CUDA_INCLUDE_DIRS "${CUDA_HOME}/include")
    set(CUDA_LIBRARY_DIRS ${CUDA_HOME}/lib64)
    include_directories(${CUDA_HOME}/include)
    link_directories(${CUDA_HOME}/lib64)
endif()


#get_property(test_LINK_DIRECTORIES DIRECTORY PROPERTY LINK_DIRECTORIES)
#get_property(test_INCLUDE_DIRECTORIES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
#message("LINK_DIRECTOREIS: ${test_LINK_DIRECTORIES}")
#message("INCLUDE_DIRECTORIES: ${test_INCLUDE_DIRECTORIES}")

add_subdirectory(pybind11)

file(GLOB minkowski_CXX_SOURCE_FILES
        "${PROJECT_SOURCE_DIR}/src/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        )
file(GLOB minkowski_CUDA_SOURCE_FILES
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        "${PROJECT_SOURCE_DIR}/src/*.cuh"
        )

file(GLOB minkowski_PYTORCH_LIBRARIES
        "${PYTORCH_LIBRARIES}/*.so"
        )

ADD_LIBRARY(MinkowskiEngineBackend STATIC ${minkowski_CXX_SOURCE_FILES})
CUDA_ADD_LIBRARY(libminkowskicu STATIC ${minkowski_CUDA_SOURCE_FILES})

set(CMAKE_CXX_FLAGS "-MMD -MP -ffast-math -funsafe-math-optimizations -fno-math-errno -DBATCH_FIRST=1 -fopenmp -fPIC\
    -fwrapv -std=c++14 -DDEBUG -g -O0\
    -I${CUDA_HOME}/include\
    ${PYTORCH_INCLUDES_I_CXX}\
    -DTORCH_API_INCLUDE_EXTENSION_H\
    -DTORCH_EXTENSION_NAME=minkowski\
    -D_GLIBCXX_USE_CXX11_ABI=0\
    -Wall\
    -Wcomment\
    -Wno-sign-compare\
    -Wno-deprecated-declarations")

#set(CMAKE_CXX_FLAGS "-I/home/ran/miniconda3/include/python3.6m -I/home/ran/miniconda3/include/python3.6m/.. -I/home/ran/.local/lib/python3.6/site-packages/torch/include -I/home/ran/.local/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -I/home/ran/.local/lib/python3.6/site-packages/torch/include/TH -I/home/ran/.local/lib/python3.6/site-packages/torch/include/THC -I./ -I/usr/local/cuda-10.2/include -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=minkowski -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wcomment -Wno-sign-compare -Wno-deprecated-declarations")

#message("######### CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ############")

if(NOT CPU_ONLY)
    target_link_libraries(libminkowskicu -lcudart -lcublas -lcusparse -lcaffe2_gpu -lc10_cuda)
    target_compile_options(libminkowskicu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -g -G -std=c++14 -ccbin=g++ -Xcompiler -fPIC -DDEBUG -g -O0 ${PYTORCH_INCLUDES_I} -I./ -I${CUDA_HOME}/include -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=minkowski -D_GLIBCXX_USE_CXX11_ABI=0 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 >)
    get_target_property(MAIN_CUDA_FLAGS libminkowskicu COMPILE_OPTIONS)
#    message("######### MAIN_CUDA_FLAGS ${MAIN_CUDA_FLAGS} ############")
endif()


# BLAS configuration: mkl, atlas, open, blas
set(BLAS openblas)
if("${BLAS}" STREQUAL "mkl")
# MKL
    target_link_libraries(libminkowskicu -lmkl_rt)
    target_link_libraries(MinkowskiEngineBackend -lmkl_rt)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKL")
    set(MKLROOT /opt/intel/mkl)
    set(BLAS_INCLUDE_DIRS "${MKLROOT}/include")
    set(BLAS_LIBRARY_DIRS "${MKLROOT}/lib ${MKLROOT}/lib/intel64")
elseif("${BLAS}" STREQUAL "openblas")
    target_link_libraries(libminkowskicu -lopenblas)
    target_link_libraries(MinkowskiEngineBackend -lopenblas)
elseif("${BLAS}" STREQUAL "blas")
    target_link_libraries(libminkowskicu -lblas)
    target_link_libraries(MinkowskiEngineBackend -lblas)
else()
    target_link_libraries(libminkowskicu -latlas)
    target_link_libraries(MinkowskiEngineBackend -latlas)
    EXECUTE_PROCESS ( COMMAND ${PYTHON} -c "import numpy.distutils.system_info as si; ai = si.atlas_info(); [print(p) for p in ai.get_lib_dirs()]" OUTPUT_VARIABLE ATLAS_PATH)
endif()

#message("######### CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ############")
#message("######### CMAKE_CXX_COMPILER ${CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES} ############")
#
#message("######### PYTHON_HEADER_DIR ${PYTHON_HEADER_DIR} ############")
#message("######### PYTHON_HEADER_DIR_PARENT_DIR ${PYTHON_HEADER_DIR_PARENT_DIR} ############")
#message("######### PYTORCH_INCLUDES ${PYTORCH_INCLUDES} ############")

target_link_libraries(libminkowskicu ${CUDA_LIBRARIES})
target_link_libraries(MinkowskiEngineBackend ${minkowski_PYTORCH_LIBRARIES} libminkowskicu -lstdc++ -lc10 -lcaffe2 -ltorch -ltorch_python -l_C)
target_link_libraries(MinkowskiEngineBackend pybind11::module)
