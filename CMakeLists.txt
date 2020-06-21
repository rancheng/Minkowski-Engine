cmake_minimum_required(VERSION 3.10)
project(minkowski)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(OpenBLAS REQUIRED)
find_package(Torch REQUIRED) # see also https://pytorch.org/tutorials/advanced/cpp_frontend.html
find_package(OpenMP REQUIRED)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()
#SET(GCC_COMPILE_FLAGS "-fwrapv -std=c++14")
find_package(Python COMPONENTS Interpreter Development)

set(CMAKE_CXX_FLAGS "-MMD -MP -ffast-math -funsafe-math-optimizations -fno-math-errno -DBATCH_FIRST=1 -fopenmp -fPIC\
    -fwrapv -std=c++14 -DDEBUG -g -O0\
    -I${Python_INCLUDE_DIRS}\
    -I${Python_INCLUDE_DIRS}/..\
    -I${Python_SITELIB}/torch/include\
    -I${Python_SITELIB}/torch/include/torch/csrc/api/include\
    -I${Python_SITELIB}/torch/include/TH\
    -I${Python_SITELIB}/torch/include/THC\
    -I./\
    -I/usr/local/cuda-10.2/include\
    -DTORCH_API_INCLUDE_EXTENSION_H\
    -DTORCH_EXTENSION_NAME=minkowski\
    -D_GLIBCXX_USE_CXX11_ABI=0\
    -Wall\
    -Wcomment\
    -Wno-sign-compare\
    -Wno-deprecated-declarations")

set(CMAKE_CUDA_FLAGS "-g -G -std=c++14 -ccbin=g++ -Xcompiler -fPIC -DDEBUG -g -O0\
    -I${Python_INCLUDE_DIRS}\
    -I${Python_INCLUDE_DIRS}/..\
    -I${Python_SITELIB}/torch/include\
    -I${Python_SITELIB}/torch/include/torch/csrc/api/include\
    -I${Python_SITELIB}/torch/include/TH\
    -I${Python_SITELIB}/torch/include/THC\
    -I./\
    -I/usr/local/cuda-10.2/include\
    -DTORCH_API_INCLUDE_EXTENSION_H\
    -DTORCH_EXTENSION_NAME=minkowski\
    -D_GLIBCXX_USE_CXX11_ABI=0")


#message("######### Python_INCLUDE_DIRS ${Python_INCLUDE_DIRS} ############")
#message("######### Python_LIBRARY_DIRS ${Python_LIBRARY_DIRS} ############")
#message("######### Python_SITELIB ${Python_SITELIB} ############")
#
#message("######### CONDA_ROOT ${CONDA_ROOT} ############")
message("######### ${CMAKE_INSTALL_PREFIX} ############")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(pybind11)

set(minkowski_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/3rdparty/robin_hood.h
        ${PROJECT_SOURCE_DIR}/src/primitives/small_vector.hpp
        ${PROJECT_SOURCE_DIR}/src/broadcast.cpp
        ${PROJECT_SOURCE_DIR}/src/broadcast.cu
        ${PROJECT_SOURCE_DIR}/src/broadcast.cuh
        ${PROJECT_SOURCE_DIR}/src/broadcast.hpp
        ${PROJECT_SOURCE_DIR}/src/common.hpp
        ${PROJECT_SOURCE_DIR}/src/convolution.cpp
        ${PROJECT_SOURCE_DIR}/src/convolution.cu
        ${PROJECT_SOURCE_DIR}/src/convolution.cuh
        ${PROJECT_SOURCE_DIR}/src/convolution.hpp
        ${PROJECT_SOURCE_DIR}/src/convolution_transpose.cpp
        ${PROJECT_SOURCE_DIR}/src/coords_key.cpp
        ${PROJECT_SOURCE_DIR}/src/coords_manager.cpp
        ${PROJECT_SOURCE_DIR}/src/coords_manager.cu
        ${PROJECT_SOURCE_DIR}/src/coords_manager.hpp
        ${PROJECT_SOURCE_DIR}/src/coordsmap.cpp
        ${PROJECT_SOURCE_DIR}/src/coordsmap.hpp
        ${PROJECT_SOURCE_DIR}/src/gpu.cu
        ${PROJECT_SOURCE_DIR}/src/gpu.cuh
        ${PROJECT_SOURCE_DIR}/src/gpu_memory_manager.hpp
        ${PROJECT_SOURCE_DIR}/src/math_functions.cpp
        ${PROJECT_SOURCE_DIR}/src/math_functions.cu
        ${PROJECT_SOURCE_DIR}/src/math_functions.hpp
        ${PROJECT_SOURCE_DIR}/src/mkl_alternate.hpp
        ${PROJECT_SOURCE_DIR}/src/pooling_avg.cpp
        ${PROJECT_SOURCE_DIR}/src/pooling_avg.cu
        ${PROJECT_SOURCE_DIR}/src/pooling_avg.cuh
        ${PROJECT_SOURCE_DIR}/src/pooling_avg.hpp
        ${PROJECT_SOURCE_DIR}/src/pooling_global_avg.cpp
        ${PROJECT_SOURCE_DIR}/src/pooling_global_max.cpp
        ${PROJECT_SOURCE_DIR}/src/pooling_max.cpp
        ${PROJECT_SOURCE_DIR}/src/pooling_max.cu
        ${PROJECT_SOURCE_DIR}/src/pooling_max.hpp
        ${PROJECT_SOURCE_DIR}/src/pooling_transpose.cpp
        ${PROJECT_SOURCE_DIR}/src/pruning.cpp
        ${PROJECT_SOURCE_DIR}/src/pruning.cu
        ${PROJECT_SOURCE_DIR}/src/pruning.cuh
        ${PROJECT_SOURCE_DIR}/src/pruning.hpp
        ${PROJECT_SOURCE_DIR}/src/quantization.cpp
        ${PROJECT_SOURCE_DIR}/src/region.cpp
        ${PROJECT_SOURCE_DIR}/src/types.hpp
        ${PROJECT_SOURCE_DIR}/src/union.cpp
        ${PROJECT_SOURCE_DIR}/src/union.cu
        ${PROJECT_SOURCE_DIR}/src/union.cuh
        ${PROJECT_SOURCE_DIR}/src/union.hpp
        ${PROJECT_SOURCE_DIR}/src/utils.hpp
        )

#message("#######Project source file:  ${TORCH_INCLUDE_DIRS} ########")
#message("#######Project source file:  ${Python_SITELIB} ########")
#message("#######Project source file:  ${Python_INTERPRETER_ID} ########")

include_directories(
        ${TORCH_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
        ${Python_SITELIB}/torch/include
        ${Python_SITELIB}/torch/include/torch/csrc/api/include
        ${PROJECT_SOURCE_DIR}/pybind
        ${OpenBLAS_INCLUDE_DIR}
)


pybind11_add_module(minkowski ${PROJECT_SOURCE_DIR}/pybind/minkowski.cpp ${minkowski_SOURCE_FILES})
set_target_properties(minkowski PROPERTIES LINK_FLAGS "-pthread -fPIC -Wall -Wcomment -Wno-sign-compare\
-Wno-deprecated-declarations -Wl,-rpath= -Wl,--no-as-needed -Wl,--sysroot=/")
set(CMAKE_SHARED_LINKER_FLAGS "-L${Python_SITELIB}/torch/lib\
    -L/usr/local/cuda-10.2/lib64 -lstdc++ -lc10 -lcaffe2 -ltorch -ltorch_python -l_C -lcudart -lcublas\
    -lcusparse -lcaffe2_gpu -lc10_cuda -lopenblas")
