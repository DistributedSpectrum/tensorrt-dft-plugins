# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# DS modifications

cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME trt_dft_plugins)
project(${PROJECT_NAME})
set(CUDA_LIBRARY_PATH /usr/local/cuda/targets/aarch64-linux/lib/)
set(AARCH_64_LINUX_GNU_PATH /usr/lib/aarch64-linux-gnu/)
# find_package(CUDAToolkit 11.0 REQUIRED)
find_library(CUDART_LIBRARY cudart                 REQUIRED HINTS ${CUDA_LIBRARY_PATH})
find_library(CUDADEVRT_LIBRARY cudadevrt           REQUIRED HINTS ${CUDA_LIBRARY_PATH})
find_library(CUFFT_LIBRARY cufft                   REQUIRED HINTS ${CUDA_LIBRARY_PATH})
find_library(NVINFER_LIBRARY nvinfer               REQUIRED HINTS ${AARCH_64_LINUX_GNU_PATH})
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin REQUIRED HINTS ${AARCH_64_LINUX_GNU_PATH})
find_library(NVONNXPARSER_LIBRARY nvonnxparser     REQUIRED HINTS ${AARCH_64_LINUX_GNU_PATH})
find_library(NVMEDIA_LIBRARY nvmedia               REQUIRED HINTS ${AARCH_64_LINUX_GNU_PATH}/tegra)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
file(GLOB ${PROJECT_NAME}_sources ./*.cpp)
set(PROJECT_SOURCES ${${PROJECT_NAME}_sources})
set(TARGET_SUFFIX "")
set(TARGET_NAME ${PROJECT_NAME}${TARGET_SUFFIX})
add_library(${TARGET_NAME} SHARED ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME}  PRIVATE
    /usr/src/tensorrt/samples/common # TODO: cmakeify this
    /usr/local/cuda/include          # TODO: cmakeify this
)
target_link_libraries(${PROJECT_NAME}
              ${CUDART_LIBRARY}
                      ${CUDADEVRT_LIBRARY}
                      ${CUFFT_LIBRARY}
                      ${NVINFER_LIBRARY}
                      ${NVINFER_PLUGIN_LIBRARY}
                      ${NVONNXPARSER_LIBRARY}
                      ${NVMEDIA_LIBRARY}
                      )
