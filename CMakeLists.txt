cmake_minimum_required(VERSION 3.22)
project(tinyvcs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS "-O0")

include_directories(./include)
link_directories(/usr/local/lib)
link_directories(/usr/lib/x86_64-linux-gnu)

####ffmpeg#####
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h)
find_library(AVDEVICE_LIBRARY avdevice)

find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
find_library(SWSCALE_LIBRARY swscale swscale)
###############

####glog#####
include_directories(/usr/local/include/glog)
###############

####fmt#####
find_package(fmt REQUIRED)
###############

####opencv#####
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
###############

####libyuv#####
include_directories(/usr/local/include/libyuv)
###############

####websocket\boost#####
set(BOOST_INCLUDE_DIR /usr/local/include/boost/)
set(BOOST_INCLUDE_SUB_DIR /usr/local/include/)
include_directories(${BOOST_INCLUDE_DIR} ${BOOST_INCLUDE_SUB_DIR})
###############


####json#####
include_directories(/usr/local/include/nlohmann)
#############

find_package(CUDA REQUIRED)
find_package(ZLIB REQUIRED)

####json#####
find_library(LIBNVINFER NAME nvinfer HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVINFER_PLUGIN NAME nvinfer_plugin HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVPARSERS NAME nvparsers HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVONNXPARSER NAME nvonnxparser HINTS ${TENSORRT_LIB_PATH} REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_INCLUDE_PATH})

include(cmake/CUDA_utils.cmake)

if (GPU_ARCH)
    set(CUDA_targeted_archs ${GPU_ARCH})
    CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
else ()
    # Discover what architectures does nvcc support
    CUDA_find_supported_arch_values(CUDA_supported_archs ${CUDA_known_archs})
    set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
    list(SORT CUDA_TARGET_ARCHS_SORTED)
    CUDA_find_supported_arch_values(CUDA_targeted_archs ${CUDA_TARGET_ARCHS_SORTED})
    if (NOT CUDA_targeted_archs)
        message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}}) \
      is supported by nvcc, please set appropriate arch via -DGPU_ARCH=XX")
    endif ()
    CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
endif ()

message(STATUS "
=============> Final Config <===============
TensorRT headers: ${TENSORRT_INCLUDE_DIR}
TensorRT library: ${LIBNVINFER}
Generated gencode flags: ${CUDA_gencode_flags}

for -DGPU_ARCH, refer to
https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
============================================
")
# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)

aux_source_directory(source/websocket DIR_WEBSOCKET_SRCS)
aux_source_directory(source/json DIR_JSON_SRCS)
aux_source_directory(source/tensorrt DIR_TENSOR_SRCS)
aux_source_directory(source DIR_SRCS)
set(ALL_SRCS ${DIR_SRCS} ${DIR_WEBSOCKET_SRCS} ${DIR_JSON_SRCS} ${DIR_TENSOR_SRCS} ${SAMPLE_SRCS} samples/engine_test.cpp)
add_executable(tinyvcs ${ALL_SRCS})
target_include_directories(tinyvcs PRIVATE ${SWSCALE_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
target_link_libraries(tinyvcs PRIVATE ${LIBNVONNXPARSER} cudart ${ZLIB_LIBRARIES} ${LIBNVINFER} ${LIBNVINFER_PLUGIN} ${LIBNVPARSERS} yuv jpeg boost_chrono boost_system boost_random swscale ${SWSCALE_LIBRARY} ${OpenCV_LIBS} ${AVSWS_LIBRARY} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY} pthread glog fmt)
