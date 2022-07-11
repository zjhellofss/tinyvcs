cmake_minimum_required(VERSION 3.22)
project(tinyvcs CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG  "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIR})

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

####benchmark####
find_package(benchmark REQUIRED)
#################

find_package(CUDA REQUIRED)
find_package(ZLIB REQUIRED)


####tensorrt#####
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)

###openmp####
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

aux_source_directory(source/websocket DIR_WEBSOCKET_SRCS)
aux_source_directory(source/json DIR_JSON_SRCS)
aux_source_directory(source/cuda DIR_CUDA_SRCS)
aux_source_directory(source/tensorrt DIR_TENSOR_SRCS)
aux_source_directory(source DIR_SRCS)

set(ALL_SRCS ${DIR_SRCS} ${DIR_WEBSOCKET_SRCS} ${DIR_JSON_SRCS} ${DIR_CUDA_SRCS} ${DIR_TENSOR_SRCS} ${SAMPLE_SRCS} samples/video_test.cpp include/cuda/post_processing.cuh source/cuda/post_processing.cu)
add_executable(tinyvcs ${ALL_SRCS})
target_include_directories(tinyvcs PRIVATE ${SWSCALE_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
target_link_libraries(tinyvcs PRIVATE nvinfer cudart gflags)
target_link_libraries(tinyvcs PRIVATE benchmark::benchmark benchmark::benchmark_main ${LIBNVONNXPARSER} cudart ${ZLIB_LIBRARIES} ${LIBNVPARSERS} yuv jpeg boost_chrono boost_system boost_random boost_program_options boost_filesystem swscale ${SWSCALE_LIBRARY} ${OpenCV_LIBS} ${AVSWS_LIBRARY} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY} pthread glog fmt)
