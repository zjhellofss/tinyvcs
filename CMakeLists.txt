cmake_minimum_required(VERSION 3.22)
project(tinyvcs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(./include)
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
link_directories(/usr/local/lib)
###############

####glog#####
find_package(glog 0.6.0 REQUIRED)
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


add_executable(tinyvcs source/chain.cpp source/player.cpp source/safevec.cpp source/frame.cpp source/convert.cpp)
target_include_directories(tinyvcs PRIVATE ${SWSCALE_INCLUDE_DIR} ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
target_link_libraries(tinyvcs PRIVATE  swscale yuv ${SWSCALE_LIBRARY} ${OpenCV_LIBS} ${AVSWS_LIBRARY} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY} pthread glog fmt)
