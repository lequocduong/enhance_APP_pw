cmake_minimum_required(VERSION 3.12)
project(test)

add_definitions( -std=c++11) 
add_definitions("-Wall")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
                    ${OpenCV_DIR}/include/
                     /home/lqd159753/C++/ncnn_Inference/ncnn/build/install/include/ncnn
                    )

link_directories(${OpenCV_DIR}/lib/
                 /home/lqd159753/C++/ncnn_Inference/ncnn/build/install/lib
                 )

aux_source_directory(. SRC_LIST)
add_executable(test ${SRC_LIST})
target_link_libraries(
  test
  ncnn
  opencv_dnn
  opencv_imgcodecs
  opencv_core
  opencv_highgui
  opencv_imgproc
  opencv_videoio
  avcodec
  avformat
  avutil
)

#set(OPENCV_PATH /opt/opencv)
#set(NCNN_PATH /opt/ncnn)

#include_directories(${OPENCV_PATH}/include/opencv4
#                    ${NCNN_PATH}/include/ncnn)

#link_directories(${OPENCV_PATH}/lib
#                 ${NCNN_PATH}/lib)