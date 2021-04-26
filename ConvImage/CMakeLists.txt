cmake_minimum_required(VERSION 2.8)

project( convImage )
find_package(CUDA QUIET REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

cuda_add_executable( convImage convImage.cu)
target_link_libraries( DisplayImage ${OpenCV_LIBS} )