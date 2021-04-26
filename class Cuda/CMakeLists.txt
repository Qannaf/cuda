cmake_minimum_required(VERSION 2.8.12)

project(testOCT)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#+------------------------------
# Add the headers(/*.h)
set(hdr "include/")
include_directories(${hdr})

#+------------------------------
# Add the sources (/*.cpp)
file(GLOB USER_CU "src/*.cpp" "src/*.cu"  )


# Thorlabs: SpectralRadard lib on the OCT computer
set(SPECTRALRADAR_DIR "C:/Program Files/Thorlabs/SpectralRadar" CACHE PATH "Location of libraries" FORCE)

include_directories(${SPECTRALRADAR_DIR}/include)
link_directories(${SPECTRALRADAR_DIR}/lib)
find_package(Eigen3 3.3.7)
find_package(PCL 1.9 REQUIRED)
find_package(CUDA)
message(${EIGEN3_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})



# VTK library
set(VTK_DIR "C:/Program Files/PCL/3rdParty/VTK" CACHE PATH "Location of libraries" FORCE)

include_directories(${VTK_DIR}/include)
link_directories(${VTK_DIR}/lib/vtk-8.2)
message (STATUS "VTK_VERSION: ${VTK_VERSION}")


#+------------------------------
# ViSP library
set(VISP_DIR "C:/visp/build" CACHE PATH "Location of libraries" FORCE)

find_package(VISP 3.0 REQUIRED)

include_directories(${VISP_INCLUDE_DIRS})
link_directories(${VISP_LIBRARY_DIRS})
link_directories(${VISP_DIR}/x64/vc16/bin/Debug)
link_directories(${VISP_DIR}/x64/vc16/bin/Release)

message (STATUS "VISP_VERSION: ${VISP_VERSION}")

#+------------------------------
# Add 1394camera library
set(1394CAMERA_DIR "C:/Program Files (x86)/CMU/1394Camera" CACHE PATH "Location of libraries" FORCE)
set(1394CAMERA_LIBS "1394camera.lib")
set(1394CAMERA_LIBS_DYN "1394camera.dll")

include_directories(${1394CAMERA_DIR}/include CACHE PATH "Location of libraries" FORCE)
link_directories(${1394CAMERA_DIR}/lib64/x64)
link_directories(${1394CAMERA_DIR}/bin-x64/x64)


link_libraries(${1394CAMERA_LIBS_DYN})

LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#+------------------------------
# Add OpenCV library
set(CMAKE_PREFIX_PATH "C:/opencv-4.5.1/build")
set(OpenCV_DIR "C:/opencv-4.5.1/build" CACHE PATH "Location of libraries" FORCE)
set(OpenCV_INCLUDE_DIR "C:/opencv-4.5.1/build/include" CACHE PATH "Location of libraries" FORCE)

include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_DIR}/x64/vc14/lib) 

message (STATUS "OPENCV_VERSION: ${OPENCV_VERSION}")


# pthread
set(PTHREAD_DIR "C:/Program Files/pthread/Pre-built.2" CACHE PATH "Location of libraries" FORCE)
set(PTHREAD_LIBS "pthreadVC2.lib")
set(PTHREAD_LIBS_DYN "pthreadVC2.dll")

include_directories(${PTHREAD_DIR}/include)
link_directories(${PTHREAD_DIR}/lib/x64)
link_directories(${PTHREAD_DIR}/dll/x64)


cuda_add_executable(${PROJECT_NAME} ${USER_CU})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)




target_include_directories (${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})

target_link_libraries (${PROJECT_NAME} SpectralRadar ${PCL_LIBRARIES} 
						${1394CAMERA_LIBS} ${1394CAMERA_LIBS_DYN}
						${VISP_LIBRARIES} ${OpenCV_LIBS} ${VTK_LIBRARIES}
						${PTHREAD_LIBS} ${PTHREAD_LIBS_DYN}
						)
						



set( PCL_LIB_DLL_DEBUG "C:/Program Files/PCL/bin/pcl_commond.dll" 
		"C:/Program Files/PCL/bin/pcl_kdtreed.dll"
		"C:/Program Files/PCL/bin/pcl_searchd.dll"
		"C:/Program Files/PCL/bin/pcl_octreed.dll"
		"C:/Program Files/PCL/bin/pcl_visualizationd.dll"
		"C:/Program Files/PCL/bin/pcl_iod.dll"
		"C:/Program Files/PCL/bin/pcl_io_plyd.dll"
		"C:/Program Files/PCL/bin/pcl_featuresd.dll"
		"C:/Program Files/PCL/bin/pcl_surfaced.dll"
		"C:/Program Files/PCL/bin/pcl_filtersd.dll"
		"C:/Program Files/PCL/bin/pcl_sample_consensusd.dll"
		"C:/Program Files/PCL/bin/pcl_segmentationd.dll"
		"${1394CAMERA_DIR}/lib64/x64/1394camera.dll"
		"${PTHREAD_DIR}/dll/x64/pthreadVC2.dll" 
		"${VISP_DIR}/x64/vc16/bin/Debug/visp_core331d.dll" 
		"${VISP_DIR}/x64/vc16/bin/Debug/visp_vs331d.dll" 
		"${VISP_DIR}/x64/vc16/bin/Debug/visp_visual_features331d.dll" 
		"${VISP_DIR}/x64/vc16/bin/Debug/visp_me331d.dll" 
		"${VISP_DIR}/x64/vc16/bin/Debug/visp_blob331d.dll" 
		"${VISP_DIR}/x64/vc16/bin/Debug/visp_gui331d.dll" 
		"C:/Program Files/libxml2/libxml2.dll" )
	message("Files .dll are ${PCL_LIB_DLL_DEBUG}" )	

	file(COPY ${PCL_LIB_DLL_DEBUG}
		 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)
	message("Done coping .dll files")
