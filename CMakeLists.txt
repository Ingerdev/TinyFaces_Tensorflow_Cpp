# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

#mute modern cmake compatibility issue
cmake_policy(SET CMP0074 NEW)

# Include sub-projects.
project (FaceDetectSolution VERSION 0.1 LANGUAGES CXX)
#set environment variable which will be used as destination for copy dlls in subprojects
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(tensorflowmodel)
add_subdirectory(videorender)
add_subdirectory(app)


#install
#install opencv dlls
if (MSVC)	
	install (DIRECTORY "${OpenCV_LIB_PATH}/../bin/" DESTINATION ${CMAKE_INSTALL_PREFIX}
          FILES_MATCHING PATTERN "*.dll")
endif(MSVC)
