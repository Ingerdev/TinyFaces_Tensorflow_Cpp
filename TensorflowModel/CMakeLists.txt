cmake_minimum_required (VERSION 3.8)
project (TensorflowModel VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} src/main.cpp)

target_compile_features (${PROJECT_NAME} PRIVATE cxx_std_17)

# Add source to this project's executable.
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_sources(${PROJECT_NAME} 
	PRIVATE 
	${PROJECT_SOURCE_DIR}/src/HDF5DatasetWrapper.cpp
	${PROJECT_SOURCE_DIR}/src/ImageParser.cpp
	${PROJECT_SOURCE_DIR}/src/ModelRunner.cpp
	${PROJECT_SOURCE_DIR}/src/TestOps.cpp
	${PROJECT_SOURCE_DIR}/src/tf_helpers.cpp
	${PROJECT_SOURCE_DIR}/src/TfBlocks.cpp
	${PROJECT_SOURCE_DIR}/src/TfH5ModelLoader)

find_package(tensorflow REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${Tensorflow_INCLUDE_DIRS})

#tensorflow MSVC-required defines
if (MSVC)
	target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
	target_compile_definitions(${PROJECT_NAME} PRIVATE MSVC)
endif(MSVC)

#add tensorflow library
target_link_libraries(${PROJECT_NAME} PUBLIC tensorflow)

#in line below I force to use native hdf5 config file instead of FindHDF5 from cmake distribution
find_package(HDF5 NAMES hdf5 NO_MODULE REQUIRED CPP)

#add opencv
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

#add hdf5
target_link_libraries(${PROJECT_NAME} PUBLIC hdf5::hdf5-static)

#add videorender library
target_link_libraries(${PROJECT_NAME} PRIVATE videorender)





#post-build dll copying
if(WIN32)	
	#tensorflow dlls
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD        # Adds a post-build event to MyTest
		COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
			"${tensorflow_DIR}/../../bin/tensorflow.dll"			
			${CMAKE_RUNTIME_OUTPUT_DIRECTORY}) 

	#hdf5 dlls
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD        # Adds a post-build event to MyTest
		COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
			"${HDF5_DIR}/../../bin/hdf5.dll"		
			"${HDF5_DIR}/../../bin/hdf5_cpp.dll"		
			${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif(WIN32)



# INSTALL section
if (MSVC)
#tensorflow dlls
	install (DIRECTORY ${tensorflow_DIR}/../../bin/ DESTINATION ${CMAKE_INSTALL_PREFIX}
          FILES_MATCHING PATTERN "*.dll")
#hdf5 dlls
	install (DIRECTORY ${HDF5_DIR}/../../bin/ DESTINATION ${CMAKE_INSTALL_PREFIX}
          FILES_MATCHING PATTERN "*.dll")
endif(MSVC)

#project binary - should we install library? dont think so.
#install (TARGETS ${PROJECT_NAME} 
#         ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX})