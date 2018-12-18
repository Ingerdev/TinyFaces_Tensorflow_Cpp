cmake_minimum_required (VERSION 3.8)
project (videorender VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} src/run_me.cpp)

# set where to find additional cmake modules if any
# comment it out if not required
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeIncludes ${CMAKE_MODULE_PATH})
find_package(FFMPEG REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Add source to this project's executable.
target_include_directories(${PROJECT_NAME} PRIVATE ${FFMPEG_INCLUDE_DIRS})
target_compile_features (${PROJECT_NAME} PUBLIC cxx_std_17)

target_sources(${PROJECT_NAME} 
	PRIVATE
	${PROJECT_SOURCE_DIR}/src/BaseStreamDecoder.cpp
	${PROJECT_SOURCE_DIR}/src/BaseStreamsSelector.cpp	
	${PROJECT_SOURCE_DIR}/src/Delayer.cpp
	${PROJECT_SOURCE_DIR}/src/MainWrapper.cpp
	${PROJECT_SOURCE_DIR}/src/RenderToDisplay.cpp
	${PROJECT_SOURCE_DIR}/src/RenderToFile.cpp
	${PROJECT_SOURCE_DIR}/src/Seeker.cpp
	${PROJECT_SOURCE_DIR}/src/SingleVideoStreamSelector.cpp
	${PROJECT_SOURCE_DIR}/src/StreamDecodersManager.cpp
	${PROJECT_SOURCE_DIR}/src/StreamSelectionHelper.cpp
	${PROJECT_SOURCE_DIR}/src/StreamVideoDecoder.cpp
	${PROJECT_SOURCE_DIR}/src/VideoDecoder.cpp	
	${PROJECT_SOURCE_DIR}/src/VideoRenderImpl.cpp
	PUBLIC
	${PROJECT_SOURCE_DIR}/src/Exceptions.cpp
	${PROJECT_SOURCE_DIR}/src/VideoRender.cpp)

#add ffmpeg
target_link_libraries(${PROJECT_NAME} PRIVATE ${FFMPEG_LIBRARIES})


#add opencv
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

#mute modern cmake compatibility issue
cmake_policy(SET CMP0074 NEW)

#add boost
find_package(Boost 1.55.0)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

if(WIN32)	
	#ffmpeg dlls
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD        # Adds a post-build event to MyTest
		COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
			"${FFMPEG_PATH}/bin/avcodec-57.dll"
			"${FFMPEG_PATH}/bin/avdevice-57.dll"
			"${FFMPEG_PATH}/bin/avfilter-6.dll"
			"${FFMPEG_PATH}/bin/avformat-57.dll"
			"${FFMPEG_PATH}/bin/avutil-55.dll"
			"${FFMPEG_PATH}/bin/postproc-54.dll"
			"${FFMPEG_PATH}/bin/swresample-2.dll"
			"${FFMPEG_PATH}/bin/swscale-4.dll"# <--this is in-file
			${CMAKE_RUNTIME_OUTPUT_DIRECTORY}) 

	#opencv dlls
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD        # Adds a post-build event to MyTest
		COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
			"${OpenCV_LIB_PATH}/../bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}.dll"		
			${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif(WIN32)

# INSTALL section

#install tensorflow,ffmpeg and opencv dlls
#if (MSVC)
#	install (DIRECTORY "${FFMPEG_PATH}/bin/" DESTINATION ${CMAKE_INSTALL_PREFIX}
#          FILES_MATCHING PATTERN "*.dll")
#endif(MSVC)

#project binary - dont install library
#install (TARGETS ${PROJECT_NAME} 
#		 ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX})