# CMakeList.txt : CMake project for FaceDetect, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
project (face_detect VERSION 0.1 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeIncludes ${CMAKE_MODULE_PATH})

#create target definition
add_executable (${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/SRC/FaceDetect.cpp)

target_compile_features (${PROJECT_NAME} PRIVATE cxx_auto_type)

#add tensorflowmodel, videorender and caf libraries
target_link_libraries(${PROJECT_NAME} PUBLIC videorender)
target_link_libraries(${PROJECT_NAME} PUBLIC TensorflowModel)

target_compile_features (${PROJECT_NAME} PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W3 /WX)
else(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror)
endif(MSVC)

#mute modern cmake compatibility issue
cmake_policy(SET CMP0074 NEW)


#add boost
#!!!important. without this variable boost will fail to find static libraries
set(Boost_USE_STATIC_LIBS ON)

find_package(Boost 1.68.0 COMPONENTS fiber REQUIRED) 

if(Boost_FOUND)
  target_include_directories(${PROJECT_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE Boost::fiber)
endif()


#tensorflow MSVC-required defines
#we need them here too (tensorflowmodel has it already)
if (MSVC)
	target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
	target_compile_definitions(${PROJECT_NAME} PRIVATE MSVC)
endif(MSVC)


#add hdf5
#in line below I force to use native hdf5 config file instead of FindHDF5 from cmake distribution
find_package(HDF5 NAMES hdf5 NO_MODULE REQUIRED CXX )
if (HDF5_FOUND)
	target_link_libraries(${PROJECT_NAME} PUBLIC hdf5::hdf5-static hdf5::hdf5_cpp-static)
endif()


#[==[

find_package(CAF COMPONENTS core REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${CAF_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${CAF_LIBRARIES})

#CAF has bug which requires manual linkage of libraries below:
if(WIN32)
  target_link_libraries(${PROJECT_NAME} PUBLIC wsock32)
  target_link_libraries(${PROJECT_NAME} PUBLIC ws2_32)
  target_link_libraries(${PROJECT_NAME} PRIVATE Iphlpapi)
endif()
#]==]

# Add source to this project's executable.
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PRIVATE src)



# INSTALL section
#project binary
#install (TARGETS ${PROJECT_NAME} 
#        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
