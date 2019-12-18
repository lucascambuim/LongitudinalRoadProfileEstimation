# project
cmake_minimum_required (VERSION 2.6)
project (LongitudinalRoadProfileEstimation)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# directories
set (LIBVISO2_SRC_DIR src/libviso2/src)
set (PROFILE_SRC_DIR src)
#set (CVM_SRC_DIR src/cvmlib/cvm/src)




find_package( OpenCV REQUIRED )

# include directory
include_directories("${LIBVISO2_SRC_DIR}")
include_directories("${PROFILE_SRC_DIR}")
#include_directories("${CVM_SRC_DIR}")

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/cvmlib/cvm/src ${CMAKE_CURRENT_SOURCE_DIR}/src/cvmlib/lib64)
#src/cvmlib/cvm/src/
# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB LIBVISO2_SRC_FILES "src/libviso2/src/*.cpp")
FILE(GLOB PROFILE_SRC_FILES "src/*.cpp")
#FILE(GLOB CVM_SRC_FILES "src/cvmlib/cvm/*.cpp")


# make release version
set(CMAKE_BUILD_TYPE Release)

# demo program
#add_executable(viso2 ${CVM_SRC_FILES} ${LIBVISO2_SRC_FILES} ${PROFILE_SRC_FILES} main.cpp)
add_executable(viso2 ${LIBVISO2_SRC_FILES} ${PROFILE_SRC_FILES} main.cpp)
#add_executable(viso2 ${LIBVISO2_SRC_FILES})
target_link_libraries (viso2 png ${OpenCV_LIBS})

