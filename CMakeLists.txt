cmake_minimum_required(VERSION 3.1)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(pyopengm)


##############################
# Check and enable C++ 14
##############################

message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
if (NOT "${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+17")
    message(STATUS "Using c++ 14")
    set(CMAKE_CXX_STANDARD 14)
else()
    message(STATUS "Using c++ 17")
endif()

# make sure the compiler supports the requested c++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Not sure if we should set this flag
# set(CMAKE_CXX_EXTENSIONS OFF)

# set default build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

# NOTE whether we need to enable the old ABI depends
# on which ABI boost is compiled with.
# If you get boost linker errors try to enable / disable the old ABI
option(USE_OLD_ABI OFF)
if(USE_OLD_ABI)
    message(STATUS "Using old GCC ABI")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()


###############################
# Include necessary libraries
###############################

### Threads
find_package(Threads)

### boost
if (MSVC)
    SET(BOOST_ROOT "${CMAKE_PREFIX_PATH}/Library")
    SET(BOOST_LIBRARYDIR "${CMAKE_PREFIX_PATH}/Library/lib")
else()
    SET(BOOST_ROOT "${CMAKE_PREFIX_PATH}")
    SET(BOOST_LIBRARYDIR "${CMAKE_PREFIX_PATH}/lib")
endif()
SET(Boost_NO_SYSTEM_PATHS ON)
find_package(Boost 1.63.0 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# set opengm boost flags
add_definitions(-DWITH_BOOST)
add_definitions(-DWITH_BOOST_GRAPH)


### cplex
find_package(CPLEX REQUIRED)
include_directories(${CPLEX_INCLUDE_DIRS})
# set opengm cplex flags
add_definitions(-DWITH_CPLEX)
add_definitions(-DIL_STD)


### opengm
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/opengm/include")



###############################
# Python-bindings
###############################

find_package(xtensor REQUIRED)
include_directories(xtensor_INCLUDE_DIRS)

find_package(xtensor-python REQUIRED)
include_directories(xtensor-python_INCLUDE_DIRS)

find_package(pybind11 REQUIRED)

if (NOT "${CMAKE_CXX_FLAGS}" MATCHES "-std=c\\+\\+17")
    if(MSVC)
        set(PYBIND11_CPP_STANDARD /std:c++14)
    else()
        message(STATUS "Using c++ 14 for pybind")
        set(PYBIND11_CPP_STANDARD -std=c++14)
    endif()
else()
    if(MSVC)
        set(PYBIND11_CPP_STANDARD /std:c++17)
    else()
        message(STATUS "Using c++ 17 for pybind")
        set(PYBIND11_CPP_STANDARD -std=c++17)
    endif()
endif()
# numpy
find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})
message(STATUS "Using numpy from ${NUMPY_INCLUDE_DIRS}")


###############################
# Set-up and install
###############################

# find global headers
# file(GLOB_RECURSE headers include/*.hxx)
# file(GLOB_RECURSE headers ${CMAKE_INSTALL_PREFIX}/include/*.hxx)
# include_directories(include)

# add subdirectories
add_subdirectory(src)
