cmake_minimum_required(VERSION 3.6)
project(patch_generator)

set(CMAKE_CXX_STANDARD 11)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# useful macros
include(${PROJECT_SOURCE_DIR}/cmake/macros.cmake)

#
#Find modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

## OpenCV
find_package(OpenCV 3.0 REQUIRED)
if(OpenCV_INCLUDE_DIRS)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})
endif(OpenCV_INCLUDE_DIRS)

## Eigen3
find_package(Eigen3 REQUIRED)
if(EIGEN3_INCLUDE_DIR)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif(EIGEN3_INCLUDE_DIR)

find_package(PCL 1.8 REQUIRED)
if(PCL_INCLUDE_DIRS)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
endif(PCL_INCLUDE_DIRS)

set(CMAKE_CXX_STANDARD 11)

# TARGETS--------------------------------------------
include_directories(${PROJECT_SOURCE_DIR}/InSegLib)
link_directories(${LIBRARY_OUTPUT_PATH})

set(SOURCE_FILES main.cpp)
add_executable(patch_generator ${SOURCE_FILES})
target_link_libraries(patch_generator ${PCL_LIBRARIES} ${OpenCV_LIBS})

set(SOURCE_FILES main_randomKP.cpp)
add_executable(patch_randomKP ${SOURCE_FILES})
target_link_libraries(patch_randomKP ${PCL_LIBRARIES} ${OpenCV_LIBS} InSegLib)

set(SOURCE_FILES testing_projection.cpp)
add_executable(patch_testing ${SOURCE_FILES})
target_link_libraries(patch_testing ${PCL_LIBRARIES} ${OpenCV_LIBS} InSegLib)