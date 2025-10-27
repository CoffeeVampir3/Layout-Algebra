cmake_minimum_required(VERSION 3.28)
project(amx_project CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TARGET_NAME algebrademo)

file(GLOB_RECURSE MODULE_FILES "${CMAKE_SOURCE_DIR}/modules/*.cppm")

add_executable(${TARGET_NAME})
target_sources(${TARGET_NAME} PRIVATE main.cpp)

set(ALL_MODULES ${MODULE_FILES} ${TEST_FILES})
if(ALL_MODULES)
    target_sources(${TARGET_NAME} PUBLIC FILE_SET CXX_MODULES FILES ${ALL_MODULES})
endif()

target_link_options(${TARGET_NAME} PRIVATE)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(${TARGET_NAME} PRIVATE -O3)
endif()
