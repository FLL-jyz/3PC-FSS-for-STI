cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(STI)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_compile_options(-O3)

add_subdirectory(application)
add_subdirectory(common)
add_subdirectory(crypto)
add_subdirectory(secure_model)
add_subdirectory(external/cnpy)

file(GLOB TEST_SOURCES "tests/*.cpp")

include_directories(${CMAKE_SOURCE_DIR}/application/ ${CMAKE_SOURCE_DIR}/common/ ${CMAKE_SOURCE_DIR}/crypto/ ${CMAKE_SOURCE_DIR}/secure_model/ ${CMAKE_SOURCE_DIR}/external)

# 添加外部头文件路径
set(FSS_PATH "/path/to/EzPC/FSS")
include_directories(${FSS_PATH}/src)
include_directories(${FSS_PATH}/src/deps/cryptoTools)

add_executable(TFssformer main.cpp ${TEST_SOURCES})

target_link_libraries(TFssformer app_lib common_lib crypto_lib model_lib OpenMP::OpenMP_CXX  ${FSS_PATH}/build/lib/libfss.a ssl crypto pthread cnpy)
