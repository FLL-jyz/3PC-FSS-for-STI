cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(STI)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_compile_options(-O3)

set(EZPC_ROOT "${CMAKE_SOURCE_DIR}/../EzPC" CACHE PATH "Path to the EzPC checkout")
get_filename_component(EZPC_ROOT "${EZPC_ROOT}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
add_subdirectory(FSS)

add_subdirectory(application)
add_subdirectory(common)
add_subdirectory(crypto)
add_subdirectory(secure_model)
add_subdirectory(external/cnpy)

file(GLOB TEST_SOURCES "tests/*.cpp")

include_directories(${CMAKE_SOURCE_DIR}/application/ ${CMAKE_SOURCE_DIR}/common/ ${CMAKE_SOURCE_DIR}/crypto/ ${CMAKE_SOURCE_DIR}/secure_model/ ${CMAKE_SOURCE_DIR}/external)

add_executable(Triforce main.cpp ${TEST_SOURCES})

target_link_libraries(Triforce app_lib common_lib crypto_lib model_lib OpenMP::OpenMP_CXX fss ssl crypto cnpy)
