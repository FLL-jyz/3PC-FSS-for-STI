cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(puma LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_compile_options(-O3)

set(FSS_PATH "" CACHE PATH "Path to the EzPC FSS source tree")
if(NOT FSS_PATH)
	message(FATAL_ERROR "FSS_PATH is required; pass -DFSS_PATH=/path/to/EzPC/FSS")
endif()

include_directories(
	${CMAKE_SOURCE_DIR}/application
	${CMAKE_SOURCE_DIR}/common
	${CMAKE_SOURCE_DIR}/crypto
	${CMAKE_SOURCE_DIR}/secure_model
	${CMAKE_SOURCE_DIR}/external
	${FSS_PATH}/src
	${FSS_PATH}/src/deps/cryptoTools
)

add_subdirectory(application)
add_subdirectory(common)
add_subdirectory(crypto)
add_subdirectory(secure_model)
add_subdirectory(external/cnpy)

file(GLOB TEST_SOURCES "tests/*.cpp")

add_executable(puma main.cpp ${TEST_SOURCES})

target_link_libraries(
	puma
	app_lib
	common_lib
	crypto_lib
	model_lib
	OpenMP::OpenMP_CXX
	${FSS_PATH}/build/lib/libfss.a
	ssl
	crypto
	pthread
	cnpy
)
