file(GLOB_RECURSE SRC_FILES "src/*.cpp")
file(GLOB_RECURSE INC_FILES "include/*.h")

find_package(Boost 1.66.0 REQUIRED COMPONENTS thread system)

set(ABY_INSTALL_PREFIX "" CACHE PATH "Path to the ABY installation prefix")
set(ABY_ENCRYPTO_BUILD_INCLUDE "" CACHE PATH "Path to the ENCRYPTO_utils build include directory")
if(NOT ABY_INSTALL_PREFIX OR NOT ABY_ENCRYPTO_BUILD_INCLUDE)
	message(FATAL_ERROR
		"ABY_INSTALL_PREFIX and ABY_ENCRYPTO_BUILD_INCLUDE are required; "
		"pass both paths to CMake")
endif()

add_library(crypto_ass STATIC ${SRC_FILES} ${INC_FILES})
target_include_directories(crypto_ass PUBLIC
	include
	${CMAKE_SOURCE_DIR}/crypto/parameters/include
	${ABY_INSTALL_PREFIX}/include
	${ABY_ENCRYPTO_BUILD_INCLUDE})
target_link_libraries(crypto_ass PUBLIC
	model_lib
	common_lib
	${ABY_INSTALL_PREFIX}/lib/libaby.a
	${ABY_INSTALL_PREFIX}/lib/libotextension.a
	${ABY_INSTALL_PREFIX}/lib/libencrypto_utils.a
	${ABY_INSTALL_PREFIX}/lib/librelic_s.a
	Boost::thread
	Boost::system
	gmpxx
	gmp
	ssl
	crypto
	pthread
	dl)
