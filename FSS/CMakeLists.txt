find_package(Threads REQUIRED)

set(FSS_EIGEN_ROOT "${EZPC_ROOT}/SCI/extern/eigen")
if(EXISTS "${FSS_EIGEN_ROOT}/Eigen/Core")
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${FSS_EIGEN_ROOT}")
    endif()
else()
    find_package(Eigen3 3.3 QUIET CONFIG)
    if(NOT TARGET Eigen3::Eigen)
        message(FATAL_ERROR
            "Eigen3 was not found. Expected it at '${FSS_EIGEN_ROOT}'. "
            "Initialize EzPC's SCI/extern/eigen dependency or install Eigen3.")
    endif()
endif()

file(GLOB FSS_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

add_library(fss STATIC
    ${FSS_HEADERS}
    src/2pcdpf.cpp
    src/add.cpp
    src/api.cpp
    src/api_varied.cpp
    src/comms.cpp
    src/dcf.cpp
    src/dpf.cpp
    src/groto_dpf.cpp
    src/input_prng.cpp
    src/lib.cpp
    src/mult.cpp
    src/prng.cpp
    src/pubdiv.cpp
    src/spline.cpp
    src/utils.cpp
    third_party/cryptoTools/cryptoTools/Common/Defines.cpp
    third_party/cryptoTools/cryptoTools/Common/Log.cpp
    third_party/cryptoTools/cryptoTools/Crypto/AES.cpp
    third_party/cryptoTools/cryptoTools/Crypto/PRNG.cpp
)

target_include_directories(fss PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cryptoTools"
)

target_link_libraries(fss PUBLIC
    Threads::Threads
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fss PRIVATE -maes -msse2 -msse4.1 -mpclmul -fPIC)
endif()
