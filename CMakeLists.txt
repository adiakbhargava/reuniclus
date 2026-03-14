cmake_minimum_required(VERSION 3.24)
project(reuniclus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(Torch REQUIRED)

# Eigen3 – prefer the git submodule if present; otherwise fall back to a
# system/vcpkg installation.  To add the submodule:
#   git submodule add https://gitlab.com/libeigen/eigen.git third_party/eigen
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/eigen/CMakeLists.txt")
    add_subdirectory(third_party/eigen EXCLUDE_FROM_ALL)
else()
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()

# Google Test
add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)

# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)

# liblsl – header-only interface; link the prebuilt library if found
find_library(LSL_LIB lsl HINTS ${CMAKE_SOURCE_DIR}/third_party/liblsl/lib)
find_path(LSL_INCLUDE lsl_cpp.h HINTS ${CMAKE_SOURCE_DIR}/third_party/liblsl/include)

# ── Compiler flag helper ───────────────────────────────────────────────────────
# GCC/Clang: -mavx512f enables AVX-512 and defines __AVX512F__.
#            -O3 -ffast-math for full optimisation.
# MSVC:      /arch:AVX2 enables AVX2 (256-bit SIMD; AVX-512 requires dedicated
#            hardware and /arch:AVX512 – not universally available).
#            /O2 /fp:fast are the MSVC equivalents of -O3 -ffast-math.
#            /W4 for high-warning level.
set(REUNICLUS_COMPILE_OPTIONS
    $<$<CXX_COMPILER_ID:GNU,Clang>:-mavx512f -O3 -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2 /O2 /fp:fast /W4>
)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(reuniclus
    src/main.cpp
)

target_include_directories(reuniclus
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${LSL_INCLUDE}
)

target_link_libraries(reuniclus
    PRIVATE
        ${TORCH_LIBRARIES}
        Eigen3::Eigen
        $<$<BOOL:${LSL_LIB}>:${LSL_LIB}>
        $<$<PLATFORM_ID:Windows>:ws2_32>
)

target_compile_options(reuniclus PRIVATE ${REUNICLUS_COMPILE_OPTIONS})

# ── Unit tests ────────────────────────────────────────────────────────────────
enable_testing()

add_executable(reuniclus_tests
    tests/test_spsc.cpp
    tests/test_signal.cpp
    tests/test_kalman.cpp
    tests/test_channel_selector.cpp
)

target_include_directories(reuniclus_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(reuniclus_tests
    PRIVATE
        GTest::gtest_main
        Eigen3::Eigen
        ${TORCH_LIBRARIES}
)

target_compile_options(reuniclus_tests PRIVATE ${REUNICLUS_COMPILE_OPTIONS})

include(GoogleTest)
gtest_discover_tests(reuniclus_tests)

# ── Benchmarks ────────────────────────────────────────────────────────────────
add_executable(reuniclus_benchmarks
    benchmarks/bench_spsc.cpp
    benchmarks/bench_signal.cpp
)

target_include_directories(reuniclus_benchmarks
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(reuniclus_benchmarks
    PRIVATE
        benchmark::benchmark_main
        Eigen3::Eigen
        ${TORCH_LIBRARIES}
)

target_compile_options(reuniclus_benchmarks PRIVATE ${REUNICLUS_COMPILE_OPTIONS})
