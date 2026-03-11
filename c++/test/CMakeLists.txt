cmake_minimum_required(VERSION 3.16)

# GoogleTest is pre-installed via libgtest-dev in the Docker image.
# The source lives at /usr/src/googletest; build it as a sub-project.
add_subdirectory(/usr/src/googletest gtest_build EXCLUDE_FROM_ALL)

set(PHASE2_SOURCES
    ../src/binary_io.cpp
    ../src/spectral_library.cpp
    ../src/hyperspectral_loader.cpp
)

add_executable(test_phase2 test_phase2.cpp ${PHASE2_SOURCES})

target_include_directories(test_phase2 PRIVATE
    ../include
)

target_compile_options(test_phase2 PRIVATE -Wall -Wextra)

target_link_libraries(test_phase2 PRIVATE
    gtest
    gtest_main
    Eigen3::Eigen
)

add_test(NAME phase2 COMMAND test_phase2)

# ---------------------------------------------------------------------------
# Phase 3 — normalization + ACE detector
# ---------------------------------------------------------------------------
set(PHASE3_SOURCES
    ../src/binary_io.cpp
    ../src/spectral_library.cpp
    ../src/hyperspectral_loader.cpp
    ../src/normalize.cpp
    ../src/ace_detector.cpp
)

add_executable(test_phase3 test_phase3.cpp ${PHASE3_SOURCES})

target_include_directories(test_phase3 PRIVATE
    ../include
)

target_compile_options(test_phase3 PRIVATE -Wall -Wextra)

target_link_libraries(test_phase3 PRIVATE
    gtest
    gtest_main
    Eigen3::Eigen
)

add_test(NAME phase3 COMMAND test_phase3)
