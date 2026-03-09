cmake_minimum_required(VERSION 3.16)
project(multimodal_detect CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# ccache (speeds up incremental builds by caching compiled objects)
# ---------------------------------------------------------------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
endif()

# ---------------------------------------------------------------------------
# Vivante VIP SDK (ARM NPU driver; headers and .so files from reference tree)
# ---------------------------------------------------------------------------
set(VIVANTE_SDK_INCLUDE
    "${CMAKE_CURRENT_SOURCE_DIR}/reference/yolov3_camera_dc/vip/driver/sdk/include"
    CACHE PATH "Vivante VIP SDK include directory")
set(VIVANTE_SDK_LIBDIR
    "${CMAKE_CURRENT_SOURCE_DIR}/reference/yolov3_camera_dc/vip/driver/sdk/drivers"
    CACHE PATH "Vivante VIP SDK shared-library directory")

# ---------------------------------------------------------------------------
# OpenCV (pre-installed on the ARM target; only headers needed for cross-build)
# ---------------------------------------------------------------------------
set(OPENCV_INCLUDE "/usr/include" CACHE PATH "OpenCV include directory")
set(OPENCV_LIBDIR  "/usr/lib/arm-linux-gnueabihf" CACHE PATH "OpenCV library directory")

# ---------------------------------------------------------------------------
# Eigen3 (header-only; installed on build host via libeigen3-dev)
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED NO_MODULE)

# ---------------------------------------------------------------------------
# Core library — all implementation sources except main.cpp (Phase 2+)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*[/\\]main\\.cpp$")

if(CORE_SOURCES)
    add_library(multimodal_core STATIC ${CORE_SOURCES})

    target_include_directories(multimodal_core
        PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
        PRIVATE "${VIVANTE_SDK_INCLUDE}" "${OPENCV_INCLUDE}"
    )

    target_compile_options(multimodal_core PRIVATE -Wall -Wextra -fPIC)

    target_link_libraries(multimodal_core PUBLIC Eigen3::Eigen)
else()
    message(STATUS "No source files found in src/ — add sources (Phase 2+) to enable build")
endif()

# ---------------------------------------------------------------------------
# Main executable — requires src/main.cpp (added in Phase 6)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(multimodal_detect "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

    target_link_directories(multimodal_detect PRIVATE
        "${VIVANTE_SDK_LIBDIR}"
        "${OPENCV_LIBDIR}"
    )

    target_link_libraries(multimodal_detect PRIVATE
        multimodal_core
        OpenVX OpenVXU CLC VSC SPIRV_viv GAL ovxlib NNArchPerf ArchModelSw
        opencv_imgproc opencv_core
        pthread m
    )
else()
    message(STATUS "src/main.cpp not found — add it (Phase 6) to enable the main executable")
endif()

# ---------------------------------------------------------------------------
# Unit tests (native build only — ARM cross-compiled binaries cannot run on host)
# ---------------------------------------------------------------------------
if(NOT CMAKE_CROSSCOMPILING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
    enable_testing()
    add_subdirectory(test)
endif()
