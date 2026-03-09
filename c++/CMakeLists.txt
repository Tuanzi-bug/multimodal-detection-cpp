cmake_minimum_required(VERSION 3.16)
project(multimodal_detect CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# Main executable (populated from Phase 2 onwards)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

if(SOURCES)
    add_executable(multimodal_detect ${SOURCES})

    target_include_directories(multimodal_detect PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${VIVANTE_SDK_INCLUDE}"
        "${OPENCV_INCLUDE}"
    )

    target_compile_options(multimodal_detect PRIVATE
        -Wall -Wextra -fPIC
    )

    target_link_directories(multimodal_detect PRIVATE
        "${VIVANTE_SDK_LIBDIR}"
        "${OPENCV_LIBDIR}"
    )

    target_link_libraries(multimodal_detect PRIVATE
        Eigen3::Eigen
        OpenVX OpenVXU CLC VSC SPIRV_viv GAL ovxlib NNArchPerf ArchModelSw
        opencv_imgproc opencv_core
        pthread m
    )
else()
    message(STATUS "No source files found in src/ — add sources (Phase 2+) to enable build target")
endif()

# ---------------------------------------------------------------------------
# Unit tests (added in Phase 7)
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/CMakeLists.txt")
    enable_testing()
    add_subdirectory(test)
endif()
