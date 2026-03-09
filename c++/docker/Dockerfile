FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Cross-compilation toolchain + build tools + Eigen3 (header-only)
# libopencv-dev provides architecture-neutral headers; the ARM target
# ships OpenCV pre-installed, so cross-linking uses the target's .so files.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-arm-linux-gnueabihf \
        g++-arm-linux-gnueabihf \
        cmake \
        make \
        libeigen3-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
