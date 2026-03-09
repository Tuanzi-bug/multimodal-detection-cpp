# syntax=docker/dockerfile:1
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install packages. BuildKit cache mounts keep downloaded .deb files and apt
# index across image rebuilds — no re-downloading when only code changes.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        gcc-arm-linux-gnueabihf \
        g++-arm-linux-gnueabihf \
        build-essential \
        cmake \
        ninja-build \
        make \
        libeigen3-dev \
        libopencv-dev \
        libgtest-dev \
        ccache

# Build and install GoogleTest (baked into image layer; only re-runs when the
# layer above changes).
RUN cmake -S /usr/src/googletest \
          -B /tmp/gtest_build \
          -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/gtest_build \
    && cmake --install /tmp/gtest_build \
    && rm -rf /tmp/gtest_build

# Point ccache to a directory that will be bind-mounted at runtime so compiled
# objects survive container restarts.
ENV CCACHE_DIR=/ccache

WORKDIR /workspace
