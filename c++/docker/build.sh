#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
IMAGE_NAME="multimodal-detect-build"
BUILD_DIR="${PROJECT_DIR}/build"

docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

mkdir -p "${BUILD_DIR}"

docker run --rm \
    -v "${PROJECT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}" \
    bash -c "
        cmake -B build \
              -DCMAKE_TOOLCHAIN_FILE=/workspace/toolchain-arm.cmake \
              -DCMAKE_BUILD_TYPE=Release && \
        cmake --build build --parallel \$(nproc)
    "

echo "Build output: ${BUILD_DIR}/"
