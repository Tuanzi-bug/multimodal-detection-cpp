#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bio {

// Spectral library .bin:
//   Header: 12 bytes, little-endian <uint32 rows, uint32 cols, uint32 dtype_size(=8)>
//   Data:   rows * cols float64 values in row-major order.
struct SpectralRaw {
    uint32_t rows;
    uint32_t cols;
    std::vector<double> data;  // row-major, size = rows * cols
};

SpectralRaw read_spectral_bin(const std::string& path);

// Hyperspectral image .bin:
//   Header: 16 bytes, little-endian <uint32 H, uint32 W, uint32 bands, uint32 dtype_size(=8)>
//   Data:   H * W * bands float64 values in [H, W, B] row-major order.
struct HyperRaw {
    uint32_t height;
    uint32_t width;
    uint32_t bands;
    std::vector<double> data;  // [H, W, B] row-major, size = height * width * bands
};

HyperRaw read_hyper_bin(const std::string& path);

}  // namespace bio
