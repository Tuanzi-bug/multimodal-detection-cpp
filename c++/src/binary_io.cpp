#include "binary_io.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace bio {

namespace {

struct FileCloser {
    void operator()(std::FILE* f) const { if (f) std::fclose(f); }
};

using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

FilePtr open_file(const std::string& path) {
    FilePtr fp(std::fopen(path.c_str(), "rb"));
    if (!fp) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    return fp;
}

void read_exact(std::FILE* fp, void* dst, std::size_t n, const std::string& path) {
    if (std::fread(dst, 1, n, fp) != n) {
        throw std::runtime_error("Unexpected end of file: " + path);
    }
}

// Decode a little-endian uint32 from 4 bytes.
uint32_t le32(const uint8_t* b) {
    return static_cast<uint32_t>(b[0])
         | static_cast<uint32_t>(b[1]) << 8
         | static_cast<uint32_t>(b[2]) << 16
         | static_cast<uint32_t>(b[3]) << 24;
}

}  // namespace

SpectralRaw read_spectral_bin(const std::string& path) {
    auto fp = open_file(path);

    uint8_t header[12];
    read_exact(fp.get(), header, 12, path);

    SpectralRaw raw;
    raw.rows              = le32(header + 0);
    raw.cols              = le32(header + 4);
    const uint32_t dsize  = le32(header + 8);

    if (dsize != 8) {
        throw std::runtime_error("Expected float64 (dtype_size=8) in: " + path);
    }

    const std::size_t n = static_cast<std::size_t>(raw.rows) * raw.cols;
    raw.data.resize(n);
    read_exact(fp.get(), raw.data.data(), n * sizeof(double), path);

    return raw;
}

HyperRaw read_hyper_bin(const std::string& path) {
    auto fp = open_file(path);

    uint8_t header[16];
    read_exact(fp.get(), header, 16, path);

    HyperRaw raw;
    raw.height            = le32(header + 0);
    raw.width             = le32(header + 4);
    raw.bands             = le32(header + 8);
    const uint32_t dsize  = le32(header + 12);

    if (dsize != 8) {
        throw std::runtime_error("Expected float64 (dtype_size=8) in: " + path);
    }

    const std::size_t n = static_cast<std::size_t>(raw.height) * raw.width * raw.bands;
    raw.data.resize(n);
    read_exact(fp.get(), raw.data.data(), n * sizeof(double), path);

    return raw;
}

}  // namespace bio
