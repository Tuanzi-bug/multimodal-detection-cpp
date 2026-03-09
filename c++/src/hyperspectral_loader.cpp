#include "hyperspectral_loader.h"

#include <stdexcept>

#include "binary_io.h"
#include "config.h"

HyperspectralCube load_hyperspectral(const std::string& path) {
    auto raw = bio::read_hyper_bin(path);

    if (raw.bands != static_cast<uint32_t>(cfg::SPECTRAL_BANDS)) {
        throw std::runtime_error("Band count mismatch in: " + path);
    }

    const int H = static_cast<int>(raw.height);
    const int W = static_cast<int>(raw.width);
    const int B = static_cast<int>(raw.bands);

    // Raw layout: raw.data[h * W * B + w * B + b]
    //
    // Python: np.transpose(data, (2,1,0))   → shape [B, W, H]
    //         np.reshape(..., (B, -1))       → shape [B, W*H], C-order → H varies fastest
    //         so column j = w*H + h          → pixel at (w = j/H, h = j%H)
    //
    // Target: mat(b, j) where j = w * H + h
    Eigen::MatrixXd mat(B, W * H);

    for (int w = 0; w < W; ++w) {
        for (int h = 0; h < H; ++h) {
            const int col = w * H + h;
            const std::size_t base = static_cast<std::size_t>(h) * W * B
                                   + static_cast<std::size_t>(w) * B;
            for (int b = 0; b < B; ++b) {
                mat(b, col) = raw.data[base + b];
            }
        }
    }

    HyperspectralCube cube;
    cube.height = H;
    cube.width  = W;
    cube.bands  = B;
    cube.data   = std::move(mat);
    return cube;
}
