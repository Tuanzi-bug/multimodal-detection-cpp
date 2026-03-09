#include <gtest/gtest.h>

#include <cmath>

#include "binary_io.h"
#include "config.h"
#include "hyperspectral_loader.h"
#include "spectral_library.h"
#include "types.h"

// Data paths inside the Docker container (/workspace maps to c++/).
static constexpr char SPEC_BIN[]     = "/workspace/data/spectral_lib/aircraft/aircraft1.bin";
static constexpr char HYPER_BIN[]    = "/workspace/data/hyperspectral/aircraft_1.bin";
static constexpr char SPEC_LIB_DIR[] = "/workspace/data/spectral_lib";

// ── config.h ────────────────────────────────────────────────────────────────

TEST(Config, Constants) {
    EXPECT_EQ(cfg::NUM_CLASSES,    5);
    EXPECT_EQ(cfg::SPECTRAL_BANDS, 129);
    EXPECT_EQ(cfg::INPUT_H,        640);
    EXPECT_EQ(cfg::INPUT_W,        640);
    EXPECT_FLOAT_EQ(cfg::ACE_THRESHOLD,  0.80f);
    EXPECT_FLOAT_EQ(cfg::CONF_THRESHOLD, 0.05f);
    EXPECT_FLOAT_EQ(cfg::NMS_IOU,        0.3f);
}

TEST(Config, AnchorCount) {
    EXPECT_EQ(cfg::ANCHORS.size(), 18u);
    // First anchor pair: 10, 13
    EXPECT_FLOAT_EQ(cfg::ANCHORS[0], 10.f);
    EXPECT_FLOAT_EQ(cfg::ANCHORS[1], 13.f);
    // Last anchor pair: 373, 326
    EXPECT_FLOAT_EQ(cfg::ANCHORS[16], 373.f);
    EXPECT_FLOAT_EQ(cfg::ANCHORS[17], 326.f);
}

TEST(Config, ClassNames) {
    EXPECT_STREQ(cfg::CLASS_NAMES[0], "ship");
    EXPECT_STREQ(cfg::CLASS_NAMES[1], "aircraft");
    EXPECT_STREQ(cfg::CLASS_NAMES[2], "roof");
    EXPECT_STREQ(cfg::CLASS_NAMES[3], "car");
    EXPECT_STREQ(cfg::CLASS_NAMES[4], "oiltank");
}

// ── types.h ─────────────────────────────────────────────────────────────────

TEST(Types, BoundingBoxDefaults) {
    BoundingBox b;
    EXPECT_FLOAT_EQ(b.top,    0.f);
    EXPECT_FLOAT_EQ(b.left,   0.f);
    EXPECT_FLOAT_EQ(b.bottom, 0.f);
    EXPECT_FLOAT_EQ(b.right,  0.f);
    EXPECT_FLOAT_EQ(b.score,  0.f);
    EXPECT_EQ(b.class_id, -1);
}

TEST(Types, HyperspectralCubeDefaults) {
    HyperspectralCube cube;
    EXPECT_EQ(cube.height, 0);
    EXPECT_EQ(cube.width,  0);
    EXPECT_EQ(cube.bands,  0);
}

// ── binary_io: spectral .bin ─────────────────────────────────────────────────

TEST(BinaryIo, SpectralBinShape) {
    auto raw = bio::read_spectral_bin(SPEC_BIN);
    EXPECT_EQ(raw.rows, 131u);
    EXPECT_EQ(raw.cols, 129u);
    EXPECT_EQ(raw.data.size(), 131u * 129u);
}

TEST(BinaryIo, SpectralBinDataFinite) {
    auto raw = bio::read_spectral_bin(SPEC_BIN);
    for (const auto v : raw.data) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// ── binary_io: hyperspectral .bin ────────────────────────────────────────────

TEST(BinaryIo, HyperBinBandCount) {
    auto raw = bio::read_hyper_bin(HYPER_BIN);
    EXPECT_GT(raw.height, 0u);
    EXPECT_GT(raw.width,  0u);
    EXPECT_EQ(raw.bands,  static_cast<uint32_t>(cfg::SPECTRAL_BANDS));
    EXPECT_EQ(raw.data.size(),
              static_cast<std::size_t>(raw.height) * raw.width * raw.bands);
}

TEST(BinaryIo, HyperBinDataFinite) {
    auto raw = bio::read_hyper_bin(HYPER_BIN);
    for (const auto v : raw.data) {
        EXPECT_TRUE(std::isfinite(v));
    }
}

// ── SpectralLibrary ──────────────────────────────────────────────────────────

TEST(SpectralLibrary, PerClassComponentCounts) {
    SpectralLibrary lib(SPEC_LIB_DIR);
    // Actual file counts: ship=6, aircraft=3, roof=3, car=5, oiltank=4
    EXPECT_EQ(lib.component_count(0), 6);  // ship
    EXPECT_EQ(lib.component_count(1), 3);  // aircraft
    EXPECT_EQ(lib.component_count(2), 3);  // roof
    EXPECT_EQ(lib.component_count(3), 5);  // car
    EXPECT_EQ(lib.component_count(4), 4);  // oiltank
}

TEST(SpectralLibrary, ComponentDimensions) {
    SpectralLibrary lib(SPEC_LIB_DIR);
    const auto& comp = lib.get(1, 0);  // aircraft, first component
    EXPECT_EQ(comp.mu.cols(),         cfg::SPECTRAL_BANDS);
    EXPECT_EQ(comp.whitening.rows(),  cfg::SPECTRAL_BANDS);
    EXPECT_EQ(comp.whitening.cols(),  cfg::SPECTRAL_BANDS);
    EXPECT_EQ(comp.signature.cols(),  cfg::SPECTRAL_BANDS);
}

TEST(SpectralLibrary, ComponentDataFinite) {
    SpectralLibrary lib(SPEC_LIB_DIR);
    const auto& comp = lib.get(1, 0);
    EXPECT_TRUE(comp.mu.allFinite());
    EXPECT_TRUE(comp.whitening.allFinite());
    EXPECT_TRUE(comp.signature.allFinite());
}

// ── HyperspectralLoader ───────────────────────────────────────────────────────

TEST(HyperspectralLoader, ShapeAndBands) {
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    EXPECT_GT(cube.height, 0);
    EXPECT_GT(cube.width,  0);
    EXPECT_EQ(cube.bands,  cfg::SPECTRAL_BANDS);
    EXPECT_EQ(cube.data.rows(), cfg::SPECTRAL_BANDS);
    EXPECT_EQ(cube.data.cols(), cube.width * cube.height);
}

// Verify w-major column ordering matches Python transpose(2,1,0) + reshape.
// Column j = pixel at (w = j / height, h = j % height).
TEST(HyperspectralLoader, WMajorColumnOrder) {
    auto raw = bio::read_hyper_bin(HYPER_BIN);
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);

    const int H = cube.height;
    const int W = cube.width;
    const int B = cube.bands;

    // Pixel (w=0, h=0) → column 0
    for (int b = 0; b < B; ++b) {
        EXPECT_DOUBLE_EQ(cube.data(b, 0),
                         raw.data[static_cast<std::size_t>(0) * W * B + 0 * B + b]);
    }

    // Pixel (w=0, h=1) → column 1
    if (H > 1) {
        for (int b = 0; b < B; ++b) {
            EXPECT_DOUBLE_EQ(cube.data(b, 1),
                             raw.data[static_cast<std::size_t>(1) * W * B + 0 * B + b]);
        }
    }

    // Pixel (w=1, h=0) → column H
    if (W > 1) {
        for (int b = 0; b < B; ++b) {
            EXPECT_DOUBLE_EQ(cube.data(b, H),
                             raw.data[static_cast<std::size_t>(0) * W * B + 1 * B + b]);
        }
    }
}

TEST(HyperspectralLoader, DataFinite) {
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    EXPECT_TRUE(cube.data.allFinite());
}
