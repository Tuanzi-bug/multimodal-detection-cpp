#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "ace_detector.h"
#include "config.h"
#include "hyperspectral_loader.h"
#include "normalize.h"
#include "spectral_library.h"
#include "types.h"

// Data paths inside the Docker container (/workspace maps to c++/).
static constexpr char SPEC_LIB_DIR[] = "/workspace/data/spectral_lib";
static constexpr char HYPER_BIN[]    = "/workspace/data/hyperspectral/aircraft_1.bin";

// ── normalize_columns ────────────────────────────────────────────────────────

TEST(Normalize, NormalColumnScaledTo01) {
    // Column with distinct values should be linearly scaled to [0, 1].
    Eigen::MatrixXd data(3, 1);
    data << 0.0, 5.0, 10.0;

    Eigen::MatrixXd out = ace::normalize_columns(data);

    EXPECT_NEAR(out(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(out(1, 0), 0.5, 1e-12);
    EXPECT_NEAR(out(2, 0), 1.0, 1e-12);
}

TEST(Normalize, NegativeValuesScaledTo01) {
    Eigen::MatrixXd data(3, 1);
    data << -10.0, 0.0, 10.0;

    Eigen::MatrixXd out = ace::normalize_columns(data);

    EXPECT_NEAR(out(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(out(1, 0), 0.5, 1e-12);
    EXPECT_NEAR(out(2, 0), 1.0, 1e-12);
}

TEST(Normalize, AllZeroColumnStaysZero) {
    // Edge case 1: constant zero column → stays all-zero (sklearn minmax_scale default).
    Eigen::MatrixXd data(4, 1);
    data << 0.0, 0.0, 0.0, 0.0;

    Eigen::MatrixXd out = ace::normalize_columns(data);

    for (int r = 0; r < out.rows(); ++r) {
        EXPECT_NEAR(out(r, 0), 0.0, 1e-12);
    }
}

TEST(Normalize, ConstantNonZeroColumnBecomesAllOnes) {
    // Edge case 2: constant non-zero column → overridden to all-ones (Python behaviour).
    Eigen::MatrixXd data(4, 1);
    data << 5.0, 5.0, 5.0, 5.0;

    Eigen::MatrixXd out = ace::normalize_columns(data);

    for (int r = 0; r < out.rows(); ++r) {
        EXPECT_NEAR(out(r, 0), 1.0, 1e-12);
    }
}

TEST(Normalize, MultipleColumnsIndependentlyScaled) {
    // Two columns: first [0..10], second [2..4] — each scaled independently.
    Eigen::MatrixXd data(3, 2);
    data << 0.0, 2.0,
            5.0, 3.0,
           10.0, 4.0;

    Eigen::MatrixXd out = ace::normalize_columns(data);

    EXPECT_NEAR(out(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(out(2, 0), 1.0, 1e-12);
    EXPECT_NEAR(out(0, 1), 0.0, 1e-12);
    EXPECT_NEAR(out(2, 1), 1.0, 1e-12);
}

TEST(Normalize, OutputValuesInRange) {
    // Output of normalize_columns should always be in [0, 1].
    Eigen::MatrixXd data = Eigen::MatrixXd::Random(129, 100);  // random bands × pixels

    Eigen::MatrixXd out = ace::normalize_columns(data);

    EXPECT_GE(out.minCoeff(), -1e-12);
    EXPECT_LE(out.maxCoeff(),  1.0 + 1e-12);
}

TEST(Normalize, RealHyperspectralData) {
    // Apply normalize_columns to a real hyperspectral cube.
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    Eigen::MatrixXd out = ace::normalize_columns(cube.data);

    EXPECT_EQ(out.rows(), cube.bands);
    EXPECT_EQ(out.cols(), cube.width * cube.height);
    EXPECT_GE(out.minCoeff(), -1e-9);
    EXPECT_LE(out.maxCoeff(),  1.0 + 1e-9);
    EXPECT_TRUE(out.allFinite());
}

// ── detect_component ─────────────────────────────────────────────────────────

TEST(AceDetector, DetectComponentShape) {
    // Output map must have shape (height, width).
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    Eigen::MatrixXd norm = ace::normalize_columns(cube.data);

    SpectralLibrary lib(SPEC_LIB_DIR);
    const auto& comp = lib.get(1, 0);  // aircraft, component 0

    AceMap map = ace::detect_component(norm, comp, cube.height, cube.width);

    EXPECT_EQ(map.rows(), cube.height);
    EXPECT_EQ(map.cols(), cube.width);
}

TEST(AceDetector, DetectComponentValuesInRange) {
    // Each component map must be min-max normalised to [0, 1].
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    Eigen::MatrixXd norm = ace::normalize_columns(cube.data);

    SpectralLibrary lib(SPEC_LIB_DIR);
    const auto& comp = lib.get(1, 0);

    AceMap map = ace::detect_component(norm, comp, cube.height, cube.width);

    EXPECT_GE(map.minCoeff(), -1e-6f);
    EXPECT_LE(map.maxCoeff(),  1.0f + 1e-6f);
    EXPECT_TRUE(map.allFinite());
}

TEST(AceDetector, DetectComponentExtrema) {
    // After per-component min-max normalisation, min ≈ 0 and max ≈ 1.
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    Eigen::MatrixXd norm = ace::normalize_columns(cube.data);

    SpectralLibrary lib(SPEC_LIB_DIR);
    const auto& comp = lib.get(1, 0);

    AceMap map = ace::detect_component(norm, comp, cube.height, cube.width);

    EXPECT_NEAR(map.minCoeff(), 0.0f, 1e-5f);
    EXPECT_NEAR(map.maxCoeff(), 1.0f, 1e-5f);
}

// ── detect_class ─────────────────────────────────────────────────────────────

TEST(AceDetector, DetectClassShape) {
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    SpectralLibrary lib(SPEC_LIB_DIR);

    // aircraft = class_id 1
    AceMap map = ace::detect_class(cube, lib, 1);

    EXPECT_EQ(map.rows(), cube.height);
    EXPECT_EQ(map.cols(), cube.width);
}

TEST(AceDetector, DetectClassValuesInRange) {
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    SpectralLibrary lib(SPEC_LIB_DIR);

    AceMap map = ace::detect_class(cube, lib, 1);

    EXPECT_GE(map.minCoeff(), -1e-6f);
    EXPECT_LE(map.maxCoeff(),  1.0f + 1e-6f);
    EXPECT_TRUE(map.allFinite());
}

TEST(AceDetector, DetectClassIsMaxOverComponents) {
    // detect_class must be >= every individual component's map element-wise.
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    Eigen::MatrixXd norm = ace::normalize_columns(cube.data);
    SpectralLibrary lib(SPEC_LIB_DIR);

    AceMap class_map = ace::detect_class(cube, lib, 1);

    int n = lib.component_count(1);
    for (int i = 0; i < n; ++i) {
        AceMap comp_map = ace::detect_component(norm, lib.get(1, i), cube.height, cube.width);
        // class_map >= comp_map element-wise (within float tolerance)
        EXPECT_GE((class_map - comp_map).minCoeff(), -1e-5f)
            << "class_map is less than component " << i << " at some pixel";
    }
}

TEST(AceDetector, AllClassesProduceFiniteMaps) {
    HyperspectralCube cube = load_hyperspectral(HYPER_BIN);
    SpectralLibrary lib(SPEC_LIB_DIR);

    for (int cls = 0; cls < cfg::NUM_CLASSES; ++cls) {
        AceMap map = ace::detect_class(cube, lib, cls);
        EXPECT_TRUE(map.allFinite()) << "class " << cls << " produced non-finite values";
        EXPECT_EQ(map.rows(), cube.height) << "class " << cls;
        EXPECT_EQ(map.cols(), cube.width)  << "class " << cls;
    }
}
