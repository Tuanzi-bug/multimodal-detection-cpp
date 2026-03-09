#pragma once

#include <Eigen/Dense>

// YOLO post-processing result: one detected object.
struct BoundingBox {
    float top      = 0.f;
    float left     = 0.f;
    float bottom   = 0.f;
    float right    = 0.f;
    float score    = 0.f;
    int   class_id = -1;
};

// ACE detection map for one class: rows=H, cols=W, values normalised to [0, 1].
using AceMap = Eigen::MatrixXf;

// One spectral signature parsed from a 131×129 .bin matrix.
//   Row 0         → mu        (1×SPECTRAL_BANDS background mean)
//   Rows 1..B     → whitening (SPECTRAL_BANDS×SPECTRAL_BANDS sqrtDxinvU matrix)
//   Row rows-1    → signature (1×SPECTRAL_BANDS target spectral signature)
struct SpectralComponent {
    Eigen::RowVectorXd mu;         // 1×SPECTRAL_BANDS
    Eigen::MatrixXd    whitening;  // SPECTRAL_BANDS×SPECTRAL_BANDS
    Eigen::RowVectorXd signature;  // 1×SPECTRAL_BANDS
};

// Hyperspectral image prepared for ACE processing.
//   data(b, j) = value at band b, pixel j.
//   Column ordering: j = w * height + h  (w-major, matching Python transpose/reshape).
struct HyperspectralCube {
    int height = 0;
    int width  = 0;
    int bands  = 0;
    Eigen::MatrixXd data;  // shape: (bands, width * height)
};
