#include "ace_detector.h"

#include "normalize.h"

namespace ace {

// Run ACE algorithm for a single spectral component.
//
// Python reference (multimode.py: ace_tarsig):
//   centered_x = x - tile(mu, (1, N))          → subtract background mean
//   arrow_x    = sqrtDxinvU @ centered_x        → apply whitening
//   norm_arrow = ||arrow_x||_2 column-wise      → L2 norm per pixel
//   unit_arrow = arrow_x / norm_arrow           → unit-normalize each pixel
//   Y          = signature @ unit_arrow         → dot with target signature
//
// Result is reshaped to (W, H) then transposed to (H, W), then min-max
// normalised to [0, 1].
AceMap detect_component(const Eigen::MatrixXd& normalized_cube,
                        const SpectralComponent& comp,
                        int height, int width) {
    const int n_pixels = width * height;

    // 1. Centre each pixel by subtracting background mean.
    //    comp.mu is RowVectorXd (1×B); transpose to (B×1) for broadcasting.
    Eigen::MatrixXd centered = normalized_cube.colwise() - comp.mu.transpose();

    // 2. Apply whitening matrix: arrow_x = whitening × centered  (B×N).
    Eigen::MatrixXd arrow_x = comp.whitening * centered;

    // 3. L2-normalise each column (per-pixel unit vector).
    Eigen::RowVectorXd col_norms = arrow_x.colwise().norm();
    // Avoid division by zero: pixels where norm == 0 stay zero.
    for (int j = 0; j < n_pixels; ++j) {
        if (col_norms(j) != 0.0) {
            arrow_x.col(j) /= col_norms(j);
        }
    }

    // 4. Dot with target signature: Y = signature × arrow_x  →  (1×N).
    Eigen::RowVectorXd raw_scores = comp.signature * arrow_x;

    // 5. Reshape to (H, W) matching Python:
    //    confid_temp.reshape((W, H)).T
    //    Column ordering: j = w * H + h  (w-major, matching the loaded cube).
    AceMap map(height, width);
    for (int w = 0; w < width; ++w) {
        for (int h = 0; h < height; ++h) {
            map(h, w) = static_cast<float>(raw_scores(w * height + h));
        }
    }

    // 6. Min-max normalise the map to [0, 1].
    const float lo = map.minCoeff();
    const float hi = map.maxCoeff();
    if (hi != lo) {
        map = (map.array() - lo) / (hi - lo);
    } else {
        map.setZero();
    }

    return map;
}

// Run ACE for all spectral components of a class and take element-wise max.
AceMap detect_class(const HyperspectralCube& cube,
                    const SpectralLibrary& lib,
                    int class_id) {
    // Column-wise normalization matching Python normalize(test_data_, 1).
    const Eigen::MatrixXd norm = normalize_columns(cube.data);

    const int n = lib.component_count(class_id);
    AceMap result = AceMap::Zero(cube.height, cube.width);

    for (int i = 0; i < n; ++i) {
        AceMap comp_map = detect_component(norm, lib.get(class_id, i),
                                           cube.height, cube.width);
        result = result.cwiseMax(comp_map);
    }

    return result;
}

}  // namespace ace
