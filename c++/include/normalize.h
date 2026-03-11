#pragma once

#include <Eigen/Dense>

namespace ace {

// Column-wise min-max normalization, matching Python normalize(x, flag=1).
//
// For each column j of 'data' (shape: bands × n_pixels):
//   let lo = min(col_j), hi = max(col_j)
//   if hi == lo == 0  → column stays all-zero
//   if hi == lo != 0  → column becomes all-ones   (sklearn override)
//   otherwise         → (col_j - lo) / (hi - lo)
//
// Returns a new matrix of the same shape with values in [0, 1].
Eigen::MatrixXd normalize_columns(const Eigen::MatrixXd& data);

}  // namespace ace
