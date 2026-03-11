#include "normalize.h"

namespace ace {

// Column-wise min-max normalization matching Python normalize(x, flag=1).
//
// For each column j:
//   lo = min(col_j), hi = max(col_j)
//   if hi == lo == 0  → column stays all-zero        (sklearn default)
//   if hi == lo != 0  → column becomes all-ones       (Python override)
//   otherwise         → (x - lo) / (hi - lo)
Eigen::MatrixXd normalize_columns(const Eigen::MatrixXd& data) {
    const int rows = static_cast<int>(data.rows());
    const int cols = static_cast<int>(data.cols());

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(rows, cols);

    for (int j = 0; j < cols; ++j) {
        const double lo = data.col(j).minCoeff();
        const double hi = data.col(j).maxCoeff();

        if (hi == lo) {
            // Constant column.
            if (hi != 0.0) {
                // Constant non-zero → all-ones (Python override of sklearn behaviour).
                out.col(j).setOnes();
            }
            // else constant zero → stays all-zero (already initialised).
        } else {
            out.col(j) = (data.col(j).array() - lo) / (hi - lo);
        }
    }

    return out;
}

}  // namespace ace
