#pragma once

#include <string>

#include "types.h"

// Load a hyperspectral .bin file and return a HyperspectralCube ready for ACE.
//
// Mirrors Python:
//   test_data_ = np.reshape(np.transpose(test_data, (2, 1, 0)), (bands, -1))
//
// Result layout: cube.data(b, j), where j = w * height + h  (w-major).
// This matches Python's column ordering after transpose(2,1,0) + C-order reshape.
HyperspectralCube load_hyperspectral(const std::string& path);
