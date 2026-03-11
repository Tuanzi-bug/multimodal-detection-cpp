#pragma once

#include "spectral_library.h"
#include "types.h"

namespace ace {

// Run ACE for a single spectral component on a pre-normalised cube.
//
// normalized_cube : shape (bands, width * height), column j = pixel (w=j/H, h=j%H)
// comp            : mu, whitening, signature loaded from a .bin file
// height, width   : spatial dimensions of the hyperspectral image
//
// Returns AceMap of shape (height, width) with values min-max normalised to [0, 1].
AceMap detect_component(const Eigen::MatrixXd& normalized_cube,
                        const SpectralComponent& comp,
                        int height, int width);

// Run ACE detection for all components of one class and return the per-pixel
// element-wise maximum over all components (already in [0, 1]).
//
// cube     : loaded with load_hyperspectral(); normalisation is applied internally.
// lib      : SpectralLibrary covering all classes.
// class_id : 0=ship, 1=aircraft, 2=roof, 3=car, 4=oiltank.
AceMap detect_class(const HyperspectralCube& cube,
                    const SpectralLibrary& lib,
                    int class_id);

}  // namespace ace
