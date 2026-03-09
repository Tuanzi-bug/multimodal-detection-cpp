#pragma once

#include <array>
#include <string>
#include <vector>

#include "config.h"
#include "types.h"

// Loads and owns all SpectralComponents for every class.
// Components are read from: base_dir/{class_name}/*.bin
// Files are sorted lexicographically within each class directory.
class SpectralLibrary {
public:
    explicit SpectralLibrary(const std::string& base_dir);

    // Number of spectral components loaded for a given class index.
    int component_count(int class_idx) const;

    // Access a specific component (bounds-checked).
    const SpectralComponent& get(int class_idx, int component_idx) const;

private:
    static SpectralComponent parse(const std::vector<double>& data,
                                   uint32_t rows, uint32_t cols);

    std::array<std::vector<SpectralComponent>, cfg::NUM_CLASSES> library_;
};
