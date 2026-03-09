#include "spectral_library.h"

#include <algorithm>
#include <stdexcept>

#include <dirent.h>

#include "binary_io.h"
#include "config.h"

namespace {

// Return sorted absolute paths of *.bin files in dir.
std::vector<std::string> list_bin_files(const std::string& dir) {
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        throw std::runtime_error("Cannot open spectral library directory: " + dir);
    }

    std::vector<std::string> paths;
    struct dirent* ent;
    while ((ent = readdir(dp)) != nullptr) {
        const std::string name(ent->d_name);
        if (name.size() > 4 && name.substr(name.size() - 4) == ".bin") {
            paths.push_back(dir + "/" + name);
        }
    }
    closedir(dp);

    std::sort(paths.begin(), paths.end());
    return paths;
}

}  // namespace

SpectralComponent SpectralLibrary::parse(const std::vector<double>& data,
                                         uint32_t rows, uint32_t cols) {
    if (rows < 3) {
        throw std::runtime_error("Spectral matrix must have at least 3 rows");
    }
    if (cols != static_cast<uint32_t>(cfg::SPECTRAL_BANDS)) {
        throw std::runtime_error("Spectral matrix column count does not match SPECTRAL_BANDS");
    }

    const int B = static_cast<int>(cols);
    const int W = static_cast<int>(rows) - 2;  // whitening rows

    // Row-major interpretation: element (i,j) = data[i*B + j].
    using RowMatXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    SpectralComponent comp;
    comp.mu        = Eigen::Map<const Eigen::RowVectorXd>(data.data(), B);
    comp.whitening = Eigen::Map<const RowMatXd>(data.data() + B, W, B);
    comp.signature = Eigen::Map<const Eigen::RowVectorXd>(
                         data.data() + static_cast<std::size_t>(rows - 1) * B, B);
    return comp;
}

SpectralLibrary::SpectralLibrary(const std::string& base_dir) {
    for (int c = 0; c < cfg::NUM_CLASSES; ++c) {
        const std::string class_dir = base_dir + "/" + cfg::CLASS_NAMES[c];
        const auto paths = list_bin_files(class_dir);
        library_[c].reserve(paths.size());
        for (const auto& path : paths) {
            auto raw = bio::read_spectral_bin(path);
            library_[c].push_back(parse(raw.data, raw.rows, raw.cols));
        }
    }
}

int SpectralLibrary::component_count(int class_idx) const {
    return static_cast<int>(library_.at(static_cast<std::size_t>(class_idx)).size());
}

const SpectralComponent& SpectralLibrary::get(int class_idx, int component_idx) const {
    return library_.at(static_cast<std::size_t>(class_idx))
                   .at(static_cast<std::size_t>(component_idx));
}
