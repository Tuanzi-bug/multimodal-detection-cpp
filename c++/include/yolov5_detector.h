#pragma once

#include <string>
#include <vector>
#include <memory>

// YoloV5Detector: wraps Vivante NPU inference for YOLOv5.
//
// The actual NPU interaction is conditionally compiled with HAVE_VIVANTE.
// In native (x86) unit-test builds the class still exists but infer() returns
// false (not implemented on host).  Tests that exercise post-processing should
// call the postprocess functions directly, not through this class.
class YoloV5Detector {
public:
    // Load the .nb model file.  Throws std::runtime_error if the file cannot
    // be opened or the NPU SDK fails to initialise (only when HAVE_VIVANTE).
    explicit YoloV5Detector(const std::string& nb_file);
    ~YoloV5Detector();

    // Run one inference frame.
    //
    // chw_input: float32 tensor, shape [3 × INPUT_H × INPUT_W], values [0,1].
    // out0: decoded output for scale-0 head (20×20 grid) — 20*20*30 floats.
    // out1: scale-1 head (40×40) — 40*40*30 floats.
    // out2: scale-2 head (80×80) — 80*80*30 floats.
    //
    // Returns true on success.  Returns false without throwing when the NPU is
    // not available (e.g. host build or hardware error).
    bool infer(const float*         chw_input,
               std::vector<float>&  out0,
               std::vector<float>&  out1,
               std::vector<float>&  out2);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
