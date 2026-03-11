#pragma once

#include <cstdint>

namespace preprocess {

// Result of letterbox transformation — needed to map predicted boxes back to
// original image coordinates.
struct LetterboxInfo {
    float scale;    // uniform scale applied (orig → target)
    int   pad_top;  // gray rows added at top (pixels in target space)
    int   pad_left; // gray columns added at left (pixels in target space)
};

// Convert NV12 (YUV 4:2:0 semi-planar, Y plane then interleaved UV plane) to
// packed RGB888.  Output buffer must hold h * w * 3 bytes.
void nv12_to_rgb888(const uint8_t* nv12, int h, int w, uint8_t* rgb);

// Letterbox-resize a packed RGB888 image (src_h × src_w × 3) into
// dst (dst_h × dst_w × 3).  Gray fill value is 114.
// Returns scale and padding applied so callers can undo the transform.
LetterboxInfo letterbox(const uint8_t* src, int src_h, int src_w,
                        uint8_t* dst,       int dst_h, int dst_w);

// Normalize packed RGB888 HWC uint8 → float32 CHW layout, dividing each
// channel value by 255.0f.  Output must hold 3 * h * w floats.
// Channel order is preserved (R→C0, G→C1, B→C2).
void normalize_to_chw_float(const uint8_t* hwc, int h, int w, float* chw);

}  // namespace preprocess
