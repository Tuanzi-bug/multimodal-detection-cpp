#pragma once

#include "types.h"
#include "config.h"

#include <array>
#include <vector>

namespace postprocess {

// Decode one YOLOv5 feature-map head into BoundingBox candidates.
//
// feats         : raw float array of shape [grid_h × grid_w × (3*(5+NUM_CLASSES))].
//                 Layout: for each (row, col, anchor, channel).
// grid_h/grid_w : spatial dimensions of this head (e.g. 20, 40, 80).
// anchor_mask   : 3 indices into cfg::ANCHORS pairs to use (e.g. {6,7,8}).
// conf_threshold: minimum objectness * max-class-prob to keep a candidate.
//
// Box coordinates (top/left/bottom/right) are returned in pixel space,
// relative to the 640×640 network-input image.
std::vector<BoundingBox> decode_head(
    const float*              feats,
    int                       grid_h,
    int                       grid_w,
    const std::array<int, 3>& anchor_mask,
    float                     conf_threshold);

// Compute Intersection-over-Union for two boxes.
float iou(const BoundingBox& a, const BoundingBox& b);

// Non-Maximum Suppression, applied per class.
// Removes boxes whose IoU with a higher-scoring box of the same class
// exceeds iou_threshold.  Input vector is consumed (sorted in-place).
std::vector<BoundingBox> nms(std::vector<BoundingBox> boxes,
                             float iou_threshold);

// Full post-processing pipeline: decode all three heads then apply NMS.
//
// out0: 20×20 head raw floats (20*20*30 elements).
// out1: 40×40 head raw floats (40*40*30 elements).
// out2: 80×80 head raw floats (80*80*30 elements).
std::vector<BoundingBox> post_process(
    const float* out0,
    const float* out1,
    const float* out2,
    float        conf_threshold = cfg::CONF_THRESHOLD,
    float        nms_iou        = cfg::NMS_IOU);

}  // namespace postprocess
