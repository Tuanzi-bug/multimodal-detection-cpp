#pragma once

#include <array>

namespace cfg {

// ---------- Image input --------------------------------------------------
constexpr int INPUT_H = 640;
constexpr int INPUT_W = 640;

// ---------- Detection thresholds -----------------------------------------
constexpr int   NUM_CLASSES    = 5;
constexpr float CONF_THRESHOLD = 0.05f;  // mAP evaluation threshold
constexpr float NMS_IOU        = 0.3f;
constexpr int   MAX_DETECTIONS = 300;

// ---------- Spectral parameters ------------------------------------------
constexpr int   SPECTRAL_BANDS = 129;
constexpr float ACE_THRESHOLD  = 0.80f;

// ---------- YOLO anchors -------------------------------------------------
// 9 anchor pairs (w, h) in pixels at 640×640 input.
// Order: scale2 (80×80), scale1 (40×40), scale0 (20×20).
constexpr std::array<float, 18> ANCHORS = {{
     10.f,  13.f,   16.f,  30.f,   33.f,  23.f,   // scale 2 — 80×80 feature map
     30.f,  61.f,   62.f,  45.f,   59.f, 119.f,   // scale 1 — 40×40 feature map
    116.f,  90.f,  156.f, 198.f,  373.f, 326.f,   // scale 0 — 20×20 feature map
}};

// ANCHOR_MASK[scale_idx][i] → index into ANCHORS pairs.
// scale 0 → 20×20, scale 1 → 40×40, scale 2 → 80×80 feature map.
constexpr std::array<std::array<int, 3>, 3> ANCHOR_MASK = {{
    {{6, 7, 8}},   // scale 0
    {{3, 4, 5}},   // scale 1
    {{0, 1, 2}},   // scale 2
}};

// ---------- Class names (matches voc_classes.txt order) ------------------
constexpr std::array<const char*, 5> CLASS_NAMES = {{
    "ship", "aircraft", "roof", "car", "oiltank"
}};

}  // namespace cfg
