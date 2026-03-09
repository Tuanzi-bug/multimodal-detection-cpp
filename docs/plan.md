# 实施计划（Plan）

## 总体架构

```
RGB Image
    │
    ▼
┌─────────────────┐     ┌──────────────────────┐
│   YOLOv5m NPU   │     │  Hyperspectral Cube   │
│  (Vivante VIP)  │     │  (131 bands, HxWxB)   │
└────────┬────────┘     └──────────┬────────────┘
         │ DetectionResult          │ normalize()
         │ (BoundingBox[])          │ ace_detector()
         │                          │ AceMap
         └──────────────┬───────────┘
                        ▼
              ┌─────────────────┐
              │ Multimodal Fuse │  center/cross ≥ 0.80
              └────────┬────────┘
                       ▼
              Final DetectionResult
```

## 文件结构

```
c++/
  tools/
    convert_h5_to_onnx_plan.md      # Phase 0.1 方案文档
    quantize_model_plan.md           # Phase 0.2 方案文档
    convert_mat_to_bin.py            # Phase 0.3 实现
    convert_hyimg_to_bin.py          # Phase 0.4 实现
  include/
    config.h                         # 全局常量（无硬编码）
    types.h                          # 核心数据结构
    binary_io.h                      # .bin 文件 I/O 接口
    spectral_library.h               # 光谱库加载接口
    hyperspectral_loader.h           # 高光谱数据加载接口
    normalize.h                      # Min-max 归一化接口
    ace_detector.h                   # ACE 算法接口
    yolov5_preprocess.h              # YOLO 预处理接口
    yolov5_detector.h                # NPU 推理封装接口
    yolov5_postprocess.h             # 解码 + NMS 接口
    multimodal_fusion.h              # 融合过滤接口
    pipeline.h                       # 主流水线接口
  src/
    binary_io.cpp
    spectral_library.cpp
    hyperspectral_loader.cpp
    normalize.cpp
    ace_detector.cpp
    yolov5_preprocess.cpp
    yolov5_detector.cpp
    yolov5_postprocess.cpp
    multimodal_fusion.cpp
    pipeline.cpp
    main.cpp
  test/
    test_normalize.cpp
    test_ace.cpp
    test_postprocess.cpp
    test_fusion.cpp
    test_pipeline.cpp
    CMakeLists.txt
  docker/
    Dockerfile
    build.sh
  data/                              # 转换后数据（工具生成，不进 git）
    spectral_lib/{class}/*.bin
    hyperspectral/*.bin
    model/
      yolov5m.nb                     # 需手动运行 Phase 0.2
      yolo_anchors.txt
      voc_classes.txt
  CMakeLists.txt
  toolchain-arm.cmake
docs/
  spec.md     ← 本项目需求规格
  plan.md     ← 本文件
  tasks.md    ← 任务追踪
  findings.md ← 技术发现
```

---

## Phase 0：离线预处理工具

### Phase 0.1：H5 → ONNX 方案（文档）
- **产出**：`c++/tools/convert_h5_to_onnx_plan.md`
- **内容**：tf2onnx 使用说明、自定义层处理（Focus、SiLU）、验证方法
- **不实现代码**

### Phase 0.2：ONNX → Vivante .nb 量化方案（文档）
- **产出**：`c++/tools/quantize_model_plan.md`
- **内容**：Acuity 5.24.6 工具链使用、uint8 量化配置、精度验证步骤
- **不实现代码**

### Phase 0.3：convert_mat_to_bin.py
- 读取 `spectral_lib/{class}/*.mat` → 写出 `c++/data/spectral_lib/{class}/*.bin`
- 二进制格式：`[uint32 rows][uint32 cols][uint32 dtype_size=8][float64 row-major]`
- 包含 round-trip 验证

### Phase 0.4：convert_hyimg_to_bin.py
- 读取 `hyperspectral/*.mat` → 写出 `c++/data/hyperspectral/*.bin`
- 二进制格式：`[uint32 H][uint32 W][uint32 bands][uint32 dtype_size=8][float64 H×W×bands]`

**提交**：`feat(tools): add data conversion scripts and model plan docs`

---

## Phase 1：Docker 环境与 CMake 构建系统

### Dockerfile 关键内容
```dockerfile
FROM ubuntu:20.04
RUN apt-get install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf cmake
# Eigen3 header-only
# Vivante SDK headers（来自 reference/vip/driver/sdk/include/）
# OpenCV ARM 预编译包（target 预装，仅需 headers）
```

### CMakeLists.txt 链接库
```cmake
target_link_libraries(multimodal_detect
  OpenVX OpenVXU CLC VSC SPIRV_viv GAL ovxlib NNArchPerf ArchModelSw
  opencv_imgproc opencv_core
)
```

### toolchain-arm.cmake
```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
```

**提交**：`feat(build): add Docker cross-compilation environment and CMake config`

---

## Phase 2：核心数据结构与 I/O

### types.h 核心结构

```cpp
struct BoundingBox {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

struct DetectionResult {
    std::vector<BoundingBox> boxes;
    int image_width, image_height;
};

struct AceMap {
    std::vector<float> data;   // row-major, [0,1]
    int rows, cols;
};

struct SpectralComponent {
    Eigen::VectorXd mu;             // (129,1) 均值向量
    Eigen::MatrixXd sqrt_dx_inv_u;  // (129,129) 白化矩阵
    Eigen::RowVectorXd signature;   // (1,129) 目标光谱
};

struct HyperspectralCube {
    std::vector<double> data;       // [H×W×bands] row-major
    int height, width, bands;
};
```

### config.h 常量（摘录）
```cpp
constexpr int YOLO_INPUT_SIZE = 640;
constexpr int NUM_CLASSES = 5;
constexpr float YOLO_ANCHORS[9][2] = {
    {10,13},{16,30},{33,23},{30,61},{62,45},
    {59,119},{116,90},{156,198},{373,326}
};
constexpr int ANCHOR_MASK[3][3] = {{6,7,8},{3,4,5},{0,1,2}};
constexpr float DEFAULT_CONFIDENCE = 0.05f;
constexpr float DEFAULT_NMS_IOU = 0.3f;
constexpr float ACE_THRESHOLD = 0.80f;
constexpr float CONTRAST_FACTOR = 1.3f;
constexpr int DILATION_KERNEL_SIZE = 3;
```

**提交**：`feat(core): add data structures, config constants, and binary I/O`

---

## Phase 3：ACE 算法与归一化

### normalize.cpp
移植 `eFUMI_VCA_initialize.py normalize()`：
- 逐列 min-max 归一化，范围 [0,1]
- 边界情况：全零列保持 0；常数非零列填 1.0

### ace_detector.cpp
移植 `multimode.py ace_tarsig()`：
```
1. centered_x = x - mu  (广播)
2. arrow_x = sqrt_dx_inv_u * centered_x
3. norm_x = ||arrow_x||₂  (逐列)
4. double_arrow_x = arrow_x / norm_x
5. Y = signature · double_arrow_x  → (1 × num_pixels)
```

完整流水线（含 reshape、膨胀、对比度增强）：
```
hyperspectral cube [H,W,B]
    → transpose(2,1,0) → reshape → [B, W*H]
    → normalize()
    → ace_tarsig() × N个光谱组件
    → reshape → [W,H] → transpose → [H,W]
    → per-component max
    → cv::dilate (3×3)
    → clip(map * 1.3, 0, 1)
    → AceMap
```

**⚠️ 关键陷阱**：Python `np.transpose(test_data, (2,1,0))` 把 [H,W,B] 变成 [B,W,H]，
reshape 到 [B, W\*H] 时列优先顺序为 (W, H)，ACE 输出 reshape 回 [W,H] 后再转置为 [H,W]。

**提交**：`feat(ace): implement ACE spectral detection algorithm and normalization`

---

## Phase 4：YOLOv5 推理流水线

### yolov5_preprocess.cpp
- Letterbox 缩放（保持宽高比，灰色填充 RGB(128,128,128)）
- 归一化：`pixel / 255.0`
- 支持 NV12 和 RGB888 输入

### yolov5_detector.cpp（Vivante NPU 封装）
```cpp
class Yolov5Detector {
    bool init(const std::string& model_nb_path);
    bool process(const uint8_t* frame, int frame_id);
    bool get_raw_outputs(std::vector<std::vector<float>>& outputs);
    void release();
};
```
参照 `c++/reference/yolov3_camera_dc/vip/` 的 FIFO 异步模式。

### yolov5_postprocess.cpp
**⚠️ 关键：YOLOv5 解码公式与 YOLOv3（reference）完全不同**

```cpp
// YOLOv5（本项目，来自 utils/utils_bbox.py）
box_xy = (sigmoid(raw[:2]) * 2 - 0.5 + grid) / grid_shape
box_wh = pow(sigmoid(raw[2:4]) * 2, 2) * anchors / 640

// YOLOv3（reference 代码，不能复用）
box_xy = sigmoid(raw[:2]) + grid
box_wh = exp(raw[2:4]) * anchors
```

NMS 纯 C++ 实现（无 TF 依赖）：
```cpp
std::vector<int> non_max_suppression(
    const std::vector<BoundingBox>& boxes,
    float iou_threshold, int max_boxes
);
```

**提交**：`feat(yolo): implement YOLOv5 inference pipeline with Vivante NPU`

---

## Phase 5：多模态融合

### multimodal_fusion.cpp
移植 `multimode.py lines 146-176`：

```
1. 确定主类（YOLO 置信度最高的类别）
2. 对主类每个检测框：
   a. 提取 ACE 图区域 region = ace_map[y1:y2, x1:x2]
   b. H = region.rows, W = region.cols
   c. 水平中心带：region[H/2, W/3 : 2*W/3].max() ≥ 0.80
   d. 垂直中心带：region[H/3 : 2*H/3, W/2].max() ≥ 0.80
   e. 两个条件都满足 → 保留，否则 → 丢弃
3. 非主类检测框全部丢弃
```

**提交**：`feat(fusion): implement multimodal ACE-YOLO fusion filter`

---

## Phase 6：主流水线与入口

### pipeline.cpp（FIFO 多阶段流水线）
```
Stage 1: 加载 RGB + 高光谱数据
Stage 2: YOLOv5 NPU 推理（异步 FIFO）
Stage 3: YOLOv5 后处理（解码+NMS）
Stage 4: ACE 光谱检测（CPU）
Stage 5: 多模态融合
Stage 6: 输出结果
```

### main.cpp（CLI 接口）
```bash
./multimodal_detect \
  --model    c++/data/model/yolov5m.nb \
  --spec-lib c++/data/spectral_lib/ \
  --rgb      test_image.jpg \
  --hyper    test_image.bin \
  --conf     0.05 \
  --nms-iou  0.3
```

**提交**：`feat(pipeline): integrate full multimodal detection pipeline`

---

## Phase 7：单元测试

| 测试文件 | 测试内容 | 验证方式 |
|----------|----------|----------|
| test_normalize.cpp | Min-max 归一化，含边界情况 | 与 Python sklearn 输出对比，tolerance 1e-10 |
| test_ace.cpp | ACE 核心算法 | 与 Python 输出对比，tolerance 1e-6 |
| test_postprocess.cpp | YOLOv5 解码 + NMS | 与 Python 输出对比，坐标误差 ≤ 1px |
| test_fusion.cpp | 融合过滤逻辑，8 个场景 | 通过/拒绝条件覆盖 |
| test_pipeline.cpp | 端到端集成测试 | mAP ≥ 91% |

**提交**：`test: add comprehensive unit and integration tests`

---

## Git 工作流

```
main (稳定)
 └── feature/cpp-port (开发)
      ├── 每个 Phase 完成后 commit
      └── 全部完成后 → PR → main
```

**已完成：**
- `chore: initial commit — Python multimodal detection project`
- GitHub 仓库：https://github.com/Tuanzi-bug/multimodal-detection-cpp

## 风险追踪

| ID | 风险 | 严重度 | 缓解方案 |
|----|------|--------|----------|
| R1 | 高光谱 Transpose 顺序错误 | 高 | 专项测试：序列化 Python 中间值对比 |
| R2 | YOLOv5 后处理公式错误 | 高 | 与 Python utils_bbox.py 逐步对比 |
| R3 | Vivante SDK 不支持 YOLOv5 特殊层 | 中 | 提前验证 SDK ops 支持（Focus, SiLU） |
| R4 | OpenCV 交叉编译困难 | 中 | 使用目标设备预装版本 |
| R5 | 内存超限（高光谱数据） | 中 | 改用 float32，必要时分块处理 |
| R6 | NMS 行为与 TF 不一致 | 低 | 允许 1px 坐标误差，0.001 分数误差 |
