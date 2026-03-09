# 任务追踪（Tasks）

## 状态说明

| 状态 | 含义 |
|------|------|
| ⬜ pending | 待开始 |
| 🔵 in_progress | 进行中 |
| ✅ done | 已完成 |
| ⏭️ doc_only | 仅文档，不实现代码 |
| ❌ blocked | 被阻塞 |

---

## Phase 0：离线预处理工具

| ID | 任务 | 产出文件 | 状态 | 备注 |
|----|------|----------|------|------|
| 0.1 | H5 → ONNX 转换方案 | `c++/tools/convert_h5_to_onnx_plan.md` | ✅ done | commit: `docs(tools/phase-0.1)` |
| 0.2 | ONNX → Vivante .nb 量化方案 | `c++/tools/quantize_model_plan.md` | ✅ done | commit: `docs(tools/phase-0.2)` |
| 0.3 | 光谱库 .mat → .bin 转换脚本 | `c++/tools/convert_mat_to_bin.py` | ✅ done | 18 tests passed；21/21 文件转换成功，max_diff=0 |
| 0.4 | 高光谱图像 .mat → .bin 转换脚本 | `c++/tools/convert_hyimg_to_bin.py` | ✅ done | 19 tests passed；22/22 文件转换成功 |

**Git commits（各自独立）**：
- `docs(tools/phase-0.1)`: H5→ONNX 方案文档
- `docs(tools/phase-0.2)`: 量化方案文档
- `feat(tools/phase-0.3)`: 光谱库转换脚本（conda env: multimodal-cpp-tools）
- `feat(tools/phase-0.4)`: 高光谱图像转换脚本（conda env: multimodal-cpp-tools）

---

## Phase 1：Docker 环境与 CMake 构建系统

| ID | 任务 | 产出文件 | 状态 | 备注 |
|----|------|----------|------|------|
| 1.1 | Dockerfile（ARM 交叉编译环境） | `c++/docker/Dockerfile` | ⬜ pending | Ubuntu 20.04 + arm-gnueabihf-g++ + Eigen + Vivante SDK |
| 1.2 | CMakeLists.txt（主构建配置） | `c++/CMakeLists.txt` | ⬜ pending | 链接 Vivante + OpenCV 库 |
| 1.3 | ARM 工具链文件 | `c++/toolchain-arm.cmake` | ⬜ pending | |
| 1.4 | Docker 构建脚本 | `c++/docker/build.sh` | ⬜ pending | |

**Git commit**：`feat(build): add Docker cross-compilation environment and CMake config`

---

## Phase 2：核心数据结构与 I/O

| ID | 任务 | 产出文件 | 状态 | 备注 |
|----|------|----------|------|------|
| 2.1 | 全局常量（无硬编码值） | `c++/include/config.h` | ⬜ pending | 锚框、阈值、类名等 |
| 2.2 | 核心数据结构 | `c++/include/types.h` | ⬜ pending | BoundingBox, AceMap, SpectralComponent, HyperspectralCube |
| 2.3 | .bin 文件 I/O | `c++/include/binary_io.h` + `c++/src/binary_io.cpp` | ⬜ pending | |
| 2.4 | 光谱库加载器 | `c++/include/spectral_library.h` + `c++/src/spectral_library.cpp` | ⬜ pending | 5类 × 21个光谱组件 |
| 2.5 | 高光谱数据加载器 | `c++/include/hyperspectral_loader.h` + `c++/src/hyperspectral_loader.cpp` | ⬜ pending | 含 transpose/reshape 操作 |

**Git commit**：`feat(core): add data structures, config constants, and binary I/O`

---

## Phase 3：ACE 算法与归一化

| ID | 任务 | 产出文件 | 状态 | 备注 |
|----|------|----------|------|------|
| 3.1 | Min-max 归一化 | `c++/include/normalize.h` + `c++/src/normalize.cpp` | ⬜ pending | 对应 sklearn minmax_scale，含边界情况 |
| 3.2 | ACE 核心算法 | `c++/include/ace_detector.h` + `c++/src/ace_detector.cpp` | ⬜ pending | Eigen 矩阵运算，含完整流水线 |

**关键陷阱**：Python `transpose(2,1,0)` 顺序 [H,W,B]→[B,W,H]，专项测试验证
**Git commit**：`feat(ace): implement ACE spectral detection algorithm and normalization`

---

## Phase 4：YOLOv5 推理流水线

| ID | 任务 | 产出文件 | 状态 | 备注 |
|----|------|----------|------|------|
| 4.1 | YOLO 图像预处理 | `c++/include/yolov5_preprocess.h` + `c++/src/yolov5_preprocess.cpp` | ⬜ pending | Letterbox + 归一化，支持 NV12/RGB888 |
| 4.2 | Vivante NPU 推理封装 | `c++/include/yolov5_detector.h` + `c++/src/yolov5_detector.cpp` | ⬜ pending | 参照 reference FIFO 模式，3个输出头 |
| 4.3 | YOLOv5 后处理（解码+NMS） | `c++/include/yolov5_postprocess.h` + `c++/src/yolov5_postprocess.cpp` | ⬜ pending | ⚠️ YOLOv5 公式与 reference YOLOv3 完全不同 |

**关键陷阱**：YOLOv5 解码公式 `(sigmoid*2 - 0.5)` vs YOLOv3 `sigmoid`
**Git commit**：`feat(yolo): implement YOLOv5 inference pipeline with Vivante NPU`

---

## Phase 5：多模态融合

| ID | 任务 | 产出文件 | 状态 | 备注 |
|----|------|----------|------|------|
| 5.1 | ACE-YOLO 融合过滤器 | `c++/include/multimodal_fusion.h` + `c++/src/multimodal_fusion.cpp` | ⬜ pending | 中心/十字截面双重阈值，移植 multimode.py lines 146-176 |

**Git commit**：`feat(fusion): implement multimodal ACE-YOLO fusion filter`

---

## Phase 6：主流水线与入口

| ID | 任务 | 产出文件 | 状态 | 备注 |
|----|------|----------|------|------|
| 6.1 | 主流水线编排器 | `c++/include/pipeline.h` + `c++/src/pipeline.cpp` | ⬜ pending | FIFO 多阶段，参照 reference 代码 |
| 6.2 | 命令行入口 | `c++/src/main.cpp` | ⬜ pending | 参数解析，输出检测结果 |

**Git commit**：`feat(pipeline): integrate full multimodal detection pipeline`

---

## Phase 7：单元测试

| ID | 任务 | 产出文件 | 状态 | 验证方式 |
|----|------|----------|------|----------|
| 7.1 | 归一化单元测试 | `c++/test/test_normalize.cpp` | ⬜ pending | vs Python sklearn，tolerance 1e-10 |
| 7.2 | ACE 算法单元测试 | `c++/test/test_ace.cpp` | ⬜ pending | vs Python，tolerance 1e-6 |
| 7.3 | YOLOv5 后处理单元测试 | `c++/test/test_postprocess.cpp` | ⬜ pending | vs Python，坐标误差 ≤ 1px |
| 7.4 | 融合过滤逻辑测试（8场景） | `c++/test/test_fusion.cpp` | ⬜ pending | 通过/拒绝条件覆盖 |
| 7.5 | 端到端集成测试 | `c++/test/test_pipeline.cpp` | ⬜ pending | mAP ≥ 91% |

**Git commit**：`test: add comprehensive unit and integration tests`

---

## 最终发布

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| R1 | PR: feature/cpp-port → main | ⬜ pending | 所有 Phase 完成后 |
| R2 | GitHub Release tag v1.0.0 | ⬜ pending | |

---

## 进度总览

```
Phase 0  [✅✅✅✅] 0.1 doc + 0.2 doc + 0.3 实现 + 0.4 实现  ← 完成
Phase 1  [⬜⬜⬜⬜] Docker + CMake
Phase 2  [⬜⬜⬜⬜⬜] 数据结构 + I/O
Phase 3  [⬜⬜] ACE + 归一化
Phase 4  [⬜⬜⬜] YOLOv5 流水线
Phase 5  [⬜] 多模态融合
Phase 6  [⬜⬜] 主流水线
Phase 7  [⬜⬜⬜⬜⬜] 测试

总进度：4 / 22 任务完成（Phase 0 ✅）
```

---

## 错误记录

| 时间 | 错误 | 尝试方案 | 解决方案 |
|------|------|----------|----------|
| — | — | — | — |
