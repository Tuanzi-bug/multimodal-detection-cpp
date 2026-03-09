# 技术发现（Findings）

## 项目结构发现

### Python 源码关键文件
| 文件 | 作用 |
|------|------|
| `my_get_map.py` | 主入口：YOLO 检测 → 高光谱加载 → 多模态融合 → mAP 计算 |
| `multimode.py` | ACE 算法核心 + 融合过滤逻辑 |
| `nets/yolo.py` | YOLOv5 模型定义（`yolo_body()`） |
| `nets/CSPdarknet.py` | Backbone 定义（含 Focus 层、SiLU 激活） |
| `utils/utils_bbox.py` | `DecodeBox`：锚框解码 + Letterbox 校正 + NMS |
| `eFUMI_VCA_initialize.py` | `normalize()`：Min-max 归一化 |
| `VCA.py` | VCA 端成分提取（仅用于光谱库初始化，非在线） |

### 参考 C++ 代码结构
| 文件 | 作用 |
|------|------|
| `c++/reference/yolov3_camera_dc/vip/vnn_pre_process.cpp` | NPU 预处理（NV12 → RGB，resize，scale） |
| `c++/reference/yolov3_camera_dc/vip/vnn_post_process.cpp` | NPU 后处理（tensor 置换，uint8 → float32） |
| `c++/reference/yolov3_camera_dc/vip/yolov3_post_process_kernel.cpp` | YOLOv3 检测框解码 |
| `c++/reference/yolov3_camera_dc/vip/fifo.h` | FIFO 数据结构（多阶段流水线） |

## ACE 算法分析

### Python 实现（multimode.py）
```python
# ace_tarsig() — lines 22-31
def ace_tarsig(x, sqrtDxinvU, mu_x_minus, double_arrow_s):
    # x: [bands, num_pixels]
    x = x - mu_x_minus.T   # 去均值
    arrow_x = sqrtDxinvU @ x   # 白化
    fro_arrow_x = norm(arrow_x, axis=0)   # 逐列 L2 范数
    double_arrow_x = arrow_x / fro_arrow_x  # 归一化
    Y = double_arrow_s @ double_arrow_x   # 与目标光谱点积
    return Y   # shape: (1, num_pixels)
```

### 光谱库矩阵结构（每个 .mat 文件）
```
data.shape ≈ (131, 129)
row 0      = mu（背景均值向量，1×129）
rows 1~129 = sqrtDxinvU（白化矩阵，129×129）
row 130    = signature（目标光谱，1×129）
```

### 高光谱数据 Reshape 顺序（⚠️ 陷阱）
```python
# my_get_map.py line 120 + multimode.py line 68
test_data = scio.loadmat(path)['data']          # [H, W, 129]
test_data_ = np.reshape(
    np.transpose(test_data, (2, 1, 0)),          # → [129, W, H]
    (test_data.shape[2], -1)                     # → [129, W*H]
)
# 注意：reshape 后列顺序是 (W, H)，即先按 W 方向遍历
```

### ACE 输出 Reshape 顺序（⚠️ 陷阱）
```python
# multimode.py line 88
confid_temp = confid_temp.reshape((W, H)).T  # [W, H] → transpose → [H, W]
```

## YOLOv5 解码分析

### 关键公式（来自 utils/utils_bbox.py lines 67-73）
```python
# YOLOv5（本项目）
box_xy = (sigmoid(raw[..., :2]) * 2 - 0.5 + grid) / grid_shape
box_wh = (sigmoid(raw[..., 2:4]) * 2) ** 2 * anchors / input_shape

# 对比 YOLOv3（reference 代码，不可复用！）
box_xy = sigmoid(raw[..., :2]) + grid
box_wh = exp(raw[..., 2:4]) * anchors
```

### 输出头结构
```
Head 0: [1, 20, 20, 30]  → anchors [6,7,8] → 大目标
Head 1: [1, 40, 40, 30]  → anchors [3,4,5] → 中目标
Head 2: [1, 80, 80, 30]  → anchors [0,1,2] → 小目标
30 = 3 anchors × (5 + 5 classes) = 3 × 10
```

### Letterbox 校正（utils/utils.py lines 27-40）
```python
# 等比缩放，灰色填充 (128, 128, 128)
scale = min(input_shape / image_shape)
new_shape = round(image_shape * scale)
pad = (input_shape - new_shape) / 2  # top/left offset
```

## 融合过滤逻辑分析

### Python 实现（multimode.py lines 146-176）
```python
# 只处理置信度最高的类别（dominant class）
# 其他类别的检测框全部丢弃

for box in boxes_of_dominant_class:
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    temp1 = ace_map[y1:y2+1, x1:x2+1]
    H, W = temp1.shape
    # 水平中心带 × 垂直中心带 双重检验
    if (temp1[H//2, W//3 : 2*W//3].max() >= 0.80 and
        temp1[H//3 : 2*H//3, W//2].max() >= 0.80):
        # 保留此检测框
```

## Vivante SDK 关键信息

### 库版本（来自 reference 代码注释）
```
Acuity 版本：5.24.6
ovxlib 版本：1.1.31
量化方式：uint8 非对称仿射量化
```

### 关键函数（来自 reference 代码）
```cpp
// 创建网络
vnn_CreateYolov3TinyUint8_NB(const char* model_path)
// 处理输入
vsi_nn_ProcessGraph(context, graph, ...)
// 获取输出 tensor
vsi_nn_GetTensor(graph, tensor_id)
// 数据类型转换
vsi_nn_DtypeToFloat32(data, out, num, &dtype)
```

### 输出 Tensor 格式（来自 vnn_post_process.cpp）
```cpp
// 输出 tensor 需要 permutation（NHWC → NCHW 等）
// 使用 vsi_nn_DtypeToFloat32 将 uint8 转为 float32
// 量化参数在 tensor 的 attr 中
```

## 数据文件发现

### 光谱库统计
| 类别 | 文件数 | 矩阵尺寸 |
|------|--------|----------|
| aircraft | 3 | ~131×129 |
| car | 5 | ~131×129 |
| oiltank | 4 | ~131×129 |
| roof | 3 | ~131×129 |
| ship | 6 | ~131×129 |
| **合计** | **21** | |

### 模型权重文件
```
model_data/best_epoch_weights.h5  ← 主模型（~85MB）
model_data/yolo_anchors.txt       ← 9个锚框参数
model_data/voc_classes.txt        ← 5个类别名称
```

### 性能基准（Python 实现）
```
YOLO 仅检测 mAP：90.53%
多模态融合 mAP：92.34%
提升：+1.81%
```
