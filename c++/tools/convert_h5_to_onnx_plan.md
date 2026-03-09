# Convert H5 to ONNX — Plan

## Overview

This document describes how to convert the YOLOv5m Keras model
(`model_data/best_epoch_weights.h5`) to ONNX format for subsequent
quantization with the Acuity Toolkit.  No code is written here;
the document records the exact commands, custom-layer decisions,
and validation procedure.

---

## Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| Python | 3.8 | tf2onnx requires Python 3.7–3.9 |
| TensorFlow | 2.2.0 | Must match the training environment exactly |
| tf2onnx | 1.16.1 | Latest version tested against TF 2.2 |
| onnxruntime | 1.17.x | CPU-only build sufficient for offline validation |
| onnx | 1.15.x | Required by both tf2onnx and onnxruntime |
| numpy | 1.21.x | Keep aligned with TF 2.2 requirement |

Install in an isolated virtualenv:

```bash
python -m venv venv_convert
source venv_convert/bin/activate
pip install tensorflow==2.2.0
pip install tf2onnx==1.16.1 onnxruntime==1.17.1 onnx==1.15.0
```

---

## Model Loading

The model is defined in `nets/yolo.py` via `yolo_body()`.  Load it
before conversion:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from nets.yolo import yolo_body

# Anchors from model_data/yolo_anchors.txt (9 anchors, 3 masks)
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
num_classes   = 5          # voc_classes.txt: ship, aircraft, roof, car, oiltank
phi           = 'm'        # YOLOv5m variant

input_tensor  = Input(shape=(640, 640, 3), name='input_1')
model         = yolo_body(input_tensor, anchors_mask, num_classes=num_classes, phi=phi)
model.load_weights('model_data/best_epoch_weights.h5')
model.summary()
```

Expected output heads:
- `output_0`: shape `(1, 20, 20, 30)` — large objects (anchors 6,7,8)
- `output_1`: shape `(1, 40, 40, 30)` — medium objects (anchors 3,4,5)
- `output_2`: shape `(1, 80, 80, 30)` — small objects (anchors 0,1,2)

where `30 = 3 anchors × (5 + 5 classes)`.

---

## Custom Layer Handling

### Focus Layer (`nets/CSPdarknet.py`)

**What it does** — splits every 2×2 patch of the input into four
slices using strided indexing, then concatenates along the channel
axis, effectively performing a space-to-depth operation:

```python
def call(self, x):
    return tf.concat(
        [x[...,  ::2,  ::2, :],   # top-left pixel of each patch
         x[..., 1::2,  ::2, :],   # bottom-left
         x[...,  ::2, 1::2, :],   # top-right
         x[..., 1::2, 1::2, :]],  # bottom-right
        axis=-1
    )
# Input [B, H, W, C] → Output [B, H/2, W/2, 4C]
```

**ONNX export issue** — tf2onnx may fail to trace the strided-slice
concatenation as a single op.  Two mitigation strategies:

1. **Use `SpaceToDepth` (preferred)**
   Replace `Focus` before conversion with a subgraph:
   - `tf.nn.space_to_depth(x, block_size=2)` produces `[B, H/2, W/2, 4C]`
     *but* with a different channel interleave order than Focus.
   - Therefore follow `SpaceToDepth` with a `tf.gather` along axis 3
     to reorder channels to match the original Focus slice order
     `[TL, BL, TR, BR]` → SpaceToDepth order `[TL, TR, BL, BR]`.
   - This ensures numerically identical output.
   - ONNX opset 13+ supports `SpaceToDepth` natively.

2. **Rewrite as explicit split + concat** (fallback)
   Before calling `tf2onnx`, monkey-patch `Focus.call` to use only
   `tf.strided_slice` and `tf.concat` ops, which tf2onnx traces
   reliably into `Slice` + `Concat` ONNX ops.

**Recommended approach**: use option 1 (SpaceToDepth + channel
reorder).  Verify numerically on a random input before committing.

### SiLU Activation (`nets/CSPdarknet.py`)

**What it does** — Swish/SiLU: `f(x) = x * sigmoid(x)`.

```python
def call(self, inputs):
    return inputs * K.sigmoid(inputs)
```

**ONNX support**:
- ONNX opset 14 added the `Swish` op, which is identical to SiLU.
- tf2onnx 1.16+ automatically emits `Mul(x, Sigmoid(x))` which
  Acuity Toolkit 5.24.6 supports via fused-op recognition.
- **No replacement is needed** if targeting opset 14.

If the downstream tool requires opset ≤ 13:
- Replace `SiLU` with `HardSwish`: `f(x) = x * relu6(x+3) / 6`.
  This approximates SiLU and is supported from opset 12.
- Expected accuracy impact: < 0.5% mAP drop (empirical on COCO).

---

## Conversion Command

Export to a `.pb` SavedModel first, then convert with tf2onnx:

```bash
# Step 1 — export Keras model to TF SavedModel format
python - <<'PY'
import tensorflow as tf
from tensorflow.keras.layers import Input
from nets.yolo import yolo_body

anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
inp = Input(shape=(640,640,3), name='input_1')
model = yolo_body(inp, anchors_mask, num_classes=5, phi='m')
model.load_weights('model_data/best_epoch_weights.h5')
model.save('/tmp/yolov5m_savedmodel', save_format='tf')
print("SavedModel written to /tmp/yolov5m_savedmodel")
PY

# Step 2 — convert SavedModel → ONNX (opset 14)
python -m tf2onnx.convert \
  --saved-model /tmp/yolov5m_savedmodel \
  --opset 14 \
  --output c++/data/model/yolov5m.onnx \
  --verbose
```

If the Focus layer causes a tracing error, add the custom-op
override flag:

```bash
python -m tf2onnx.convert \
  --saved-model /tmp/yolov5m_savedmodel \
  --opset 14 \
  --output c++/data/model/yolov5m.onnx \
  --custom-ops nets.CSPdarknet.Focus \
  --verbose
```

Expected output file: `c++/data/model/yolov5m.onnx` (~85 MB).

---

## Validation

Compare onnxruntime output with the original TF model output on
the same image.  Both forward passes must produce numerically
identical results (tolerance 1e-4 per element).

```python
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tensorflow.keras.layers import Input
from nets.yolo import yolo_body
from PIL import Image

# ── load test image ──────────────────────────────────────────
img   = Image.open('test/sample.jpg').resize((640, 640))
data  = np.array(img, dtype=np.float32)[np.newaxis, ...] / 255.0  # [1,640,640,3]

# ── TensorFlow reference output ───────────────────────────────
anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
inp   = Input(shape=(640,640,3), name='input_1')
model = yolo_body(inp, [[6,7,8],[3,4,5],[0,1,2]], num_classes=5, phi='m')
model.load_weights('model_data/best_epoch_weights.h5')
tf_out = model.predict(data)   # list of 3 arrays

# ── ONNX output ──────────────────────────────────────────────
sess    = ort.InferenceSession('c++/data/model/yolov5m.onnx',
                               providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
onnx_out = sess.run(None, {in_name: data})

# ── compare ──────────────────────────────────────────────────
for i, (t, o) in enumerate(zip(tf_out, onnx_out)):
    diff = np.abs(t - o).max()
    status = 'PASS' if diff < 1e-4 else 'FAIL'
    print(f'Head {i}: max_diff={diff:.2e}  [{status}]')
```

All three heads must print `PASS`.

---

## Potential Issues and Mitigations

| Issue | Root cause | Mitigation |
|-------|-----------|------------|
| Focus layer tracing fails | tf2onnx cannot trace strided-slice concat | Replace with SpaceToDepth + channel reorder before conversion (see above) |
| SiLU exported as Mul+Sigmoid, not fused | opset 13 | Use opset 14 (`--opset 14`) |
| HardSwish fallback hurts mAP | Activation approximation | Use opset 14 to avoid substitution |
| Shape mismatch on output heads | Model loaded without `include_top` | Always use `yolo_body()` from `nets/yolo.py`, not `load_model()` |
| tf2onnx fails with TF 2.2 | API changes in newer tf2onnx | Pin to tf2onnx==1.16.1 exactly |
| Large ONNX file (>100 MB) | External data not split | Add `--large_model` flag to tf2onnx if needed |
| BatchNormalization folded differently | Training vs inference mode | Call `model(data, training=False)` for TF reference output |
| Anchors hardcoded in ONNX graph | tf2onnx folds constants | This is expected and correct; no action needed |

---

## Output Artifacts

| File | Description |
|------|-------------|
| `c++/data/model/yolov5m.onnx` | Converted ONNX model, input `[1,640,640,3]` float32 |
| `/tmp/yolov5m_savedmodel/` | Intermediate TF SavedModel (can be deleted after ONNX is verified) |
