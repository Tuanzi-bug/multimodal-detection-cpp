# 项目规则：multimodal-detection-cpp

本文件定义了该项目的个人工作规范，Claude 在每次会话中必须严格遵守。

---

## 1. Python 执行规范

**在运行任何 Python 脚本之前，必须：**

1. 检查是否存在适合的 conda 环境：
   ```bash
   conda env list
   ```
2. 若不存在，先创建并安装依赖：
   ```bash
   conda create -n <env-name> python=3.8 <packages> -y
   ```
3. 使用完整路径调用环境中的 Python（不使用 `conda activate`，因为子 shell 不继承激活状态）：
   ```bash
   /Users/yay/anaconda3/envs/<env-name>/bin/python script.py
   ```

**当前项目 conda 环境：**

| 环境名 | Python | 用途 | 主要包 |
|--------|--------|------|--------|
| `multimodal-cpp-tools` | 3.8 | 离线数据转换工具（Phase 0） | scipy, numpy, pytest |

---

## 2. Git / GitHub 版本管理规范

**全程使用 `gh` CLI 和 `git` 进行版本管理：**

- 仓库：https://github.com/Tuanzi-bug/multimodal-detection-cpp
- 主分支：`main`（稳定）
- 开发分支：`feature/cpp-port`（当前开发）

**Commit 规则：**

- 每个**子阶段**（如 0.1、0.2、0.3）完成后立即单独 commit，**不合并到一个大 commit**
- Commit message 格式：
  ```
  <type>(scope/phase-X.Y): <简短描述>

  <详细说明：做了什么、测试结果、产出文件>
  ```
- Type 参考：`feat`、`docs`、`test`、`chore`、`fix`、`refactor`
- 示例：
  ```
  feat(tools/phase-0.3): convert spectral library .mat to .bin

  18 unit tests passed; 21/21 files converted, max_diff=0
  ```

**PR 规则：**

- 所有 Phase 完成后，用 `gh pr create` 创建 PR 合并到 main
- PR body 包含：测试结果、mAP 对比、已知限制

---

## 3. 文档同步规范

**每个子阶段完成后，必须同步更新以下文档：**

| 文档 | 何时更新 | 更新内容 |
|------|----------|----------|
| `docs/tasks.md` | 每个子任务完成后 | 状态从 `⬜ pending` 改为 `✅ done`，填写 commit hash |
| `docs/progress.md` | 每个 Session 结束前 | 记录完成项、commit 列表、下一步 |
| `docs/findings.md` | 发现新技术细节时 | 记录算法细节、陷阱、验证结果 |

文档更新本身也需要 commit（可与最后一个子任务合并或单独提交）。

---

## 4. 执行节奏规范

**计划与确认：**

- 开始新 Phase 前，输出该 Phase 的完整执行计划（含测试策略）
- **等待用户明确确认后再开始执行**（"确认"、"继续"、"yes" 等均可）
- 用户也可说 "修改：..." 来调整计划

**Phase 隔离：**

- **一次只执行一个 Phase**，完成后主动停止并汇报结果
- 不得在用户确认前自动跳到下一个 Phase
- 每个 Phase 内部的子阶段可连续执行，无需逐一等待

---

## 5. TDD 规范（C++ 阶段）

**对于 C++ 实现阶段（Phase 2-7）：**

1. 先定义接口（头文件）
2. 先写测试（`c++/test/test_*.cpp`），运行确认 **FAIL**
3. 实现最小代码使测试 **PASS**
4. 重构（保持测试绿色）
5. 覆盖率目标：核心算法（ACE、后处理）100%，其余 ≥ 80%

**对于 Python 工具（Phase 0）：**

- 测试嵌入脚本文件末尾（`unittest.TestCase`）
- 用 `pytest <script.py>` 运行
- 先运行测试确认通过，再在真实数据上执行

---

## 6. 项目快速参考

```
Python 基准 mAP：
  YOLO 仅检测：90.53%
  多模态融合：92.34%

C++ 目标 mAP：
  YOLO：≥ 89%（允许 1.5% 量化损失）
  融合：≥ 91%

关键常量（config.h）：
  输入尺寸：640×640
  类别数：5（aircraft/car/oiltank/roof/ship）
  置信度阈值：0.05（mAP 评估）
  NMS IoU：0.3
  ACE 阈值：0.80
  光谱维度：129 bands

⚠️ 陷阱：
  YOLOv5 解码 ≠ YOLOv3（公式不同，不可复用 reference 代码）
  高光谱 transpose(2,1,0)：[H,W,B]→[B,W,H]，reshape 列顺序为(W,H)
```
