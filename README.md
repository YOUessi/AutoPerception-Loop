# AutoPerception-Loop

**自动驾驶视觉感知 · 自动标注 · 时序伪标签选择 · 数据闭环**

> Status: `v0.1.0 - code/bootstrap`. This repository includes initial **implemented** pipelines and CPU tests. **No real GPU training, KITTI/BDD benchmark numbers, or validated accuracy improvements are claimed.** This is research code, **not** a safety-certified perception component.

## 目标与研究假设

目标岗位：自动驾驶感知算法实习生（感知模型迭代、数据自动标注）。核心问题：有少量标注数据和大量未标注驾驶视频时，如何通过目标跟踪、时序一致性、困难样本复核，改善伪标签质量和检测模型泛化？

研究假设（**待实验验证**）：相对于仅使用置信度阈值，在固定标注预算下，结合跟踪支持度、类别一致性和尺度稳定性，有可能提高伪标签质量；**短暂出现的行人/骑行者不能因缺少轨迹长度而被自动排除**。

### 已实现 / 未实现边界

| 功能 | 状态 |
|---|---|
| BDD100K 2D Detection JSON → YOLO（官方 train/val 保留） | ✅ 代码已实现、合成数据单测 |
| YOLO 检测训练 / COCO 风格 mAP 评测入口 | ✅ 代码已实现，待 Tang 验证 |
| YOLO + ByteTrack 视频跟踪、帧导出 JSONL | ✅ 代码已实现，待真实视频验证 |
| 三种伪标签筛选方法：置信度 / 时序 / 时序+不确定性 | ✅ 代码已实现，待统计显著性测试 |
| 保守伪标签导出、review 队列、训练集合并、标签审计 | ✅ 代码已实现、合成数据单测 |
| 真正在线 EMA Teacher/Student 双模型联合训练（Soft Teacher 完整复现） | ⏳ 待开发 |
| BDD100K 真实数据训练、人工复核耗时统计、完整消融 | ⏳ 待 Tang 运行 |
| PointPillars/KITTI 3D、相机-点云融合 | ⏳ 第二阶段，不假装已实现 |

## 快速开始（Tang Linux + Python 3.10/3.11）

```bash
git clone https://github.com/YOUessi/AutoPerception-Loop.git
cd AutoPerception-Loop
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[vision,dev]'
pytest -q
autoperception --help
```

**注意**：`[vision]` 安装 Ultralytics，第三方库可能采用 AGPL-3.0 或其他独立许可；本仓库 MIT 仅适用于本仓库自编代码。企业复用请单独完成第三方许可评估。训练需配套 CUDA PyTorch，CPU-only CI 不包含 Ultralytics 实际推理。

### 1. 获取 BDD100K 数据

数据集文件按官方规则自行下载；勿将数据/权重上传 GitHub。需要 `labels/det_20/det_train.json`、`det_val.json` 和对应图像（文件名、路径以实际官方数据包为准）。项目不会绕过数据集授权。

```text
data/bdd100k/
  images/100k/train/*.jpg
  images/100k/val/*.jpg
  labels/det_20/det_train.json
  labels/det_20/det_val.json
```

### 2. BDD 2D 检测训练集转换

```bash
autoperception prepare-bdd \
  --annotations data/bdd100k/labels/det_20/det_train.json \
  --images-dir data/bdd100k/images/100k/train \
  --out data/bdd-yolo --split train
autoperception prepare-bdd \
  --annotations data/bdd100k/labels/det_20/det_val.json \
  --images-dir data/bdd100k/images/100k/val \
  --out data/bdd-yolo --split val
```

BDD 类别固定 8 个：`pedestrian, rider, car, truck, bus, train, motorcycle, bicycle`。交通灯/交通标志等其他类不会导出；如果以后需要完整 BDD 类别，必须明确更改配置并重新标注训练，不能混用 80 类 COCO 模型的输出 ID。此转换器不会自动切分数据；必须使用官方提供的 train/val，严禁相邻视频帧随机拆分。

### 3. 监督学习基线 / mAP 评测

```bash
autoperception train --data data/bdd-yolo/data.yaml --weights yolov8n.pt \
  --name b0_supervised_seed42 --out runs/det --epochs 30 --batch 8 --seed 42
autoperception eval --data data/bdd-yolo/data.yaml \
  --weights runs/det/b0_supervised_seed42/weights/best.pt --out runs/eval_b0
```

第一个训练 run 可以从 COCO 预训练骨干初始化，训练器将按 BDD 类数适配检测头。**不能直接拿未经 BDD 微调的 COCO 模型生成 BDD 8 类伪标签**：两者的类别 ID 不对应。

### 4. 未标注视频跟踪和伪标签

准备一段 **训练分区中未标注的原始视频**，使用已在 BDD 上训练的检测器：

```bash
autoperception track-video \
  --video data/unlabeled/train/clip_demo.mp4 \
  --weights runs/det/b0_supervised_seed42/weights/best.pt \
  --frames-dir data/track_frames \
  --out outputs/tracks/clip_demo.jsonl --source-split train

autoperception select-pseudo \
  --tracks outputs/tracks/clip_demo.jsonl \
  --config configs/pseudo.yaml \
  --out data/pseudo_round1

# 研究消融：只换筛选策略，不换教师权重/验证集/采样视频
# configs/pseudo_confidence.yaml / pseudo_temporal.yaml / pseudo.yaml
```

`data/pseudo_round1/review_queue.jsonl` 保存需要人工复核的检测结果；`decisions.jsonl` 保留每帧接受/复核/拒绝记录；`summary.json` 包含接受数量和筛选方法。**不完整的帧不进入训练集**，避免把被拒绝目标默认为背景。未检出的目标仍然可能污染伪标签，必须抽样人工审计。

### 5. 伪标签审计 + 学生模型训练

使用**独立、未参与筛选调参**的人工 GT（JSONL 格式详见 `docs/DATA_FORMAT.md`）对 `decisions.jsonl` 评估伪标签。实验可在训练集内部预留一组人工标注的研究审计片段作为“伪装未标注”素材；绝不把官方 `val/test` 并入训练。

```bash
autoperception audit-pseudo \
  --decisions data/pseudo_round1/decisions.jsonl \
  --ground-truth data/audit/ground_truth.jsonl \
  --out outputs/pseudo_audit.json

autoperception make-student \
  --labeled data/bdd-yolo --pseudo data/pseudo_round1 --out data/student_round1

autoperception train --data data/student_round1/data.yaml \
  --weights runs/det/b0_supervised_seed42/weights/best.pt \
  --name b1_student_seed42 --out runs/det --epochs 30 --batch 8 --seed 42
```

当前是可复现的**离线自训练回合**：教师推理 → 伪标签选择 → 学生微调 → 冻结验证。**不是**完整 Soft Teacher 在线 EMA 更新实现，参见 `docs/EXPERIMENT_PROTOCOL.md`。

## 项目结构

```text
src/auto_perception/
  bdd.py          BDD 标签转换与 manifest
  track.py        YOLO + ByteTrack 推理
  selection.py    时序质量评分与三路决策
  pseudo.py       训练伪标签导出、人工复核
  metrics.py      带 GT 的接受质量评估
  mix.py          标签来源记录及学生数据集构建
  training.py     YOLO 基线训练与验证适配器
  cli.py          统一命令行
configs/          固定数据配方、评测/消融配置
tests/            CPU 单元测试
docs/             数据格式、方法设计、实验协议与日期日志
.github/workflows CI 中的无 GPU 测试
```

## 科研原则

1. **先冻结验证集**，再调伪标签策略；不同模型对照必须共享测试分区。
2. B0/B1/B3/B4 与 B2 完整 Soft Teacher **严格区分**，尚未实现的对照不要报告成已运行。
3. 设定 `seed`、数据版本、教师检查点 SHA、CUDA/框架版本，保存完整运行命令。
4. 不仅比较 mAP，还比较行人/骑行者召回、伪标签 Precision/Recall、人工复核耗时，以及天气/昼夜子分布。
5. 结果必须是实际测得，**不填写示意数值**。负面结果照实记录。

## 参考工程（各自遵循其开源许可）

- BDD100K: https://github.com/bdd100k/bdd100k
- ByteTrack: https://github.com/FoundationVision/ByteTrack
- Soft Teacher: https://github.com/microsoft/SoftTeacher
- OpenPCDet (下一阶段): https://github.com/open-mmlab/OpenPCDet
- CVAT: https://github.com/cvat-ai/cvat

## 安全、隐私和许可

该仓库面向离线研究，不能直接用于真实道路车辆控制。道路视频可能含人脸、车牌、地理位置，禁止未经授权向公共仓库上传原视频、原始标签及个人信息。`data/`, `runs/`, `outputs/` 已默认忽略。
