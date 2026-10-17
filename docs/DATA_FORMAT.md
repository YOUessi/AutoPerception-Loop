# 数据格式与数据集隔离

## 1. 来源

- BDD100K 2D：`[{"name":"xxx.jpg", "attributes":{...}, "labels":[{"category":"car", "box2d":{"x1":1,"y1":2,"x2":30,"y2":40}}]}]`
- 转换后 `images/{train,val}/`、`labels/{train,val}/`，YOLO 标签行 `class cx cy w h`，坐标均为 `[0,1]`。
- 导出 classes 为 BDD 原始的 8 个动态目标类，交通牌、交通灯等不导出（可能造成非目标类背景混入；并不是误标成车辆）。

## 2. 跟踪 JSONL / 人工 GT JSONL 共同结构

一帧对应一行 JSON；详情见 `examples/frame_record.example.jsonl`。

```json
{
 "video_id": "clip_A", "frame_index": 0,
 "image_path": "/absolute/path/to/frame.jpg", "width": 1280, "height": 720,
 "source_split": "train",
 "detections": [
  {"bbox":{"x1":10,"y1":20,"x2":100,"y2":160},
   "class_id":2,"confidence":0.94,"track_id":7}
 ]
}
```

人工真值无需 confidence/track_id。检测 ID 与 BDD 类别一致：`0 pedestrian; 1 rider; 2 car; 3 truck; 4 bus; 5 train; 6 motorcycle; 7 bicycle`。

## 3. 划分规则与防泄漏

- 官方 `train` / `val` 必须保持不变，禁止把同一视频相邻帧随机拆为不同集合。
- 半监督模拟中，从 **训练分区** 隐藏某一部分标签，将之视为 `train/unlabeled`；该隐藏标签仅用作独立内部审计时要单独冻结，并与训练、阈值选择数据隔离。
- `select-pseudo` 会拒绝标记为 val/test 的帧；`make-student` 只从人工标注 train 与伪标注 train 合并，val 一律用人工标签。
- 本地数据 JSONL 中的绝对路径仅供本地训练使用，不可直接移动到另一个主机复用，迁移时需重新生成 manifest。
- 某一帧只要有接受之外的伪标签，就整帧送 review，不部分导出训练。即使全部预测框通过，**漏掉的真实物体依然不可由此发现**；需要独立 GT 抽样审计。
