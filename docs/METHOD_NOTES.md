# 方法说明：边界条件与假设

## 1. 分类级别

BDD 2D detection 的 8 个动态目标类不会自动合并；尤其 `rider`、`bicycle`、`motorcycle` 类之间需要特别关注类别错位和标注歧义。

## 2. 三路选择

分数 `quality = w_c·confidence + w_t·support + w_a·class_agreement + w_s·area_stability - w_u·uncertainty`，当前默认系数位于 `configs/pseudo.yaml`。其中 `support=min(1,length/min_track_frames)×coverage`，`class_agreement` 为轨迹中多数类别的比例，`area_stability=1/(1+mean(abs(log(area ratio))))`，`uncertainty` 为同轨迹置信度标准差的有界变换。这是可消融的启发式，不是经验证的最优方法。

保守规则：短暂的行人、骑行者、摩托车与自行车轨迹即使置信度很高，仍送人工 review，不因时序证据不足而拒绝。某些遮挡/快速接近场景的面积变化是合理运动，所以 area_stability 权重必须通过消融证明。

## 3. ByteTrack 的作用与限制

使用 Ultralytics 内置 ByteTrack 跟踪器。跟踪 ID 在一个视频内部持续，换视频重新初始化模型与跟踪状态。不使用跟踪 ID 跨视频推断物体身份。

## 4. 为什么审计分数不能替代检测 mAP

`audit-pseudo` 计算的是所有 GT 帧上的 accepted 伪标签匹配情况，衡量筛选策略的 precision/recall；而 `eval` 才评估目标检测器的 COCO-style mAP。两者不是同一个指标。

## 5. 预期失败模式

低光照、遮挡、模糊、雨天、相机运动、高速目标与剧烈尺度变化都可能导致跟踪不稳定。用轨迹长度当作真值可能系统性排斥短暂重要事件。自动筛选质量不等于整帧标注完备性，必须核实漏检目标。
