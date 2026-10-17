# Tang GPU 实施计划（尚未实际运行）

设备历史记录：Tang 曾配置 RTX 4090 Laptop 16 GB VRAM、Ubuntu 22.04；这些属于历史信息，**训练前必须重新核实**。

进入训练前：

```bash
nvidia-smi
python --version
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('available:', torch.cuda.is_available())
PY
```

建议先训练 `yolov8n.pt`、batch 8，发生 OOM 降为 4 或 2。不要同时运行 Isaac Sim、LLM GRPO 和 BDD 检测训练。确认可用 VRAM 后才能扩大图片分辨率与模型规模。

日志至少包含：时间、Git commit、实际运行命令、数据版本与划分、GPU 型号、PyTorch/CUDA 版本、模型权重 hash、seed、完整运行日志、结果、问题与修复、下一阶段目标。

仓库为唯一事实源：GitHub 放代码、固定配置、实验总结；真实训练/benchmark 在 Tang GPU 执行，原始数据、权重和大日志不上传普通 GitHub 仓库。必要时用受控对象存储保存大型产物，并在 GitHub 文档只记录索引与校验和。
