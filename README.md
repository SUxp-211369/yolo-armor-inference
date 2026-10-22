# YOLO 装甲板视频推理与 ONNX 模型导出对比

## 1. 项目完成的功能
本项目包含两次作业的核心内容：
1. **视频推理**：使用自训练的 YOLOv8n-pose 装甲板模型（6 个类别：`B1, B2, B3, R1, R2, R3`），对 `raw_video.mp4` 进行推理，输出带检测框、类别、置信度和 4 个关键点的结果视频。
2. **模型导出与性能对比**：将 `.pt` 模型导出为 `.onnx` 格式，并编写原生推理代码，在 RTX 5070 GPU 环境下进行严格的性能对比测试。

## 2. Python 版本与依赖
- **Python**: 3.11.9
- **硬件**: NVIDIA RTX 5070 (8GB) + Windows 11
- **核心依赖**:
  - `torch==2.11.0+cu128` (适配 RTX 50 系 Blackwell 架构)
  - `onnxruntime-gpu==1.27.0` (使用微软官方 CUDA 12 源编译版本)
  - `ultralytics`, `opencv-python`, `numpy`

## 3. 模型权重准备与导出
- **原始权重**：`best.pt`（由于体积较小已包含在仓库中）。
- **导出方法**：运行 `python export.py`。核心参数为 `imgsz=640, opset=12, simplify=True`，生成的 `best.onnx` 约 9.7 MB。

## 4. 运行命令
```bash
# 1. 运行第一次作业的视频推理（输出到 runs/infer/result.mp4）
python infer.py --weights best.pt --source raw_video.mp4 --output runs/infer/result.mp4 --conf 0.25 --device 0

# 2. 运行第二次作业的 ONNX 图片推理并保存结果（输出到 results/onnx_result.jpg）
python onnx_infer.py

# 3. 运行性能对比测试（包含 5 次预热，50 次循环取平均）
python benchmark.py