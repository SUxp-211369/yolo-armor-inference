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

---

# YOLO 装甲板识别与工业相机实时推理实验

## 1. 实验环境与设备
- Windows 端：Python 3.11.9, NVIDIA RTX 5070, torch 2.11.0+cu128
- Linux 端：Ubuntu 24.04 (VMware虚拟机, CPU推理), 海康威视 MV-CS016-10UC 工业相机

## 2. 项目文件说明
- `infer.py` / `export.py` / `benchmark.py`：Windows 端视频推理与模型导出代码
- `run_camera_yolo.py`：Linux 端海康相机实时取流+YOLO推理代码（本次新增）
- `best.pt` / `best.onnx`：模型权重

## 3. 运行方式
(1) Windows 视频推理（GPU）：
`python infer.py --weights best.pt --source raw_video.mp4 --output runs/infer/result.mp4 --conf 0.25 --device 0`

(2) Linux 工业相机实时推理：
`python3 run_camera_yolo.py 3000 0`
*快捷键：按 S 保存截图，按 Y 开关 YOLO，按 Q 退出并释放相机资源。*

## 4. 实验数据与结果分析（截图见 images 文件夹）

### 4.1 曝光测试
- 500us：画面极暗，YOLO 丢失特征（无检测框）。
- 3000us：装甲板轮廓清晰，YOLO 以 0.77 置信度成功识别出 B1 装甲板。
- 运动模糊：CPU 推理帧率仅约 5 FPS，高曝光下移动物体会产生严重拖影导致漏检。

### 4.2 增益测试
- 固定 500us 曝光，将增益提升至 12dB。画面变亮，YOLO 成功识别（置信度 0.66），但暗部噪点明显增加，容易导致误识别。

### 4.3 格式与资源管理
- 图像格式：程序强制相机输出 RGB8 并转为 BGR，解决了拔插后红蓝反转或重影问题。
- 异常处理：在 finally 块中严格执行断流、关设备、销毁句柄。实测拔掉相机后能优雅退出，重插后可正常恢复。