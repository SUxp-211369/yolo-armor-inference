# YOLO 装甲板视频推理

## 1. 项目完成的功能
使用指定的 YOLO26n 装甲板四点模型 `best.pt`，对自拍装甲板视频进行推理，输出带检测框、类别、置信度和 4 个关键点的结果视频。

## 2. Python 版本与依赖
- Python 3.11.9
- 主要依赖见 `requirements.txt`
- 使用设备：GPU (RTX 5070)

## 3. 模型权重准备与放置
从作业材料下载 `best.pt`，放到 `D:\yolo_armor\best.pt`。
模型文件未提交至 Git 仓库。

## 4. 运行命令
```bash
python infer.py --weights best.pt --source raw_video.mp4 --output runs/infer/result.mp4 --conf 0.25 --device 0