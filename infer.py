import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 装甲板视频推理")
    parser.add_argument("--weights", type=str, default="best.pt", help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output", type=str, default="runs/infer/result.mp4", help="输出视频路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", type=str, default="0", help="0 表示 GPU")
    return parser.parse_args()

def main():
    args = parse_args()
    weights, source, output = Path(args.weights), Path(args.source), Path(args.output)
    if not weights.exists(): raise FileNotFoundError(f"找不到模型权重：{weights.resolve()}")
    if not source.exists(): raise FileNotFoundError(f"找不到输入视频：{source.resolve()}")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise RuntimeError(f"无法打开视频：{source.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened(): raise RuntimeError("无法创建输出视频，请尝试将 --output 改为 .avi")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1
        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        result = results[0]
        annotated = result.plot()

        # 额外画出 4 个关键点的编号（1-4），方便作业展示
        if result.keypoints is not None and result.keypoints.xy is not None:
            for i in range(len(result.keypoints.xy)):
                pts = result.keypoints.xy[i].cpu().numpy().astype(int)
                for j, (px, py) in enumerate(pts):
                    cv2.circle(annotated, (int(px), int(py)), 4, (0, 0, 255), -1)
                    cv2.putText(annotated, str(j + 1), (int(px) + 5, int(py) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        writer.write(annotated)

        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes[i].cls.item())
                conf = float(result.boxes[i].conf.item())
                xyxy = result.boxes[i].xyxy.cpu().numpy().astype(int).tolist()
                name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                kpts = result.keypoints.xy[i].cpu().numpy().astype(int).tolist() if result.keypoints is not None and i < len(result.keypoints.xy) else []
                print(f"[frame {frame_id}] {name} conf={conf:.3f} box={xyxy} kpts={kpts}")

    cap.release()
    writer.release()
    print(f"完成！结果视频已保存到：{output.resolve()}")

if __name__ == "__main__":
    main()