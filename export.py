import os
from ultralytics import YOLO

def main():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print(f"错误: 找不到 {model_path}")
        return

    model = YOLO(model_path)
    
    # 导出 ONNX 格式，opset 固定为 12 保证兼容性
    model.export(
        format="onnx", 
        imgsz=640, 
        opset=12, 
        simplify=True,
        dynamic=False 
    )
    print("导出完成。")

if __name__ == "__main__":
    main()