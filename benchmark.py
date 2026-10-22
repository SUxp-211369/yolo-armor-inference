import os
import sys

# 同理，加载 dll 路径
site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
nvidia_dirs = [
    os.path.join(site_packages, "nvidia", "cudnn", "bin"),
    os.path.join(site_packages, "nvidia", "cublas", "bin"),
    os.path.join(site_packages, "nvidia", "cuda_nvrtc", "bin")
]
for d in nvidia_dirs:
    if os.path.exists(d):
        os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass

import time
import numpy as np
import cv2
from ultralytics import YOLO
import onnxruntime as ort
from onnx_infer import letterbox

IMG_PATH = "test.jpg"
PT_MODEL_PATH = "best.pt"
ONNX_MODEL_PATH = "best.onnx"

def test_pt():
    model = YOLO(PT_MODEL_PATH)
    
    # 预热，消除首次加载的额外开销
    for _ in range(5):
        model.predict(IMG_PATH, verbose=False, device=0)
        
    times = []
    # 跑 50 次取平均
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict(IMG_PATH, verbose=False, device=0)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

def test_onnx():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    
    # 提前把数据处理好，排除预处理的时间干扰
    img0 = cv2.imread(IMG_PATH)
    img_640, _, _ = letterbox(img0, (640, 640))
    img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
    blob = img_rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

    # 预热
    for _ in range(5):
        session.run(None, {input_name: blob})
        
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        session.run(None, {input_name: blob})
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)

if __name__ == "__main__":
    print("开始测试 PyTorch...")
    pt_mean, pt_std = test_pt()
    
    print("开始测试 ONNX...")
    onnx_mean, onnx_std = test_onnx()
    
    print(f"\nPyTorch 耗时: {pt_mean:.2f} ± {pt_std:.2f} ms")
    print(f"ONNX 耗时: {onnx_mean:.2f} ± {onnx_std:.2f} ms")
    
    # 算一下提升倍数
    if onnx_mean < pt_mean:
        print(f"ONNX 比 PyTorch 快了约 {pt_mean / onnx_mean:.2f} 倍")