import os
import sys
import ctypes
import glob
# 1. 加载 NVIDIA 动态链接库 

site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
nvidia_base = os.path.join(site_packages, "nvidia")

if os.path.exists(nvidia_base):
    for pkg_dir in glob.glob(os.path.join(nvidia_base, "*", "bin")):
        os.environ["PATH"] = pkg_dir + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(pkg_dir)
            except OSError:
                pass
                
        # 强制预加载该目录下的所有 DLL，防止 onnxruntime 找不到
        for dll_path in glob.glob(os.path.join(pkg_dir, "*.dll")):
            try:
                ctypes.WinDLL(dll_path)
            except OSError:
                pass

import cv2
import numpy as np
import onnxruntime as ort

# 装甲板类别
CLASSES = ['B1', 'B2', 'B3', 'R1', 'R2', 'R3']
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # 等比例缩放并填充灰边
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def process_image(img_path, onnx_path="best.onnx"):
    # 优先使用 CUDA，回退到 CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    img0 = cv2.imread(img_path)
    if img0 is None:
        print(f"错误: 找不到图片 {img_path}")
        return

    # 图像预处理
    img_640, ratio, (dw, dh) = letterbox(img0, (640, 640))
    img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
    blob = img_rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

    # 执行推理
    output = session.run(None, {input_name: blob})[0]
    
    # 输出形状转换: (1, 18, 8400) -> (8400, 18)
    output = np.transpose(np.squeeze(output)) 
    
    boxes, scores, class_ids, keypoints_list = [], [], [], []

    for row in output:
        # row 的结构: [x, y, w, h, 类别1~6分数, 4个关键点x,y]
        cls_scores = row[4:10]
        class_id = np.argmax(cls_scores)
        score = cls_scores[class_id]
        
        if score > 0.5:
            # 还原坐标到原图尺寸
            x, y, w, h = row[0:4]
            left = (x - w/2 - dw) / ratio
            top = (y - h/2 - dh) / ratio
            w = w / ratio
            h = h / ratio
            boxes.append([int(left), int(top), int(w), int(h)])
            scores.append(float(score))
            class_ids.append(int(class_id))
            
            # 提取关键点坐标
            kpts = []
            for i in range(4):
                kx = (row[10 + i*2] - dw) / ratio
                ky = (row[11 + i*2] - dh) / ratio
                kpts.append([int(kx), int(ky)])
            keypoints_list.append(kpts)

    # NMS 过滤重叠框
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.45)
    
    # 绘制结果
    if len(indices) > 0:
        # 兼容不同版本 OpenCV 返回格式 (防报错)
        indices = np.array(indices).flatten()
        for idx in indices:
            x, y, w, h = boxes[idx]
            color = COLORS[class_ids[idx] % len(COLORS)]
            
            # 画框
            cv2.rectangle(img0, (x, y), (x + w, y + h), color, 2)
            # 写标签
            cv2.putText(img0, f"{CLASSES[class_ids[idx]]} {scores[idx]:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 画四个角的关键点
            for pt in keypoints_list[idx]:
                cv2.circle(img0, (pt[0], pt[1]), 3, (0, 0, 255), -1)
                
    # 保存结果
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/onnx_result.jpg", img0)
    print("结果已保存至 results/onnx_result.jpg")

if __name__ == "__main__":
    process_image("test.jpg")