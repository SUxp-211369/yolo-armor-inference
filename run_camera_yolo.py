import sys
import os
import time
import cv2
import numpy as np
from ctypes import *

# ==========================================
# 配置区
# ==========================================
MVS_SDK_PATH = "/opt/MVS/Samples/64/Python/MvImport"
MODEL_PATH = "best.pt"

DEFAULT_EXPOSURE = 3000.0  # 微秒
DEFAULT_GAIN = 0.0         # dB
# ==========================================

# 导入海康 SDK
sys.path.append(MVS_SDK_PATH)
try:
    from MvCameraControl_class import *
except Exception as e:
    print(f"❌ 导入海康SDK失败: {MVS_SDK_PATH}\n报错: {e}")
    sys.exit(1)

# 导入 YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    print(f"❌ 导入 Ultralytics 失败\n报错: {e}")
    sys.exit(1)

cam = None
model = None
current_pixel_format = "Unknown"

def init_yolo():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型文件: {MODEL_PATH}")
        sys.exit(1)
    print(f"⏳ 正在加载 YOLO 模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    print("✅ YOLO 模型加载成功！")

def init_camera(exposure, gain):
    global cam, current_pixel_format
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        print(f"❌ 未找到相机，错误码: {ret}")
        sys.exit(1)

    print(f"✅ 找到 {deviceList.nDeviceNum} 个设备")

    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print(f"❌ 创建句柄失败，错误码: {ret}")
        sys.exit(1)

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"❌ 打开设备失败，错误码: {ret}")
        sys.exit(1)

    # 1. 先确保关闭自动曝光，否则设置曝光时间无效
    cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    # 2. 设置曝光和增益
    cam.MV_CC_SetFloatValue("ExposureTime", exposure)
    cam.MV_CC_SetFloatValue("Gain", gain)
    # 3. 关闭触发模式，连续采集
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

    # 4. 强制设置相机输出 RGB8 格式，彻底解决拔插后变成三幅黑白图的问题
    # 0x02180014 是海康 RGB8 的固定枚举值
    ret_fmt = cam.MV_CC_SetEnumValue("PixelFormat", 0x02180014)
    if ret_fmt == 0:
        print("📷 成功强制设置相机格式为 RGB8 (彩色)")
        current_pixel_format = "RGB8"
    else:
        print(f"⚠️ 强制设置 RGB8 失败(错误码 {ret_fmt})，将使用默认格式解析。")
        current_pixel_format = "Unknown"

    print(f"✅ 相机参数设置: 曝光={exposure}us, 增益={gain}dB")

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"❌ 开始取流失败，错误码: {ret}")
        sys.exit(1)
    print("🎥 相机已开始取流...")

def release_resources():
    global cam
    print("\n🛑 正在释放资源...")
    if cam is not None:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    cv2.destroyAllWindows()
    print("✅ 资源释放完毕，程序安全退出。")

def main():
    exp = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXPOSURE
    gain = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GAIN

    init_yolo()
    init_camera(exp, gain)

    data_buf = (c_ubyte * (1920 * 1080 * 3))()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    frame_count = 0
    start_time = time.time()
    fps = 0.0
    show_yolo = True

    print("\n💡 快捷键说明：按 'Y' 切换 YOLO，按 'S' 保存截图，按 'Q' 退出")

    try:
        while True:
            memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
            ret = cam.MV_CC_GetOneFrameTimeout(data_buf, sizeof(data_buf), stFrameInfo, 1000)

            if ret != 0:
                continue

            w, h = stFrameInfo.nWidth, stFrameInfo.nHeight

            # 图像解析
            try:
                # 正常情况下，因为前面强制了 RGB8，这里按三通道解析即可
                frame_rgb = np.frombuffer(data_buf, dtype=np.uint8, count=w*h*3).reshape(h, w, 3).copy()
                # 将 RGB 转为 OpenCV 需要的 BGR
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                # 如果相机是纯黑白相机（不支持 RGB8），走备用解析逻辑
                try:
                    frame_raw = np.frombuffer(data_buf, dtype=np.uint8, count=w*h).reshape(h, w)
                    frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_GRAY2BGR)
                except:
                    print(f"⚠️ 画面解析失败，分辨率: {w}x{h}")
                    continue

            if show_yolo:
                results = model.predict(source=frame_bgr, conf=0.25, imgsz=640, device='cpu', verbose=False)
                display_frame = results[0].plot()
            else:
                display_frame = frame_bgr.copy()

            # FPS 计算
            frame_count += 1
            if time.time() - start_time >= 1.0:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            status = "YOLO: ON" if show_yolo else "YOLO: OFF"
            cv2.putText(display_frame, f"{status} | FPS: {fps:.1f} | Exp: {exp}us | Gain: {gain}dB",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Armor Detection (Press 'S' Save, 'Y' YOLO, 'Q' Quit)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('y') or key == ord('Y'):
                show_yolo = not show_yolo
                print(f"🔄 YOLO 状态: {'开启' if show_yolo else '关闭'}")
            elif key == ord('s'):
                filename = f"capture_exp{exp}_gain{gain}.png"
                cv2.imwrite(filename, display_frame)
                print(f"📸 已保存截图: {filename}")

    except KeyboardInterrupt:
        print("\n用户手动中断...")
    except Exception as e:
        print(f"\n❌ 程序发生异常: {e}")
    finally:
        release_resources()

if __name__ == "__main__":
    main()