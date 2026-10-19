# -*- coding: utf-8 -*-
"""摄像头实时检测演示（俯视安装：相机装在夹爪正上方，垂直向下拍小车顶面）。

测量模型见 tag_utils.py：小车绕管道轴滚转 phi 时，标签中心在图像里
沿垂直于管道的方向平移，所以 phi 由中心位移反解（不再用面内角度）。

操作:
  - 相机大致垂直向下，标签贴小车顶面、上沿箭头平行于管道轴线
  - 把小车摆到夹爪正好能抓取的位置，按 r : 记录参考位
  - 手动转动/移动标签，屏幕显示 phi(周向滚转角)/psi(歪头角)/arc(应走弧长)/axial(轴向偏差)
  - |phi| < 4° 时文字变绿（夹取判据）
  - 按 s : 保存当前帧到 输出/
  - 按 q : 退出

运行: python live_detect.py [摄像头序号, 默认0] [宽度, 默认1280]
"""
import os
import sys
import math
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_utils import detect_tag, delta_report, arc_mm_for, imwrite_u

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, "..", "输出")

cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# 分辨率可选第2个参数: python live_detect.py 0 960
# 树莓派4B建议960x540(检测更快); Windows桌面默认720p
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
H = round(W * 9 / 16)
if sys.platform == "win32":                       # Windows 用 DSHOW 打开更快
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
else:                                             # 树莓派/Linux
    cap = cv2.VideoCapture(cam_index)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)   # 台架测试用自动对焦; 实机定焦镜头此设置无效

print("俯视模式: 标签贴小车顶面, 上沿箭头平行管道轴线; "
      "小车摆到夹爪对中位置后按 r 记录参考位")

ref = None
fps_t = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("读取摄像头失败")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tag = detect_tag(gray)
    disp = frame.copy()

    if tag is not None:
        c = (int(tag["u"]), int(tag["v"]))
        cv2.polylines(disp, [tag["corners"].astype(np.int32)], True, (0, 0, 255), 2)
        cv2.circle(disp, c, 5, (0, 255, 0), -1)
        txt = f"u={tag['u']:.0f} v={tag['v']:.0f} edge={tag['edge_a']:.0f}px"
        cv2.putText(disp, txt, (c[0] + 10, c[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if ref is not None:
            r = delta_report(tag, ref)
            # 俯视映射: phi=周向滚转角(位置法反解), arc=周向纠偏应走弧长,
            # axial=沿管道方向的进/退偏差(夹取深度用)
            line2 = (f"phi={r['phi_deg']:+.2f} psi={r['psi_deg']:+.2f} "
                     f"(shape~{r['phi_shape_deg']:.2f})  "
                     f"arc={arc_mm_for(r['phi_deg']):+.1f}mm  "
                     f"axial={r['axial_mm']:+.1f}mm")
            color = (0, 255, 0) if abs(r["phi_deg"]) < 4.0 else (0, 0, 255)  # 夹爪容差±5°,目标±4°
            cv2.putText(disp, line2, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # 参考位十字 + 位移箭头 + 管道方向线(紫色, 方向来自参考位theta)
            pr = (int(ref["u"]), int(ref["v"]))
            cv2.drawMarker(disp, pr, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.arrowedLine(disp, pr, c, (255, 0, 255), 2, tipLength=0.08)
            L = tag["edge_a"] / 2 + 50
            dx, dy = math.cos(ref["theta"]), math.sin(ref["theta"])
            cv2.line(disp, (int(tag["u"] - L * dx), int(tag["v"] - L * dy)),
                     (int(tag["u"] + L * dx), int(tag["v"] + L * dy)),
                     (255, 0, 255), 1)
    else:
        cv2.putText(disp, "NO TAG - check light/blur", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_t, 1e-6))
    fps_t = now
    status = "REF SET" if ref else "press r to set REF"
    cv2.putText(disp, f"{fps:.0f}fps  [{status}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("AprilTag top-view (r=ref, s=save, q=quit)", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        if tag is not None:
            ref = tag
            print(f"参考位已记录: theta={math.degrees(ref['theta']):+.3f}deg (图像里的管道方向)")
        else:
            print("当前没有识别到标签, 无法记录参考位")
    elif key == ord("s") and ok:
        p = os.path.abspath(os.path.join(OUT_DIR, f"live_{int(time.time())}.png"))
        imwrite_u(p, frame)
        print("已保存:", os.path.abspath(p))

cap.release()
cv2.destroyAllWindows()