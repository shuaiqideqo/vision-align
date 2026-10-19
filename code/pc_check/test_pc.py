# -*- coding: utf-8 -*-
"""固件板端 AprilTag 检测器的 PC 端验证 (编译真实固件 tag_detector.cpp)。

流程:
  1. 用 zig(pip 装的 ziglang) 把 摄像头子机/tag_detector.cpp 编译成 test.exe
  2. 合成 320x240 灰度场景 (AprilTag ID0, 旋转/平移/缩放/透视/噪声/模糊)
  3. 逐帧调用 test.exe, 校验: ID=0、角点误差、theta、以及 phi/psi/arc 解算

运行: conda activate vision && python test_pc.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
FW_DIR = r"C:\Users\Administrator\Desktop\wallbot\摄像头子机"
EXE = os.path.join(HERE, "test.exe")
FRAME = os.path.join(HERE, "frame.bin")

W, H = 320, 240
TAG_MM = 50.0          # 与固件 config.h 一致
R_TAG_MM = 60.0
PIPE_R = 60.0
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def build():
    """编译固件 tag_detector.cpp + main.cpp (拷到 ASCII 路径避开编码问题)"""
    import shutil
    tmp = os.path.join(os.environ.get("TEMP", tempfile.gettempdir()), "fw_check")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    for f in ("tag_detector.cpp", "tag_detector.h", "config.h", "tag36h11_table.h"):
        shutil.copy(os.path.join(FW_DIR, f), os.path.join(tmp, f))
    py = sys.executable
    cmd = [py, "-m", "ziglang", "c++", "-O2",
           os.path.join(HERE, "main.cpp"),
           os.path.join(tmp, "tag_detector.cpp"),
           "-I", os.path.join(HERE, "shims"),
           "-I", tmp,
           "-o", os.path.join(tmp, "test.exe")]
    if os.environ.get("DET_DEBUG"):
        cmd.append("-DDET_DEBUG")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        sys.exit("编译失败")
    global EXE
    EXE = os.path.join(tmp, "test.exe")
    print("[build] ok")


def make_frame(rot_deg=0.0, center=(160.0, 120.0), edge_px=100.0,
               tilt=0.0, noise=0.0, blur=0):
    """合成一帧 320x240 灰度图, 返回 (gray, gt_corners)。"""
    marker_px = 200
    try:
        marker = cv2.aruco.generateImageMarker(ARUCO_DICT, 0, marker_px)
    except AttributeError:
        marker = cv2.aruco.drawMarker(ARUCO_DICT, 0, marker_px)
    pad = 50
    tag = cv2.copyMakeBorder(marker, pad, pad, pad, pad,
                             cv2.BORDER_CONSTANT, value=255)

    # 源图上黑框外角 (TL TR BR BL)
    src = np.float32([[pad, pad], [pad + marker_px, pad],
                      [pad + marker_px, pad + marker_px], [pad, pad + marker_px]])
    # 目标: 先绕黑框中心转 rot, 缩放到 edge_px, 平移到 center, 可选透视歪斜
    scale = edge_px / marker_px
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    R = np.float32([[ca, -sa], [sa, ca]])

    def xform(p):
        v = R.dot(np.float32([p[0] - pad - marker_px / 2, p[1] - pad - marker_px / 2]))
        v = v * scale + np.float32(center)
        return v

    dst = np.float32([xform(p) for p in src])
    if tilt:  # 简易透视: 顶边向右推, 底边向左推 (模拟相机不垂直)
        dst[0][0] += tilt
        dst[1][0] += tilt
        dst[2][0] -= tilt
        dst[3][0] -= tilt
    M = cv2.getPerspectiveTransform(src, dst)

    scene = cv2.warpPerspective(tag, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderValue=255)
    if blur:
        scene = cv2.GaussianBlur(scene, (blur, blur), 0)
    if noise:
        scene = np.clip(scene.astype(np.float64) +
                        np.random.normal(0, noise, scene.shape), 0, 255).astype(np.uint8)
    return scene, dst


def run_exe(frame):
    frame.tofile(FRAME)
    r = subprocess.run([EXE, FRAME, str(W), str(H)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = r.stdout.strip()
    if r.returncode != 0 or not m.startswith("OK"):
        return None
    import re as _re
    kv = {k: float(v) for k, v in _re.findall(r"(\w+)=([-\d.]+)", m)}
    cs = _re.findall(r"\(([-\d.]+),([-\d.]+)\)", m)
    kv["corners"] = [(float(x), float(y)) for x, y in cs]
    return kv


def err(msg):
    print("  [FAIL]", msg)
    return 1


def check(name, rot, center, edge_px, tilt=0.0, noise=0.0, blur=0):
    print(f"[{name}] rot={rot} center={center} edge={edge_px} tilt={tilt} noise={noise} blur={blur}")
    frame, gt = make_frame(rot, center, edge_px, tilt, noise, blur)
    o = run_exe(frame)
    fails = 0
    if o is None:
        return err("未检出")
    if int(o["id"]) != 0:
        fails += err(f"id={o['id']} 应为 0")
    if o["hamming"] > 1:
        fails += err(f"hamming={o['hamming']}")
    for i in range(4):
        d = math.hypot(o["corners"][i][0] - gt[i][0], o["corners"][i][1] - gt[i][1])
        if d > 2.0:
            fails += err(f"角点{i} 误差 {d:.2f}px > 2.0  det={o['corners'][i]} gt={tuple(gt[i])}")
    th_gt = math.atan2(gt[1][1] - gt[0][1], gt[1][0] - gt[0][0])
    dth = abs((o["theta_deg"] - th_gt * 180 / math.pi + 90 + 180) % 180 - 90)
    if dth > 1.0:
        fails += err(f"theta 误差 {dth:.2f}deg")
    edge_gt = 0.5 * (math.hypot(gt[1][0] - gt[0][0], gt[1][1] - gt[0][1]) +
                     math.hypot(gt[3][0] - gt[2][0], gt[3][1] - gt[2][1]))
    if abs(o["edge_a"] - edge_gt) / edge_gt > 0.05:
        fails += err(f"edge_a={o['edge_a']:.1f} 应为 {edge_gt:.1f}")
    print("  " + ("PASS" if fails == 0 else "FAIL") +
          f"  hamming={int(o['hamming'])} margin={o['margin']:.0f} contrast={o['contrast']:.0f}")
    return fails


def measure(name, ref_frame_kw, meas_frame_kw, expect):
    """参考帧 + 测量帧 -> 固件输出的 phi/psi/arc 是否符合几何预期"""
    print(f"[{name}] expect={expect}")
    fails = 0
    _, ref_gt = make_frame(**ref_frame_kw)
    o = run_exe(make_frame(**ref_frame_kw)[0])
    if o is None:
        return err("参考帧未检出")
    th0 = math.atan2(ref_gt[1][1] - ref_gt[0][1], ref_gt[1][0] - ref_gt[0][0])
    ref = dict(u=o["u"], v=o["v"], theta=th0, edge_a=o["edge_a"])

    o = run_exe(make_frame(**meas_frame_kw)[0])
    if o is None:
        return err("测量帧未检出")
    # 固件侧 align_measure 的解算在这里复现
    th = math.radians(o["theta_deg"])
    dir_ax = (math.cos(th), math.sin(th))
    dir_pp = (-math.sin(th), math.cos(th))
    scale = TAG_MM / o["edge_a"]
    du, dv = o["u"] - ref["u"], o["v"] - ref["v"]
    axial = (du * dir_ax[0] + dv * dir_ax[1]) * scale
    perp = (du * dir_pp[0] + dv * dir_pp[1]) * scale
    phi = math.degrees(math.asin(max(-1, min(1, perp / R_TAG_MM))))
    psi = (o["theta_deg"] - math.degrees(th0) + 180) % 360 - 180
    arc = math.radians(phi) * PIPE_R

    dphi = abs(phi - expect["phi"])
    dpsi = abs(psi - expect["psi"])
    darc = abs(arc - expect["arc"])
    dax = abs(axial - expect["axial"])
    for label, d, tol in (("phi", dphi, 0.6), ("psi", dpsi, 0.6),
                          ("arc", darc, 0.8), ("axial", dax, 1.5)):
        flag = "ok" if d <= tol else "FAIL"
        print(f"  {label}={locals()[label]:+.2f} expect={expect[label]:+.2f} err={d:.2f} {flag}")
        if d > tol:
            fails += 1
    print("  " + ("PASS" if fails == 0 else "FAIL"))
    return fails


def main():
    np.random.seed(42)
    build()
    total = 0

    # ---- 检测/解码/角点精度 ----
    for rot in (0, 15, 30, 60, 90, 135, 180, 270):
        total += check(f"旋转 rot={rot}", rot, (160, 120), 100)
    for edge in (40, 70, 140):
        total += check(f"尺寸 edge={edge}", 20, (160, 120), edge)
    total += check("离中心 (60,70)", 25, (60, 70), 90)
    total += check("透视 tilt=15", 35, (160, 120), 100, tilt=15)
    total += check("噪声 sigma=10", 50, (160, 120), 100, noise=10)
    total += check("模糊 5x5", 40, (160, 120), 100, blur=5)
    total += check("综合: 倾斜+噪声+模糊", 65, (150, 130), 80, tilt=10, noise=8, blur=3)

    # ---- 纠偏测量 (几何换算) ----
    ref_kw = dict(rot_deg=0, center=(160.0, 120.0), edge_px=100.0)
    mmpp = TAG_MM / 100.0
    # phi=8°: 中心沿垂直管道方向移 R*sin(phi)
    perp_px = R_TAG_MM * math.sin(math.radians(8)) / mmpp
    total += measure("纯 phi=8°", ref_kw,
                     dict(rot_deg=0, center=(160.0, 120.0 + perp_px), edge_px=100.0),
                     dict(phi=8.0, psi=0.0, arc=math.radians(8) * PIPE_R, axial=0.0))
    # psi=-6°: 标签原地转 6°
    total += measure("纯 psi=-6°", ref_kw,
                     dict(rot_deg=-6, center=(160.0, 120.0), edge_px=100.0),
                     dict(phi=0.0, psi=-6.0, arc=0.0, axial=0.0))
    # 组合: phi + psi=5° + 轴向前进 +10px (用当前帧比例尺换算, 与固件一致)
    perp_px = R_TAG_MM * math.sin(math.radians(4)) / mmpp
    scale_meas = TAG_MM / 90.0
    total += measure("组合 phi+psi+axial", ref_kw,
                     dict(rot_deg=5, center=(160.0 + 10, 120.0 + perp_px), edge_px=90.0),
                     dict(phi=math.degrees(math.asin(8 * scale_meas / R_TAG_MM)),
                          psi=5.0, arc=math.asin(8 * scale_meas / R_TAG_MM) * PIPE_R,
                          axial=10 * scale_meas))

    print()
    if total == 0:
        print("=== 全部通过 ===")
    else:
        print(f"=== {total} 项失败 ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
