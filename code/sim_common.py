# -*- coding: utf-8 -*-
"""俯视仿真公共件：把标签图像按已知三维姿态投影成相机看到的画面。
test_synthetic.py / test_tilt_leak.py 共用，只用于离线测试，不参与实机运行。

世界坐标(mm)：
  管道轴线 = X 轴（y=0, z=0），小车绕 X 轴滚转
  标签贴在小车顶面：参考位时标签黑框中心在 (0, 0, R_TAG_MM)，标签面水平
  相机在夹爪上方，无安装误差时位于 (0,0,H_MM)、光轴竖直向下

投影：u = CX + F*x_cam/z_cam, v = CY + F*y_cam/z_cam
（无倾斜时等价于 u=600+F*x/(H-z), v=450+F*y/(H-z)，+y 画向图像下方）
"""
import math
import cv2
import numpy as np

from tag_utils import TAG_REAL_MM, R_TAG_MM

CANVAS_W, CANVAS_H = 1200, 900
CX, CY = 600.0, 450.0
H_MM = 400.0    # 相机到管道轴线的高度；工作距离 = H_MM - R_TAG_MM
F_PX = 2000.0   # 焦距(px)，标签在画面里约占 1/4 宽，接近实机


def _cam_basis(tilt_along_deg, tilt_across_deg):
    """相机安装倾斜时的光轴/图像右/图像下方向向量。

    无倾斜: f=(0,0,-1)竖直向下, r=(1,0,0), up=(0,1,0)。
    tilt_along  = 光轴向管道方向倾斜（绕世界Y轴）
    tilt_across = 光轴向垂直管道方向倾斜（绕世界X轴）
    """
    a = math.radians(tilt_along_deg)
    b = math.radians(tilt_across_deg)
    Ry = np.array([[math.cos(a), 0, math.sin(a)],
                   [0, 1, 0],
                   [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(b), -math.sin(b)],
                   [0, math.sin(b), math.cos(b)]])
    R = Rx @ Ry
    f = R @ np.array([0.0, 0.0, -1.0])
    r = R @ np.array([1.0, 0.0, 0.0])
    up = R @ np.array([0.0, 1.0, 0.0])
    return f, r, up


def make_scene(tag_img, marker_px, phi_deg=0.0, axial_mm=0.0,
               tilt_along_deg=0.0, tilt_across_deg=0.0, roll_deg=0.0,
               brightness=1.0):
    """渲染俯视画面（灰度或彩色均可）。

    phi_deg          小车绕管道轴滚转角。取号约定: phi>0 时标签中心向图像 +v
                     方向移动，与 tag_utils.delta_report 的符号约定一致
    axial_mm         小车沿管道轴线的位移(mm)
    tilt_along_deg   相机安装: 沿管道方向倾斜(deg)
    tilt_across_deg  相机安装: 垂直管道方向倾斜(deg)。倾斜后相机仍对准标签
                     参考位中心、工作距离不变（模拟"装歪了但瞄着工作区"）
    roll_deg         相机安装: 绕光轴旋转(deg)
    brightness       全图亮度系数
    """
    h_px, w_px = tag_img.shape[:2]
    mm_per_png_px = TAG_REAL_MM / marker_px      # PNG含白边，同一打印比例
    src = np.float32([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]])

    phi = math.radians(phi_deg)
    ca, sa = math.cos(phi), math.sin(phi)
    f, r, up = _cam_basis(tilt_along_deg, tilt_across_deg)
    # 相机位置：沿光轴后退工作距离，保证无倾斜时就是 (0,0,H_MM)
    cam = np.array([0.0, 0.0, R_TAG_MM]) - f * (H_MM - R_TAG_MM)

    dst = np.zeros((4, 2), np.float32)
    for i, (pu, pv) in enumerate(src):
        # PNG像素 -> 标签平面局部坐标(mm)：x沿管道轴, y沿周向, 平面高 R_TAG_MM
        x = (pu - w_px / 2.0) * mm_per_png_px + axial_mm
        y = (pv - h_px / 2.0) * mm_per_png_px
        z = R_TAG_MM
        # 小车滚转 phi：绕管道轴(X)旋转，z>0 的点转向 y>0 侧（图像 +v 方向）
        y, z = y * ca + z * sa, -y * sa + z * ca
        p = np.array([x, y, z]) - cam
        zc = float(p @ f)                         # 深度(>0)
        dst[i] = (CX + F_PX * float(p @ r) / zc,
                  CY + F_PX * float(p @ up) / zc)

    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(tag_img, M, (CANVAS_W, CANVAS_H),
                              flags=cv2.INTER_LINEAR, borderValue=210)
    if roll_deg != 0.0:                            # 相机绕光轴旋转 = 整幅图像旋转
        Mr = cv2.getRotationMatrix2D((CX, CY), roll_deg, 1.0)
        out = cv2.warpAffine(out, Mr, (CANVAS_W, CANVAS_H),
                             flags=cv2.INTER_LINEAR, borderValue=210)
    out = np.clip(out.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return out