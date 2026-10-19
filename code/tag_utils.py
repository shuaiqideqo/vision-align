# -*- coding: utf-8 -*-
"""公共工具：AprilTag 检测、俯视位姿测量、弧长换算。
母机视觉程序的核心算法都在这里，桌面测试和树莓派上用的是同一份代码。

测量模型（俯视安装，v2）：
  相机装在夹爪正上方、大致垂直向下拍，看到小车顶面（俯视图）。
  标签贴在小车顶面，且使标签的 c0→c1 边（theta 方向）平行于管道轴线。
  小车绕管道轴滚转 phi 时，标签中心在图像里沿"垂直于管道"的方向平移
  约 R_TAG_MM*sin(phi)，标签面只发生透视倾斜（theta 基本不变）。
  所以周向偏差用【中心位移】反解，而不是用角度（旧侧面视角用角度已不适用）。
"""
import math
import os
import cv2
import numpy as np

# tag36h11 字典（执行文档第 2.2 节：角度精度最好）
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# 兼容不同 OpenCV 版本：4.7+/5.x 用 ArucoDetector；旧版(如树莓派 apt 的 4.6)退回旧 API
try:
    _DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

    def _detect(img):
        corners, ids, _ = _DETECTOR.detectMarkers(img)
        return corners, ids
except AttributeError:
    _PARAMS = cv2.aruco.DetectorParameters_create()

    def _detect(img):
        corners, ids, _ = cv2.aruco.detectMarkers(img, ARUCO_DICT, parameters=_PARAMS)
        return corners, ids

TAG_REAL_MM = 50.0   # 标签黑框真实边长(mm)，打印后用卡尺/直尺实测，改这里
R_TAG_MM = 60.0      # ★标签黑框中心平面到管道轴线的垂直距离(mm)，贴好后必须实测填入
PIPE_RADIUS_MM = 60.0  # 管道内半径(mm)，仅用于把 phi 换算成沿管壁应走的弧长，粗估即可


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """支持中文路径的 imread（Windows 上 cv2.imread 遇中文路径会失败）。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_u(path, img):
    """支持中文路径的 imwrite。"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def detect_tag(img, tag_id=0):
    """在图像中找指定 ID 的 AprilTag。

    返回 dict(u=中心x像素, v=中心y像素, theta=c0→c1边方向(弧度),
              edge_a=轴向两边平均边长px, edge_p=周向两边平均边长px,
              edge_px=同edge_a(兼容旧字段), corners=4x2角点)，找不到返回 None。

    角点顺序为标签自身坐标系: c0左上 c1右上 c2右下 c3左下。
    按粘贴约定 c0→c1 边平行于管道轴线，因此:
      edge_a(=c0c1与c2c3平均) 滚转时不缩短 -> 用作比例尺基准;
      edge_p(=c1c2与c3c0平均) 滚转 phi 后约等于 edge_a*cos(phi) -> 可反推 phi。
    """
    corners, ids = _detect(img)
    if ids is None or len(ids) == 0:
        return None
    for i, marker_id in enumerate(ids.ravel()):
        if int(marker_id) != tag_id:
            continue
        c = corners[i][0].astype(np.float64)          # 4x2: 左上 右上 右下 左下
        u, v = float(c[:, 0].mean()), float(c[:, 1].mean())
        theta = math.atan2(c[1, 1] - c[0, 1], c[1, 0] - c[0, 0])
        e01 = float(np.linalg.norm(c[1] - c[0]))
        e12 = float(np.linalg.norm(c[2] - c[1]))
        e23 = float(np.linalg.norm(c[3] - c[2]))
        e30 = float(np.linalg.norm(c[0] - c[3]))
        edge_a = 0.5 * (e01 + e23)
        edge_p = 0.5 * (e12 + e30)
        return dict(u=u, v=v, theta=theta, edge_a=edge_a, edge_p=edge_p,
                    edge_px=edge_a, corners=c)
    return None


def wrap_deg(d):
    """角度差归一化到 [-180, 180)。"""
    return (d + 180.0) % 360.0 - 180.0


def delta_report(tag, ref):
    """俯视测量：当前标签姿态 vs 参考位（参考位 = 夹爪已对中、可抓取时的姿态）。

    用参考位的 theta 定义图像里的"管道轴线方向"，自动补偿相机安装旋转
    （相机没装正也能测，前提是标签粘贴方向正确）。
    比例尺用当前帧的轴向边长 edge_a：滚转/进退导致的高度变化会被自动跟踪。

    返回 dict:
      phi_deg        周向滚转角(deg)，由中心位移反解 phi = asin(perp_mm/R_TAG_MM)
                     —— 夹取判据用（|phi|<4° 变绿）
      phi_shape_deg  滚转角参考值(deg)，由边长收缩反解 acos(edge_p/edge_a)。
                     无正负号；相机有俯仰倾斜时自带固定偏置，仅作交叉验证
      psi_deg        机身航向/歪头角(deg) = 面内角差 theta − theta0（绕壁面法线的扭转）。
                     与 phi 天然解耦：φ 表现为标签中心平移，ψ 表现为标签原地旋转。
                     注意：标签中心偏离车身旋转中心时，ψ 会经位移耦合污染 φ/axial，
                     闭环务必先 SPIN 纠 ψ、再 MV 纠 φ，每步重测即可吸收耦合
      axial_mm       沿管道方向位移(mm)，正负号=进/退方向（上机时确认一次）
      perp_mm        垂直管道方向位移(mm) = R_TAG_MM*sin(phi)
      mm_per_px      当前帧比例尺
    """
    th = ref["theta"]
    dir_ax = (math.cos(th), math.sin(th))       # 图像里管道轴线方向
    dir_pp = (-math.sin(th), math.cos(th))      # 垂直管道方向
    scale = TAG_REAL_MM / max(tag["edge_a"], 1e-9)
    du, dv = tag["u"] - ref["u"], tag["v"] - ref["v"]
    axial_mm = (du * dir_ax[0] + dv * dir_ax[1]) * scale
    perp_mm = (du * dir_pp[0] + dv * dir_pp[1]) * scale
    phi = math.degrees(math.asin(max(-1.0, min(1.0, perp_mm / R_TAG_MM))))
    psi_deg = wrap_deg(math.degrees(tag["theta"] - ref["theta"]))
    ratio = tag["edge_p"] / max(tag["edge_a"], 1e-9)
    phi_shape = math.degrees(math.acos(max(-1.0, min(1.0, ratio))))
    return dict(phi_deg=phi, psi_deg=psi_deg, phi_shape_deg=phi_shape,
                axial_mm=axial_mm, perp_mm=perp_mm, mm_per_px=scale)


def arc_mm_for(deg):
    """周向偏转 deg 度 -> 沿管壁应行走的弧长(mm)。正负号即行走方向（上机确认）。"""
    return math.radians(deg) * PIPE_RADIUS_MM