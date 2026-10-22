# -*- coding: utf-8 -*-
"""VGA 640x480 + 22mm@25cm 尺度的双码合成测试。

按 config.h 注释: 水平 FOV≈65°, STREAM=VGA。
像素换算:
  fov_w_mm = 2 * dist_mm * tan(FOV/2)
  px_per_mm = 640 / fov_w_mm
  tag_edge_px = 22 * px_per_mm
  sep_px     = DUAL_TAG_SEP_MM * px_per_mm

运行 (视觉目录下):
  .venv\\Scripts\\python.exe code\\test_dual_tag_vga25.py
"""
from __future__ import print_function

import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_utils import (
    detect_dual_tags, detect_all_tags, delta_report, wrap_deg, imwrite_u,
    DUAL_TAG_SEP_MM, TAG_ID_A, TAG_ID_B, R_TAG_MM, TAG_REAL_MM,
)
from gen_apriltag import make_tag_image

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, "..", "输出")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 相机/几何 (与固件 config 对齐) ----
CANVAS_W, CANVAS_H = 640, 480
CX, CY = CANVAS_W / 2.0, CANVAS_H / 2.0
HFOV_DEG = 65.0
DIST_MM = 250.0          # 用户关心的工作距 25cm
TAG_MM = TAG_REAL_MM     # 22
SEP_MM = DUAL_TAG_SEP_MM # 45

# 容差 (OpenCV 理想合成; 小码边缘插值会有亚像素误差)
TOL_EDGE_PX = 6.0
TOL_PSI_DEG = 2.5
TOL_PHI_DEG = 1.5
TOL_AXIAL_MM = 2.5
TOL_EDGE_BLUR_PX = 10.0  # 模糊/JPEG 后基线放宽


def px_scale(dist_mm, hfov_deg=HFOV_DEG, img_w=CANVAS_W):
    fov_w_mm = 2.0 * dist_mm * math.tan(math.radians(hfov_deg) / 2.0)
    ppm = img_w / fov_w_mm
    return fov_w_mm, ppm


def _scaled_tag(tag_id, edge_px):
    """生成黑框约 edge_px 的标签图 (含静区, 无箭头以免干扰检测)。"""
    img, mpx = make_tag_image(TAG_MM, with_arrow=False, tag_id=tag_id)
    s = float(edge_px) / float(mpx)
    h, w = img.shape[:2]
    nw = max(8, int(round(w * s)))
    nh = max(8, int(round(h * s)))
    # 小目标用 AREA 缩小时更锐; 若放大则 LINEAR
    inter = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=inter)


def place_tag(canvas, tag_img, cx, cy, yaw_deg=0.0):
    h, w = tag_img.shape[:2]
    if abs(yaw_deg) > 1e-6:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), yaw_deg, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw = int(h * sin + w * cos)
        nh = int(h * cos + w * sin)
        M[0, 2] += (nw - w) / 2.0
        M[1, 2] += (nh - h) / 2.0
        tag_img = cv2.warpAffine(tag_img, M, (nw, nh),
                                 flags=cv2.INTER_LINEAR, borderValue=255)
        h, w = tag_img.shape[:2]
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1, y1 = x0 + w, y0 + h
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    sx1 = w - max(0, x1 - CANVAS_W)
    sy1 = h - max(0, y1 - CANVAS_H)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(CANVAS_W, x1), min(CANVAS_H, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    canvas[dy0:dy1, dx0:dx1] = tag_img[sy0:sy1, sx0:sx1]


def make_dual_scene(sep_px, edge_px, yaw_deg=0.0, phi_shift_px=0.0,
                    ax_shift_px=0.0, missing=None):
    canvas = np.full((CANVAS_H, CANVAS_W), 255, np.uint8)
    img_a = _scaled_tag(TAG_ID_A, edge_px)
    img_b = _scaled_tag(TAG_ID_B, edge_px)
    ang = math.radians(yaw_deg)
    dx = 0.5 * sep_px * math.cos(ang)
    dy = 0.5 * sep_px * math.sin(ang)
    mx = CX + ax_shift_px
    my = CY + phi_shift_px
    if missing not in ("a", "both"):
        place_tag(canvas, img_a, mx - dx, my - dy, yaw_deg=yaw_deg)
    if missing not in ("b", "both"):
        place_tag(canvas, img_b, mx + dx, my + dy, yaw_deg=yaw_deg)
    return canvas


def degrade(img, blur_ksize=0, jpeg_q=None):
    out = img
    if blur_ksize and blur_ksize > 1:
        k = blur_ksize | 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    if jpeg_q is not None:
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
        assert ok
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return out


def main():
    print("=" * 60)
    print("VGA dual-tag geometry check (OpenCV ArUco path)")
    print("=" * 60)

    fov25, ppm25 = px_scale(250.0)
    fov30, ppm30 = px_scale(300.0)
    edge25 = TAG_MM * ppm25
    sep25 = SEP_MM * ppm25
    edge30 = TAG_MM * ppm30
    sep30 = SEP_MM * ppm30

    print("\n[pixel scale] HFOV=%.1fdeg  W=%dx%d" % (HFOV_DEG, CANVAS_W, CANVAS_H))
    print("  @25cm: fov_w=%.2f mm  px/mm=%.4f  tag22=%.2f px  sep45=%.2f px"
          % (fov25, ppm25, edge25, sep25))
    print("  @30cm: fov_w=%.2f mm  px/mm=%.4f  tag22=%.2f px  sep45=%.2f px"
          % (fov30, ppm30, edge30, sep30))
    print("  (config comment: ~37px black border @30cm -> got %.2f)" % edge30)
    assert abs(edge30 - 37.0) < 1.5, "30cm scale mismatch vs config comment"

    edge_px = edge25
    sep_px = sep25
    print("\nUsing dist=25cm: edge_px=%.2f sep_px=%.2f" % (edge_px, sep_px))
    print("IDs=%d/%d  DUAL_TAG_SEP_MM=%.1f  R_TAG_MM=%.1f"
          % (TAG_ID_A, TAG_ID_B, SEP_MM, R_TAG_MM))
    print("Tolerances: edge<%.1fpx psi<%.1fdeg phi<%.1fdeg axial<%.1fmm"
          % (TOL_EDGE_PX, TOL_PSI_DEG, TOL_PHI_DEG, TOL_AXIAL_MM))

    # ---- 参考位 ----
    ref_img = make_dual_scene(sep_px, edge_px, yaw_deg=0.0)
    ref, reason = detect_dual_tags(ref_img)
    assert ref is not None, "ref detect failed: %s" % reason
    print("\n[ref] u=%.1f v=%.1f theta=%+.2fdeg edge=%.2fpx ids=%s/%s"
          % (ref["u"], ref["v"], math.degrees(ref["theta"]),
             ref["edge_a"], ref["id"], ref["id_b"]))
    edge_err = abs(ref["edge_a"] - sep_px)
    print("  edge vs true sep: err=%.2f px (tol %.1f)" % (edge_err, TOL_EDGE_PX))
    assert edge_err < TOL_EDGE_PX, "baseline px too far: %.2f" % ref["edge_a"]
    assert abs(math.degrees(ref["theta"])) < 3.0

    # ---- 缺码 ----
    print("\n[missing tags]")
    for miss, expect in [("both", "no_tag"), ("a", "need_two_tags"), ("b", "need_two_tags")]:
        img = make_dual_scene(sep_px, edge_px, missing=miss)
        obs, why = detect_dual_tags(img)
        assert obs is None and why == expect, "missing=%s got %r" % (miss, (obs, why))
        print("  missing=%s -> %s OK" % (miss, why))

    # ---- psi (小幅 yaw) ----
    print("\n[psi yaw]")
    worst_psi = 0.0
    for yaw in (-15, -8, -4, 0, 4, 8, 15):
        img = make_dual_scene(sep_px, edge_px, yaw_deg=yaw)
        obs, why = detect_dual_tags(img)
        assert obs is not None, "yaw=%s fail %s" % (yaw, why)
        r = delta_report(obs, ref)
        err = abs(wrap_deg(r["psi_deg"] - yaw))
        worst_psi = max(worst_psi, err)
        print("  yaw=%+d -> psi=%+.2f err=%.2f" % (yaw, r["psi_deg"], err))
    assert worst_psi < TOL_PSI_DEG, "psi err %.3f" % worst_psi
    print("==> psi max err %.3f deg  PASS" % worst_psi)

    # ---- phi (中点垂直位移) ----
    print("\n[phi midpoint shift]")
    worst_phi = 0.0
    # 位移约对应 |phi|<=~15deg: R*sin(15)≈15.5mm -> ~31px @25cm
    for shift in (-24, -12, -6, 0, 6, 12, 24):
        img = make_dual_scene(sep_px, edge_px, phi_shift_px=shift)
        obs, why = detect_dual_tags(img)
        assert obs is not None, "shift=%s fail %s" % (shift, why)
        r = delta_report(obs, ref)
        true_perp = shift * (SEP_MM / sep_px)
        true_phi = math.degrees(math.asin(max(-1.0, min(1.0, true_perp / R_TAG_MM))))
        err = abs(r["phi_deg"] - true_phi)
        worst_phi = max(worst_phi, err)
        print("  shift=%+dpx -> phi=%+.2f (true %+.2f) err=%.2f"
              % (shift, r["phi_deg"], true_phi, err))
    assert worst_phi < TOL_PHI_DEG, "phi err %.3f" % worst_phi
    print("==> phi max err %.3f deg  PASS" % worst_phi)

    # ---- axial ----
    print("\n[axial]")
    worst_a = 0.0
    for ax_px in (-30, -15, 0, 15, 30):
        img = make_dual_scene(sep_px, edge_px, ax_shift_px=ax_px)
        obs, why = detect_dual_tags(img)
        assert obs is not None, "ax=%s fail %s" % (ax_px, why)
        r = delta_report(obs, ref)
        true_ax = ax_px * (SEP_MM / sep_px)
        err = abs(r["axial_mm"] - true_ax)
        worst_a = max(worst_a, err)
        print("  ax_px=%+d -> axial=%+.2f (true %+.2f) err=%.2f"
              % (ax_px, r["axial_mm"], true_ax, err))
    assert worst_a < TOL_AXIAL_MM, "axial err %.3f" % worst_a
    print("==> axial max err %.2f mm  PASS" % worst_a)

    # ---- 180 消歧 ----
    print("\n[180-disambiguation]")
    canvas = np.full((CANVAS_H, CANVAS_W), 255, np.uint8)
    place_tag(canvas, _scaled_tag(TAG_ID_B, edge_px), CX - sep_px / 2, CY)
    place_tag(canvas, _scaled_tag(TAG_ID_A, edge_px), CX + sep_px / 2, CY)
    obs, why = detect_dual_tags(canvas)
    assert obs is not None, why
    th = math.degrees(obs["theta"])
    assert abs(abs(th) - 180.0) < 8.0, "unexpected theta %.1f" % th
    print("  swapped A/B placement theta=%.1f deg OK" % th)

    # ---- 退化: 模糊 / JPEG (模拟板端 JPEG + 轻微失焦) ----
    print("\n[degrade: blur / JPEG]")
    for label, img in [
        ("blur3", degrade(ref_img, blur_ksize=3)),
        ("jpeg_q12", degrade(ref_img, jpeg_q=12)),  # STREAM_JPEG_QUALITY=12
        ("blur3+jpeg12", degrade(ref_img, blur_ksize=3, jpeg_q=12)),
    ]:
        obs, why = detect_dual_tags(img)
        if obs is None:
            print("  %s -> FAIL detect (%s)" % (label, why))
            raise AssertionError("%s detect failed: %s" % (label, why))
        eerr = abs(obs["edge_a"] - sep_px)
        print("  %s -> ok edge=%.2f err=%.2fpx" % (label, obs["edge_a"], eerr))
        assert eerr < TOL_EDGE_BLUR_PX, "%s edge err %.2f" % (label, eerr)

    # ---- 可选: 也报一下 30cm 尺度能否检出 (同画布) ----
    print("\n[bonus @30cm scale on same 640x480]")
    img30 = make_dual_scene(sep30, edge30)
    obs30, why30 = detect_dual_tags(img30)
    if obs30 is None:
        print("  detect FAIL (%s) — 22mm@30cm already marginal for OpenCV" % why30)
    else:
        print("  detect OK edge=%.2f (true sep %.2f) err=%.2f"
              % (obs30["edge_a"], sep30, abs(obs30["edge_a"] - sep30)))

    # 标注图
    vis = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    for t in detect_all_tags(ref_img):
        cv2.polylines(vis, [t["corners"].astype(np.int32)], True, (0, 255, 0), 1)
        cv2.putText(vis, "id%d" % t["id"], (int(t["u"]) - 15, int(t["v"]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.circle(vis, (int(ref["u"]), int(ref["v"])), 4, (255, 0, 255), -1)
    cv2.putText(vis, "22mm@25cm VGA", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
    out = os.path.join(OUT_DIR, "test_dual_vga25_annotated.png")
    imwrite_u(out, vis)
    print("\nannotated: %s" % out)
    print("NOTE: pc_check/main.cpp only calls TagDetector_Detect (single-tag),")
    print("      not DualTagDetector_DetectEx; firmware dual path NOT exercised here.")
    print("\nALL VGA@25cm dual-tag OpenCV tests PASSED.")


if __name__ == "__main__":
    main()
