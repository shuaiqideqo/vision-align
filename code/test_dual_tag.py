# -*- coding: utf-8 -*-
"""双码融合 + AlignMeasure 兼容性离线测试。

合成两枚 tag36h11 (默认 ID 0/1), 按已知中心距并排放置, 检验:
  1) detect_dual_tags 必须两码齐全, 缺一即 need_two_tags / no_tag
  2) 融合观测的 edge_a / theta (小ID->大ID)
  3) delta_report 用 DUAL_TAG_SEP_MM 比例尺

运行: python test_dual_tag.py
"""
import os
import sys
import math
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_utils import (detect_dual_tags, detect_all_tags, delta_report, wrap_deg,
                       imwrite_u, TAG_REAL_MM, DUAL_TAG_SEP_MM, LANDMARK_MM,
                       TAG_ID_A, TAG_ID_B, R_TAG_MM)
from gen_apriltag import make_tag_image

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, "..", "输出")
os.makedirs(OUT_DIR, exist_ok=True)

CANVAS_W, CANVAS_H = 800, 600
CX, CY = CANVAS_W / 2.0, CANVAS_H / 2.0
TARGET_EDGE = 80.0  # 黑框目标像素边长


def _scaled_tag(tag_id):
    img, mpx = make_tag_image(TAG_REAL_MM, with_arrow=False, tag_id=tag_id)
    s = TARGET_EDGE / mpx
    h, w = img.shape[:2]
    nw, nh = max(8, int(round(w * s))), max(8, int(round(h * s)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def place_tag(canvas, tag_img, cx, cy, yaw_deg=0.0):
    """把已缩放过的标签贴到画布中心 (cx,cy), 可选面内旋转。"""
    h, w = tag_img.shape[:2]
    if abs(yaw_deg) > 1e-6:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), yaw_deg, 1.0)
        # 旋转后扩展画布避免裁切
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
    # 裁到画布内
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    sx1 = w - max(0, x1 - CANVAS_W); sy1 = h - max(0, y1 - CANVAS_H)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(CANVAS_W, x1), min(CANVAS_H, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    patch = tag_img[sy0:sy1, sx0:sx1]
    canvas[dy0:dy1, dx0:dx1] = patch


def make_dual_scene(sep_px, yaw_deg=0.0, phi_shift_px=0.0, missing=None):
    canvas = np.full((CANVAS_H, CANVAS_W), 255, np.uint8)
    img_a = _scaled_tag(TAG_ID_A)
    img_b = _scaled_tag(TAG_ID_B)
    ang = math.radians(yaw_deg)
    dx, dy = 0.5 * sep_px * math.cos(ang), 0.5 * sep_px * math.sin(ang)
    mx, my = CX, CY + phi_shift_px
    if missing not in ("a", "both"):
        place_tag(canvas, img_a, mx - dx, my - dy, yaw_deg=yaw_deg)
    if missing not in ("b", "both"):
        place_tag(canvas, img_b, mx + dx, my + dy, yaw_deg=yaw_deg)
    return canvas


def main():
    print(f"LANDMARK_MM={LANDMARK_MM} DUAL_TAG_SEP_MM={DUAL_TAG_SEP_MM} "
          f"IDs={TAG_ID_A},{TAG_ID_B}")
    sep_px = 160.0

    ref_img = make_dual_scene(sep_px, yaw_deg=0.0)
    ref, reason = detect_dual_tags(ref_img)
    assert ref is not None, f"参考位失败: {reason}"
    print(f"参考位: u={ref['u']:.1f} v={ref['v']:.1f} "
          f"theta={math.degrees(ref['theta']):+.2f}deg edge={ref['edge_a']:.1f}px "
          f"ids={ref['id']}/{ref['id_b']}")
    assert abs(ref["edge_a"] - sep_px) < 8.0, f"基线像素偏差过大: {ref['edge_a']}"
    assert abs(math.degrees(ref["theta"])) < 3.0

    print("\n测试0: 缺码必须失败且 reason 正确")
    for miss, expect in [("both", "no_tag"), ("a", "need_two_tags"), ("b", "need_two_tags")]:
        img = make_dual_scene(sep_px, missing=miss)
        obs, why = detect_dual_tags(img)
        assert obs is None and why == expect, f"missing={miss} got {(obs, why)}"
        print(f"  missing={miss}: reason={why} OK")

    print("\n测试1: psi (整体 yaw)")
    worst_psi = 0.0
    for yaw in [-20, -10, -5, 0, 5, 10, 20]:
        img = make_dual_scene(sep_px, yaw_deg=yaw)
        obs, why = detect_dual_tags(img)
        assert obs is not None, f"yaw={yaw} fail {why}"
        r = delta_report(obs, ref)
        err = abs(wrap_deg(r["psi_deg"] - yaw))
        worst_psi = max(worst_psi, err)
        print(f"  yaw={yaw:+.0f} -> psi={r['psi_deg']:+.2f} err={err:.2f}")
    assert worst_psi < 2.0, f"psi 误差过大 {worst_psi}"
    print(f"==> psi 最大误差 {worst_psi:.3f} deg")

    print("\n测试2: phi (中点垂直位移)")
    worst_phi = 0.0
    for shift in [-40, -20, -10, 0, 10, 20, 40]:
        img = make_dual_scene(sep_px, phi_shift_px=shift)
        obs, why = detect_dual_tags(img)
        assert obs is not None, f"shift={shift} fail {why}"
        r = delta_report(obs, ref)
        true_perp = shift * (DUAL_TAG_SEP_MM / sep_px)
        true_phi = math.degrees(math.asin(max(-1, min(1, true_perp / R_TAG_MM))))
        err = abs(r["phi_deg"] - true_phi)
        worst_phi = max(worst_phi, err)
        print(f"  shift={shift:+.0f}px -> phi={r['phi_deg']:+.2f} "
              f"(true {true_phi:+.2f}) err={err:.2f}")
    assert worst_phi < 1.0, f"phi 误差过大 {worst_phi}"
    print(f"==> phi 最大误差 {worst_phi:.3f} deg")

    print("\n测试3: axial")
    worst_a = 0.0
    for ax_px in [-50, -25, 0, 25, 50]:
        canvas = np.full((CANVAS_H, CANVAS_W), 255, np.uint8)
        place_tag(canvas, _scaled_tag(TAG_ID_A), CX - sep_px / 2 + ax_px, CY)
        place_tag(canvas, _scaled_tag(TAG_ID_B), CX + sep_px / 2 + ax_px, CY)
        obs, why = detect_dual_tags(canvas)
        assert obs is not None, why
        r = delta_report(obs, ref)
        true_ax = ax_px * (DUAL_TAG_SEP_MM / sep_px)
        err = abs(r["axial_mm"] - true_ax)
        worst_a = max(worst_a, err)
        print(f"  ax_px={ax_px:+.0f} -> axial={r['axial_mm']:+.2f} err={err:.2f}")
    assert worst_a < 2.0, f"axial 误差过大 {worst_a}"
    print(f"==> axial 最大误差 {worst_a:.2f} mm")

    print("\n测试4: 180 消歧 (ID 顺序固定朝向)")
    canvas = np.full((CANVAS_H, CANVAS_W), 255, np.uint8)
    place_tag(canvas, _scaled_tag(TAG_ID_B), CX - sep_px / 2, CY)
    place_tag(canvas, _scaled_tag(TAG_ID_A), CX + sep_px / 2, CY)
    obs, why = detect_dual_tags(canvas)
    assert obs is not None, why
    th = math.degrees(obs["theta"])
    assert abs(abs(th) - 180.0) < 8.0, f"unexpected theta {th}"
    print(f"  swapped placement theta={th:.1f}deg OK")

    vis = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    for t in detect_all_tags(ref_img):
        cv2.polylines(vis, [t["corners"].astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(vis, f"id{t['id']}", (int(t["u"]) - 20, int(t["v"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.circle(vis, (int(ref["u"]), int(ref["v"])), 6, (255, 0, 255), -1)
    out = os.path.join(OUT_DIR, "test_dual_annotated.png")
    imwrite_u(out, vis)
    print(f"\n标注图: {out}")
    print("全部双码测试通过。")


if __name__ == "__main__":
    main()
