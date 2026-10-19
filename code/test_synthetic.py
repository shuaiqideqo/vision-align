# -*- coding: utf-8 -*-
"""离线精度测试（不需要摄像头）：俯视测量模型的回归测试。

用 sim_common 把标签按已知三维姿态(滚转 phi / 轴向平移 / 亮度)投影成
相机画面，检验 tag_utils.delta_report 的反解精度：
  phi       由中心位移反解 asin(perp_mm/R_TAG_MM) —— 主判据
  psi       面内角差 theta − theta0 —— 歪头角(航向)
  phi_shape 由边长收缩反解 acos(edge_p/edge_a)   —— 交叉验证(无符号)

运行: python test_synthetic.py
"""
import os
import sys
import math
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_utils import (detect_tag, delta_report, wrap_deg, imwrite_u,
                       TAG_REAL_MM, R_TAG_MM)
from gen_apriltag import make_tag_image
import sim_common

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, "..", "输出")


def main():
    tag_img, marker_px = make_tag_image(TAG_REAL_MM, with_arrow=False)
    scene = sim_common.make_scene

    # 参考位：phi=0, axial=0（相当于小车摆到夹爪对中位置后按 r）
    ref = detect_tag(scene(tag_img, marker_px, 0.0))
    assert ref is not None, "参考位识别失败！"
    print(f"参考位: theta0={math.degrees(ref['theta']):+.3f}deg, "
          f"中心=({ref['u']:.1f},{ref['v']:.1f}), 轴向边长={ref['edge_a']:.1f}px")
    print(f"(仿真参数: R_TAG_MM={R_TAG_MM}, 工作距离={sim_common.H_MM - R_TAG_MM:.0f}mm)\n")

    print("测试1: 滚转角 phi 测量精度（位置法 vs 边长法, 模拟小车绕管道轴滚转）")
    print(f"{'真实phi(deg)':>12} {'位置法(deg)':>12} {'误差':>8} {'边长法(deg)':>12} {'误差':>8}")
    worst = 0.0
    for phi in [-30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30]:
        t = detect_tag(scene(tag_img, marker_px, phi_deg=phi))
        if t is None:
            print(f"{phi:>12.1f} {'识别失败':>20}")
            continue
        r = delta_report(t, ref)
        err = abs(wrap_deg(r["phi_deg"] - phi))
        err_s = abs(r["phi_shape_deg"] - abs(phi))
        worst = max(worst, err)
        print(f"{phi:>12.1f} {r['phi_deg']:>12.3f} {err:>8.3f} "
              f"{r['phi_shape_deg']:>12.3f} {err_s:>8.3f}")
    print(f"==> 位置法最大误差: {worst:.3f} deg（要求 |phi|<=10 时 < 0.5）\n")

    print("测试2: 轴向位移测量精度（模拟小车沿管道进/退）")
    print(f"{'真实axial(mm)':>14} {'测得(mm)':>10} {'误差(mm)':>10}")
    worst_a = 0.0
    for a in [-40, -20, -10, 0, 10, 20, 40]:
        t = detect_tag(scene(tag_img, marker_px, axial_mm=a))
        if t is None:
            print(f"{a:>14.1f} {'识别失败':>10}")
            continue
        r = delta_report(t, ref)
        worst_a = max(worst_a, abs(r["axial_mm"] - a))
        print(f"{a:>14.1f} {r['axial_mm']:>10.2f} {abs(r['axial_mm'] - a):>10.2f}")
    print(f"==> 轴向最大误差: {worst_a:.2f} mm（要求 < 1）\n")

    print("测试3: 亮度变化鲁棒性 (phi=7.5, axial=15, 亮度0.55)")
    t = detect_tag(scene(tag_img, marker_px, phi_deg=7.5, axial_mm=15.0,
                         brightness=0.55))
    assert t is not None, "综合工况识别失败"
    r = delta_report(t, ref)
    print(f"  phi={r['phi_deg']:+.3f}deg (误差 {abs(r['phi_deg']-7.5):.3f}), "
          f"axial={r['axial_mm']:+.2f}mm (误差 {abs(r['axial_mm']-15):.2f}), "
          f"比例尺 {r['mm_per_px']*1000:.1f}um/px\n")

    print("测试4: 歪头角 psi 测量精度（模拟车身绕壁面法线原地转 = 标签面内旋转）")
    print("注: 仿真 roll 绕画面中心旋转, 图像坐标 y 向下, 故测得 psi 约等于")
    print("    -psi_sim(符号相反); 实机 'psi 正方向=车身往哪边歪' 由台架空跑标定一次即可, 与 arc 同理。")
    print(f"{'真实psi(deg)':>12} {'测得psi(deg)':>12} {'误差':>8} {'phi(deg)':>10} {'axial(mm)':>10}")
    worst_psi = 0.0
    for psi in [-30, -15, -10, -5, -2, 0, 2, 5, 10, 15, 30]:
        t = detect_tag(scene(tag_img, marker_px, roll_deg=psi))
        if t is None:
            print(f"{psi:>12.1f} {'识别失败':>20}")
            continue
        r = delta_report(t, ref)
        err = abs(wrap_deg(r["psi_deg"] + psi))   # 仿真符号: psi_deg 约等于 -roll
        worst_psi = max(worst_psi, err)
        print(f"{psi:>12.1f} {r['psi_deg']:>12.3f} {err:>8.3f} "
              f"{r['phi_deg']:>+10.3f} {r['axial_mm']:>+10.3f}")
    print(f"==> psi 最大误差: {worst_psi:.3f} deg\n")

    print("测试5: phi/psi/axial 联合工况 (phi=6, axial=10, psi=8) 解耦检查")
    t = detect_tag(scene(tag_img, marker_px, phi_deg=6.0, axial_mm=10.0, roll_deg=8.0))
    assert t is not None, "联合工况识别失败"
    r = delta_report(t, ref)
    print(f"  psi={r['psi_deg']:+.3f}deg (误差 {abs(wrap_deg(r['psi_deg'] + 8.0)):.3f})  "
          f"phi={r['phi_deg']:+.3f}deg (误差 {abs(r['phi_deg'] - 6.0):.3f})  "
          f"axial={r['axial_mm']:+.2f}mm (误差 {abs(r['axial_mm'] - 10.0):.2f})")
    print("  注: 仿真把歪头加在整幅图像上(绕画面中心旋转), phi/axial 的少量混叠来自位移向量")
    print("      被整体旋转; 实机混叠量 = 标签中心偏离车身旋转中心的距离 × psi, 标签贴在")
    print("      车身旋转中心附近即可忽略, 且闭环先 SPIN 纠 psi 再 MV 纠 phi 可进一步吸收。\n")

    # 保存一张带标注的结果图供人工查看
    s = scene(tag_img, marker_px, phi_deg=12.0, axial_mm=20.0)
    t = detect_tag(s)
    assert t is not None
    vis = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    r = delta_report(t, ref)
    cv2.polylines(vis, [t["corners"].astype(np.int32)], True, (0, 0, 255), 2)
    c = (int(t["u"]), int(t["v"]))
    cv2.circle(vis, c, 5, (0, 255, 0), -1)
    pr = (int(ref["u"]), int(ref["v"]))
    cv2.drawMarker(vis, pr, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
    cv2.arrowedLine(vis, pr, c, (255, 0, 255), 2, tipLength=0.08)
    cv2.putText(vis, f"phi={r['phi_deg']:+.2f} psi={r['psi_deg']:+.2f} axial={r['axial_mm']:+.1f}mm",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    imwrite_u(os.path.join(OUT_DIR, "test_annotated.png"), vis)
    print("带标注的结果图已保存: 输出/test_annotated.png")

    print(f"\n提醒: 实机前把 R_TAG_MM 实测值填入 tag_utils.py (当前为 {R_TAG_MM:.0f})")


if __name__ == "__main__":
    main()