# -*- coding: utf-8 -*-
"""安装误差测试：相机没装正时，俯视测量被带偏多少？

真实安装不可能完全垂直向下。本测试用 sim_common 对相机施加:
  tilt_along  = 沿管道方向倾斜
  tilt_across = 垂直管道方向倾斜
  roll        = 绕光轴旋转
在标准工况 phi=+10deg、axial=+15mm 下测量。参考位也用同一倾斜相机的
phi=0 画面记录（与实机一致），看 phi/axial 的残余误差。
phi_shape 一栏显示边长法在相机倾斜时的固定偏置（说明它只能作参考）。

运行: python test_tilt_leak.py
"""
import os
import sys
import math
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_utils import detect_tag, delta_report, wrap_deg, imwrite_u, TAG_REAL_MM
from gen_apriltag import make_tag_image
import sim_common

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CODE_DIR, "..", "输出")

PHI_TRUE = 10.0
AXIAL_TRUE = 15.0


def main():
    tag_img, marker_px = make_tag_image(TAG_REAL_MM, with_arrow=False)
    scene = sim_common.make_scene

    print(f"标准工况: 真实 phi=+{PHI_TRUE:.0f}deg, axial=+{AXIAL_TRUE:.0f}mm\n")
    print(f"{'相机安装误差':<24} {'phi测得':>9} {'phi误差':>9} "
          f"{'axial测得':>10} {'axial误差':>10} {'shape参考':>10}")
    for name, kw in [("无(基准)", dict()),
                     ("沿管道倾斜5deg", dict(tilt_along_deg=5)),
                     ("沿管道倾斜10deg", dict(tilt_along_deg=10)),
                     ("沿管道倾斜15deg", dict(tilt_along_deg=15)),
                     ("垂直管道倾斜5deg", dict(tilt_across_deg=5)),
                     ("垂直管道倾斜10deg", dict(tilt_across_deg=10)),
                     ("滚转10deg", dict(roll_deg=10)),
                     ("组合(沿5+垂5+滚5)", dict(tilt_along_deg=5,
                                               tilt_across_deg=5, roll_deg=5))]:
        ref = detect_tag(scene(tag_img, marker_px, phi_deg=0.0, **kw))
        if ref is None:
            print(f"{name:<24} 参考位识别失败")
            continue
        t = detect_tag(scene(tag_img, marker_px, phi_deg=PHI_TRUE,
                             axial_mm=AXIAL_TRUE, **kw))
        if t is None:
            print(f"{name:<24} 工况识别失败")
            continue
        r = delta_report(t, ref)
        pe = wrap_deg(r["phi_deg"] - PHI_TRUE)
        ae = r["axial_mm"] - AXIAL_TRUE
        print(f"{name:<24} {r['phi_deg']:>9.2f} {pe:>+9.2f} "
              f"{r['axial_mm']:>10.2f} {ae:>+10.2f} {r['phi_shape_deg']:>10.2f}")

    print("\n结论参考: phi误差<0.5deg、axial误差<1mm 对应的安装误差范围内可放心使用;")
    print("phi_shape 在相机倾斜时有固定偏置, 只作交叉验证不作判据。")

    # 保存一张倾斜安装下的识别效果图供人工查看
    s = scene(tag_img, marker_px, phi_deg=10.0, axial_mm=15.0, tilt_along_deg=10.0)
    t = detect_tag(s)
    vis = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
    if t is not None:
        cv2.polylines(vis, [t["corners"].astype(np.int32)], True, (0, 0, 255), 2)
        cv2.putText(vis, "camera tilted 10deg along pipe", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    imwrite_u(os.path.join(OUT_DIR, "tilt_test_10deg.png"), vis)
    print("\n倾斜安装10°的识别效果图已存: 输出/tilt_test_10deg.png")


if __name__ == "__main__":
    main()