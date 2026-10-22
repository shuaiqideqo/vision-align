# -*- coding: utf-8 -*-
"""生成可打印的 AprilTag 标签 PNG。
运行: python gen_apriltag.py [边长mm]
输出: 输出/apriltag_id0_50mm.png (打印时选"实际大小/100%缩放"，打印后量黑框边长应为 50mm)

标签上沿印有黑色箭头 = 标签的 c0→c1 方向(theta 方向)。
粘贴时必须让箭头平行于管道轴线（俯视测量的坐标约定，见 tag_utils.py）。
"""
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tag_utils import ARUCO_DICT, TAG_REAL_MM, imwrite_u

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "输出")


def make_tag_image(size_mm=TAG_REAL_MM, with_arrow=True, tag_id=0):
    """生成标签图像(含白色静区; with_arrow=True 时上沿加方向箭头)。
    返回 (img, marker_px)：marker_px = 黑框边长像素，仿真测试用它换算世界尺寸。"""
    px = round(size_mm * 300 / 25.4)          # 300dpi
    try:
        marker = cv2.aruco.generateImageMarker(ARUCO_DICT, int(tag_id), px)
    except AttributeError:
        marker = cv2.aruco.drawMarker(ARUCO_DICT, int(tag_id), px)
    # 白边 = 静区(识别必需) + 箭头区；箭头画在外圈，紧贴标签的约1圈静区保持干净
    pad = max(90, round(px * 0.25))
    img = cv2.copyMakeBorder(marker, pad, pad, pad, pad,
                             cv2.BORDER_CONSTANT, value=255)
    if with_arrow:
        y = round(pad * 0.35)                      # 外圈，避免挤占静区
        x0 = round(pad * 0.45)
        x1 = img.shape[1] - round(pad * 0.45)
        cv2.arrowedLine(img, (x0, y), (x1, y), 0, max(3, px // 150),
                        cv2.LINE_AA, tipLength=0.03)
    return img, px


def main():
    # 用法: python gen_apriltag.py [边长mm] [id]
    # 双码: python gen_apriltag.py 22 0 && python gen_apriltag.py 22 1
    size_mm = float(sys.argv[1]) if len(sys.argv) > 1 else TAG_REAL_MM
    tag_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tag, _ = make_tag_image(size_mm, tag_id=tag_id)
    out_path = os.path.abspath(
        os.path.join(OUT_DIR, "apriltag_id%d_%dmm.png" % (tag_id, round(size_mm))))
    if not imwrite_u(out_path, tag):
        print("写入失败:", out_path); sys.exit(1)
    print("已生成:", out_path)
    print("打印要求: A4纸, 黑白, 缩放=实际大小/100%%, 黑框边长应为 %.0fmm, 打印后用尺子量一下" % size_mm)
    print("粘贴方向: 标签上沿的黑色箭头必须平行于管道轴线 (见 tag_utils.py 测量模型)")
    print("还要实测: 标签黑框中心平面到管道轴线的垂直距离 R_TAG_MM, 填入 tag_utils.py")
    if size_mm != TAG_REAL_MM:
        print("注意: 该尺寸与 tag_utils.py 中 TAG_REAL_MM=%.0f 不一致, 若用此标签请同步修改" % TAG_REAL_MM)


if __name__ == "__main__":
    main()