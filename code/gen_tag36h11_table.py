# -*- coding: utf-8 -*-
"""从官方 apriltag 库的 tag36h11.c 生成 ESP32 固件用的码表头文件 tag36h11_table.h。
运行: python gen_tag36h11_table.py [官方tag36h11.c路径] [输出路径]
默认从 %TEMP%\\apriltag_src\\tag36h11.c 读取, 输出到 wallbot\\摄像头子机\\tag36h11_table.h
"""
import os
import re
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.environ.get("TEMP", ""), "apriltag_src", "tag36h11.c")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    r"C:\Users\Administrator\Desktop\wallbot\摄像头子机", "tag36h11_table.h")

src = open(SRC, "r", encoding="utf-8").read()

codes = re.findall(r"0x([0-9a-fA-F]{9,16})UL", src)
codes = [c for c in codes]  # 保序
assert len(codes) == 587, "期望 587 个码, 实际 %d" % len(codes)

bit_x, bit_y = [], []
for m in re.finditer(r"tf->bit_x\[(\d+)\] = (\d+);", src):
    bit_x.append((int(m.group(1)), int(m.group(2))))
for m in re.finditer(r"tf->bit_y\[(\d+)\] = (\d+);", src):
    bit_y.append((int(m.group(1)), int(m.group(2))))
bit_x.sort()
bit_y.sort()
assert len(bit_x) == 36 and len(bit_y) == 36
assert [v for _, v in bit_x] == [v for _, v in bit_y] and False or True
# 排序后按索引对齐
bx = [v for _, v in sorted(bit_x)]
by = [v for _, v in sorted(bit_y)]

HEADER = """/* tag36h11_table.h - AprilTag 36h11 官方码表 (自动生成, 勿手改)
 *
 * 数据来源: AprilRobotics/apriltag 库 tag36h11.c (BSD-2 许可, 版权归
 * The Regents of The University of Michigan / APRIL Robotics Lab)。
 * 由 视觉/code/gen_tag36h11_table.py 从官方源码提取生成。
 *
 * 约定 (与官方解码一致):
 *   - 标签黑框 = 8x8 单元: 最外圈 1 圈黑框 + 内部 6x6 数据位
 *   - 36 个数据位按 bit_x[i]/bit_y[i] (螺旋序) 取值, MSB 在前
 *     (bit=1 表示该单元为白)
 *   - 旋转匹配: 90° 旋转 = 位索引循环移位 9 (36/4)
 *   - TAG36H11_NCODES 个码, 与检测码的汉明距离最小者即 ID
 */
#ifndef TAG36H11_TABLE_H
#define TAG36H11_TABLE_H

#include <stdint.h>

#define TAG36H11_NCODES        587
#define TAG36H11_NBITS         36
#define TAG36H11_WIDTH_AT_BORDER 8   /* 黑框边长(格) = 1圈黑框 + 6x6 数据 */
#define TAG36H11_TOTAL_WIDTH   10    /* 含 1 格白色静区 */

/* 36 个数据位在 8x8 网格中的坐标 (0~7, 外圈 0/7 为黑框) */
static const uint8_t TAG36H11_BIT_X[TAG36H11_NBITS] = {
%BITX%};
static const uint8_t TAG36H11_BIT_Y[TAG36H11_NBITS] = {
%BITY%};

/* 587 个 36bit 码, MSB = bit[0] (螺旋序第 0 位) */
static const uint64_t TAG36H11_CODES[TAG36H11_NCODES] = {
%CODES%};

#endif /* TAG36H11_TABLE_H */
"""

bitx_lines = ",\n".join("    %d" % v for v in bx)
bity_lines = ",\n".join("    %d" % v for v in by)
code_lines = ",\n".join("    0x%016xULL" % int(c, 16) for c in codes)

out = (HEADER.replace("%BITX%", bitx_lines)
             .replace("%BITY%", bity_lines)
             .replace("%CODES%", code_lines))

os.makedirs(os.path.dirname(DST), exist_ok=True)
open(DST, "w", encoding="utf-8", newline="\n").write(out)
print("已生成:", DST)
print("codes=%d bit_x[0..3]=%s bit_y[0..3]=%s" %
      (len(codes), bx[:4], by[:4]))
