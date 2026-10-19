# -*- coding: utf-8 -*-
"""检查官方码表位坐标与 cv2 渲染的 tag36h11 图案之间的朝向关系。
回答: 用官方 bit 坐标 (y 向下) 直接采样 cv2 渲染的 ID0 图案, 得到的码等于
codes[0] 本身, 还是它的某个旋转? -> 决定固件 c0 约定是否需要 +180° 修正。
"""
import re
import os
import cv2
import numpy as np

TABLE = r"C:\Users\Administrator\Desktop\wallbot\摄像头子机\tag36h11_table.h"
src = open(TABLE, encoding="utf-8").read()
codes = [int(x, 16) for x in re.findall(r"0x([0-9a-f]{16})ULL", src)]
bx = [int(v) for v in re.findall(r"TAG36H11_BIT_X\[TAG36H11_NBITS\] = \{(.*?)\};", src, re.S)[0].replace("\n", "").split(",")]
by = [int(v) for v in re.findall(r"TAG36H11_BIT_Y\[TAG36H11_NBITS\] = \{(.*?)\};", src, re.S)[0].replace("\n", "").split(",")]
print("codes:", len(codes), "bits:", len(bx))

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
try:
    marker = cv2.aruco.generateImageMarker(DICT, 0, 400)
except AttributeError:
    marker = cv2.aruco.drawMarker(DICT, 0, 400)
cell = marker.shape[0] / 8.0


def sample_code(img):
    code = 0
    for i in range(36):
        cx = int((bx[i] + 0.5) * cell)
        cy = int((by[i] + 0.5) * cell)
        v = img[cy, cx]
        code = (code << 1) | (1 if v > 127 else 0)
    return code


def rot90(code):  # 官方索引旋转 (90° = 移 9 位)
    return ((code << 9) | (code >> 27)) & ((1 << 36) - 1)


c = sample_code(marker)
print("采样码 = 0x%09x" % c)
print("codes[0] = 0x%09x" % codes[0])
r = c
for k in range(4):
    if r == codes[0]:
        print(f"匹配: codes[0] 直接相等 或 旋转{k}*90°")
        break
    r = rot90(r)
else:
    print("不匹配任何旋转!")
