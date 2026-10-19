# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.insert(0, r'C:\Users\Administrator\Desktop\视觉\code\pc_check')
import test_pc as T

frame, gt = T.make_frame(60, (160, 120), 100)
print('gt corners:')
for p in gt:
    print('   ', float(p[0]), float(p[1]))
for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
    d = float(((gt[b] - gt[a]) ** 2).sum() ** 0.5)
    print(f'e{a}{b} = {d:.2f}')

# 对比: 用检测器输出
o = T.run_exe(frame)
print('det corners:', [(round(x, 2), round(y, 2)) for x, y in o['corners']])
print('det edge_a =', o['edge_a'], ' edge_p =', o['edge_p'])
