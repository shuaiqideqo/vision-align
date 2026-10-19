# -*- coding: utf-8 -*-
"""生成失败场景帧并直接运行 exe, 打印全部 stderr 诊断。"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pc as T

cases = [
    ("rot30", dict(rot_deg=30, center=(160.0, 120.0), edge_px=100.0)),
    ("rot135", dict(rot_deg=135, center=(160.0, 120.0), edge_px=100.0)),
    ("edge140", dict(rot_deg=20, center=(160.0, 120.0), edge_px=140.0)),
    ("tilt15", dict(rot_deg=35, center=(160.0, 120.0), edge_px=100.0, tilt=15)),
    ("noise10", dict(rot_deg=50, center=(160.0, 120.0), edge_px=100.0, noise=10)),
    ("blur5", dict(rot_deg=40, center=(160.0, 120.0), edge_px=100.0, blur=5)),
]

T.build()
for name, kw in cases:
    frame, gt = T.make_frame(**kw)
    binp = os.path.join(os.path.dirname(T.EXE), name + ".bin")
    frame.tofile(binp)
    r = subprocess.run([T.EXE, binp, str(T.W), str(T.H)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    print(f"===== {name}: rc={r.returncode} stdout={r.stdout.strip()[:120]}")
    err = r.stderr.strip()
    if err:
        print(err[-1500:])
