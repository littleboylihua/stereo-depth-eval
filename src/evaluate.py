# -*- coding: utf-8 -*-
"""
evaluate.py: 把 SGBM 视差与 Middlebury 官方真值对比, 输出统一指标并写 results.md
这是 Task3「多模型统一评测」的雏形: 以后 DA3/MoGe-2 的深度图也走同一套指标。

用法:
    python src/evaluate.py                    # 默认 0.25 缩放
    python src/evaluate.py --scale 0.5
"""
import argparse
import os
import datetime                      # 结果表里记录评测时间
import cv2
import numpy as np
from eval_metrics import load_pfm, summarize            # 指标库 (同目录, 直接 import)
from sgbm_depth import imread_gray, parse_calib, estimate_disparity  # 复用管线


def evaluate_disparity(data_dir, scale):
    """对指定场景计算 SGBM 视差并对比真值, 返回 (指标dict, 备注)"""
    # ---- 1. 真值: 缩放 + 手动缩数值 (resize 不缩数值, 血的教训) ----
    gt = load_pfm(os.path.join(data_dir, "disp0.pfm"))               # 全分辨率视差真值
    gt_s = cv2.resize(gt, None, fx=scale, fy=scale,
                      interpolation=cv2.INTER_NEAREST) * scale       # 最近邻缩放 + 数值等比

    # ---- 2. SGBM 预测 (与 sgbm_depth.py 完全同参) ----
    left = imread_gray(os.path.join(data_dir, "im0.png"))
    right = imread_gray(os.path.join(data_dir, "im1.png"))
    left_s = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    right_s = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    c = parse_calib(os.path.join(data_dir, "calib.txt"))
    ndisp = int(np.ceil(c["ndisp"] * scale / 16)) * 16               # 上取整到 16 倍数
    disp = estimate_disparity(left_s, right_s, ndisp)                # 浮点视差

    # ---- 3. 统一指标 ----
    m = summarize(disp, gt_s, name=f"SGBM@x{scale} vs GT", unit="px")
    note = (f"视差 AbsRel 中位 {m['AbsRel(median)']}, δ1 {m['δ1 (<25%)']}。"
            f"均值被遮挡区/弱纹理拉高属正常; 边缘离散率见 EdgeDisagree。")
    print(note)
    return m, note


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="SGBM vs Middlebury 真值评测")
    ap.add_argument("--data-dir", default=os.path.join(root, "data", "Adirondack-perfect"))
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--results", default=os.path.join(root, "outputs", "results.md"))
    args = ap.parse_args()

    scene = os.path.basename(os.path.normpath(args.data_dir))        # 场景名, 写进结果表
    m, note = evaluate_disparity(args.data_dir, args.scale)

    # ---- 追加写 results.md, 形成可提交的评测记录 ----
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    header = "" if os.path.exists(args.results) else (
        "# 评测结果记录\n\n| 时间 | 模型 | 场景 | scale | AbsRel(mean) | AbsRel(median) | MAE | δ1 | EdgeDisagree |\n"
        "|---|---|---|---|---|---|---|---|---|\n")
    row = (f"| {datetime.datetime.now():%Y-%m-%d %H:%M} | SGBM | {scene} | {args.scale} "
           f"| {m['AbsRel(mean)']} | {m['AbsRel(median)']} | {m['MAE']} | {m['δ1 (<25%)']} | {m['EdgeDisagree']} |\n")
    with open(args.results, "a", encoding="utf-8") as f:             # 追加模式, 保留历史记录
        f.write(header + row)
    print(f"结果已追加到 {args.results}")


if __name__ == "__main__":
    main()
