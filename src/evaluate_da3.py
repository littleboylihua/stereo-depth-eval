# -*- coding: utf-8 -*-
"""
evaluate_da3.py: DA3-Metric 接入后的四方对比 —— 真值 / SGBM(几何) / DA2(单目) / DA3(单目)
复用 evaluate_mono.py 的网格构造函数, 三路几何统一在同一真值网格上以米为单位对比。

用法:
    python src/evaluate_da3.py
"""
import argparse
import os
import datetime
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_metrics import summarize                                     # 统一指标库
from evaluate_mono import gt_depth_grid, sgbm_depth_grid               # 网格构造复用
from mono_depth import load_model as load_da2, infer as infer_da2      # DA2 单目
from mono_da3 import (load_model as load_da3, infer as infer_da3,
                      to_metric as da3_to_metric)                      # DA3 单目


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="GT/SGBM/DA2/DA3 四方评测")
    ap.add_argument("--data-dir", default=os.path.join(root, "data", "Adirondack-perfect"))
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--results", default=os.path.join(root, "outputs", "results.md"))
    args = ap.parse_args()

    import time

    # 1. 真值与 SGBM 网格 (复用 evaluate_mono 的构造, 同一套标定/换算)
    z_gt, gt_valid, c = gt_depth_grid(args.data_dir, args.scale)
    h, w = z_gt.shape
    z_sgbm, _ = sgbm_depth_grid(args.data_dir, args.scale)

    # 2. DA2 单目米制 (GPU; DA2 官方 infer_image 内部用 .cuda(), 模型必须在卡上)
    img_path = os.path.join(args.data_dir, "im0.png")
    buf = np.fromfile(img_path, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    m2 = load_da2(device="cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    z2_full = infer_da2(m2, img_bgr, args.input_size)
    dt2 = time.time() - t0
    del m2                                                          # 释放内存再载 DA3
    z_da2 = cv2.resize(z2_full, (w, h), interpolation=cv2.INTER_LINEAR)

    # 3. DA3 单目米制 (GPU; net_output 需按真实焦距换算米, 见 mono_da3.to_metric)
    m3 = load_da3()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    raw3, _ = infer_da3(m3, img_bgr)                      # net_output @ process_res 网格
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt3 = time.time() - t0
    del m3
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # 真实焦距换到 DA3 输出网格: f_out = f * (输出宽/原图宽), 再套官方公式 focal*net/300
    focal_px = c["f"] * (raw3.shape[1] / img_bgr.shape[1])
    z3_grid = da3_to_metric(raw3, focal_px)               # 米 @ process_res 网格
    z_da3 = cv2.resize(z3_grid, (w, h), interpolation=cv2.INTER_LINEAR)  # 米制缩放数值不变
    print(f"DA3 focal_px={focal_px:.1f} @ {raw3.shape[1]}x{raw3.shape[0]} 网格")

    # 4. 统一指标 (单位米)
    r_sgbm = summarize(z_sgbm, z_gt, name="SGBM vs GT", unit="m")
    r_da2 = summarize(z_da2, z_gt, name="DA2 vs GT", unit="m")
    r_da3 = summarize(z_da3, z_gt, name="DA3 vs GT", unit="m")

    # 5. 四联图: 同一色标目视对比
    zmax = float(np.percentile(z_gt[gt_valid], 99))
    fig, axes = plt.subplots(1, 4, figsize=(21, 5), dpi=110)
    for ax, (title, z) in zip(axes, [
            ("GT depth", z_gt), ("SGBM (stereo)", z_sgbm),
            ("DA2-Metric ViT-S (mono)", z_da2), ("DA3-Metric Large (mono)", z_da3)]):
        im = ax.imshow(np.where(z > 0, z, np.nan), cmap="plasma_r", vmin=0, vmax=zmax)
        ax.set_title(title); ax.axis("off")
    fig.colorbar(im, ax=axes, label="depth (m)", shrink=0.8)
    fig.savefig(os.path.join(root, "outputs", "compare_depth_4way.png"), bbox_inches="tight")

    # 6. 结果追加到 results.md
    scene = os.path.basename(os.path.normpath(args.data_dir))
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    header = "" if os.path.exists(args.results) else (
        "# 评测结果记录\n\n| 时间 | 模型 | 场景 | scale | AbsRel(mean) | AbsRel(median) | MAE | δ1 | EdgeDisagree |\n"
        "|---|---|---|---|---|---|---|---|---|\n")
    rows = ""
    for name, r, extra in [("SGBM", r_sgbm, ""), ("DA2-Metric-ViTS", r_da2, f" cpu={dt2:.1f}s"),
                           ("DA3-Metric-Large", r_da3, f" gpu={dt3:.2f}s")]:
        rows += (f"| {datetime.datetime.now():%m-%d %H:%M} | {name}{extra} | {scene} | {args.scale} "
                 f"| {r['AbsRel(mean)']} | {r['AbsRel(median)']} | {r['MAE']} | {r['δ1 (<25%)']} | {r['EdgeDisagree']} |\n")
    with open(args.results, "a", encoding="utf-8") as f:
        f.write(header + rows)

    print(f"\n===== 四方对比 (GT 深度 {z_gt[gt_valid].min():.2f}~{zmax:.2f}m) =====")
    for name, r in [("SGBM", r_sgbm), ("DA2", r_da2), ("DA3", r_da3)]:
        print(f"{name:>5}: AbsRel中位 {r['AbsRel(median)']:>7} | MAE {r['MAE']:>8} | δ1 {r['δ1 (<25%)']:>7} | Edge {r['EdgeDisagree']}")
    print(f"结果已追加 {args.results}, 对比图 outputs/compare_depth_4way.png")


if __name__ == "__main__":
    import torch                                                     # 放底部: 仅本脚本需要
    main()
