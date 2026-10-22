# -*- coding: utf-8 -*-
"""
evaluate_mono.py: 单目米制模型 vs SGBM vs Middlebury 真值 —— 三方对比
关键点: 单目输出的是"米", 不是视差, 所以三者在同一张真值深度图上以米为单位对比。
米制深度 resize 时数值不变(和视差相反!), 直接 bilinear 缩放到统一网格即可。

用法:
    python src/evaluate_mono.py
    python src/evaluate_mono.py --scale 0.5
"""
import argparse
import os
import datetime
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_metrics import load_pfm, summarize          # 统一指标库
from sgbm_depth import imread_gray, parse_calib, estimate_disparity  # SGBM 管线复用
from mono_depth import load_model, infer              # 单目米制模型


def gt_depth_grid(data_dir, scale):
    """真值视差 -> 真值深度(米), 在 x-scale 网格上"""
    d = load_pfm(os.path.join(data_dir, "disp0.pfm"))                      # 全分辨率视差真值
    d_s = cv2.resize(d, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST) * scale              # 最近邻缩放+数值等比(老坑)
    c = parse_calib(os.path.join(data_dir, "calib.txt"))
    f, doffs = c["f"] * scale, c["doffs"] * scale                          # 像素量等比换算
    z = c["baseline_m"] * f / (d_s + doffs)                                # Middlebury 深度公式
    z[~np.isfinite(z) | (z <= 0)] = 0                                      # inf/非正深度置无效
    return z, (d_s > 0), c


def sgbm_depth_grid(data_dir, scale):
    """SGBM 视差 -> 深度(米), 与真值同一网格"""
    left = imread_gray(os.path.join(data_dir, "im0.png"))
    right = imread_gray(os.path.join(data_dir, "im1.png"))
    ls = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rs = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    c = parse_calib(os.path.join(data_dir, "calib.txt"))
    f, doffs = c["f"] * scale, c["doffs"] * scale
    ndisp = int(np.ceil(c["ndisp"] * scale / 16)) * 16
    disp = estimate_disparity(ls, rs, ndisp)
    z = c["baseline_m"] * f / np.where(disp > 0, disp + doffs, np.nan)     # 无效视差 -> NaN
    return np.nan_to_num(z, nan=0.0), c


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="单目 vs SGBM vs 真值 三方评测")
    ap.add_argument("--data-dir", default=os.path.join(root, "data", "Adirondack-perfect"))
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--results", default=os.path.join(root, "outputs", "results.md"))
    args = ap.parse_args()

    # 1. 真值深度 (米)
    z_gt, gt_valid, c = gt_depth_grid(args.data_dir, args.scale)
    h, w = z_gt.shape

    # 2. SGBM 深度 (米)
    z_sgbm, _ = sgbm_depth_grid(args.data_dir, args.scale)

    # 3. DA2 单目米制深度: 原图推理 -> 缩到统一网格
    import time
    model = load_model()
    buf = np.fromfile(os.path.join(args.data_dir, "im0.png"), dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)                          # BGR 原图
    t0 = time.time()
    z_full = infer(model, img_bgr, args.input_size)                        # 原图分辨率, 单位米
    dt = time.time() - t0
    z_mono = cv2.resize(z_full, (w, h), interpolation=cv2.INTER_LINEAR)    # 米制深度缩放数值不变!
    print(f"DA2 CPU 推理 {dt:.1f}s (input_size={args.input_size})")

    # 4. 统一指标 (单位米) —— 这就是选型文档说的"三路对比"
    r_mono = summarize(z_mono, z_gt, name="DA2-Metric-ViTS(hypersim) vs GT", unit="m")
    r_sgbm = summarize(z_sgbm, z_gt, name="SGBM vs GT", unit="m")

    # 5. 三联图: 真值 / SGBM / DA2, 同色标便于目视对比
    zmax = float(np.percentile(z_gt[gt_valid], 99))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=110)
    for ax, (title, z) in zip(axes, [("GT depth", z_gt), ("SGBM", z_sgbm), ("DA2-Metric ViT-S", z_mono)]):
        im = ax.imshow(np.where(z > 0, z, np.nan), cmap="plasma_r", vmin=0, vmax=zmax)
        ax.set_title(title); ax.axis("off")
    fig.colorbar(im, ax=axes, label="depth (m)", shrink=0.8)
    fig.savefig(os.path.join(root, "outputs", "compare_depth.png"), bbox_inches="tight")

    # 6. 结果追加到 results.md
    scene = os.path.basename(os.path.normpath(args.data_dir))
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    header = "" if os.path.exists(args.results) else (
        "# 评测结果记录\n\n| 时间 | 模型 | 场景 | scale | AbsRel(mean) | AbsRel(median) | MAE | δ1 | EdgeDisagree |\n"
        "|---|---|---|---|---|---|---|---|---|\n")
    rows = ""
    for name, r, extra in [("SGBM", r_sgbm, ""), ("DA2-Metric-ViTS", r_mono, f" cpu={dt:.1f}s")]:
        rows += (f"| {datetime.datetime.now():%m-%d %H:%M} | {name}{extra} | {scene} | {args.scale} "
                 f"| {r['AbsRel(mean)']} | {r['AbsRel(median)']} | {r['MAE']} | {r['δ1 (<25%)']} | {r['EdgeDisagree']} |\n")
    with open(args.results, "a", encoding="utf-8") as f:
        f.write(header + rows)
    print(f"\n结果已追加到 {args.results}, 对比图 outputs/compare_depth.png")


if __name__ == "__main__":
    main()
