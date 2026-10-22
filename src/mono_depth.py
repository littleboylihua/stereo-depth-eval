# -*- coding: utf-8 -*-
"""
mono_depth.py: 单目米制深度模型适配层 (Depth Anything V2 Metric, ViT-S, Hypersim 室内权重)
官方代码内嵌在 third_party/depth_anything_v2 (Apache-2.0), 权重在 models/ (99MB, 不入库)

用法:
    python src/mono_depth.py --image data/Adirondack-perfect/im0.png
"""
import argparse                    # 命令行参数
import os
import sys                         # 把 third_party 加进 import 路径
import cv2
import numpy as np
import torch
import matplotlib                  # 仅用于出图
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party"))     # 内嵌官方代码的 import 路径
from depth_anything_v2.dpt import DepthAnythingV2         # 米制版模型类

CKPT = os.path.join(ROOT, "models", "depth_anything_v2_metric_hypersim_vits.pth")

# ViT-S 在官方 run.py 里的构造参数 (Hypersim 室内数据集训练, max_depth=20m)
MODEL_KWARGS = dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0)


def load_model(ckpt=CKPT, device="cpu"):
    """加载 DA2-Metric ViT-S; checkpoint 是完整 state_dict(含 DINOv2 骨干), 无需额外下载"""
    model = DepthAnythingV2(**MODEL_KWARGS)               # 按官方参数构建网络
    state = torch.load(ckpt, map_location="cpu")          # CPU 加载, 避免无卡机器报错
    model.load_state_dict(state)                          # 全量载入(骨干+深度头)
    model = model.to(device).eval()                       # eval 模式: 关掉 BN/dropout 等训练行为
    return model


def infer(model, img_bgr, input_size=518):
    """
    单目米制深度推理。
    输入: BGR 图 (官方 infer_image 内部自己转 RGB/归一化/按 14 倍数缩放)
    输出: 与输入同分辨率的深度图, 单位米 (输出尺度来自训练先验, 与拍摄相机无关!)
    """
    with torch.inference_mode():                          # 推理不需梯度, 省内存提速
        depth = model.infer_image(img_bgr, input_size)    # 返回 float, shape = 原图 HxW
    return depth.astype(np.float32)


def save_vis(depth, out_path, title="DA2-Metric ViT-S depth (m)"):
    """伪彩色 + 色条保存深度图; 量程取有效深度 2~99 分位, 避免离群值拉爆色标"""
    valid = np.isfinite(depth) & (depth > 0)              # 有效区
    vmin = np.percentile(depth[valid], 2) if valid.any() else 0
    vmax = np.percentile(depth[valid], 99) if valid.any() else 10
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    im = ax.imshow(np.where(valid, depth, np.nan), cmap="plasma_r", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="depth (m)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="DA2-Metric 单目米制深度 demo")
    ap.add_argument("--image", default=os.path.join(ROOT, "data", "Adirondack-perfect", "im0.png"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--input-size", type=int, default=518, help="推理分辨率下界, 越大越慢越准")
    args = ap.parse_args()

    buf = np.fromfile(args.image, dtype=np.uint8)         # fromfile 绕开中文路径坑
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)             # BGR 三通道
    assert img is not None, f"读图失败: {args.image}"

    model = load_model()                                  # 加载模型 (CPU)
    import time
    t0 = time.time()                                      # 记录推理耗时, 报告里要用
    depth = infer(model, img, args.input_size)
    dt = time.time() - t0
    print(f"推理耗时 {dt:.2f}s (CPU, input_size={args.input_size}), 深度范围 "
          f"{depth[depth>0].min():.2f}~{depth[depth>0].max():.2f}m")

    out = os.path.join(args.out_dir, "mono_depth.png")
    save_vis(depth, out)
    print(f"深度图已存: {out}")


if __name__ == "__main__":
    main()
