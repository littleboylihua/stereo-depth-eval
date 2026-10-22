# -*- coding: utf-8 -*-
"""
mono_da3.py: 单目米制深度模型适配层 v2 (Depth Anything 3 Metric-Large, 0.35B, Apache-2.0)
官方代码内嵌 third_party/depth_anything_3, 权重 models/DA3METRIC-LARGE/ (1.34GB, 不入库)

与 DA2 的区别:
  - DA3 网络内部用"深度-射线"表示, forward 时已按预测内参完成米制换算
    (utils/alignment.py: apply_metric_scaling, focal*net_output/300), 返回即米, 无需后处理
  - 推理分辨率 504 (upper_bound_resize), 输出深度在该网格上, 需自行缩放回原图
  - 输入约定 RGB (api.py L302: ndarray 按 RGB 处理) —— DA2 是 BGR, 别搞混!

用法:
    python src/mono_da3.py --image data/Adirondack-perfect/im0.png
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
from depth_anything_3.api import DepthAnything3           # DA3 高层 API

WEIGHT_DIR = os.path.join(ROOT, "models", "DA3METRIC-LARGE")


def load_model(weight_dir=WEIGHT_DIR, device=None):
    """
    从本地权重目录加载 DA3-Metric-Large。
    from_pretrained 会读目录里的 config.json(model_name=da3metric-large) + model.safetensors,
    这是 PyTorchModelHubMixin 的标准行为 —— 换模型只要换目录。
    """
    if device is None:                                    # 不指定就自动选
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(weight_dir)    # 按目录内 config 构建网络并载权重
    model = model.to(device).eval()                       # eval: 关闭训练期行为
    print(f"DA3-Metric-Large loaded on {device} "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    return model


def infer(model, img_bgr, process_res=504):
    """
    DA3 单目推理(原始网络输出)。
    输入: BGR 图 (内部转 RGB 再喂, 因为 DA3 的 ndarray 约定是 RGB)
    输出: (raw_depth, conf) —— 注意! 这是 net_output, 不是米!
          DA3-Metric-Large 的米制换算必须由调用方完成: to_metric(raw, focal_px)
          (实验验证: inference() 的 intrinsics 参数对本 preset 无效, pred.is_metric 也不是 1)
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)        # BGR -> RGB (DA3 约定)
    with torch.inference_mode():                          # 推理无需梯度
        pred = model.inference(image=[rgb], process_res=process_res,
                               export_format="none")      # 只要深度, 不导出文件
    d = np.asarray(pred.depth)                            # (N,H,W) 取 N=1
    conf = np.asarray(pred.conf) if pred.conf is not None else None
    return d[0].astype(np.float32), (conf[0].astype(np.float32) if conf is not None else None)


def to_metric(raw, focal_px):
    """
    net_output -> 米制深度 (官方 README 公式: metric_depth = focal * net_output / 300)。
    focal_px 必须是"推理输出网格"上的真实焦距:
        focal_px = 真实焦距 * (输出图宽 / 原图宽)   # upper_bound_resize 的等比系数
    原理: 米制模型的尺度锚定在"典型相机焦距"先验上(训练集统计), 你的镜头越长,
    深度被低估得越多, 必须用真实焦距重新标定 —— 这就是"单目尺度漂移"的工程修正。
    """
    return raw * (focal_px / 300.0)


def resize_to(depth, size_hw):
    """米制深度缩放到目标网格; 米制量缩放数值不变(与视差相反), bilinear 即可"""
    return cv2.resize(depth, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)


def save_vis(depth, out_path, title="DA3-Metric-Large depth (m)"):
    """伪彩色 + 色条保存; 量程取有效深度 2~99 分位, 避免离群值拉爆色标"""
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
    ap = argparse.ArgumentParser(description="DA3-Metric-Large 单目米制深度 demo")
    ap.add_argument("--image", default=os.path.join(ROOT, "data", "Adirondack-perfect", "im0.png"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--process-res", type=int, default=504, help="推理分辨率上界, 越大越慢越准")
    ap.add_argument("--focal", type=float, default=None, help="原图真实焦距(px), 给了才输出米制深度")
    ap.add_argument("--img-width", type=int, default=None, help="原图宽度(px), 与 --focal 配合换算")
    args = ap.parse_args()

    buf = np.fromfile(args.image, dtype=np.uint8)         # fromfile 绕开中文路径坑
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)             # BGR 三通道
    assert img is not None, f"读图失败: {args.image}"

    model = load_model()                                  # 自动 cuda/cpu
    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None  # GPU 计时要同步后再计时
    t0 = time.time()
    raw, _ = infer(model, img, args.process_res)          # net_output @ process_res 网格
    if torch.cuda.is_available():
        torch.cuda.synchronize()                          # 不同步的话拿到的是异步提交时间
    dt = time.time() - t0

    if args.focal and args.img_width:                     # 有标定才做米制换算
        focal_px = args.focal * (raw.shape[1] / args.img_width)   # 换到输出网格
        depth = to_metric(raw, focal_px)                  # 米
        v = depth[depth > 0]
        print(f"推理耗时 {dt:.2f}s ({'GPU' if torch.cuda.is_available() else 'CPU'}, "
              f"process_res={args.process_res}), 网格 {raw.shape}, "
              f"米制深度范围 {v.min():.2f}~{v.max():.2f}m (focal_px={focal_px:.1f})")
    else:                                                 # 没标定就只展示 net_output
        depth = raw
        v = depth[depth > 0]
        print(f"推理耗时 {dt:.2f}s, 网格 {raw.shape}, net_output 范围 {v.min():.3f}~{v.max():.3f} "
              f"(未乘焦距, 不是米! 传 --focal --img-width 可得米制深度)")

    out = os.path.join(args.out_dir, "mono_da3_depth.png")
    save_vis(depth, out)
    print(f"深度图已存: {out}")


if __name__ == "__main__":
    main()
