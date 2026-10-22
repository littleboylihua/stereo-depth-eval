# -*- coding: utf-8 -*-
"""
eval_metrics.py: 深度/视差评测指标库 (统一口径, 后续所有模型都用它评)
指标: AbsRel / MAE / δ1 / 边缘离散率 (edge disagreement)

约定: pred 与 gt 为同尺寸 float 数组, 无效值用 NaN 或 <=0 表示, 由本模块的 mask 逻辑统一处理。
"""
import numpy as np       # 数值计算


def load_pfm(path):
    """解析 Middlebury 的 .pfm 视差真值文件 -> float32 二维数组 (无效处为 inf)"""
    with open(path, "rb") as f:                          # 二进制打开
        header = f.readline().decode().strip()           # 首行: 'Pf'=单通道 / 'PF'=三通道
        assert header in ("Pf", "PF"), f"不是合法 PFM 文件: {header}"
        w, h = map(int, f.readline().decode().split())   # 第二行: 宽 高
        scale = float(f.readline().decode().strip())     # 第三行: 字节序线索 (负数=小端)
        dtype = "<f4" if scale < 0 else ">f4"            # 按符号确定端序
        data = np.fromfile(f, dtype=dtype)               # 读原始字节
    img = data.reshape(h, w)                             # 行优先排布
    return np.flipud(img).astype(np.float32)             # PFM 原点在左下, 需上下翻转成图像惯例


def valid_mask(*arrays):
    """取多个数组共同的有效区: 全为有限值且 >0 (视差/深度的物理有效范围)"""
    m = None                                             # 累积掩膜
    for a in arrays:                                     # 逐个数组求 AND
        cur = np.isfinite(a) & (a > 0)                   # 有限且为正才算有效
        m = cur if m is None else (m & cur)              # 第一轮直接赋值, 之后按位与
    return m


def abs_rel(pred, gt):
    """AbsRel = mean(|pred-gt| / gt): 深度估计领域最常用的相对误差, 返回 (均值, 中位数)"""
    m = valid_mask(pred, gt)                             # 共同有效区
    e = np.abs(pred[m] - gt[m]) / gt[m]                  # 逐像素相对误差
    return float(e.mean()), float(np.median(e))          # 均值受离群值影响大, 中位数更抗噪, 两个都报


def mae(pred, gt):
    """MAE = mean(|pred-gt|): 绝对误差, 深度场景下单位是米, 直接回答'差几米'"""
    m = valid_mask(pred, gt)
    return float(np.abs(pred[m] - gt[m]).mean())


def delta1(pred, gt, thresh=0.25):
    """δ1 = 相对误差 < 25% 的像素占比 (KITTI 惯例), 越接近 1 越好"""
    m = valid_mask(pred, gt)
    e = np.abs(pred[m] - gt[m]) / gt[m]
    return float((e < thresh).mean())


def edge_disagreement(pred, gt, rel_grad_thresh=0.5, flip=0.25):
    """
    边缘离散率: 深度突变处 (物体轮廓) 的错分程度。
    参考 NODAR 2026 Orin 横评的动机: AbsRel 对边缘错误不敏感 (边缘像素占比小),
    但人眼/避障最在乎边缘。单目模型在此指标上 17~46%, 学习双目 2~9%。

    定义 (简化版):
      边缘像素 = GT 相对梯度 |∇gt|/gt > rel_grad_thresh 的像素
      离散 = 边缘像素中 |pred-gt|/gt > flip (前后景判断翻转) 的比例
    """
    gt_clean = np.where(np.isfinite(gt), gt, np.nanmedian(gt[valid_mask(gt)]))  # 先把 inf/NaN 填成中位数,
    gy, gx = np.gradient(gt_clean)                       # 否则梯度在真值无效区附近传播出 inf 污染边缘判定
    rel_grad = np.sqrt(gx**2 + gy**2) / np.maximum(gt, 1e-6)   # 相对梯度, 防除零
    edge = (rel_grad > rel_grad_thresh) & valid_mask(pred, gt) # 边缘且两边都有效
    if edge.sum() == 0:                                  # 没有边缘像素时返回 NaN, 别除零
        return float("nan"), 0
    flip_ratio = (np.abs(pred[edge] - gt[edge]) / gt[edge] > flip).mean()
    return float(flip_ratio), int(edge.sum())            # 离散比例 + 边缘像素数


def summarize(pred, gt, name="model", unit=""):
    """打印并返回一份指标汇总 (供 evaluate 流程与 README 表格使用)"""
    ar_mean, ar_med = abs_rel(pred, gt)                  # 相对误差
    m = valid_mask(pred, gt)                             # 有效区
    d1 = delta1(pred, gt)                                # δ1
    ed, n_edge = edge_disagreement(pred, gt)             # 边缘离散率
    rows = [
        ("AbsRel(mean)", f"{ar_mean*100:.2f}%"),
        ("AbsRel(median)", f"{ar_med*100:.2f}%"),
        ("MAE", f"{mae(pred, gt):.4f} {unit}"),
        ("δ1 (<25%)", f"{d1*100:.1f}%"),
        ("EdgeDisagree", f"{ed*100:.2f}%  (n={n_edge})"),
    ]
    print(f"\n[{name}]  有效像素 {m.sum()}")
    for k, v in rows:                                    # 逐行打印, 对齐好看
        print(f"  {k:<14} {v}")
    return {k: v for k, v in rows}                       # 方便调用方写进结果表
