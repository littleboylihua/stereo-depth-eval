# -*- coding: utf-8 -*-
"""
sgbm_depth.py: SGBM 双目测距管线 (几何基线)
流程: 读图 -> 缩放 -> SGBM 视差 -> 深度 (Middlebury 约定) -> 可视化 + 采样点

用法:
    python src/sgbm_depth.py --data-dir data/Adirondack-perfect
    python src/sgbm_depth.py --data-dir data/Adirondack-perfect --scale 0.5
"""
import argparse                        # 命令行参数
import os                              # 路径
import cv2                             # SGBM / 图像 IO
import numpy as np                     # 数组
import matplotlib                      # 绘图
matplotlib.use("Agg")                  # 无窗口后端, 服务器/脚本环境通用
import matplotlib.pyplot as plt


def imread_gray(path):
    """读灰度图; 走 fromfile+imdecode 绕开 Windows 中文路径下 cv2.imread 返回 None 的坑"""
    buf = np.fromfile(path, dtype=np.uint8)             # 先按字节读入, 路径编码问题不影响
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)       # 灰度解码: 匹配只依赖亮度
    assert img is not None, f"读图失败: {path}"          # 早失败, 不带病运行
    return img


def parse_calib(path):
    """解析 Middlebury calib.txt -> {f, cx0, cx1, doffs, baseline_m, ndisp, vmin, vmax}"""
    calib = {}                                           # 结果字典
    with open(path) as f:                                # 文本格式, 逐行 key=value
        for line in f:
            if "=" not in line:                          # 跳过空行/异常行
                continue
            k, v = line.split("=", 1)                    # 只按第一个 = 切
            calib[k.strip()] = v.strip()                 # 去空白存入
    cx0 = float(calib["cam0"].replace("[", "").replace("]", "").split(";")[0].split()[2])  # cam0 第1行第3个数 = cx0
    cx1 = float(calib["cam1"].replace("[", "").replace("]", "").split(";")[0].split()[2])  # 同理 cx1
    fx = float(calib["cam0"].replace("[", "").replace("]", "").split(";")[0].split()[0])   # 焦距 fx (已校正, fx=fy)
    return dict(
        f=fx,                                            # 焦距
        doffs=cx1 - cx0,                                 # 主点差, 深度公式的补偿项
        baseline_m=float(calib["baseline"]) / 1000.0,    # 基线 mm -> m
        ndisp=int(calib["ndisp"]),                       # 真值最大视差
        vmin=float(calib.get("vmin", 0)),                # 真值视差下界(可用来设色标)
        vmax=float(calib.get("vmax", 1e9)),              # 真值视差上界
    )


def estimate_disparity(left, right, ndisp):
    """SGBM 视差估计; 返回 float 视差图 (定点值已 /16 还原)"""
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,          # 已校正收敛对, 视差非负
        numDisparities=ndisp,    # 搜索范围 (0, ndisp], 必须 16 的倍数
        blockSize=7,             # 匹配窗口, 奇数; 精度/速度折中
        P1=8 * 1 * 49,           # 平滑惩罚(视差差1): 8*通道*blockSize^2
        P2=32 * 1 * 49,          # 平滑惩罚(跳变>1): 必须 > P1
        uniquenessRatio=10,      # 最优/次优区分度门槛
        speckleWindowSize=200,   # 小连通斑清除窗口
        speckleRange=1,          # 斑内允许视差波动
        disp12MaxDiff=1,         # 左右一致性检查容差
    )
    raw = sgbm.compute(left, right)                      # 输出 int16 定点数
    return raw.astype(np.float32) / 16.0                 # 还原浮点视差; 不除 16 深度差 16 倍


def disparity_to_depth(disp, f, baseline_m, doffs, z_far=200.0):
    """Middlebury 约定: Z = B*f/(d+doffs); 无效视差(<=0 或超范围)掩掉"""
    valid = (disp > 0) & np.isfinite(disp)               # 基础有效性
    depth = np.zeros_like(disp, dtype=np.float32)        # 全 0 深度图
    depth[valid] = baseline_m * f / (disp[valid] + doffs)  # 只算有效像素
    depth[depth > z_far] = 0                             # 离群远距压掉, 防止色标被拉爆
    return depth, depth > 0                              # 深度图 + 最终有效掩膜


def run(data_dir, scale, out_dir):
    """完整管线: 返回 (视差图, 深度图, 标定) 供评测脚本复用"""
    os.makedirs(out_dir, exist_ok=True)                  # 输出目录
    left = imread_gray(os.path.join(data_dir, "im0.png"))    # 左目
    right = imread_gray(os.path.join(data_dir, "im1.png"))   # 右目
    c = parse_calib(os.path.join(data_dir, "calib.txt"))     # 标定参数
    print(f"原图 {left.shape}, 标定: f={c['f']:.1f}px B={c['baseline_m']*1000:.1f}mm doffs={c['doffs']:.2f}")

    # 缩放: resize 只重采样像素, 不缩数值! 所有像素量(f/doffs/ndisp)必须手动乘 scale
    left_s = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    right_s = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    f, doffs = c["f"] * scale, c["doffs"] * scale        # 像素量等比缩放, 深度不变
    ndisp = int(np.ceil(c["ndisp"] * scale / 16)) * 16   # 视差上限向上取整到 16 倍数
    print(f"缩放 x{scale}: f={f:.1f}px doffs={doffs:.2f}px ndisp={ndisp}")

    disp = estimate_disparity(left_s, right_s, ndisp)    # 视差
    depth, valid = disparity_to_depth(disp, f, c["baseline_m"], doffs)  # 深度

    # ---- 可视化: 视差伪彩 (近=大视差=红) ----
    disp_vis = np.clip(disp / ndisp * 255, 0, 255).astype(np.uint8)     # 归一化 8bit
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)          # JET 伪彩
    disp_color[~valid] = 0                                              # 无效区涂黑
    cv2.imencode(".png", disp_color)[1].tofile(os.path.join(out_dir, "disparity.png"))

    # ---- 可视化: 深度图 + 米制色条, 量程取有效深度 99 分位 (自适应, 不拍脑袋) ----
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    vmax = float(np.percentile(depth[valid], 99)) if valid.any() else 10.0
    im = ax.imshow(np.where(valid, depth, np.nan), cmap="plasma_r", vmin=0, vmax=vmax)
    ax.set_title(f"SGBM depth (B={c['baseline_m']*1000:.0f}mm f={f:.0f}px scale={scale})")
    fig.colorbar(im, ax=ax, label="depth (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "depth.png"))
    plt.close(fig)
    print(f"可视化已存: {out_dir}/disparity.png, depth.png")
    return disp, depth, c


def main():
    ap = argparse.ArgumentParser(description="SGBM 双目测距基线")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根
    ap.add_argument("--data-dir", default=os.path.join(root, "data", "Adirondack-perfect"))
    ap.add_argument("--out-dir", default=os.path.join(root, "outputs"))
    ap.add_argument("--scale", type=float, default=0.25, help="缩放系数, 1.0=原分辨率(慢)")
    args = ap.parse_args()

    disp, depth, _ = run(args.data_dir, args.scale, args.out_dir)
    h, w = depth.shape                                   # 打印 3 个采样点
    for name, (y, x) in {"中心": (h // 2, w // 2), "上1/3": (h // 3, w // 2), "下1/3": (2 * h // 3, w // 2)}.items():
        z = depth[y, x]
        print(f"  {name} ({x},{y}): 视差={disp[y, x]:.2f}px 深度={z:.2f}m" if z > 0 else f"  {name}: 无效区")


if __name__ == "__main__":
    main()
