# -*- coding: utf-8 -*-
"""
下载 Middlebury 2014 双目数据集 (Adirondack-perfect 默认场景) 到 data/<scene>/
文件: im0.png(左) im1.png(右) calib.txt(标定) disp0.pfm(视差真值, 约 22MB)

用法:
    python scripts/download_middlebury.py                    # 默认 Adirondack-perfect
    python scripts/download_middlebury.py --scene Art-lr     # 换场景
"""
import argparse                      # 命令行参数解析(标准库)
import os                            # 路径与目录操作
import urllib.request                # 下载; 自动走系统代理环境变量

BASE = "https://vision.middlebury.edu/stereo/data/scenes2014/datasets"
FILES = ["im0.png", "im1.png", "calib.txt", "disp0.pfm"]   # 跑通全部流程所需的四个文件


def download(url, dst):
    """下载单个文件, 已存在则跳过 (断点续传不做了, 文件都不大)"""
    if os.path.exists(dst):                       # 本地已有 -> 跳过, 支持重复执行
        print(f"  已存在, 跳过: {dst}")
        return
    print(f"  下载: {url}")

    def report(n, bs, total):                     # 进度回调: 已下载块数 x 块大小
        done = min(n * bs, total)                 # 最后一块会超过 total, min 防止 >100%
        print(f"\r  {done/1e6:7.1f} / {total/1e6:.1f} MB", end="", flush=True)

    print()
    urllib.request.urlretrieve(url, dst, reporthook=report)   # 实际下载
    print(f"\r  完成: {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="下载 Middlebury 2014 双目数据")
    ap.add_argument("--scene", default="Adirondack-perfect", help="场景名, 见 Middlebury 2014 页面")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data"), help="下载根目录 (默认 项目根/data)")
    args = ap.parse_args()

    dst_dir = os.path.join(args.out, args.scene)              # data/<scene>/
    os.makedirs(dst_dir, exist_ok=True)                       # 建目录
    print(f"场景: {args.scene} -> {dst_dir}")
    for f in FILES:                                           # 逐个下载四个文件
        download(f"{BASE}/{args.scene}/{f}", os.path.join(dst_dir, f))
    print("全部就绪。")


if __name__ == "__main__":
    main()
