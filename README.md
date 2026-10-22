# stereo-depth-eval — 双目/单目深度估计精度评测

用标准双目数据集 + 真值，量化评测「模型输出的距离 vs 真实距离」。
目标场景：宇树 G1 机器人头部双目相机的测距精度验证（几何基线 → 学习双目 → 单目米制模型三路对比）。

![depth](outputs/sample/depth.png)

## 当前结果（SGBM 基线 @ Middlebury 2014 Adirondack-perfect, x0.25）

| 模型 | AbsRel(mean) | AbsRel(median) | MAE | δ1 | EdgeDisagree |
|---|---|---|---|---|---|
| SGBM (OpenCV) | 8.33% | **1.06%** | 1.33 px | **95.2%** | 44.0%* |

> \* EdgeDisagree 为本项目自定义的简化边缘指标（GT 相对梯度 >0.5 的像素中，预测相对误差 >25% 的比例），
> 定义将在引入学习模型时统一校准。均值被遮挡/弱纹理区拉高属正常，看中位数与 δ1。

真值深度范围 1.62~3.07m（Middlebury 近距离摆拍场景），采样点误差与官方真值一致。

## 快速开始

```bash
# 环境: Python 3.11+ (项目在 3.11 / OpenCV 5.0 下开发)
pip install -r requirements.txt

# 1. 下载数据 (Middlebury 2014, ~35MB, 已配代理环境可直接下)
python scripts/download_middlebury.py

# 2. 跑 SGBM 测距管线, 输出视差图/深度图到 outputs/
python src/sgbm_depth.py

# 3. 对真值评测, 指标追加到 outputs/results.md
python src/evaluate.py
```

## 项目结构

```
├── scripts/download_middlebury.py   # 数据下载
├── src/
│   ├── sgbm_depth.py                # SGBM 测距管线 (标定解析/视差/深度/可视化)
│   ├── eval_metrics.py              # 统一指标库: AbsRel/MAE/δ1/边缘离散率 + PFM 解析
│   └── evaluate.py                  # 真值对比评测, 结果落 results.md
├── data/                            # 数据集 (gitignore, 用脚本下载)
├── outputs/                         # 结果图 + results.md 评测记录
└── notes/                           # 学习/复盘笔记
```

## Roadmap

- [x] Task 1 — SGBM 几何基线 + 真值评测闭环
- [ ] Task 2 — 单目米制模型接入（Depth Anything 3 Metric / MoGe-2）
- [ ] Task 3 — 多模型统一评测脚本（含边缘指标校准）
- [ ] Task 4 — G1 机器人实测协议（标定 → 分距离点实测 → 误差曲线）

### 关键技术笔记

- 双目深度误差 δZ = Z²·σd/(f·B)：G1 小基线（~6cm）在 3~4m 外撑不住厘米级精度；
- `cv2.resize` 只重采样像素、不缩放数值——视差/焦距/doffs 等像素量缩放后必须手动等比换算；
- `StereoSGBM.compute()` 输出 int16 定点数，真实视差 = 值/16；
- 单目米制模型换分辨率会发生尺度漂移（arXiv 2610.27199），评测时锁定分辨率。

## 数据与许可

数据来自 [Middlebury Stereo Dataset 2014](https://vision.middlebury.edu/stereo/data/)（学术用途），
不入库，通过脚本下载。
