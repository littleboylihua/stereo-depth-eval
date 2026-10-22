# stereo-depth-eval — 双目/单目深度估计精度评测

用标准双目数据集 + 真值，量化评测「模型输出的距离 vs 真实距离」。
目标场景：宇树 G1 机器人头部双目相机的测距精度验证（几何基线 → 学习双目 → 单目米制模型三路对比）。

![depth](outputs/sample/depth.png)

## 当前结果（Middlebury 2014 Adirondack-perfect, 深度以米为单位统一口径）

| 模型 | 类型 | AbsRel(median) | MAE | δ1 | EdgeDisagree |
|---|---|---|---|---|---|
| SGBM (OpenCV) | 双目几何 | **0.32%** | **4.0 cm** | 98.2% | 6.8% |
| DA3-Metric Large (0.33B, GPU) | 单目米制 | 10.4% | 24.0 cm | **99.9%** | **2.6%** |
| DA2-Metric ViT-S (Hypersim) | 单目米制 | 29.2% | 70.3 cm | 22.0% | 81.2% |

> 真值深度范围 1.62~3.07m。结论与 [arXiv 2610.27199](https://arxiv.org/html/2610.27199v1) 的 Orin 横评一致：
> 双目几何的 AbsRel 最好；但 DA3 在焦距校正后 **δ1=99.9%**（99.9% 像素相对误差 <25%）、边缘离散率全场最低
> （2.6%，比 SGBM 还干净）——学习模型的完整性 + 结构精度已经非常能打，剩余误差几乎全是 **-8% 的系统性尺度偏置**
> （域外场景 + 焦距先验残差），用一个已知距离做单点尺度标定即可消除——这正是 G1 实测协议（Task 4）的核心思想。
> DA2 无焦距校正机制可喂，域外尺度漂移严重（整体偏近 30%），作为对照保留。
> 视差口径的历史评测记录见 `outputs/results.md`。

![compare](outputs/sample/compare_depth_4way.png)

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

# 4. (Task2) 单目米制模型对比: 先下载权重 (~99MB, 国内走 hf-mirror)
curl -L -o models/depth_anything_v2_metric_hypersim_vits.pth ^
  https://hf-mirror.com/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
python src/evaluate_mono.py    # SGBM vs DA2 vs 真值 三方对比, 输出 compare_depth.png

# 5. (Task3) DA3-Metric-Large 四方对比: 权重 ~1.34GB (config.json + model.safetensors)
#   从 https://hf-mirror.com/depth-anything/DA3METRIC-LARGE 下载到 models/DA3METRIC-LARGE/
python src/evaluate_da3.py     # GT vs SGBM vs DA2 vs DA3, 输出 compare_depth_4way.png
```

需要 torch（GPU 可跑 DA3, 8GB 显存充裕：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`；
纯 CPU 也能跑全流程, DA3 单图约 12s）。

## 项目结构

```
├── scripts/download_middlebury.py   # 数据下载
├── src/
│   ├── sgbm_depth.py                # SGBM 测距管线 (标定解析/视差/深度/可视化)
│   ├── mono_depth.py                # 单目米制模型适配层 (DA2-Metric ViT-S)
│   ├── mono_da3.py                  # 单目米制模型适配层 v2 (DA3-Metric Large, 含焦距换算)
│   ├── eval_metrics.py              # 统一指标库: AbsRel/MAE/δ1/边缘离散率 + PFM 解析
│   ├── evaluate.py                  # SGBM vs 真值, 结果落 results.md
│   ├── evaluate_mono.py             # 单目 vs SGBM vs 真值 三方对比
│   └── evaluate_da3.py              # GT vs SGBM vs DA2 vs DA3 四方对比
├── third_party/
│   ├── depth_anything_v2/           # DA2 官方模型代码 (Apache-2.0, 已内嵌)
│   └── depth_anything_3/            # DA3 官方模型代码 (Apache-2.0, 已内嵌+精简)
├── models/                          # 权重 (gitignore, 见快速开始)
├── data/                            # 数据集 (gitignore, 用脚本下载)
├── outputs/                         # 结果图 + results.md 评测记录
└── notes/                           # 学习/复盘笔记
```

## Roadmap

- [x] Task 1 — SGBM 几何基线 + 真值评测闭环
- [x] Task 2 — 单目米制模型接入（Depth Anything V2 Metric ViT-S）+ 三方对比
- [x] Task 2.5 — GPU 环境（RTX 4060 8GB）+ DA3-Metric Large 接入 + 四方对比
- [ ] Task 3 — 学习双目接入（IGEV / Lite Any Stereo），边缘指标校准
- [ ] Task 4 — G1 机器人实测协议（标定 → 分距离点实测 → 误差曲线）

### 关键技术笔记

- 双目深度误差 δZ = Z²·σd/(f·B)：G1 小基线（~6cm）在 3~4m 外撑不住厘米级精度；
- `cv2.resize` 只重采样像素、不缩放数值——视差/焦距/doffs 等像素量缩放后必须手动等比换算；
- `StereoSGBM.compute()` 输出 int16 定点数，真实视差 = 值/16；
- 单目米制模型换分辨率会发生尺度漂移（arXiv 2610.27199），评测时锁定分辨率；
- **DA3-Metric 的"米"要自己算**：`metric_depth = focal_px * net_output / 300`，其中 focal_px 是
  **推理输出网格**上的真实焦距（真实焦距 × 输出宽/原图宽）；`inference()` 的 intrinsics 参数对纯米制 preset 无效；
- DA3 输出在 process_res(默认504) 网格上，评规则需要 resize 回 GT 网格（米制量缩放数值不变）。

## 数据与许可

数据来自 [Middlebury Stereo Dataset 2014](https://vision.middlebury.edu/stereo/data/)（学术用途），
不入库，通过脚本下载。
