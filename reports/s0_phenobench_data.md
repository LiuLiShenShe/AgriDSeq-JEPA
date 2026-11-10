# S0-C：PhenoBench v1.1.0 原始数据与官方评价协议核实

状态：**数据审计通过**。本轮实际读取原始 train/val RGB 文件、16-bit semantic masks；未复制、重新下载或重新解压原始数据，未使用 test 标签。以下数字来自完整文件审计，并非沿用历史报告。

## 版本、位置、来源与完整性

- 原始数据根目录：`/data/datasets/PhenoBench/PhenoBench-v110`。
- 本地 `README.MD` 首行明确 `PhenoBench version 1.1.0 (December 2023)`；其 SHA-256 为 `2b54ae5b6b9c6265d47546d1377b5b020132917a19b77c2c5815440de7201be8`。
- 原始 zip **实际仍存在**：`/data/datasets/PhenoBench/PhenoBench-v110.zip`，7,630,658,167 bytes。本轮重新运行 `sha256sum` 与 `md5sum`，不把用户给出的旧摘要当作新测量。
- zip SHA-256：`a310c1c1dcfec4ca4897097fd38c02af39ea24112f658fd9408d5deb3c7c8bfc`。
- zip MD5：`5168bba762053725890478432cdbdb1d`，与本轮直接读取的[官方数据页](https://www.phenobench.org/dataset.html)所公布 `PhenoBench-v110.zip` MD5 一致。
- 固定轻量 manifest：`reports/phenobench_train_val_manifest.csv`，SHA-256 `6ce0232f323c1bf055f4b6f588872ca5605ae964bb26a541109780e63d307f3b`。它对每个 train/val 成员记录文件名、RGB/mask 字节大小和 SHA-256、尺寸、mask 位深、映射后三类像素数。

## 完整划分与标注类型

| 划分 | RGB 图像 | semantics masks | 文件名匹配 | 尺寸 | 图像格式 | mask 格式 |
|---|---:|---:|---|---|---|---|
| train | 1407 | 1407 | 1407/1407 一一匹配 | 1024×1024 | PNG RGB，8-bit/通道 | PNG 单通道整数，16-bit |
| val | 772 | 772 | 772/772 一一匹配 | 1024×1024 | PNG RGB，8-bit/通道 | PNG 单通道整数，16-bit |

审计对 **全部 2179 图像及其 mask** 验证 PNG signature 与 IHDR，RGB 头为 `(1024,1024,8,2)`，mask 头为 `(1024,1024,16,0)`；对全部 mask 使用 Pillow 整数数组读取，验证所有像素均在 `0..4`。样本的 Pillow mode 是 `I;16`，NumPy dtype 为 `uint16`，未转成 RGB，未进行 ADE20K 的减 1 映射。

仅计数 `test/images/*.png` 得到 693；本轮没有读取 test 标签、将 test 图像用于训练或调参。数据版本 README 和[官方数据页](https://www.phenobench.org/dataset.html)均确认 1407/772/693 划分以及 1024×1024 RGB 图像和 16-bit 标注格式。

## 标签语义与完整像素统计

严格沿用官方 `convert_partial_semantics`：`0→0 soil/background`，`1→1 crop`，`2→2 weed`，`3→1 partial crop`，`4→2 partial weed`。官方网页与 zip README 对 ID 4 的说明有文字笔误；[官方转换函数源码](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/auxiliary/convert.py)和[官方教程](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/phenobench_tutorial.ipynb)明确 ID 4 是 partial weed，因此映射为 2。

| 划分 | raw 0 | raw 1 | raw 2 | raw 3 | raw 4 |
|---|---:|---:|---:|---:|---:|
| train | 1293238516 | 153463715 | 7078797 | 21310522 | 254882 |
| val | 727889307 | 66418271 | 4088634 | 10916383 | 188077 |

| 划分 | Soil 像素/比例 | Crop 像素/比例 | Weed 像素/比例 |
|---|---|---|---|
| train | 1293238516 / 87.656600% | 174774237 / 11.846318% | 7333679 / 0.497082% |
| val | 727889307 / 89.918308% | 77334654 / 9.553377% | 4276711 / 0.528315% |

`0/1/2` 均是有效计算类别，几何 padding 才可设专门的 ignore index，例如 255。RGB 与 mask 的 resize/crop/flip 必须同步，mask resize 必须 NEAREST。验证预测应在原始 1024×1024 坐标逐图导出，不能以缩小图的指标代替官方评价。

审计实现位于本轮 `/tmp/agridseq_data_audit.py`，逐成员读取与哈希、检查 IHDR、按 `np.bincount` 计算 raw/mapped 像素分布；使用 8 个读文件线程。实测完整审计用时 97.617 秒。精确汇总在 `/tmp/agridseq_data_audit.json`，上述固定 manifest 为提交的长期证据。

## 官方 devkit、CLI 与安全的全量评价边界

- 官方源代码：[PRBonn/phenobench](https://github.com/PRBonn/phenobench)。本轮独立克隆至 `runs/phenobench-devkit`（Git 忽略，不纳入新项目源代码）。
- 固定完整 SHA：`0edc128ef7f67c8c6577554c7d1a2e382e2ea81f`。
- `pyproject.toml` 包版本 `0.1.0`（这是 devkit 版本，**不等于数据版本 1.1.0**）；eval extras 明确 `torchmetrics==0.10.3`。
- 官方安装方式为 `pip install "phenobench[eval]"`；为了锁定实际源码，本项目环境可以安装 `./runs/phenobench-devkit[eval]`。

正式命令来自[官方 README](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/README.md)：

```bash
phenobench-eval --task semantics \
  --phenobench_dir /data/datasets/PhenoBench/PhenoBench-v110 \
  --prediction_dir runs/phenobench_dseq_vitb16_s20261110/predictions/val \
  --split val
```

要求文件布局为 `prediction_dir/semantics/*.png`。运行前外层必须检查 val 772 个 GT 文件名与 772 个预测文件名完全相同，所有预测为 1024×1024 单通道 PNG，ID 仅 `0/1/2`。理由是[官方评价函数](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/evaluate_semantics.py)使用 `zip(gt_fnames,pred_original_fnames)`，少文件时会静默截断；只运行 CLI 不足以证明完整 772 张验证集被评价。

官方函数接口（未改动 devkit）：

```python
from pathlib import Path
from phenobench.evaluation.evaluate_semantics import evaluate_semantics

results = evaluate_semantics({
    "phenobench_dir": Path("/data/datasets/PhenoBench/PhenoBench-v110"),
    "prediction_dir": Path("runs/phenobench_dseq_vitb16_s20261110/predictions/val"),
    "split": "val",
    # 可选："export": Path("runs/.../metrics/official_export"),
})
# results: {"soil": float, "crop": float, "weed": float, "mIoU": float}
```

官方实现使用 `MulticlassJaccardIndex(num_classes=3,average=None)`，随后乘 `100.0` 并四项 `round(...,2)`；结果已经是百分比。`export` 存在时创建 `export/all/eval_semantics.yaml`。`phenobench-eval --task semantics` 的[CLI 实现](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/phenobench_eval.py)只打印 crop、weed、mIoU；soil 应从同一未改动的官方 `evaluate_semantics` 返回值取得，不能用自实现指标冒充官方 soil。

本轮还按用户要求使用 Context7：先 `npx ctx7@latest library PhenoBench "<完整问题>"`，得到 `/prbonn/phenobench`；再 `npx ctx7@latest docs /prbonn/phenobench "<完整问题>"`，实际拉到官方 README 与 tutorial 的安装、CLI、目录、partial-label 文档。博查搜索已由主任务实际尝试并返回 HTTP 403（余额/套餐配额不足）；上述证据来自当前直接读取的官方页面和固定源码，未把搜索失败隐藏为成功。

## 官方 baseline 的比较边界（没有增加模型训练）

[官方 baseline 仓库](https://github.com/PRBonn/phenobench-baselines)当前 main SHA `78db625441e54c5b64efabeb0d886d020961665b`，仅为核实公开方法与配置读取，不纳入新项目模型训练。其[ERFNet 配置](https://github.com/PRBonn/phenobench-baselines/blob/78db625441e54c5b64efabeb0d886d020961665b/semantic_segmentation/config/config_erfnet.yaml)和[DeepLabV3+ 配置](https://github.com/PRBonn/phenobench-baselines/blob/78db625441e54c5b64efabeb0d886d020961665b/semantic_segmentation/config/config_deeplab.yaml)采用：

- 768×768 random train crop，1024×1024 val/test center crop；随机 hflip/vflip、1.0–1.1 scale、颜色增强。
- 1 GPU，batch 4，`max_epoch=4096`，学习率 `5e-4`，weight decay `2e-4`。
- 加权 cross entropy，class weights `[1.47,5.06,10.02]`，`pretrained=false`，single-image normalization。

这些公开配置参数不能被称为与本轮 JEPA 512×512 / 10k updates 协议相同。[官方论文 Table III](https://arxiv.org/html/2306.04557)报告的是 **test set**：

| 官方方法 | Soil IoU | Crop IoU | Weed IoU | mIoU | 拆分 |
|---|---:|---:|---:|---:|---|
| ERFNet | 99.28% | 94.30% | 64.37% | 85.98% | test |
| DeepLabV3+ | 99.25% | 94.07% | 64.59% | 85.97% | test |

本轮 best-val 不得据此声称独立 test 提升。官方仓库同时发布 [ERFNet val 预测](https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/predictions/erfnet/erfnet-val.zip)和[DeepLabV3+ val 预测](https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/predictions/deeplab/deeplab-val.zip)；如需同拆分参照，可对这些已公开预测运行相同官方评价，不涉及训练第二模型，但应单独记录下载与评价事实。**本报告未下载或评价 baseline 预测，未提供虚构 val baseline 数字。**
