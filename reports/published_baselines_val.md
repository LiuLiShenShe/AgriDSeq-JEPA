# 官方现成 baseline 预测：同 PhenoBench val 拆分实测参照

本轮没有训练或推理第二个模型。只下载作者公开的 ERFNet 与 DeepLabV3+ **val prediction archives**，对当前本地 PhenoBench v1.1.0 原始分辨率 772 张 val 标签运行未改动的官方 devkit。以下是本轮重新评价的 **val** 数字，未混用论文 Table III 的 test 结果。

## 正式官方 val 指标

| 官方现成预测 | Soil IoU | Crop IoU | Weed IoU | mIoU | 已验证 PNG |
|---|---:|---:|---:|---:|---:|
| ERFNet | 99.33% | 95.34% | 68.65% | 87.77% | 772/772 |
| DeepLabV3+ | 99.30% | 95.13% | 68.00% | 87.47% | 772/772 |

评价结果来自[官方 `evaluate_semantics`](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/evaluate_semantics.py)，devkit 固定 SHA `0edc128ef7f67c8c6577554c7d1a2e382e2ea81f`。同时实际运行 `phenobench-eval --task semantics`，CLI 输出 crop/weed/mIoU 与函数返回结果一致；soil 从同一官方函数返回值取得。官方结果已经乘 100 并四舍五入到两位小数，不能再次乘 100。

## 下载与完整性证据

来源由[官方 baseline README](https://github.com/PRBonn/phenobench-baselines/blob/78db625441e54c5b64efabeb0d886d020961665b/README.md)的 Validation 预测链接给出。

| 方法 | 本轮实际下载来源 | 归档文件 | Bytes | 完整 SHA-256 |
|---|---|---|---:|---|
| ERFNet | [作者 ERFNet val 预测](https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/predictions/erfnet/erfnet-val.zip) | `runs/published_baselines/erfnet-val.zip` | 5721202 | `2e21129799861b2683a76028cfcc4e719b82dc183b02c408f4b49c4277dd6ada` |
| DeepLabV3+ | [作者 DeepLabV3+ val 预测](https://www.ipb.uni-bonn.de/html/projects/phenobench/semantic_segmentation/predictions/deeplab/deeplab-val.zip) | `runs/published_baselines/deeplab-val.zip` | 5600792 | `846f67be3e331f8b1ff85deae2d90ab1bcf8fe568015445640a9496ba331ed13` |

两个 HTTP 下载实际成功；归档各有 772 个 PNG，解压后结构为 `<method>/val/semantics/*.png`。评价前外层严格核对：

- 772/772 预测文件名与 `/data/datasets/PhenoBench/PhenoBench-v110/val/semantics` 完全一致。
- 全部预测大小为原始 1024×1024，单通道 PNG，Pillow mode `L`。
- 全部预测类别 ID 仅为 `[0,1,2]`，无需改变作者预测。
- GT 使用官方 `convert_partial_semantics` 合并 `3→1、4→2`；soil 0 参与计分。

该校验在调用官方评价前完成，避免官方循环 `zip(gt,pred)` 在缺文件时静默截断。每份预测均实际经过 CLI 与官方函数两次全量评分。

## 资源、运行产物与复现

CPU 执行，明确 `CUDA_VISIBLE_DEVICES=''`、`OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`，没有访问 GPU0/GPU1，没有触碰正式 AgriDSeq-JEPA run。解压、核验与两次官方评价总用时 ERFNet 79.263 秒、DeepLabV3+ 79.092 秒，不计下载时间。

只提交轻量报告和 `reports/published_baselines_val.json`。下载 zip、作者预测 PNG 和官方 CLI 日志均位于 Git 忽略目录：

```text
runs/published_baselines/
  erfnet-val.zip
  deeplab-val.zip
  erfnet/val/semantics/*.png
  erfnet/metrics/phenobench_official_cli.txt
  erfnet/metrics/phenobench_official.json
  deeplab/val/semantics/*.png
  deeplab/metrics/phenobench_official_cli.txt
  deeplab/metrics/phenobench_official.json
  published_baselines_val.json
```

从 `/fj/AgriDSeq-JEPA` 复评已有公开预测：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  .venv/bin/python - <<'PY'
from pathlib import Path
from src.dataset.phenobench import PhenoBenchSegmentation
from scripts.agriculture.evaluation import official_metrics

dataset = PhenoBenchSegmentation(
    '/data/datasets/PhenoBench/PhenoBench-v110', train=False)
assert len(dataset) == 772 and dataset.test_scale is None
for method in ('erfnet', 'deeplab'):
    base = Path('runs/published_baselines') / method
    print(method, official_metrics(dataset, base / 'val', base / 'metrics'))
PY
```

本轮下载、严格 zip 成员检查、解压及评分执行脚本为 `/tmp/agridseq_published_baselines_eval.py`；Git 中的 `scripts/agriculture/evaluation.py` 已提供长期可复用的全量检查与官方评价入口。

## 比较边界

该参照解决了 **同一 val 拆分、同一 1024×1024 GT、同一官方指标** 的问题，但不是控制了全部训练条件的公平预训练对照。官方[ERFNet 配置](https://github.com/PRBonn/phenobench-baselines/blob/78db625441e54c5b64efabeb0d886d020961665b/semantic_segmentation/config/config_erfnet.yaml)和[DeepLabV3+ 配置](https://github.com/PRBonn/phenobench-baselines/blob/78db625441e54c5b64efabeb0d886d020961665b/semantic_segmentation/config/config_deeplab.yaml)指定 768×768 训练 crop、1024×1024 val/test center crop、batch 4、`max_epoch=4096`、lr `5e-4`、weight decay `2e-4`、class-weighted cross entropy `[1.47,5.06,10.02]`、single-image normalization、`pretrained=false`。这是公开配置，归档预测并未携带足以确认全部实际训练更新数的日志。

本轮 AgriDSeq-JEPA 使用官方 JEPA ViT-B/16 预训练、512×512 crop、UPerNet 主/辅损失、AdamW 与 poly/warmup、10k optimizer-update 农业探索预算，验证采用滑窗融合；架构、训练 crop、损失、更新预算及初始化均有差异。因此可报告同 val 数字与差值，并清楚标注协议差异；不能据此把性能差异单独归因于 JEPA，也不能把 best-val 称为独立 test 表现。

信息来源：本轮直接读取的官方 baseline README、公开 val 归档、固定 devkit 源码及本地执行日志；博查搜索 HTTP 403 的配额问题已记录于 S0 报告，未使用训练数据记忆替代这些测量。
