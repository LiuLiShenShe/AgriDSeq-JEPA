# AgriDSeq-JEPA

独立农业迁移工程，采用作者DSeq-JEPA ViT-B/16与官方UPerNet，全参数微调PhenoBench v1.1.0三类语义分割。上游固定SHA `123082374389f78e4516157187681f9d5edb4310`，完整保留原MIT许可证及作者署名。历史AgriSA-JEPA只读归档，本项目无旧训练框架依赖。

当前真实完成进度与指标见 [最终阶段报告](reports/s1_final_report.md)。正式预算是seed20261110、单卡A6000、microbatch2×accum8=16、10000 optimizer updates；smoke不计入正式结果。

本轮已实际完成10000 optimizer updates，并恢复step10000的best checkpoint重新导出全部772张1024×1024 val预测。官方Soil/Crop/Weed IoU为99.47%/95.97%/74.62%，mIoU90.02%；这是val选择结果。详见[正式结果](reports/s1_phenobench_results.md)、[农业失败案例](reports/s1_agricultural_error_analysis.md)及[实测成本](reports/s1_gpu_compute.md)。

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r configs/agriculture/requirements-lock.txt
# 使用已有PhenoBench根目录，按本机位置修改data.root_path。
# 从作者Drive下载真实dseqjepa_vit_base_latest.pth到runs/pretrained/，SHA在配置中严格校验。
PYTHONPATH=. .venv/bin/python -m pytest tests/agriculture -q
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/agriculture/run_phenobench.py --mode verify
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONPATH=. .venv/bin/python scripts/agriculture/run_phenobench.py --mode smoke
bash scripts/agriculture/train_phenobench.sh
bash scripts/agriculture/eval_phenobench.sh
.venv/bin/phenobench-eval --task semantics \
  --phenobench_dir /data/datasets/PhenoBench/PhenoBench-v110 \
  --prediction_dir runs/phenobench_dseq_vitb16_s20261110/predictions/val --split val
```

恢复使用train脚本的`--checkpoint <latest路径>`。禁止对已有run不带checkpoint重新启动；训练每1k评价全部772val，最终脚本恢复best再次导出完整1024×1024预测。soil/crop/weed为0/1/2，partial3→1/4→2，padding才255。官方devkit锁定SHA `0edc128ef7f67c8c6577554c7d1a2e382e2ea81f`；CLI不打印soil，由未修改的官方函数补全。正式指标优先官方评价，confusion只交叉核对。

官方权重来源：[下载](https://drive.google.com/uc?export=download&id=1CjueAvk52g5YIf30yI7ZFDIbsfqYq5tt)，343239900bytes，SHA256 `3e3ca279c9a4929bfb66c10c3a21f92a702899bb556cda9e528ee57e72a2cc5e`。公开文件为裸state_dict，项目只为官方loader补格式兼容，保持strict=True。

最终预测的CPU误差分析及轻量报告可重新生成：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONPATH=. \
  .venv/bin/python scripts/agriculture/analyze_errors.py
.venv/bin/python scripts/agriculture/summarize_run.py
```

完整下游权重的文件名、字节数和SHA见[checkpoint清单](reports/s1_checkpoint_manifest.json)，最终恢复来源见[评价追溯记录](reports/s1_evaluation_provenance.json)。正式训练和周期验证3.38小时、最终恢复评价37.22分钟；GPU共享及I/O等待的实际记录保留在报告中。

轻量数据manifest与报告提交Git；原始数据不复制，runs/预训练与下游权重/预测PNG全部忽略。报告包含完整环境、来源、split、协议与实际GPU成本。本轮仅完成这一模型和数据集。

---

以下为原作者README（保留引用与使用说明）。

# DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2406.11838-b31b1b.svg)](https://arxiv.org/abs/2511.17354)&nbsp;


<p align="center">
  <img src="demo/motivation.png" width="720">
</p>

This is a official PyTorch implementation of the paper [DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2511.17354):

```
@article{he2025dseqjepa,
      title={DSeq-JEPA: Discriminative Sequential Joint-Embedding Predictive Architecture}, 
      author={Xiangteng He and Shunsuke Sakai and Shivam Chandhok and Sara Beery and Kun Yuan and Nicolas Padoy and Tatsuhito Hasegawa and Leonid Sigal},
      year={2025},
      url={https://arxiv.org/abs/2511.17354}, 
}
```

## Preparation

### Installation
Download the code:
```
git clone https://github.com/SkyShunsuke/DSeq-JEPA.git
cd DSeq-JEPA
```


Create virtual environment, then you can install the required packages with:
```
pip install -r requirements.txt
```

### Dataset
Download [ImageNet](https://www.image-net.org/download.php) dataset for pre-training and place it in the `data` directory. 

## Usage

### Pre-training
To pre-train the model, run the following command:

```
bash scripts/pretrain/pretrain_dseqjepa.sh
```
And you can change training parameters in `configs/dseqjepa/xxx.yaml` file.

### Downstream Tasks
All downstream evaluations start from a pre-training checkpoint: set
`model.pretrained_weights` in the corresponding config, and make sure
`model.use_masked_vit` / `model.use_class_token` match the pre-training config.

| Task | Datasets | Config | Script |
|---|---|---|---|
| Linear probing | ImageNet-1K | `configs/classification/probing.yaml` | `bash scripts/downstream/probing.sh` |
| Fine-grained visual categorization | iNaturalist21, CUB-200-2011, Stanford-Cars | `configs/classification/probing_{inat21,cub200,cars}.yaml` | `bash scripts/downstream/probing_fgvc.sh` |
| Detection / instance segmentation | MS-COCO 2017 | `configs/detection/coco_maskrcnn_vitb16.yaml` | `bash scripts/downstream/detection.sh` |
| Semantic segmentation | ADE20K | `configs/segmentation/ade20k_upernet_vitb16.yaml` | `bash scripts/downstream/segmentation.sh` |
| Low-level reasoning | Clevr/Count, Clevr/Dist | `configs/lowlevel/clevr_{count,dist}.yaml` | `bash scripts/downstream/clevr.sh` |
| kNN | ImageNet-1K | `configs/classification/knn.yaml` | `bash scripts/downstream/knn.sh` |

Datasets are downloaded by the helpers in `scripts/setup_data/` (`fgvc.sh`,
`inaturalist.sh`, `clervr.sh`, `coco.sh`, `ade20k.sh`) into `data/`.

#### Linear probing (ImageNet & FGVC)
A linear head with batch normalization is trained on the frozen encoder, on the
concatenated pooled features of the last four transformer blocks: SGD (momentum
0.9, Nesterov), weight decay 5e-4, 28 epochs, base learning rate 0.01 decayed at
epochs 8/16/24, batch size 32 per replica, random-resized 224 crops from
256-pixel images.  iNaturalist21, CUB-200-2011 and Stanford-Cars reuse exactly
the same protocol.

```
bash scripts/downstream/probing.sh       # ImageNet-1K
bash scripts/downstream/probing_fgvc.sh  # iNaturalist21 / CUB / Stanford-Cars
```

#### Detection and instance segmentation (MS-COCO)
Mask R-CNN with an FPN built from blocks 3, 5, 7 and 11 of the pre-trained
encoder, fine-tuned end-to-end for 25 epochs with AdamW (lr 1e-4, weight decay
0.1), linear warmup over the first epoch followed by cosine decay, global batch
size 16 and stochastic depth with a maximum drop-path rate of 0.1.  AP^box and
AP^mask are reported on `val2017`.

```
bash scripts/downstream/detection.sh
```

#### Semantic segmentation (ADE20K)
A UPerNet decoder over the feature maps of blocks 3, 5, 7 and 11, fine-tuned
with 512x512 crops using AdamW (lr 1e-4, weight decay 0.05) and a poly schedule,
and evaluated with sliding-window inference (512x512 window, stride 341x341).

```
bash scripts/downstream/segmentation.sh
```

#### Low-level reasoning (Clevr/Count, Clevr/Dist)
The target encoder is frozen and a task-specific linear classifier is trained on
the final-layer `[CLS]` token of 224x224 inputs -- 8-way for Clevr/Count and
6-way for Clevr/Dist -- with AdamW (lr 1e-3, weight decay 0.05), batch size 256
and 100 epochs with a cosine schedule, on the VTAB-1k split (800 train / 200 val
/ 15,000 test).

```
bash scripts/downstream/clevr.sh   # set CONFIG_FILE to clevr_dist.yaml for Clevr/Dist
```

## Pre-trained Models
We provide pre-trained models for accelerating reproduction process. You can download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1a6HA5BKXohARqpP6BrRqgQhbWxyRFSJG?usp=sharing).

### Contact
For any questions, please open an issue on GitHub or contact Shunsuke Sakai (sshunsuke0102@gmail.com)



