# S2 冻结 MAE 训练协议

配置逐字段归一后与S1相等。仅更改model.pretrained_type=mae、官方权重路径/SHA、logging.log_dir。
PhenoBench v1.1.0 train1407/val772；固定manifest SHA `6ce0232f323c1bf055f4b6f588872ca5605ae964bb26a541109780e63d307f3b`；zip SHA `a310c1c1dcfec4ca4897097fd38c02af39ea24112f658fd9408d5deb3c7c8bfc`。
512 crop、base_size[1024,512]、ratio[0.5,2]、category-balanced crop/hflip/photometric/ImageNet normalize全部沿用；val原1024、512滑窗/stride341/float32 logits融合。
同一ViT-B/16 MaskedVisionTransformer + ViTDenseBackbone + neck + UPerNet + aux；全活跃参数微调。主CE1.0、auxCE0.4、soil0有效、padding255忽略。
AdamW lr1e-4/WD0.05，bias/norm无WD，backbone倍率1；poly power1/minLR0，warmup1500/ratio1e-6，10000 optimizer updates。
物理GPU1 RTX A6000，microbatch2×accum8×world1=effective16，单卡BN实际统计batch2；BF16、FP32优化器，无FP16 GradScaler；activation checkpoint use_reentrant=False，drop_path0.1。
seed20261110，OMP/CPU threads4、workers8；模型创建顺序复用S1，head初始张量重建校验证据见s2_model_equivalence.md。
每1000步772张官方val，按官方round2百分比mIoU严格大于当前best时选择best；不称独立test。
保存step01000/05000/10000、latest、best；完整model/optimizer/poly/RNG/consumed_batches/elapsed状态原子存储，中断按原轨迹恢复。
Smoke独立runs/s2_mae_smoke，先10步保存、再恢复到20步，总计160 microbatches；不计正式更新。

正式训练：`bash scripts/agriculture/train_phenobench.sh --config configs/agriculture/phenobench_mae_vitb16_upernet.yaml`。
恢复追加：`--checkpoint runs/phenobench_mae_vitb16_s20261110/checkpoints/phenobench_upernet_latest.pth`。
最终best评价：`bash scripts/agriculture/eval_phenobench.sh --config configs/agriculture/phenobench_mae_vitb16_upernet.yaml --checkpoint runs/phenobench_mae_vitb16_s20261110/checkpoints/phenobench_upernet_best.pth`（须显式传入MAE checkpoint；shell默认仍为S1）。
不修改S1配置、日志、metrics/checkpoints；未启动DSeq重训或其它数据集/模型矩阵。
来源：本地S1固定配置/脚本、S2逐字段审计；[PyTorch2.5.1 AMP文档](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/amp_examples.rst)。
