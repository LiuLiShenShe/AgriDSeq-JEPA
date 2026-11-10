# S1：固定正式训练协议

唯一轨迹：官方DSeq-JEPA ViT-B/16 + 官方UPerNet，PhenoBench v1.1.0，train1407/val772，seed20261110，10000 optimizer updates预算；不以ADE20K160k论文配置复刻命名。
使用官方层3/5/7/11，全活跃backbone/neck/main/aux参数微调，freeze_backbone=False。官方aux CE权重0.4，主CE1.0，不加类别权重；soil0有效、padding255忽略。
输入512×512：base long/short=(1024,512)，ratio0.5–2，作者category-balanced crop/hflip/photometric/ImageNet normalization；验证不resize，原1024×1024滑窗512/stride341，融合float32 logits后argmax，PNG ID0/1/2。

物理GPU1单A6000；microbatch2×accum8×world1=effective16。每microbatch的(main+0.4aux)loss除以8，累计8次backward后一次AdamW.step，再一次poly scheduler.step。LR时间单位为optimizer updates；auto_lr_scaling=False，baseLR1e-4，WD0.05，bias/norm不衰减、backboneLRscale1。warmup1500updates、ratio1e-6、power1.0/minLR0。保留有效batch16并不代表重现8卡SyncBN统计：单卡BN每microbatch2，累积不改变BN样本数。
BF16显式dtype，模型/梯度/优化器FP32；BF16不使用FP16 GradScaler。作者ViT activation checkpoint use_reentrant=False启用，window_size0全局attention，drop_path0.1。设置OMP4、PyTorch CPU threads4、num_workers8。

每1000updates全772val用官方CLI评价，再调用同一未修改官方evaluate_semantics返回soil/crop/weed/mIoU（已百分比、round2）。外层预检文件名、尺寸、ID完整一致，防止官方zip静默截断。best按官方val mIoU选择，不称独立test性能。
保存latest、step01000/05000/10000、best；state含完整model(encoder/neck/main/aux)、optimizer、poly、step/已消费microbatch、best、config和Python/NumPy/Torch/CUDA随机状态；原子替换。按sample+epoch确定augmentation，独立DataLoader generator不消耗模型RNG；resume从sampler epoch/batch位置恢复，不新建第二正式轨迹。

日志逐optimizer update记录loss/LR/microbatch/accum/effective/秒数/显存，run provenance含config、官方权重、固定manifest、zip SHA与源代码commit。run/权重/原图/预测不提交Git。轻量审计证据提交reports。

训练：`bash scripts/agriculture/train_phenobench.sh`
恢复：`bash scripts/agriculture/train_phenobench.sh --checkpoint runs/phenobench_dseq_vitb16_s20261110/checkpoints/phenobench_upernet_latest.pth`
最终best全量评价：`bash scripts/agriculture/eval_phenobench.sh`

来源：[官方segmentation.py](https://github.com/SkyShunsuke/DSeq-JEPA/blob/123082374389f78e4516157187681f9d5edb4310/src/eval/segmentation.py)、[PyTorch梯度累积文档](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/amp_examples.rst)、实际本项目配置与执行日志。
