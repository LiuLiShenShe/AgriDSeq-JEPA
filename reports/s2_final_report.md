# S2 最终验收报告

MAE唯一正式轨迹已真实完成10000 optimizer updates和80000 microbatches。best-val在第10000步；完整772张官方val重新评价得到mIoU89.51%。DSeq为90.02%，本次额外迁移收益为0.51个百分点，Weed IoU收益为1.44个百分点。当前暂保留DSeq-JEPA为主baseline，优先用另一个匹配的有监督训练seed复验两种初始化。

分支为 `agri-s2-mae-control`，从冻结S1 `20ffaadcfa0f1172303c2ccbc09a456458f80e3d` 创建。正式训练和最终评价代码HEAD均为 `6a537745067e4b4b2d9041425a75206525d0288c`，见 [运行来源](s2_run_provenance.json)。最终结果commit由该分支 `git rev-parse HEAD` / GitHub提交记录识别，避免在提交文件内写入自引用SHA。S1本地和实时远端分支、32项冻结文件及五个下游checkpoint重新校验，证据见 [成果完整性审计](s2_artifact_integrity.json)。

采用 [Meta官方MAE仓库](https://github.com/facebookresearch/mae/tree/efb2a8062c206524e35e47d04501ed4f544c0ae8)，上游SHA `efb2a8062c206524e35e47d04501ed4f544c0ae8`；[原始自监督预训练文件](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)实际343,249,461 bytes，SHA-256：

`aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d`

实际官方发布文件只有150个encoder张量，已去除decoder。内存映射150/150后严格加载，逐张量验证与源相等，包括12个blocks、patch、QKV/projection、MLP和norm。位置编码从[1,197,768]取patch部分为[1,196,768]；CLS不进入原S1 dense前向，未改变计算结构。完整键名、形状、转换和SHA见 [权重映射](s2_mae_weight_mapping.json) 与 [来源报告](s2_mae_checkpoint_source.md)。

两个模型都为126,907,910总参数、126,756,614可训练参数，四层输出[ B,768,32,32 ]、3类logits[ B,3,512,512 ]，neck、UPerNet和aux完全复用。唯一冻结参数pos_embed与cls_token符合S1。Neck8、main head74、aux8，共90项初始state与S1固定启动代码/config/seed重建张量逐项相同，实际S2初始化SHA再次匹配；S1没有历史step-zero独立存档，不能混称为直接检查原始S1初始存档。详见 [模型等价报告](s2_model_equivalence.md)。

真实MAE/PhenoBench smoke独立完成20步，先10步保存完整状态，再恢复至20步，共160 microbatches；backbone、neck、main及aux均有梯度和参数更新，原图滑窗及官方devkit读取通过。针对性MAE测试6项和已有农业测试合计12项通过。正式日志连续1—10000，最终199组optimizer参数状态全部step10000，poly scheduler为10000，RNG和完整状态存在，12个blocks及patch/final norm/head实际更新。详见 [最终训练状态](s2_final_verification.json)、[Smoke证据](s2_smoke_verification.json) 和 [正式梯度证据](s2_training_gradient_evidence.json)。

每1000步完整评价772张，MAE mIoU依次为77.87、85.38、87.43、86.71、88.06、88.25、88.82、88.83、89.03、89.51%。保存step1000、5000、10000、latest、best五个完整checkpoint。best SHA-256为：

`910306bfcf3717f0369c1552ab1a9d0154d86568a360319a0cde9490a11e3b07`

最佳权重保留在 `runs/phenobench_mae_vitb16_s20261110/checkpoints/phenobench_upernet_best.pth`；[五个checkpoint清单](s2_checkpoint_manifest.json)、[完整val曲线](s2_val_log.csv) 和 [训练结果图](s2_mae_training_results.md) 已归档。正式训练未中断或启动替代轨迹；smoke不计入10000次正式更新。

最终恢复best生成772/772张1024×1024 uint8预测，成员与GT完全一致、类别仅0/1/2。使用 `phenobench-eval --task semantics`，soil/crop/weed都参与mIoU；官方CLI显示crop/weed/mIoU，soil由同一官方评价函数记录到JSON，混淆矩阵另行交叉核对。见 [官方原始输出](s2_official_cli.txt)、[官方指标](s2_official_metrics.json)、[评价来源](s2_evaluation_provenance.json)。这是val选择结果，不是独立test。

| 指标 (%) | DSeq-JEPA | MAE | DSeq − MAE (百分点) |
|---|---:|---:|---:|
| Soil IoU | 99.47 | 99.47 | 0.00 |
| Crop IoU | 95.97 | 95.89 | +0.08 |
| Weed IoU | 74.62 | 73.18 | +1.44 |
| mIoU | 90.02 | 89.51 | +0.51 |

| 杂草度量 | DSeq-JEPA | MAE | DSeq − MAE |
|---|---:|---:|---:|
| 小组件召回 | 494/1102 (44.83%) | 465/1102 (42.20%) | +29 / +2.63 pp |
| Weed precision (%) | 88.4015 | 89.0998 | −0.6983 pp |
| Weed recall (%) | 82.7191 | 80.3799 | +2.3393 pp |
| Weed→Soil (pixels) | 547,960 | 587,001 | −39,041 |
| Weed→Crop (pixels) | 191,094 | 252,096 | −61,002 |

DSeq少漏分100,043个weed像素，MAE少误报43,601个weed像素。小组件沿用GT四连接≤100px、预测覆盖≥50%定义，不能称真实独立植株。两者共同漏分657,306个weed像素，逐图weed IoU相关0.9275；共同困难显著存在。复用S1病例后，05-26_00242由MAE更好（65.47对54.31），两个06-05密集叶片病例由DSeq更好，微小weed和细叶缘仍有共同失败。真实图、局部框及逐图记录见 [错误分析](s2_error_analysis.md)。

DSeq首次观测达到88%/89% mIoU为3000/6000步，MAE为5000/9000步；MAE在2000步短暂领先，之后DSeq更高。每1000步评价不能确定精确阈值穿越时刻，相关val节点也不能当作独立复验。

| 成本 | DSeq-JEPA | MAE |
|---|---:|---:|
| 训练+10次周期val (h) | 3.3767 | 4.5090 |
| 平均update (s) | 0.8683 | 1.2500 |
| 训练allocator峰值 (GiB) | 3.5138 | 3.5214 |
| 训练reserved峰值 (GiB) | 3.7656 | 3.7930 |
| 最后完整best评价 (s) | 2233.40 | 406.73 |

MAE训练及最终评价合计4.6220任务运行小时。模型显存接近，MAE审计额外保留9MiB neck抽样clone。GPU1上的其他任务在约4000步val时启动，MAE前4000步0.8895s/update、后6000步1.4902s/update；因此墙钟不能归因为初始化计算效率。遥测板卡能量1.17744kWh、设备峰值47520.9MiB包含共享进程，不能全部算作MAE或换算成整机电费。详见 [计算成本](s2_compute_cost.md) 与 [实时环境事件](s2_runtime_events.md)。

科研决策：本次实测支持DSeq checkpoint在统一下游协议中的mIoU、weed和小组件召回优势，收益有限但方向一致，暂保留DSeq为主baseline，保留MAE正式参照。只有一条DSeq和一条MAE轨迹，不能宣称跨下游seed或预训练seed稳定、统计显著的优势。原始预训练方法、数据曝光和预算未完全控制，归因层级是“预训练checkpoint迁移性能”，不能推导JEPA目标的因果优势。

下一步优先对两种初始化追加相同的另一个有监督seed，复核0.51 pp与召回/precision取舍。保持方向后再进入CropAndWeed外部验证；若方向反转或落入波动，则重新评估或并列主baseline。小杂草单一适配应在基线复验后开展。本轮没有扩大矩阵、重训DSeq或运行其他农业数据集。

Git仅提交必要代码、配置、指标、报告及选定真实派生对比图；权重、全部预测PNG、原始图像和runs均保留本地。结论来源为冻结S1与本次真实训练日志、官方评价和原始预测；上游实现见 [MAE固定源码](https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py) 和 [PhenoBench官方评价源码](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/evaluate_semantics.py)。
