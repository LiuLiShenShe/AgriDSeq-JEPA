# S1 PhenoBench正式验证结果

真实状态：TRAINING_IN_PROGRESS；已完成 4420/10000 optimizer updates。以下周期指标只来自完整772val官方devkit。

| optimizer updates | Soil IoU | Crop IoU | Weed IoU | mIoU |
|---:|---:|---:|---:|---:|
| 1000 | 99.3 | 93.53 | 45.35 | 79.39 |
| 2000 | 99.38 | 94.76 | 58.98 | 84.37 |
| 3000 | 99.43 | 95.45 | 69.57 | 88.15 |
| 4000 | 99.43 | 95.49 | 70.17 | 88.36 |


最终恢复best-val并重新导出全部772张1024×1024PNG的官方结果：尚未完成最终best-val重新导出和官方评价。
best选择步数 4000，按官方val mIoU选择，不是独立test。
原始预测路径 `runs/phenobench_dseq_vitb16_s20261110/predictions/val/semantics/*.png`；权重与PNG不提交Git。官方CLI及官方函数返回JSON保存在run/metrics。
交叉confusion核对结果见reports/s1_confusion_crosscheck.json（若已生成）；正式指标以官方devkit为准。

农业辅助指标：Weed precision 未计算%，recall 未计算%；面积<=100px且至少50%像素识别的4连接weed组件召回 未计算%。这些是辅助度量，不替代官方IoU。
具体错误成员与数值见reports/s1_error_summary.json，局部可视化在run/analysis，主观失败类型应以直接看图描述为准。
官方baseline同val公开预测参照见reports/published_baselines_val.md（若存在）；论文TableIII为test，不能声称val相对test提升。

| 方法（同772张val，原始1024坐标） | Soil IoU | Crop IoU | Weed IoU | mIoU |
|---|---:|---:|---:|---:|
| DSeq-JEPA当前best-val | 99.43% | 95.49% | 70.17% | 88.36% |
| ERFNet | 99.33% | 95.34% | 68.65% | 87.77% |
| DeepLabV3+ | 99.30% | 95.13% | 68.00% | 87.47% |


相对ERFNet公开val预测重评：mIoU +0.59个百分点，Weed IoU +1.52个百分点。此表是同拆分参照：本方法经过val选择，baseline的backbone、训练crop、损失权重、优化协议和预算不同，因此差值不能单独归因于JEPA预训练。详细公开配置及下载/评价证据见published_baselines_val.md。

来源：[固定官方评价实现](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/evaluate_semantics.py)，本项目实际val日志与预测。
