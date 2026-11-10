# S0–S1 最终/实时报告

当前状态 **TRAINING_IN_PROGRESS**；正式完成 **6031/10000 optimizer updates**。只有TRAINING_COMPLETE且最终772官方评价已生成才满足完整训练验收。

1. 独立路径 `/fj/AgriDSeq-JEPA`；新远端 https://github.com/LiuLiShenShe/AgriDSeq-JEPA ，私有，分支agri-phenobench-transfer；上游SHA123082374389f78e4516157187681f9d5edb4310。启动代码commit见s1_run_provenance.json，最终报告commit通过git log查询，避免在自身内容写自引用SHA。
2. 真实作者dseqjepa_vit_base_latest.pth，343239900bytes，SHA2563e3ca279c9a4929bfb66c10c3a21f92a702899bb556cda9e528ee57e72a2cc5e；[下载来源](https://drive.google.com/uc?export=download&id=1CjueAvk52g5YIf30yI7ZFDIbsfqYq5tt)。严格加载+512前向通过，详细s0_checkpoint_verification.md。
3. train1407/val772，原1024×1024 RGB8/灰度16-bit mask；3→1/4→2，soil0有效，padding255；zip和固定逐文件manifest真实hash验证，s0_phenobench_data.md。
4. 四层[3,5,7,11]各[B,768,32,32]进入作者UPerNet/aux；freeze_backbone=False，全部活跃参数梯度存在，backbone/main/aux实测非零权重变化。保留官方固定pos_embed和unused cls_token冻结。
5. microbatch2×accum8×1=16，BF16/AdamW/poly/warmup；峰值allocated3.5138GiB，墙钟7033.34s；成本详情s1_gpu_compute.md。
6. 真实正式updates6031；下列checkpoint为实际存在文件，不存在者不会列出。所有权重保留在Git忽略run中。

- phenobench_upernet_best.pth: 1522024148 bytes, SHA256 `5b09b16a0713e5bd15770e7814b0c6008a3d47b59b66c9017ae7ffdccc21f8f2`
- phenobench_upernet_latest.pth: 1522025834 bytes, SHA256 `d2f657a7ebdb1211e5e6b45d9271619895957f8ecf19c70e12ffd409d590a64d`
- phenobench_upernet_step01000.pth: 1522025834 bytes, SHA256 `38367a870a89e4fc803a94f4cc1d88a142460d8b6fd01ddc962e639e629d9162`
- phenobench_upernet_step05000.pth: 1522025834 bytes, SHA256 `0f45adda9af0a83174410c58ce61b5c840c28ebc81302b27e54ea0026e881d55`

7. 最终772val官方指标：尚未完成最终best-val重新导出和官方评价。best-val选择步数6000。
8. 农业失败数值见s1_error_summary.json与s1_phenobench_results.md；相对ERFNet公开val预测重评：mIoU +1.28个百分点，Weed IoU +3.29个百分点。baseline协议不同，不能将差值单独归因于JEPA，也不能以本轮val对公开test宣称胜出。
9. 当前协议验证后优先MAE ViT-B/16公平对照：先固定同backbone容量、head、crop、updates和effectivebatch，判断JEPA初始化的贡献；外部CropAndWeed验证留后续阶段。该建议不意味着本轮已启动任何新对照。

许可证/环境/复用：s0_environment_and_upstream.md；固定协议：s1_training_protocol.md；README有真实训练、恢复、评价命令。
博查搜索HTTP403配额不足已明确记录，证据来自实时官方源码、页面与本地执行。未添加JEPA预训练、旧语料处理、第二模型/数据集训练。
