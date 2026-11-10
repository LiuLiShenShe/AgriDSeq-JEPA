"""Produce lightweight reports exclusively from real run artifacts."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

run = Path('runs/phenobench_dseq_vitb16_s20261110')
reports = Path('reports')


def rows(path):
    if not path.exists(): return []
    with path.open() as f: return list(csv.DictReader(f))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


train, val, telemetry = rows(run/'train_log.csv'), rows(run/'val_log.csv'), rows(run/'gpu_telemetry.csv')
last = train[-1] if train else {}
updates = int(last.get('optimizer_updates',0))
status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'status':'TRAINING_IN_PROGRESS'}
metrics_file=run/'metrics'/'phenobench_official.json'
metrics=json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
analysis=json.loads((run/'analysis'/'summary.json').read_text()) if (run/'analysis'/'summary.json').exists() else {}
eval_provenance=json.loads((run/'metrics'/'evaluation_provenance.json').read_text()) if (run/'metrics'/'evaluation_provenance.json').exists() else {}
best = max(val, key=lambda r:float(r['mIoU'])) if val else {}
for source, target in [(run/'val_log.csv',reports/'s1_val_log.csv'),
                       (run/'gradient_evidence.json',reports/'s1_gradient_evidence.json'),
                       (run/'provenance.json',reports/'s1_run_provenance.json'),
                       (run/'metrics'/'phenobench_official.json',reports/'s1_official_metrics.json'),
                       (run/'metrics'/'confusion_crosscheck.json',reports/'s1_confusion_crosscheck.json'),
                       (run/'metrics'/'evaluation_provenance.json',reports/'s1_evaluation_provenance.json'),
                       (run/'analysis'/'summary.json',reports/'s1_error_summary.json'),
                       (run/'analysis'/'per_image_errors.csv',reports/'s1_per_image_errors.csv')]:
    if source.exists(): shutil.copyfile(source,target)
if train:
    with (reports/'s1_train_log_sample.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(train[0]),lineterminator='\n');writer.writeheader()
        writer.writerows([r for r in train if int(r['optimizer_updates'])%100==0 or r is last])
checkpoints=[]
for path in sorted((run/'checkpoints').glob('*.pth')):
    checkpoints.append(dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path)))
(reports/'s1_checkpoint_manifest.json').write_text(json.dumps(checkpoints,indent=2))
elapsed=float(status.get('elapsed_seconds',last.get('elapsed_seconds',0)))
energy=0.; sensor_seconds=0.
if len(telemetry)>1:
    times=np.array([float(r['unix_seconds']) for r in telemetry]);powers=np.array([float(r['power_watts']) for r in telemetry])
    energy=float(np.trapz(powers,times)/3600000);sensor_seconds=float(times[-1]-times[0])
peak_alloc=max((float(r['peak_allocated_gib']) for r in train),default=0)
peak_reserved=max((float(r['peak_reserved_gib']) for r in train),default=0)
peak_nvml=max((float(r['memory_used_mib']) for r in telemetry),default=0)
peak_temperature=max((float(r['temperature_c']) for r in telemetry),default=0)
mean_update=float(np.mean([float(r['update_seconds']) for r in train])) if train else 0.
(reports/'s1_gpu_compute.md').write_text(f'''# S1 GPU成本（实测）

状态 {status['status']}，正式optimizer updates {updates}/10000，microbatch2×accum8=有效16。
物理GPU1 RTX A6000，UUID GPU-6836f890-98b2-5891-07ef-675c25199c65；BF16+作者activation checkpoint。
训练和周期val累计墙钟 {elapsed:.2f}s ({elapsed/3600:.4f} 单GPU任务运行小时，非独占GPU小时)；平均每optimizer update {mean_update:.4f}s。
训练allocator峰值 {peak_alloc:.4f}GiB，reserved峰值 {peak_reserved:.4f}GiB；NVML监控峰值 {peak_nvml:.1f}MiB。
10s采样遥测区间 {sensor_seconds:.2f}s，GPU板卡功率梯形积分 {energy:.5f}kWh。初始GPU1空闲，但训练期间另有用户任务启动，设备显存/功率含其他进程，不能全部归因于本实验，也不含整机CPU/磁盘能耗或代表电费账单。PyTorch allocator峰值属于本任务。遥测开始于正式训练约170秒后；不外推缺失区间功耗。实测设备温度峰值{peak_temperature:.1f}°C。遥测原始CSV在Git忽略的run中。
烟雾工程检查另行记录reports/smoke_verification.json和smoke_resume_verification.json，不计正式10kupdates。最终best重新评价计入持续遥测区间，训练status耗时终点为训练完成。
最终best恢复、全772张预测和官方评价另耗时 {float(eval_provenance.get('elapsed_seconds',0)):.2f}s；评价checkpoint和执行代码SHA见s1_evaluation_provenance.json（生成后）。
训练及最终评价合计 {(elapsed+float(eval_provenance.get('elapsed_seconds',0))):.2f}s ({(elapsed+float(eval_provenance.get('elapsed_seconds',0)))/3600:.4f}任务运行小时)。控制会话与存储等待的实际记录见s1_runtime_events.md，未追加或重启训练。

来源：实际train_log.csv、status.json、gpu_telemetry.csv；没有估算未执行的更新。
''')
table='| optimizer updates | Soil IoU | Crop IoU | Weed IoU | mIoU |\n|---:|---:|---:|---:|---:|\n'
for row in val:
    table+='| '+' | '.join(row[k] for k in ('optimizer_updates','soil','crop','weed','mIoU'))+' |\n'
metric_text=', '.join(f'{k}={v:.2f}%' for k,v in metrics.items()) if metrics else '尚未完成最终best-val重新导出和官方评价'
comparison = metrics or {k:float(best[k]) for k in ('soil','crop','weed','mIoU') if k in best}
baseline_table=''
delta_text='尚无可比较的完整val结果'
baseline_path=reports/'published_baselines_val.json'
if baseline_path.exists() and comparison:
    baselines=json.loads(baseline_path.read_text())['models']
    baseline_table='| 方法（同772张val，原始1024坐标） | Soil IoU | Crop IoU | Weed IoU | mIoU |\n|---|---:|---:|---:|---:|\n'
    for name,values in [('DSeq-JEPA当前best-val',comparison)]+[(n,b['official_metrics_percent']) for n,b in baselines.items()]:
        baseline_table+='| '+name+' | '+' | '.join(f'{values[k]:.2f}%' for k in ('soil','crop','weed','mIoU'))+' |\n'
    erf=baselines['ERFNet']['official_metrics_percent']
    delta_text=f'相对ERFNet公开val预测重评：mIoU {comparison["mIoU"]-erf["mIoU"]:+.2f}个百分点，Weed IoU {comparison["weed"]-erf["weed"]:+.2f}个百分点'
(reports/'s1_phenobench_results.md').write_text(f'''# S1 PhenoBench正式验证结果

真实状态：{status['status']}；已完成 {updates}/10000 optimizer updates。以下周期指标只来自完整772val官方devkit。

{table}

最终恢复best-val并重新导出全部772张1024×1024PNG的官方结果：{metric_text}。
best选择步数 {best.get('optimizer_updates','未产生')}，按官方val mIoU选择，不是独立test。
原始预测路径 `{run}/predictions/val/semantics/*.png`；权重与PNG不提交Git。官方CLI及官方函数返回JSON保存在run/metrics。
交叉confusion核对结果见reports/s1_confusion_crosscheck.json（若已生成）；正式指标以官方devkit为准。

农业辅助指标：Weed precision {analysis.get('weed_precision','未计算')}%，recall {analysis.get('weed_recall','未计算')}%；面积<=100px且至少50%像素识别的4连接weed组件召回 {analysis.get('small_component_recall_100px','未计算')}%。这些是辅助度量，不替代官方IoU。
具体错误成员与数值见reports/s1_error_summary.json，局部可视化在run/analysis，主观失败类型应以直接看图描述为准。
直接查看最终真实RGB/GT/预测后的农业案例分析见[s1_agricultural_error_analysis.md](s1_agricultural_error_analysis.md)。
官方baseline同val公开预测参照见reports/published_baselines_val.md（若存在）；论文TableIII为test，不能声称val相对test提升。

{baseline_table}

{delta_text}。此表是同拆分参照：本方法经过val选择，baseline的backbone、训练crop、损失权重、优化协议和预算不同，因此差值不能单独归因于JEPA预训练。详细公开配置及下载/评价证据见published_baselines_val.md。

来源：[固定官方评价实现](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/evaluate_semantics.py)，本项目实际val日志与预测。
''')
checkpoint_text='\n'.join(f'- {c["file"]}: {c["bytes"]} bytes, SHA256 `{c["sha256"]}`' for c in checkpoints) or '尚未保存正式checkpoint。'
(reports/'s1_final_report.md').write_text(f'''# S0–S1 {'完成报告' if metrics and status['status']=='TRAINING_COMPLETE' else '进度报告'}

当前状态 **{status['status']}**；正式完成 **{updates}/10000 optimizer updates**。只有TRAINING_COMPLETE且最终772官方评价已生成才满足完整训练验收。

1. 独立路径 `/fj/AgriDSeq-JEPA`；新远端 https://github.com/LiuLiShenShe/AgriDSeq-JEPA ，私有，分支agri-phenobench-transfer；上游SHA123082374389f78e4516157187681f9d5edb4310。启动代码commit见s1_run_provenance.json，最终报告commit通过git log查询，避免在自身内容写自引用SHA。
2. 真实作者dseqjepa_vit_base_latest.pth，343239900bytes，SHA2563e3ca279c9a4929bfb66c10c3a21f92a702899bb556cda9e528ee57e72a2cc5e；[下载来源](https://drive.google.com/uc?export=download&id=1CjueAvk52g5YIf30yI7ZFDIbsfqYq5tt)。严格加载+512前向通过，详细s0_checkpoint_verification.md。
3. train1407/val772，原1024×1024 RGB8/灰度16-bit mask；3→1/4→2，soil0有效，padding255；zip和固定逐文件manifest真实hash验证，s0_phenobench_data.md。
4. 四层[3,5,7,11]各[B,768,32,32]进入作者UPerNet/aux；freeze_backbone=False，全部活跃参数梯度存在，backbone/main/aux实测非零权重变化。保留官方固定pos_embed和unused cls_token冻结。
5. microbatch2×accum8×1=16，BF16/AdamW/poly/warmup；峰值allocated{peak_alloc:.4f}GiB，墙钟{elapsed:.2f}s；成本详情s1_gpu_compute.md。
6. 真实正式updates{updates}；下列checkpoint为实际存在文件，不存在者不会列出。所有权重保留在Git忽略run中。

{checkpoint_text}

7. 最终772val官方指标：{metric_text}。best-val选择步数{best.get('optimizer_updates','未产生')}。
8. 主要失败为小杂草漏分成soil、相邻/交叠叶片处weed被并入crop、局部叶缘错误；数值及真实成员见s1_agricultural_error_analysis.md与s1_error_summary.json；{delta_text}。baseline协议不同，不能将差值单独归因于JEPA，也不能以本轮val对公开test宣称胜出。
9. 当前协议验证后优先MAE ViT-B/16公平对照：先固定同backbone容量、head、crop、updates和effectivebatch，判断JEPA初始化的贡献；外部CropAndWeed验证留后续阶段。该建议不意味着本轮已启动任何新对照。

许可证/环境/复用：s0_environment_and_upstream.md；固定协议：s1_training_protocol.md；README有真实训练、恢复、评价命令。
博查搜索HTTP403配额不足已明确记录，证据来自实时官方源码、页面与本地执行。未添加JEPA预训练、旧语料处理、第二模型/数据集训练。
''')
print(json.dumps(dict(status=status['status'],updates=updates,final_metrics=metrics,best=best,energy_kwh=energy),indent=2))
