"""Summarize only a completed MAE trajectory and its frozen S1 paired control."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage

from src.dataset.phenobench import convert_partial_semantics
from src.utils.mae_checkpoint_adapter import tensor_sha256

RUN = Path('runs/phenobench_mae_vitb16_s20261110')
DSEQ = Path('runs/phenobench_dseq_vitb16_s20261110')
REPORT = Path('reports')
ROOT = Path('/data/datasets/PhenoBench/PhenoBench-v110')
FIGURES = REPORT/'s2_figures'


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


status = read_json(RUN/'status.json')
assert status['status'] == 'TRAINING_COMPLETE' and status['optimizer_updates'] == 10000
train, val = read_rows(RUN/'train_log.csv'), read_rows(RUN/'val_log.csv')
assert [int(r['optimizer_updates']) for r in train] == list(range(1,10001))
assert [int(r['consumed_microbatches']) for r in train] == list(range(8,80001,8))
assert [int(r['optimizer_updates']) for r in val] == list(range(1000,10001,1000))
assert all(int(r['effective_batch']) == 16 for r in train)
for path, expected in read_json(Path('runs/s2_source/s1_frozen_hashes.json')).items():
    assert digest(Path(path)) == expected, f'Frozen S1 artifact changed: {path}'
init = read_json(RUN/'initialization.json')
reconstruction = read_json(REPORT/'s2_initialization_evidence.json')
assert init['downstream_tensors'] == {r['name']:r['mae_sha256'] for r in reconstruction['downstream_tensors']}
state = torch.load(RUN/'checkpoints/phenobench_upernet_latest.pth',map_location='cpu',weights_only=False)
assert state['step'] == 10000 and state['consumed_batches'] == 80000
assert state['lr_scheduler']['_step'] == state['lr_scheduler']['total_steps'] == 10000
assert all(g['lr'] == 0. for g in state['opt']['param_groups'])
optimizer_steps = sorted({int(s['step']) for s in state['opt']['state'].values()})
assert optimizer_steps == [10000]
mapping = {r['target']:r['sha256'] for r in read_json(REPORT/'s2_mae_weight_mapping.json')['entries']}
for key in ('cls_token','pos_embed'):
    assert tensor_sha256(state['model']['backbone.vit.'+key]) == mapping[key]
block_changes = {}
for block in range(12):
    block_changes[block] = {}
    for suffix in ('attn.qkv.weight','attn.proj.weight','mlp.fc1.weight','mlp.fc2.weight','norm1.weight','norm2.weight'):
        key=f'blocks.{block}.{suffix}'
        changed=tensor_sha256(state['model']['backbone.vit.'+key]) != mapping[key]
        assert changed, key
        block_changes[block][suffix] = changed
for key in ('patch_embed.proj.weight','norm.weight'):
    assert tensor_sha256(state['model']['backbone.vit.'+key]) != mapping[key]
for key in ('neck.stages.0.0.weight','decode_head.cls_seg.weight','auxiliary_head.cls_seg.weight'):
    assert tensor_sha256(state['model'][key]) != init['downstream_tensors'][key]
assert state['config'] == yaml.safe_load((RUN/'config.yaml').read_text())
assert state['config'] == yaml.safe_load(Path('configs/agriculture/phenobench_mae_vitb16_upernet.yaml').read_text())
final_state=dict(step=state['step'],consumed_batches=state['consumed_batches'],optimizer_steps=optimizer_steps,
                 optimizer_parameter_state_count=len(state['opt']['state']),scheduler=state['lr_scheduler'],
                 rng_state_present=sorted(state['rng']),frozen_cls_and_pos_equal_MAE_source=True,
                 all_twelve_blocks_updated=block_changes,patch_final_norm_neck_main_aux_updated=True,
                 actual_head_initialization_equal_S1_reconstruction=True)
(REPORT/'s2_final_verification.json').write_text(json.dumps(final_state,indent=2))
del state
dm, mm = read_json(REPORT/'s1_official_metrics.json'), read_json(RUN/'metrics/phenobench_official.json')
de, me = read_json(REPORT/'s1_error_summary.json'), read_json(RUN/'analysis/summary.json')
assert me['totals']['small_components'] == de['totals']['small_components'] == 1102
evaluation = read_json(RUN/'metrics/evaluation_provenance.json')
best = max(val, key=lambda r:float(r['mIoU']))
assert evaluation['selected_optimizer_updates'] == int(best['optimizer_updates'])
assert evaluation['prediction_count'] == 772
assert mm == {k:float(best[k]) for k in ('soil','crop','weed','mIoU')}
comparison = [dict(metric=k, DSeq_JEPA=dm[k], MAE=mm[k], DSeq_minus_MAE_pp=round(dm[k]-mm[k],2))
              for k in ('soil','crop','weed','mIoU')]
write_rows(REPORT/'s2_comparison.csv', comparison)
for source, name in [('val_log.csv','val_log.csv'), ('provenance.json','run_provenance.json'),
                     ('metrics/phenobench_official.json','official_metrics.json'),
                     ('metrics/phenobench_official_cli.txt','official_cli.txt'),
                     ('metrics/confusion_crosscheck.json','confusion_crosscheck.json'),
                     ('metrics/evaluation_provenance.json','evaluation_provenance.json'),
                     ('analysis/summary.json','error_summary.json'),
                     ('analysis/per_image_errors.csv','per_image_errors.csv')]:
    shutil.copyfile(RUN/source, REPORT/('s2_'+name))
write_rows(REPORT/'s2_train_log_sample.csv', [r for r in train if int(r['optimizer_updates'])%100 == 0])
checkpoints = [dict(file=p.name, bytes=p.stat().st_size, sha256=digest(p))
               for p in sorted((RUN/'checkpoints').glob('*.pth'))]
assert len(checkpoints) == 5
(REPORT/'s2_checkpoint_manifest.json').write_text(json.dumps(checkpoints,indent=2))

# Pair every original pixel and retain the same S1-selected images.
drows, mrows = read_rows(REPORT/'s1_per_image_errors.csv'), read_rows(RUN/'analysis/per_image_errors.csv')
assert [r['filename'] for r in drows] == [r['filename'] for r in mrows]
paired = []
totals = dict(both_wrong=0, dseq_only_wrong=0, mae_only_wrong=0, disagree=0,
              both_weed_false_negative=0, dseq_only_weed_false_negative=0,
              mae_only_weed_false_negative=0)
palette = np.array([[95,65,40],[30,200,65],[220,45,180]],dtype=np.uint8)
FIGURES.mkdir(exist_ok=True)
selected = [r['filename'] for r in de['worst_cases']]
small_cases = sorted((r for r in drows if int(r['small_components'])>int(r['small_recalled'])),
                     key=lambda r:int(r['small_components'])-int(r['small_recalled']),reverse=True)[:2]
selected += [r['filename'] for r in small_cases if r['filename'] not in selected]
small_crops = []
for dr, mr in zip(drows, mrows):
    name = dr['filename']
    gt = convert_partial_semantics(np.array(Image.open(ROOT/'val/semantics'/name)))
    dp = np.array(Image.open(DSEQ/'predictions/val/semantics'/name))
    mp = np.array(Image.open(RUN/'predictions/val/semantics'/name))
    assert gt.shape == dp.shape == mp.shape == (1024,1024)
    dw, mw, weed = dp!=gt, mp!=gt, gt==2
    for row, prediction in ((dr,dp),(mr,mp)):
        actual = dict(weed_pixels=int(weed.sum()),weed_tp=int((weed & (prediction==2)).sum()),
                      weed_fp=int((~weed & (prediction==2)).sum()),weed_fn=int((weed & (prediction!=2)).sum()),
                      weed_as_soil=int((weed & (prediction==0)).sum()),
                      weed_as_crop=int((weed & (prediction==1)).sum()))
        assert all(value == int(row[key]) for key,value in actual.items()), f'Prediction/statistics mismatch: {name}'
    both = dw & mw
    counts = dict(both_wrong=int(both.sum()),dseq_only_wrong=int((dw & ~mw).sum()),
                  mae_only_wrong=int((~dw & mw).sum()),disagree=int((dp!=mp).sum()),
                  both_weed_false_negative=int((weed & (dp!=2) & (mp!=2)).sum()),
                  dseq_only_weed_false_negative=int((weed & (dp!=2) & (mp==2)).sum()),
                  mae_only_weed_false_negative=int((weed & (dp==2) & (mp!=2)).sum()))
    for k,v in counts.items(): totals[k] += v
    paired.append(dict(filename=name,dseq_weed_IoU=float(dr['weed_IoU']),mae_weed_IoU=float(mr['weed_IoU']),
                       dseq_weed_as_soil=int(dr['weed_as_soil']),mae_weed_as_soil=int(mr['weed_as_soil']),
                       dseq_weed_as_crop=int(dr['weed_as_crop']),mae_weed_as_crop=int(mr['weed_as_crop']),**counts))
    if name not in selected: continue
    rgb = Image.open(ROOT/'val/images'/name).convert('RGB')
    diff = (np.array(rgb)*0.35).astype(np.uint8)
    diff[both] = [210,45,45]
    diff[dw & ~mw] = [35,100,255]
    diff[~dw & mw] = [255,155,30]
    panels = [('RGB',rgb),('GT',Image.fromarray(palette[gt])),('DSeq',Image.fromarray(palette[dp])),
              ('MAE',Image.fromarray(palette[mp])),('red=both; blue=DSeq; orange=MAE errors',Image.fromarray(diff))]
    sheet = Image.new('RGB',(2560,556),'white'); draw=ImageDraw.Draw(sheet)
    draw.text((8,4),f'{name}: weed IoU DSeq {float(dr["weed_IoU"]):.2f}%, MAE {float(mr["weed_IoU"]):.2f}%',fill='black')
    for i,(label,im) in enumerate(panels):
        draw.text((i*512+8,24),label,fill='black')
        sheet.paste(im.resize((512,512),Image.Resampling.BILINEAR if i==0 else Image.Resampling.NEAREST),(i*512,42))
    sheet.save(FIGURES/(Path(name).stem+'_paired.jpg'),quality=92)
    if name in [r['filename'] for r in small_cases]:
        cc, n = ndimage.label(weed); sizes=np.bincount(cc.ravel())
        overlap=np.bincount(cc[dp==2],minlength=n+1)
        candidates=[i for i in range(1,n+1) if 1<sizes[i]<=100 and overlap[i]<0.5*sizes[i]]
        if not candidates: continue
        component=max(candidates,key=lambda i:sizes[i]); ys,xs=np.where(cc==component)
        cx,cy=int(xs.mean()),int(ys.mean()); x0=min(768,max(0,cx-128)); y0=min(768,max(0,cy-128))
        box=(x0,y0,x0+256,y0+256)
        crop=Image.new('RGB',(2560,556),'white'); draw=ImageDraw.Draw(crop)
        bounds=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        note=dict(filename=name,component_area_pixels=int(sizes[component]),crop_box=list(box),
                  component_bounds_original=bounds,
                  dseq_covered_pixels=int(overlap[component]),mae_covered_pixels=int((mp[cc==component]==2).sum()))
        small_crops.append(note)
        draw.text((8,4),f'{name}: yellow box=GT small area {note["component_area_pixels"]}; DSeq covered={note["dseq_covered_pixels"]}; MAE covered={note["mae_covered_pixels"]}',fill='black')
        for i,(label,im) in enumerate(panels):
            draw.text((i*512+8,24),label,fill='black')
            crop.paste(im.crop(box).resize((512,512),Image.Resampling.BILINEAR if i==0 else Image.Resampling.NEAREST),(i*512,42))
            # Identify the exact GT component without changing the stored masks.
            draw.rectangle((i*512+2*(bounds[0]-x0)-4,42+2*(bounds[1]-y0)-4,
                            i*512+2*(bounds[2]-x0)+4,42+2*(bounds[3]-y0)+4),
                           outline='yellow',width=2)
        crop.save(FIGURES/(Path(name).stem+'_small_crop.jpg'),quality=92)
write_rows(REPORT/'s2_paired_image_errors.csv',paired)
weed_confusions = {}
for key in ('weed_as_soil','weed_as_crop'):
    weed_confusions[key] = dict(DSeq=sum(int(r[key]) for r in drows),MAE=sum(int(r[key]) for r in mrows))
paired_summary = dict(totals=totals,weed_confusions=weed_confusions,selected_images=selected,small_crops=small_crops,
                     images_with_shared_wrong_pixels=sum(r['both_wrong']>0 for r in paired),
                     images_with_shared_weed_false_negatives=sum(r['both_weed_false_negative']>0 for r in paired),
                     weed_present_image_count=sum(int(r['weed_pixels'])>0 for r in drows),
                     weed_IoU_per_image_pearson=float(np.corrcoef(
                         [float(d['weed_IoU']) for d in drows if int(d['weed_pixels'])>0],
                         [float(m['weed_IoU']) for d,m in zip(drows,mrows) if int(d['weed_pixels'])>0])[0,1]))
(REPORT/'s2_paired_error_summary.json').write_text(json.dumps(paired_summary,indent=2))

# Scientific plots are exported artifacts, never generated substitute images.
dv=read_rows(REPORT/'s1_val_log.csv')
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,key,title in zip(axes.flat,('soil','crop','weed','mIoU'),('Soil IoU','Crop IoU','Weed IoU','Mean IoU')):
    for values,label,color in [(dv,'DSeq-JEPA','#2373a8'),(val,'MAE','#d46b29')]:
        ax.plot([int(r['optimizer_updates']) for r in values],[float(r[key]) for r in values],
                marker='o',markersize=3,label=label,color=color)
    ax.set(xlabel='Optimizer updates',ylabel='IoU (%)',title=title,xlim=(0,10000));ax.grid(alpha=.25);ax.legend()
fig.savefig(FIGURES/'validation_curves.png',dpi=180);fig.savefig(FIGURES/'validation_curves.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for values,label,color in [(read_rows(DSEQ/'train_log.csv'),'DSeq-JEPA','#2373a8'),(train,'MAE','#d46b29')]:
    xs=[int(values[i]['optimizer_updates']) for i in range(99,len(values),100)]
    for ax,key in zip(axes,('main_loss','aux_loss')):
        ys=[np.mean([float(r[key]) for r in values[i-99:i+1]]) for i in range(99,len(values),100)]
        ax.plot(xs,ys,label=label,color=color);ax.set(xlabel='Optimizer updates',ylabel='CE loss',title=key)
        ax.grid(alpha=.25);ax.legend()
fig.savefig(FIGURES/'training_loss.png',dpi=180);plt.close(fig)

telemetry=read_rows(RUN/'gpu_telemetry.csv')
times=np.array([float(r['unix_seconds']) for r in telemetry]);powers=np.array([float(r['power_watts']) for r in telemetry])
compute=dict(training_and_periodic_val_seconds=status['elapsed_seconds'],final_best_evaluation_seconds=evaluation['elapsed_seconds'],
             mean_optimizer_update_seconds=float(np.mean([float(r['update_seconds']) for r in train])),
             mean_updates_1_to_4000_seconds=float(np.mean([float(r['update_seconds']) for r in train[:4000]])),
             mean_updates_4001_to_10000_seconds=float(np.mean([float(r['update_seconds']) for r in train[4000:]])),
             peak_allocated_gib=max(float(r['peak_allocated_gib']) for r in train),
             peak_reserved_gib=max(float(r['peak_reserved_gib']) for r in train),
             device_peak_memory_mib=max(float(r['memory_used_mib']) for r in telemetry),
             sensor_interval_seconds=float(times[-1]-times[0]),device_energy_kwh=float(np.trapz(powers,times)/3600000),
             device_peak_temperature_c=max(float(r['temperature_c']) for r in telemetry))
(REPORT/'s2_compute_cost.json').write_text(json.dumps(compute,indent=2))
cost_table='| 度量 | DSeq-JEPA | MAE |\n|---|---:|---:|\n'
dt=read_rows(DSEQ/'train_log.csv');ds=read_json(DSEQ/'status.json');dep=read_json(REPORT/'s1_evaluation_provenance.json')
cost_table+=f'| 训练+10次周期val墙钟(s) | {ds["elapsed_seconds"]:.2f} | {status["elapsed_seconds"]:.2f} |\n'
cost_table+=f'| 平均optimizer update(s) | {np.mean([float(r["update_seconds"]) for r in dt]):.4f} | {compute["mean_optimizer_update_seconds"]:.4f} |\n'
cost_table+=f'| 训练allocator峰值(GiB) | {max(float(r["peak_allocated_gib"]) for r in dt):.4f} | {compute["peak_allocated_gib"]:.4f} |\n'
cost_table+=f'| 训练reserved峰值(GiB) | {max(float(r["peak_reserved_gib"]) for r in dt):.4f} | {compute["peak_reserved_gib"]:.4f} |\n'
cost_table+=f'| 最终best全量评价(s) | {dep["elapsed_seconds"]:.2f} | {evaluation["elapsed_seconds"]:.2f} |\n'
(REPORT/'s2_compute_cost.md').write_text(f'''# S2 计算成本（实测）

{cost_table}

MAE训练及最终评价合计 {(status['elapsed_seconds']+evaluation['elapsed_seconds'])/3600:.4f}任务运行小时。
同一物理GPU1 A6000/BF16/activation checkpoint；CPU threads4/workers8。
MAE额外的neck参数更新检查保留了neck.stages.0.0.weight抽样clone（2359296个FP32值，9MiB）；allocator比较包含这项审计内存，模型本身的参数数量和形状完全一致。
MAE前4000步平均{compute['mean_updates_1_to_4000_seconds']:.4f}s/update，后6000步平均{compute['mean_updates_4001_to_10000_seconds']:.4f}s/update；另一个GPU1进程在第4000步val期间启动，具体实时进程证据见s2_runtime_events.md。时间分段是环境观察，不能作为初始化导致速度差异的因果证据。
MAE设备遥测峰值显存{compute['device_peak_memory_mib']:.1f}MiB、温度{compute['device_peak_temperature_c']:.1f}°C；10秒采样区间{compute['sensor_interval_seconds']:.2f}s，板卡功率积分{compute['device_energy_kwh']:.5f}kWh。设备采样涵盖训练、最终评价及分析启动，含其它进程时不能全部归因于本任务；不等于整机能耗/电费。任务allocator峰值更适合比较模型显存。
S1设备遥测存在共享任务及缺失起始170秒，不能把设备能耗差直接解释成模型差异；本轮时间同样受到设备调度和I/O影响。Smoke20步单列，不算正式10000步。
来源：两条真实train_log/status/evaluation_provenance、gpu_telemetry.csv；S1固定成本见s1_gpu_compute.md。
''')
table='| 指标(%) | DSeq-JEPA | MAE | DSeq − MAE(百分点) |\n|---|---:|---:|---:|\n'
for r in comparison: table+=f'| {r["metric"]} | {r["DSeq_JEPA"]:.2f} | {r["MAE"]:.2f} | {r["DSeq_minus_MAE_pp"]:+.2f} |\n'
curve='| 更新数 | DSeq mIoU | MAE mIoU | DSeq Weed IoU | MAE Weed IoU |\n|---:|---:|---:|---:|---:|\n'
for d,m in zip(dv,val): curve+=f'| {d["optimizer_updates"]} | {d["mIoU"]} | {m["mIoU"]} | {d["weed"]} | {m["weed"]} |\n'
(REPORT/'s2_mae_training_results.md').write_text(f'''# S2 MAE真实训练结果

TRAINING_COMPLETE：10000/10000 optimizer updates，80000 microbatches，有效batch16；逐步日志连续、10次完整val已验证。
best-val step{best['optimizer_updates']}、mIoU{mm['mIoU']:.2f}%；最终严格恢复best并重新输出772/772张1024×1024单通道预测，ID仅0/1/2，官方phenobench-eval --task semantics输出保存在run/metrics。

{table}

{curve}

![逐类验证曲线](s2_figures/validation_curves.png)
![训练损失100步均值](s2_figures/training_loss.png)

权重大小/完整SHA见s2_checkpoint_manifest.json；正式启动代码、config/pretrain/data哈希见s2_run_provenance.json，最终best及772预测证据见s2_evaluation_provenance.json。
曲线是val选择过程，不是独立test，也没有多seed置信区间。
来源：真实正式日志、官方指标JSON和全量预测；[官方PhenoBench评价实现](https://github.com/PRBonn/phenobench/blob/0edc128ef7f67c8c6577554c7d1a2e382e2ea81f/src/phenobench/evaluation/evaluate_semantics.py)。
''')
# The comparison, visual interpretation, and final decision are reviewed prose.
# Keep those reports intact when recomputing numeric evidence.
print(json.dumps(dict(status=status,metrics=mm,best=best,error=me,compute=compute),indent=2))
