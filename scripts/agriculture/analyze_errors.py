"""Read final original-resolution predictions and describe agricultural errors.

Creates numerical case evidence and local contact sheets; no further training.
Small weeds are defined by image pixels, without claiming physical leaf area.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from src.dataset.phenobench import convert_partial_semantics

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('/data/datasets/PhenoBench/PhenoBench-v110'))
parser.add_argument('--run', type=Path, default=Path('runs/phenobench_dseq_vitb16_s20261110'))
parser.add_argument('--output', type=Path, help='Separate directory for an intermediate validation snapshot')
args = parser.parse_args()
rows = []
totals = dict(weed_tp=0, weed_fp=0, weed_fn=0, small_components=0, small_recalled=0)
palette = np.array([[95,65,40], [30,200,65], [220,45,180]], dtype=np.uint8)
destination = args.output or args.run/'analysis'
destination.mkdir(exist_ok=True)
for path in sorted((args.root/'val'/'semantics').glob('*.png')):
    gt = convert_partial_semantics(np.array(Image.open(path)))
    pred = np.array(Image.open(args.run/'predictions'/'val'/'semantics'/path.name))
    assert pred.shape == gt.shape == (1024,1024)
    weed, pw = gt == 2, pred == 2
    tp, fp, fn = int((weed & pw).sum()), int((~weed & pw).sum()), int((weed & ~pw).sum())
    components, count = ndimage.label(weed)
    sizes = np.bincount(components.ravel())
    overlap = np.bincount(components[pw], minlength=count+1)
    small = (sizes[1:] <= 100)
    recalled = overlap[1:] >= 0.5*sizes[1:]
    row = dict(filename=path.name, weed_pixels=int(weed.sum()), weed_IoU=100*tp/max(1,tp+fp+fn),
               weed_precision=100*tp/max(1,tp+fp), weed_recall=100*tp/max(1,tp+fn),
               weed_tp=tp, weed_fp=fp, weed_fn=fn, small_components=int(small.sum()),
               small_recalled=int((small & recalled).sum()),
               weed_as_crop=int((weed & (pred==1)).sum()), weed_as_soil=int((weed & (pred==0)).sum()),
               crop_as_weed=int(((gt==1) & pw).sum()))
    rows.append(row)
    for key in totals:
        totals[key] += row[key]
with (destination/'per_image_errors.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
selected = sorted((r for r in rows if r['weed_pixels'] >= 100), key=lambda r:r['weed_IoU'])[:8]
for row in selected:
    name = row['filename']
    rgb = Image.open(args.root/'val'/'images'/name).convert('RGB')
    gt = convert_partial_semantics(np.array(Image.open(args.root/'val'/'semantics'/name)))
    pred = np.array(Image.open(args.run/'predictions'/'val'/'semantics'/name))
    error = np.array(rgb).copy()
    error[gt != pred] = [255,40,40]
    sheet = Image.new('RGB', (2048,550), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8,4), f'{name}; weed IoU {row["weed_IoU"]:.2f}%; brown=soil green=crop magenta=weed', fill='black')
    for i,(label,im) in enumerate([('RGB',rgb),('Ground truth',Image.fromarray(palette[gt])),
                                  ('Prediction',Image.fromarray(palette[pred])),('Errors (red)',Image.fromarray(error))]):
        draw.text((i*512+8,24), label, fill='black')
        sheet.paste(im.resize((512,512), Image.Resampling.NEAREST if i in (1,2) else Image.Resampling.BILINEAR), (i*512,38))
    sheet.save(destination/(Path(name).stem+'_case.png'))
summary = dict(weed_precision=100*totals['weed_tp']/max(1,totals['weed_tp']+totals['weed_fp']),
               weed_recall=100*totals['weed_tp']/max(1,totals['weed_tp']+totals['weed_fn']),
               small_component_recall_100px=100*totals['small_recalled']/max(1,totals['small_components']),
               small_component_definition='GT weed 4-connected component area <=100px; >=50% pixels predicted weed',
               totals=totals, worst_cases=selected)
(destination/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
