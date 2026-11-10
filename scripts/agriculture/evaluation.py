"""Export validation predictions, check coverage, invoke the official devkit."""
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.eval.segmentation import slide_inference, confusion_update


def validate_predictions(dataset, prediction_dir):
    directory = Path(prediction_dir) / 'semantics'
    actual = {p.name for p in directory.glob('*.png')}
    if actual != set(dataset.names):
        raise ValueError(f'Prediction membership mismatch: {len(actual)} vs {len(dataset)}')
    for name in dataset.names:
        with Image.open(directory / name) as p, Image.open(dataset.img_dir / name) as source:
            values = np.array(p)
            if p.size != source.size or values.ndim != 2 or not np.isin(values, [0, 1, 2]).all():
                raise ValueError(f'Invalid prediction dimensions/IDs: {name}')


def official_metrics(dataset, prediction_dir, output_dir):
    from phenobench.evaluation.evaluate_semantics import evaluate_semantics
    validate_predictions(dataset, prediction_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = str(dataset.img_dir.parent.parent)
    command = [str(Path(sys.executable).parent / 'phenobench-eval'),
               '--task', 'semantics', '--phenobench_dir', root,
               '--prediction_dir', str(prediction_dir), '--split', 'val']
    result = subprocess.run(command, capture_output=True, text=True, check=True,
                            env={**__import__('os').environ, 'OMP_NUM_THREADS': '4',
                                 'MKL_NUM_THREADS': '4'})
    (output_dir / 'phenobench_official_cli.txt').write_text(result.stdout + result.stderr)
    args = dict(phenobench_dir=Path(root), prediction_dir=Path(prediction_dir),
                split='val', task='semantics')
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = evaluate_semantics(args)
    metrics = {k: float(v) for k, v in metrics.items()}
    (output_dir / 'phenobench_official.json').write_text(json.dumps(metrics, indent=2))
    return metrics


@torch.no_grad()
def evaluate(model, dataset, device, prediction_dir, stride=341, limit=None):
    model.eval()
    destination = Path(prediction_dir) / 'semantics'
    destination.mkdir(parents=True, exist_ok=True)
    conf = torch.zeros((3, 3), dtype=torch.int64, device=device)
    for index in range(len(dataset) if limit is None else min(limit, len(dataset))):
        image, label = dataset[index]
        logits = slide_inference(model, image[None].to(device), 3, 512, stride, True)
        pred = logits.argmax(1)[0]
        confusion_update(conf, pred.flatten(), label.to(device).flatten(), 3)
        Image.fromarray(pred.cpu().numpy().astype(np.uint8)).save(destination / dataset.names[index])
        if (index + 1) % 100 == 0:
            print(f'Validation PNGs: {index+1}/{len(dataset)}', flush=True)
    conf = conf.cpu().numpy()
    intersection = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - intersection
    iou = 100 * intersection / np.maximum(union, 1)
    return dict(soil=float(iou[0]), crop=float(iou[1]), weed=float(iou[2]),
                mIoU=float(iou.mean()), confusion=conf.tolist())
