from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset.phenobench import PhenoBenchSegmentation
from src.eval.segmentation import slide_inference
from scripts.agriculture.evaluation import official_metrics, validate_predictions


def test_sliding_windows_fuse_logits_and_cover_borders():
    class ConstantModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 3, *x.shape[2:]) * torch.tensor([1.,3.,2.])[None,:,None,None]
    out = slide_inference(ConstantModel(), torch.zeros(1,3,1024,1024), 3)
    assert out.shape == (1,3,1024,1024)
    assert (out.argmax(1) == 1).all()
    assert (out[:,1] == 3).all()


def test_official_evaluator_and_coverage_guard(tmp_path):
    root, predictions = tmp_path/'data', tmp_path/'predictions'
    for p in (root/'val'/'images', root/'val'/'semantics', predictions/'semantics'):
        p.mkdir(parents=True)
    Image.fromarray(np.zeros((8,8,3), dtype=np.uint8)).save(root/'val'/'images'/'field.png')
    raw = np.tile(np.array([0,1,2,3,4,0,1,2],dtype=np.uint16), (8,1))
    Image.fromarray(raw).save(root/'val'/'semantics'/'field.png')
    ds = PhenoBenchSegmentation(root, train=False)
    with pytest.raises(ValueError):
        validate_predictions(ds, predictions)
    pred = raw.copy(); pred[pred==3]=1; pred[pred==4]=2
    Image.fromarray(pred.astype(np.uint8)).save(predictions/'semantics'/'field.png')
    scores = official_metrics(ds, predictions, tmp_path/'metrics')
    assert scores == {'soil':100., 'crop':100., 'weed':100., 'mIoU':100.}
    Image.fromarray(np.zeros((4,4),dtype=np.uint8)).save(predictions/'semantics'/'field.png')
    with pytest.raises(ValueError):
        validate_predictions(ds, predictions)
