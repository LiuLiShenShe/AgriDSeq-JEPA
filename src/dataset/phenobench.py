"""PhenoBench v1.1.0 semantics, using the upstream ADE20K augmentation helpers.

Soil is a scored class. Only geometric padding uses 255. Validation never
rescales the original image. Raw integer PNG labels follow the official devkit.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

from src.dataset.ade20k import (
    ADE20KSegmentation, IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD,
    collate_segmentation,
)

PHENOBENCH_CLASSES = ("soil", "crop", "weed")


def convert_partial_semantics(mask):
    """Same mapping as PRBonn/phenobench's convert_partial_semantics."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Expected a two-dimensional integer semantic mask")
    if not np.isin(mask, [0, 1, 2, 3, 4]).all():
        raise ValueError(f"Unknown PhenoBench labels: {np.unique(mask)}")
    result = mask.astype(np.uint8, copy=True)
    result[result == 3] = 1
    result[result == 4] = 2
    return result


class PhenoBenchSegmentation(ADE20KSegmentation):
    def __init__(self, root, train=True, crop_size=512, base_size=(1024, 512),
                 ratio_range=(0.5, 2.0), cat_max_ratio=0.75,
                 photometric_distortion=True, test_scale=None,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD, seed=20261110):
        if test_scale is not None:
            raise ValueError("PhenoBench validation must retain original coordinates")
        split = "train" if train else "val"
        self.img_dir = Path(root) / split / "images"
        self.ann_dir = Path(root) / split / "semantics"
        self.names = sorted(p.name for p in self.img_dir.glob("*.png"))
        if not self.names:
            raise FileNotFoundError(self.img_dir)
        labels = {p.name for p in self.ann_dir.glob("*.png")}
        if set(self.names) != labels:
            raise ValueError(f"Image/mask membership differs in {split}")
        self.train, self.crop_size = train, crop_size
        self.base_size, self.ratio_range = base_size, ratio_range
        self.cat_max_ratio = cat_max_ratio
        self.photometric_distortion = photometric_distortion
        self.test_scale = None
        self.mean, self.std = mean, std
        self.seed, self.epoch = seed, 0

    def _remap(self, ann):
        return convert_partial_semantics(ann)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.img_dir / name) as source:
            img = source.convert("RGB")
        with Image.open(self.ann_dir / name) as source:
            raw = np.array(source)  # do not convert integer labels to RGB
        ann = Image.fromarray(self._remap(raw))
        if img.size != ann.size:
            raise ValueError(f"Mismatched image/mask dimensions: {name}")
        if self.train:
            # Augmentation depends on sample and epoch, rather than worker
            # prefetch timing; resuming at a saved batch position is reproducible.
            previous_rng = random.getstate()
            random.seed(self.seed + self.epoch * len(self) + index)
            try:
                ratio = random.uniform(*self.ratio_range)
                img, ann = self._resize(img, ann, tuple(int(v * ratio) for v in self.base_size))
                pw, ph = max(0, self.crop_size-img.width), max(0, self.crop_size-img.height)
                # PIL.crop outside an image fills with zero, which would turn
                # padding into soil. Explicitly pad the annotation first.
                if pw or ph:
                    img = ImageOps.expand(img, (0, 0, pw, ph), fill=0)
                    ann = ImageOps.expand(ann, (0, 0, pw, ph), fill=IGNORE_INDEX)
                img, ann = self._random_crop(img, ann)
                if random.random() < 0.5:
                    img, ann = TF.hflip(img), TF.hflip(ann)
                if self.photometric_distortion:
                    img = self._photometric_distortion(img)
            finally:
                random.setstate(previous_rng)
        return self._normalize(img), torch.from_numpy(np.array(ann)).long()


def make_phenobench(transform=None, batch_size=2, collator=collate_segmentation,
                   pin_mem=True, num_workers=8, world_size=1, rank=0,
                   root_path=None, training=True, drop_last=True, **kwargs):
    keys = ("crop_size", "base_size", "ratio_range", "cat_max_ratio",
            "photometric_distortion", "test_scale", "seed")
    dataset = PhenoBenchSegmentation(root_path, train=training,
                                    **{k: v for k, v in kwargs.items() if k in keys})
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=training,
        seed=dataset.seed, drop_last=drop_last)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
        pin_memory=pin_mem, drop_last=drop_last, collate_fn=collator,
        generator=torch.Generator().manual_seed(dataset.seed))
    return dataset, loader, sampler
