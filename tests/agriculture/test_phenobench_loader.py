import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset.phenobench import PhenoBenchSegmentation, convert_partial_semantics


def sample(root, width=32):
    for split in ('train', 'val'):
        for folder in ('images', 'semantics'):
            (root/split/folder).mkdir(parents=True)
        mask = np.tile(np.arange(width) % 5, (width, 1)).astype(np.uint16)
        rgb = np.repeat((convert_partial_semantics(mask)*80)[:, :, None], 3, axis=2)
        Image.fromarray(rgb).save(root/split/'images'/'field.png')
        Image.fromarray(mask).save(root/split/'semantics'/'field.png')


def test_official_partial_mapping_preserves_soil_and_input():
    raw = np.array([[0,1,2,3,4]], dtype=np.uint16)
    result = convert_partial_semantics(raw)
    assert result.tolist() == [[0,1,2,1,2]]
    assert raw.tolist() == [[0,1,2,3,4]]
    for invalid in (np.array([[5]]), np.zeros((2,2,3)), np.ones((1,1), dtype=float)):
        with pytest.raises(ValueError):
            convert_partial_semantics(invalid)


def test_integer_png_and_original_validation_coordinates(tmp_path):
    sample(tmp_path)
    dataset = PhenoBenchSegmentation(tmp_path, train=False)
    image, mask = dataset[0]
    assert image.shape == (3,32,32) and mask.shape == (32,32)
    assert mask.dtype == torch.int64 and set(mask.unique().tolist()) == {0,1,2}
    with pytest.raises(ValueError):
        PhenoBenchSegmentation(tmp_path, train=False, test_scale=(512,512))


def test_synchronized_crop_flip_and_padding_ignore(tmp_path):
    sample(tmp_path)
    dataset = PhenoBenchSegmentation(tmp_path, crop_size=48, base_size=(32,32),
                                    ratio_range=(1,1), photometric_distortion=False,
                                    mean=(0,0,0), std=(1,1,1))
    image, label = dataset[0]
    valid = label != 255
    assert image.shape == (3,48,48) and not valid.all()
    np.testing.assert_allclose(image[0][valid].numpy()*255,
                               label[valid].numpy()*80, atol=1e-4)
    # Worker timing and repeated reads cannot alter sample augmentation.
    image2, label2 = dataset[0]
    assert torch.equal(image, image2) and torch.equal(label, label2)


def test_nearest_resize_never_creates_new_label_ids(tmp_path):
    sample(tmp_path)
    ds = PhenoBenchSegmentation(tmp_path, crop_size=48, base_size=(48,48),
                               ratio_range=(1,1), photometric_distortion=False)
    _, labels = ds[0]
    assert set(labels.unique().tolist()) <= {0,1,2,255}
