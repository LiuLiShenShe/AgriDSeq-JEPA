"""Focused mapping tests; the actual ViT-B source is checked by the GPU audit."""
import copy

import pytest
import torch

from src.models.masked_vision_transformer import MaskedVisionTransformer
from src.utils.mae_checkpoint_adapter import adapt_mae_encoder, load_mae_encoder_weights


@pytest.fixture
def pair():
    encoder = MaskedVisionTransformer(img_size=[32], patch_size=16, embed_dim=32,
                                     depth=12, num_heads=4, use_class_token=True)
    source = copy.deepcopy(encoder.state_dict())
    source['pos_embed'] = torch.cat([torch.full((1,1,32), 99.), source['pos_embed']], 1)
    return encoder, source


def test_strict_load_positions_and_rng(pair, tmp_path):
    encoder, source = pair
    source['blocks.11.attn.qkv.weight'].fill_(0.125)
    source['decoder_pred.weight'] = torch.zeros(4,4)
    path = tmp_path/'mae.pth'
    torch.save({'model':source}, path)
    before = torch.get_rng_state().clone()
    load_mae_encoder_weights(encoder, path)
    assert torch.equal(before, torch.get_rng_state())
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, source[key][:,1:] if key == 'pos_embed' else source[key])
    assert encoder.mae_weight_mapping['excluded_source_keys'] == ['decoder_pred.weight']
    assert encoder.mae_weight_mapping['missing_keys'] == []


@pytest.mark.parametrize('fault', ['missing', 'unexpected', 'position', 'shape', 'nonfinite'])
def test_rejects_incomplete_or_incompatible_source(pair, fault):
    encoder, source = pair
    if fault == 'missing': del source['blocks.7.mlp.fc2.weight']
    elif fault == 'unexpected': source['head.weight'] = torch.zeros(3,32)
    elif fault == 'position': source['pos_embed'] = source['pos_embed'][:,1:]
    elif fault == 'shape': source['norm.weight'] = torch.zeros(33)
    else: source['patch_embed.proj.weight'][0,0,0,0] = float('nan')
    with pytest.raises(ValueError): adapt_mae_encoder({'model':source}, encoder)
