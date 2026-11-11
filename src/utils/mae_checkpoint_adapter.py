"""Strict, in-memory adaptation of Meta MAE encoder weights for dense ViT.

The public pretrain file already contains only the encoder. Full MAE pretrain
states may also contain decoder tensors; only those explicitly named below
are excluded. No missing target tensor is filled from random initialization.
"""
import hashlib

import torch


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def adapt_mae_encoder(checkpoint, encoder):
    source = checkpoint.get('model')
    if not isinstance(source, dict) or not source:
        raise ValueError('MAE checkpoint requires a nonempty model state_dict')
    target = encoder.state_dict()
    if len(encoder.blocks) != 12 or 'cls_token' not in target:
        raise ValueError('Expected the S1 twelve-block ViT with class-token parameter')
    excluded = []
    for name in source.keys() - target.keys():
        if name == 'mask_token' or name == 'decoder_pos_embed' or name.startswith((
                'decoder_embed.', 'decoder_blocks.', 'decoder_norm.', 'decoder_pred.')):
            excluded.append(name)
        else:
            raise ValueError(f'Unexpected MAE source tensor: {name}')
    missing = target.keys() - source.keys()
    if missing:
        raise ValueError(f'MAE encoder tensors missing: {sorted(missing)}')
    mapped, entries = {}, []
    for name, expected in target.items():
        original = source[name]
        if not isinstance(original, torch.Tensor):
            raise ValueError(f'Non-tensor MAE parameter: {name}')
        value = original
        operation = 'identity'
        if name == 'pos_embed':
            shape = (expected.shape[0], expected.shape[1] + 1, expected.shape[2])
            if tuple(original.shape) != shape:
                raise ValueError(f'MAE CLS+patch position shape must be {shape}')
            value = original[:, 1:, :]
            operation = 'remove CLS positional entry; preserve patch order'
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f'MAE tensor shape/dtype mismatch: {name}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Non-finite MAE tensor: {name}')
        mapped[name] = value
        entries.append(dict(source=name, target=name, source_shape=list(original.shape),
                            target_shape=list(value.shape), operation=operation,
                            dtype=str(value.dtype), sha256=tensor_sha256(value)))
    return mapped, dict(mapped_tensor_count=len(mapped), mapped_parameters=sum(v.numel() for v in mapped.values()),
                        blocks=list(range(12)), entries=entries, excluded_source_keys=sorted(excluded),
                        discarded_position='pos_embed[:, :1, :] (unused CLS position)',
                        cls_token='loaded from MAE, frozen and unused by S1 ViTDenseBackbone',
                        random_target_fallbacks=[], missing_target_keys=[])


def load_mae_encoder_weights(encoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    mapped, evidence = adapt_mae_encoder(checkpoint, encoder)
    result = encoder.load_state_dict(mapped, strict=True)
    for name, loaded in encoder.state_dict().items():
        if not torch.equal(loaded.detach().cpu(), mapped[name]):
            raise ValueError(f'MAE loaded tensor differs from source: {name}')
    evidence.update(strict=True, missing_keys=list(result.missing_keys),
                    unexpected_keys=list(result.unexpected_keys), all_loaded_tensors_equal_source=True)
    encoder.mae_weight_mapping = evidence
    return encoder
