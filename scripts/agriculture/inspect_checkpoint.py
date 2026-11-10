"""CPU-only inspection of a real complete downstream checkpoint."""
import argparse
import json
from pathlib import Path

import torch

parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
assert {'model','opt','lr_scheduler','step','consumed_batches','rng','config'} <= state.keys()
steps=[float(v['step']) for v in state['opt']['state'].values()]
assert min(steps)==max(steps)==state['step']==state['lr_scheduler']['_step']
assert {'python','numpy','torch','cuda'} <= state['rng'].keys()
components={p:sum(k.startswith(p+'.') for k in state['model'])
            for p in ('backbone','neck','decode_head','auxiliary_head')}
assert all(components.values())
source=torch.load(state['config']['model']['pretrained_weights'],map_location='cpu',weights_only=False)
deltas={}
for key in ['patch_embed.proj.weight']+[f'blocks.{i}.attn.qkv.weight' for i in range(12)]:
    delta=state['model']['backbone.vit.'+key]-source[key]
    deltas[key]={'l2_delta':float(delta.norm()),'max_abs_delta':float(delta.abs().max())}
assert all(v['max_abs_delta']>0 for v in deltas.values())
fixed={k:float((state['model']['backbone.vit.'+k]-source[k]).abs().max())
       for k in ('pos_embed','cls_token')}
result=dict(checkpoint=str(args.checkpoint),bytes=args.checkpoint.stat().st_size,
            optimizer_updates=state['step'],consumed_microbatches=state['consumed_batches'],
            adamw_parameter_states=len(steps),adamw_step_min=min(steps),adamw_step_max=max(steps),
            poly_step=state['lr_scheduler']['_step'],component_state_tensor_counts=components,
            rng_keys=list(state['rng']),active_backbone_weight_changes=deltas,
            official_fixed_token_max_abs_delta=fixed)
args.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
