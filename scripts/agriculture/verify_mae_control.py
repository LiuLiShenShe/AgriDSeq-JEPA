"""Reconstruct S1 initialization and audit the one MAE control before training."""
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.agriculture.run_phenobench import build_model
from src.utils.mae_checkpoint_adapter import tensor_sha256

S1 = '20ffaadcfa0f1172303c2ccbc09a456458f80e3d'
LAUNCH = '399f9a84fe09471fe3d990f68734ec096fe540ca'
assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
torch.set_num_threads(4)
device = torch.device('cuda:0')
torch.cuda.set_device(device)
configs = [yaml.safe_load(Path(f'configs/agriculture/phenobench_{kind}_vitb16_upernet.yaml').read_text())
           for kind in ('dseqjepa', 'mae')]
normalized = json.loads(json.dumps(configs[1]))
del normalized['model']['pretrained_type']
for key in ('pretrained_weights', 'pretrained_sha256'):
    normalized['model'][key] = configs[0]['model'][key]
normalized['logging']['log_dir'] = configs[0]['logging']['log_dir']
assert normalized == configs[0], 'Downstream protocol changed'
original = subprocess.check_output(['git','show',f'{LAUNCH}:scripts/agriculture/run_phenobench.py'],text=True)
namespace = {'__name__':'s1_launch_reconstruction'}
exec(compile(original, 's1_launch_run_phenobench.py', 'exec'), namespace)
dependency_paths = ['src/models', 'src/eval/common.py', 'src/utils/opt', 'src/dataset']
unchanged = subprocess.check_output(['git','diff','--name-only',LAUNCH,S1,'--',*dependency_paths],text=True)
assert not unchanged, f'S1 launch dependencies changed: {unchanged}'


def seed():
    s = configs[0]['opt']['global_seed']
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False


def rng():
    return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda=[hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in torch.cuda.get_rng_state_all()])


seed()
dseq = namespace['build_model'](configs[0], device)
dseq_rng = rng()
seed()
mae = build_model(configs[1], device)
mae_rng = rng()
assert repr(dseq) == repr(mae)
dp, mp = dict(dseq.named_parameters()), dict(mae.named_parameters())
assert {n:(list(p.shape),p.requires_grad) for n,p in dp.items()} == {
    n:(list(p.shape),p.requires_grad) for n,p in mp.items()}
assert dseq_rng == mae_rng, 'Initialization consumed different RNG streams'
downstream = []
for name, value in dseq.state_dict().items():
    if name.startswith('backbone.'): continue
    other = mae.state_dict()[name]
    equal = torch.equal(value, other)
    assert equal, name
    downstream.append(dict(name=name, shape=list(value.shape), tensor_equal=equal,
                           dseq_sha256=tensor_sha256(value), mae_sha256=tensor_sha256(other)))
groups = {}
for name in ('backbone','neck','decode_head','auxiliary_head'):
    sub = getattr(mae, name)
    groups[name] = dict(parameters=sum(p.numel() for p in sub.parameters()),
                       trainable_parameters=sum(p.numel() for p in sub.parameters() if p.requires_grad),
                       initial_state_tensors=sum(r['name'].startswith(name+'.') for r in downstream))
mae.eval()
# The shared verify entry independently checks real image output shapes.
from src.dataset import make_dataset
d = configs[1]['data']
ds, _, _ = make_dataset(dataset_name='phenobench', root_path=d['root_path'], batch_size=1,
                       training=False, crop_size=512, base_size=d['base_size'],
                       ratio_range=d['ratio_range'], test_scale=None, num_workers=0,
                       pin_mem=False, drop_last=False, seed=configs[1]['opt']['global_seed'])
x = ds[0][0][None,:,:512,:512].to(device)
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    features = mae.backbone(x)
    logits = mae(x)
assert [list(f.shape) for f in features] == [[1,768,32,32]]*4
assert list(logits.shape) == [1,3,512,512]
evidence = dict(s1_frozen_commit=S1, s1_training_launch_commit=LAUNCH,
                s1_launch_dependencies_equal_frozen=True, normalized_configs_equal=True,
                architecture_repr_equal=True, parameter_shapes_and_trainability_equal=True,
                rng_after_initialization_equal=True, rng=dseq_rng, groups=groups,
                total_parameters=sum(p.numel() for p in mae.parameters()),
                trainable_parameters=sum(p.numel() for p in mae.parameters() if p.requires_grad),
                downstream_tensors=downstream, features=[list(f.shape) for f in features],
                logits=list(logits.shape), frozen=[n for n,p in mp.items() if not p.requires_grad],
                reconstruction_limit='Exact reconstructed S1 and actual S2 initialization; S1 did not archive step-zero tensors, so historical S1 tensors cannot be independently measured.')
Path('reports/s2_initialization_evidence.json').write_text(json.dumps(evidence,indent=2))
Path('reports/s2_mae_weight_mapping.json').write_text(json.dumps(mae.backbone.vit.mae_weight_mapping,indent=2))
print(json.dumps({k:v for k,v in evidence.items() if k not in ('downstream_tensors','rng')},indent=2))
