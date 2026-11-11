"""Single-GPU adaptation of src/eval/segmentation.py, without new model code.

All step counts are optimizer updates. BF16 backward stays in FP32 without
FP16 gradient scaling. The official model, main/auxiliary CE, optimizer, poly
LR and sliding-window predictor are reused directly.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.dataset import make_dataset
from src.eval.common import build_pretrained_encoder
from src.models import init_segmentation_model
from src.utils.opt.optimzer import get_finetune_optimizer
from src.utils.opt.scheduler import get_poly_lr_scheduler
from scripts.agriculture.evaluation import evaluate, official_metrics


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_row(path, row):
    new = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def build_model(config, device):
    m, s = config['model'], config['segmentation']
    if sha256(m['pretrained_weights']) != m['pretrained_sha256']:
        raise ValueError('Official pretrained file SHA-256 mismatch')
    if m['freeze_backbone'] or not m['strict_load'] or m['out_layers'] != [3, 5, 7, 11]:
        raise ValueError('Protocol requires strict loading and full backbone fine tuning')
    encoder = build_pretrained_encoder(config, device)
    return init_segmentation_model(
        encoder, num_classes=3, out_layers=m['out_layers'],
        channels=s['channels'], pool_scales=s['pool_scales'], dropout=s['dropout'],
        use_aux_head=s['use_aux_head'], aux_index=s['aux_index'], aux_channels=s['aux_channels'],
        use_checkpoint=m['use_checkpoint'], freeze_backbone=False, window_size=m['window_size']).to(device)


def batches(dataset, loader, sampler, consumed):
    epoch, offset = divmod(consumed, len(loader))
    while True:
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        # Workers are recreated each epoch, receiving its deterministic seed.
        for index, batch in enumerate(loader):
            if index >= offset:
                yield batch
        epoch, offset = epoch + 1, 0


def save_state(path, model, optimizer, scheduler, step, consumed, best, config, elapsed):
    state = dict(model=model.state_dict(), opt=optimizer.state_dict(),
                 lr_scheduler=scheduler.state_dict(), step=step, consumed_batches=consumed,
                 best_metric=best, config=config, elapsed_seconds=elapsed,
                 rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                          torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)
    return sha256(path)


def restore(path, model, optimizer=None, scheduler=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['opt'])
        scheduler.load_state_dict(state['lr_scheduler'])
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/agriculture/phenobench_dseqjepa_vitb16_upernet.yaml')
    parser.add_argument('--mode', choices=['verify', 'smoke', 'train', 'eval'], default='train')
    parser.add_argument('--checkpoint')
    parser.add_argument('--smoke-updates', type=int, default=20)
    parser.add_argument('--output')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This entry requires exactly one process')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Set CUDA_VISIBLE_DEVICES=1 to use physical GPU1')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    config = yaml.safe_load(Path(args.config).read_text())
    seed = config['opt']['global_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    model = build_model(config, device)
    run = Path(args.output or config['logging']['log_dir'])
    if args.mode in ('verify', 'smoke') and args.output is None:
        run = Path('runs') / args.mode
    if args.mode == 'train' and (run/'train_log.csv').exists() and not (
            args.checkpoint or config.get('resume', {}).get('resume_path')):
        raise ValueError('Existing trajectory: resume its checkpoint instead of restarting')
    run.mkdir(parents=True, exist_ok=True)
    (run / 'checkpoints').mkdir(exist_ok=True)
    (run / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    if config['model'].get('pretrained_type') == 'mae' and args.mode != 'eval' and not args.checkpoint:
        from src.utils.mae_checkpoint_adapter import tensor_sha256
        (run/'mae_weight_mapping.json').write_text(json.dumps(model.backbone.vit.mae_weight_mapping, indent=2))
        (run/'initialization.json').write_text(json.dumps(dict(
            seed=seed, total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            downstream_tensors={n:tensor_sha256(t) for n,t in model.state_dict().items()
                                if not n.startswith('backbone.')}
        ), indent=2))
    data = config['data']
    if sha256(data['manifest_path']) != data['manifest_sha256']:
        raise ValueError('Fixed data manifest SHA-256 mismatch')
    provenance = dict(
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=sha256(args.config), pretrained_sha256=config['model']['pretrained_sha256'],
        data_manifest_sha256=data['manifest_sha256'], original_zip_sha256=data['original_zip_sha256'],
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        device_name=torch.cuda.get_device_name(device), torch=torch.__version__,
        cuda=torch.version.cuda)
    # A final evaluation must preserve the training launch provenance.
    if args.mode != 'eval':
        (run/'provenance.json').write_text(json.dumps(provenance, indent=2))
    common = dict(dataset_name='phenobench', root_path=data['root_path'], crop_size=512,
                  base_size=data['base_size'], ratio_range=data['ratio_range'], test_scale=None,
                  num_workers=data['num_workers'], pin_mem=data['pin_memory'], seed=seed)
    train_ds, loader, sampler = make_dataset(batch_size=data['batch_size_per_replica'], training=True, **common)
    val_ds, _, _ = make_dataset(batch_size=1, training=False, drop_last=False, **common)
    if len(train_ds) != 1407 or len(val_ds) != 772:
        raise ValueError('Expected the fixed PhenoBench v1.1.0 train/val membership')
    if args.mode == 'verify':
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            image, _ = train_ds[0]
            x = image[None].to(device)
            features = model.backbone(x)
            neck = model.neck(features)
            logits = model(x)
        evidence = dict(checkpoint_sha256=sha256(config['model']['pretrained_weights']),
                        features=[list(f.shape) for f in features], neck=[list(f.shape) for f in neck],
                        logits=list(logits.shape), logits_dtype=str(logits.dtype),
                        frozen=[n for n,p in model.named_parameters() if not p.requires_grad])
        assert evidence['features'] == [[1,768,32,32]]*4
        assert evidence['logits'] == [1,3,512,512]
        (run / 'verification.json').write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence), flush=True)
        return
    if args.mode == 'eval':
        if not args.checkpoint:
            raise ValueError('--checkpoint required for evaluation')
        eval_start = time.monotonic()
        state = restore(args.checkpoint, model)
        predictions = run / 'predictions' / 'val'
        cross = evaluate(model, val_ds, device, predictions, config['logging']['eval']['slide_stride'])
        official = official_metrics(val_ds, predictions, run / 'metrics')
        (run / 'metrics' / 'confusion_crosscheck.json').write_text(json.dumps(cross, indent=2))
        (run / 'metrics' / 'evaluation_provenance.json').write_text(json.dumps(dict(
            **provenance, checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
            selected_optimizer_updates=state['step'], prediction_count=len(val_ds),
            original_resolution=[1024, 1024], elapsed_seconds=time.monotonic()-eval_start), indent=2))
        print(json.dumps(dict(step=state['step'], official=official, crosscheck=cross)), flush=True)
        return
    opt = config['opt']; lr = opt['lr']
    optimizer = get_finetune_optimizer(
        model, base_lr=lr['base_lr'], weight_decay=opt['weight_decay'],
        bias_decay=opt['bias_decay'], norm_decay=opt['norm_decay'],
        backbone_lr_scale=lr['backbone_lr_scale'], auto_lr_scaling=False, world_size=1,
        batch_size_per_replica=data['batch_size_per_replica'])
    scheduler = get_poly_lr_scheduler(optimizer, lr['base_lr'], opt['iterations'],
                                     lr['warmup_iters'], lr['warmup_ratio'], lr['power'], lr['min_lr'])
    step, consumed, best, elapsed_before = 0, 0, -1., 0.
    resume = args.checkpoint or config.get('resume', {}).get('resume_path')
    if resume:
        state = restore(resume, model, optimizer, scheduler)
        if state['config'] != config:
            raise ValueError('Resume requires the identical fixed training config')
        step, consumed, best = state['step'], state['consumed_batches'], state['best_metric']
        elapsed_before = state['elapsed_seconds']
    total = args.smoke_updates if args.mode == 'smoke' else opt['iterations']
    accumulation = opt['accumulation_steps']
    stream = batches(train_ds, loader, sampler, consumed)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    samples = dict(backbone='backbone.vit.patch_embed.proj.weight',
                   decoder='decode_head.cls_seg.weight', auxiliary='auxiliary_head.cls_seg.weight')
    if config['model'].get('pretrained_type') == 'mae':
        samples['neck'] = 'neck.stages.0.0.weight'
    params = dict(model.named_parameters())
    initial = {k: params[n].detach().clone() for k,n in samples.items()}
    gradient_evidence = {}
    while step < total:
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        lr_used = scheduler.get_current_lr()
        for _ in range(accumulation):
            images, labels = next(stream)
            consumed += 1
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=opt['use_bfloat16']):
                logits, aux = model(images.to(device, non_blocking=True))
                labels = labels.to(device, non_blocking=True)
                main_loss = F.cross_entropy(logits, labels, ignore_index=255)
                aux_loss = F.cross_entropy(aux, labels, ignore_index=255)
                loss = (main_loss + config['segmentation']['aux_loss_weight']*aux_loss) / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            losses.append((main_loss.item(), aux_loss.item()))
        if step == 0:
            gradient_evidence = {k:dict(gradient_norm=float(params[n].grad.norm())) for k,n in samples.items()}
            missing = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
            if missing:
                raise ValueError(f'Trainable parameters without gradients: {missing}')
            (run/'all_parameter_gradients.json').write_text(json.dumps(
                {n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad}, indent=2))
        if opt['clip_grad'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt['clip_grad'])
        optimizer.step()
        scheduler.step()
        step += 1
        if step <= 20 and gradient_evidence:
            for k,n in samples.items():
                gradient_evidence[k]['max_abs_update'] = float((params[n].detach()-initial[k]).abs().max())
                if gradient_evidence[k]['gradient_norm'] <= 0:
                    raise ValueError('Expected non-zero backbone/head gradients')
            (run / 'gradient_evidence.json').write_text(json.dumps(gradient_evidence, indent=2))
        elapsed = elapsed_before + time.monotonic()-start
        row = dict(optimizer_updates=step, consumed_microbatches=consumed,
                   main_loss=float(np.mean(losses, axis=0)[0]), aux_loss=float(np.mean(losses, axis=0)[1]),
                   lr=lr_used, next_lr=scheduler.get_current_lr(), microbatch=data['batch_size_per_replica'],
                   accumulation_steps=accumulation, effective_batch=data['batch_size_per_replica']*accumulation,
                   update_seconds=time.monotonic()-tick, elapsed_seconds=elapsed,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        csv_row(run/'train_log.csv', row)
        if step % config['logging']['log_step_freq'] == 0 or step == total:
            print(json.dumps(row), flush=True)
        if step % config['logging']['ckpt']['ckpt_iter_freq'] == 0 or step == total:
            path = run/'checkpoints'/'phenobench_upernet_latest.pth'
            digest = save_state(path, model, optimizer, scheduler, step, consumed, best, config, elapsed)
            csv_row(run/'checkpoint_manifest.csv', dict(step=step, file=str(path), sha256=digest))
            if step in config['logging']['ckpt']['milestones'] or args.mode == 'smoke':
                milestone = run/'checkpoints'/f'phenobench_upernet_step{step:05d}.pth'
                # One immutable inode per milestone; later latest saves are atomic replacements.
                os.link(path, milestone)
                csv_row(run/'checkpoint_manifest.csv', dict(step=step, file=str(milestone), sha256=digest))
        if args.mode == 'smoke' and step == total:
            restored = restore(path, model, optimizer, scheduler)
            assert restored['step'] == step and restored['consumed_batches'] == consumed
            cross = evaluate(model, val_ds, device, run/'predictions'/'val', limit=1)
            # A real one-image official-devkit smoke, with symlinks to original
            # data. Its score is deliberately separate from the 772-image val.
            subset = copy.copy(val_ds)
            subset.names = val_ds.names[:1]
            subset.img_dir = run/'smoke_dataset'/'val'/'images'
            subset.ann_dir = run/'smoke_dataset'/'val'/'semantics'
            for directory, source in [(subset.img_dir, val_ds.img_dir), (subset.ann_dir, val_ds.ann_dir)]:
                directory.mkdir(parents=True, exist_ok=True)
                for name in subset.names:
                    link = directory/name
                    if not link.exists():
                        link.symlink_to(source/name)
            smoke_official = official_metrics(subset, run/'predictions'/'val', run/'metrics')
            if gradient_evidence and not all(v['max_abs_update'] > 0 for v in gradient_evidence.values()):
                raise ValueError('Smoke must show actual backbone/main/auxiliary weight updates')
            (run/'smoke.json').write_text(json.dumps(dict(step=step, restore=True, gradients=gradient_evidence,
                                                         one_val_prediction=cross, official_one_image=smoke_official,
                                                         compute=row), indent=2))
        elif args.mode == 'train' and (step % config['logging']['eval']['eval_iter_freq'] == 0 or step == total):
            predictions = run/'predictions'/'val'
            cross = evaluate(model, val_ds, device, predictions, config['logging']['eval']['slide_stride'])
            metrics = official_metrics(val_ds, predictions, run/'metrics'/f'step{step:05d}')
            csv_row(run/'val_log.csv', dict(optimizer_updates=step, **metrics))
            print(json.dumps(dict(step=step, official=metrics, crosscheck=cross)), flush=True)
            if metrics['mIoU'] > best:
                best = metrics['mIoU']
                best_path = run/'checkpoints'/'phenobench_upernet_best.pth'
                digest = save_state(best_path, model, optimizer, scheduler, step, consumed, best, config,
                                    elapsed_before+time.monotonic()-start)
                csv_row(run/'checkpoint_manifest.csv', dict(step=step, file=str(best_path), sha256=digest))
            # Latest must contain the best score and elapsed time after evaluation.
            latest = run/'checkpoints'/'phenobench_upernet_latest.pth'
            digest = save_state(latest, model, optimizer, scheduler,
                                step, consumed, best, config, elapsed_before+time.monotonic()-start)
            csv_row(run/'checkpoint_manifest.csv', dict(step=step, file=str(latest), sha256=digest))
            model.train()
    (run/'status.json').write_text(json.dumps(dict(status='SMOKE_COMPLETE' if args.mode=='smoke' else 'TRAINING_COMPLETE',
                                                 optimizer_updates=step, best_val_mIoU=best,
                                                 elapsed_seconds=elapsed_before+time.monotonic()-start), indent=2))


if __name__ == '__main__':
    main()
