"""Training routines adapted from the archived C2/C3 experiment scripts."""

import copy
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .evaluation import metrics, write_predictions as writepred

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEV.type == "cuda"


def progress(component, **fields):
    if os.environ.get("MDGF_PROGRESS") == "1":
        print(component, json.dumps(fields), flush=True)


def restore_cuda_state(state):
    if USE_CUDA:
        torch.cuda.set_rng_state_all(state)


def pin(tensor):
    return tensor.pin_memory() if USE_CUDA else tensor


class CenterOnly(nn.Module):

    def __init__(self, template):
        super().__init__()
        self.center_projection = template.center_projection
        self.classifier = template.classifier

    def forward(self, xx):
        c = self.center_projection(xx[:, 7, :768])
        return self.classifier(torch.cat([c, torch.zeros_like(c), c], 1)).squeeze(1)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def put(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(d, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf8')
    os.replace(q, p)

def torchsave(obj, p):
    import torch
    p = Path(p)
    q = p.with_suffix('.writing')
    torch.save(obj, q)
    os.replace(q, p)

def seed(n=42):
    import torch
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_context(emb, seq, aux, y, sp, folder, cfg, model_class, component='gpu'):
    import torch
    from torch import nn
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'validation_record.json').exists():
        return json.loads((folder / 'validation_record.json').read_text(encoding='utf8'))
    assert np.isfinite(emb[np.r_[sp['train'], sp['val']]]).all(), 'Non-finite encoder inputs: repair extraction before context training'
    dev = DEV
    torch.set_num_threads(2)
    seed(cfg['seed'])
    template = model_class(dropout=cfg['dropout'])
    model = CenterOnly(template) if cfg.get('no_context') else template
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scaler = torch.cuda.amp.GradScaler(enabled=USE_CUDA)
    X = {s: pin(torch.from_numpy(np.concatenate([emb[seq[sp[s]]], aux[sp[s]]], axis=-1).astype('float32'))) for s in ['train', 'val']}
    Y = torch.tensor(y[sp['train']], dtype=torch.float32, device=dev)
    sd = torch.tensor(emb[sp['train']].std(0), device=dev)
    best = -1.0
    wait = 0
    history = []
    start = 1
    tstart = time.time()
    if (folder / 'last.pt').exists():
        last = torch.load(folder / 'last.pt', map_location='cpu')
        model.load_state_dict(last['model_state_dict'])
        opt.load_state_dict(last['optimizer'])
        scaler.load_state_dict(last['scaler'])
        best = last['best']
        wait = last['wait']
        history = last['history']
        start = last['epoch'] + 1
        random.setstate(last['python_rng'])
        np.random.set_state(last['numpy_rng'])
        torch.set_rng_state(last['torch_rng'])
        restore_cuda_state(last['cuda_rng'])

    def pred(x):
        model.eval()
        out = []
        with torch.inference_mode():
            for j in range(0, len(x), 128):
                with torch.cuda.amp.autocast(enabled=USE_CUDA):
                    logit = model(x[j:j + 128].to(dev, non_blocking=True))
                out.append(logit.float().sigmoid().cpu().numpy())
        return np.concatenate(out)
    put(folder / 'configuration.json', cfg)
    for epoch in range(start, cfg['max_epochs'] + 1):
        model.train()
        order = torch.randperm(len(X['train']))
        losses = []
        t = time.time()
        progress(component, status='training', experiment=cfg.get('experiment'), candidate=cfg['name'], epoch=epoch, max_epochs=cfg['max_epochs'], best_validation_AP=best)
        for j in range(0, len(order), 128):
            ix = order[j:j + 128]
            xx = X['train'][ix].to(dev, non_blocking=True)
            yy = Y[ix.to(dev)]
            if cfg.get('noise', 0):
                xx[:, :, :768] += torch.randn_like(xx[:, :, :768]) * sd * cfg['noise']
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_CUDA):
                logits = model(xx).float()
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yy, pos_weight=torch.tensor(cfg['pw'], device=dev))
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))
            if os.environ.get('PAPER_VISIBLE_CONSOLE') == '1' and (j // 128 + 1) % 20 == 0:
                progress(component, status='training', experiment=cfg.get('experiment'), candidate=cfg['name'], epoch=epoch, max_epochs=cfg['max_epochs'], step=j // 128 + 1, steps=(len(order) + 127) // 128, train_loss=float(np.mean(losses)), best_validation_AP=best)
        pp = pred(X['val'])
        mm = metrics(y[sp['val']], pp)
        if mm['AP'] > best + 1e-06:
            best = mm['AP']
            wait = 0
            torchsave({'model_state_dict': model.state_dict(), 'epoch': epoch, 'validation': mm, 'configuration': cfg}, folder / 'best.pt')
            writepred(folder / 'validation_predictions.npz', sp['val'], y, pp, mm['threshold'])
        else:
            wait += 1
        rec = dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_AP=mm['AP'], validation_AUROC=mm['AUROC'], validation_MCC=mm['MCC'], best_validation_AP=best, wait=wait, seconds=time.time() - t)
        history.append(rec)
        pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
        torchsave({'model_state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'wait': wait, 'history': history, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, folder / 'last.pt')
        progress(component, status='training', experiment=cfg.get('experiment'), candidate=cfg['name'], max_epochs=cfg['max_epochs'], **rec)
        print('CONTEXT_EPOCH', cfg['name'], json.dumps(rec), flush=True)
        if wait >= cfg['patience']:
            break
    ck = torch.load(folder / 'best.pt', map_location='cpu')
    row = dict(name=cfg['name'], family='MDGF-TCN', kind='context', path=str(folder / 'best.pt'), embedding_path=cfg['embedding_path'], configuration=cfg, validation=ck['validation'], best_epoch=ck['epoch'], seconds=time.time() - tstart, fresh=True)
    put(folder / 'validation_record.json', row)
    del model, opt, scaler, X, Y, sd
    torch.cuda.empty_cache()
    return row


def image_eval_forward(net, im, attributes=None, return_embedding=False):
    args = (im,) if attributes is None else (im, attributes)
    kw = {'return_embedding': True} if return_embedding else {}
    with torch.cuda.amp.autocast(enabled=USE_CUDA):
        out = net(*args, **kw)
    values = out if isinstance(out, tuple) else (out,)
    if not all((torch.isfinite(v).all() for v in values)):
        with torch.cuda.amp.autocast(enabled=False):
            out = net(*(v.float() for v in args), **kw)
        values = out if isinstance(out, tuple) else (out,)
        assert all((torch.isfinite(v).all() for v in values)), 'Non-finite image forward remains in full precision'
        print('IMAGE_FP32_FALLBACK', len(im), flush=True)
    return out


def batch_images(cache, ids, aug=0.0):
    a = torch.from_numpy(np.array(cache[ids], copy=True)).permute(0, 3, 1, 2).contiguous().to(DEV, non_blocking=True).float() / 255.0
    if aug:
        gain = 1.0 + (torch.rand((len(a), 3, 1, 1), device=DEV) * 2 - 1) * aug
        a = (a * gain).clamp(0, 1)
    return (a - torch.tensor([0.485, 0.456, 0.406], device=DEV)[None, :, None, None]) / torch.tensor([0.229, 0.224, 0.225], device=DEV)[None, :, None, None]


def make_image_model(ns, cfg):
    Base = ns['C2DirectionalResNet18Top96Fusion']
    if cfg.get('variant') == 'no_gate':

        class ConcatFusion(Base):

            def forward(self, image, attributes, return_embedding=False, return_gate=False):
                v = self.image_projection(self.extract_image_feature(image))
                a = self.attribute_encoder(attributes)
                w = self.gate_network(torch.cat([v, a], dim=1))
                h = torch.cat([v, a, w], dim=1)
                logits = self.classifier(h).squeeze(1)
                if return_embedding:
                    return (logits, h)
                return logits
        model = ConcatFusion(image_dropout=cfg.get('image_dropout', 0.3), attribute_dropout=0.2, fusion_dropout=cfg.get('fusion_dropout', 0.35), direction_dropout=0.1)
        model.gate_network = nn.Sequential(*list(model.gate_network.children())[:3])
        return model
    model = Base(image_dropout=cfg.get('image_dropout', 0.3), attribute_dropout=0.2, fusion_dropout=cfg.get('fusion_dropout', 0.35), direction_dropout=0.1)
    if cfg.get('variant') == 'no_direction':
        model.direction_block1 = nn.Identity()
        model.direction_block2 = nn.Identity()
    return model


def initialize(model, ns, cfg):
    source = Path(cfg['initialization'])
    state = torch.load(source, map_location='cpu')
    if cfg.get('warm_full'):
        state = state.get('model', state.get('model_state_dict', state))
        model.load_state_dict(state, strict=True)
    else:
        state = state.get('model', state.get('model_state_dict', state))
        if any((k.startswith('backbone.') for k in state)):
            state = {k[len('backbone.'):]: v for (k, v) in state.items() if k.startswith('backbone.')}
        state = {k: v for (k, v) in state.items() if not k.startswith('fc.')}
        model.backbone.load_state_dict(state, strict=True)


def train_image(ns, cache, x, y, sp, folder, cfg):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'image_record.json').exists():
        return json.loads((folder / 'image_record.json').read_text(encoding='utf8'))
    seed(cfg['seed'])
    if cfg.get('variant') in ['no_gate', 'no_direction']:
        fullcfg = dict(cfg, variant='full')
        full = make_image_model(ns, fullcfg)
        common = copy.deepcopy(full.state_dict())
        del full
        seed(cfg['seed'])
        model = make_image_model(ns, cfg)
        current = model.state_dict()
        compatible = {k: v for (k, v) in common.items() if k in current and current[k].shape == v.shape}
        model.load_state_dict(compatible, strict=False)
    else:
        model = make_image_model(ns, cfg)
    initialize(model, ns, cfg)
    model.to(DEV)
    other = [p for (n, p) in model.named_parameters() if not n.startswith('backbone.')]
    opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': cfg['backbone_lr']}, {'params': other, 'lr': cfg['head_lr']}], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2, min_lr=1e-06)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_CUDA)
    best = -1.0
    wait = 0
    history = []
    start = 1
    tstart = time.time()
    attr = torch.tensor(x, dtype=torch.float32, device=DEV)
    if (folder / 'last.pt').exists():
        z = torch.load(folder / 'last.pt', map_location='cpu')
        model.load_state_dict(z['model_state_dict'])
        opt.load_state_dict(z['optimizer'])
        scaler.load_state_dict(z['scaler'])
        scheduler.load_state_dict(z['scheduler'])
        best = z['best']
        wait = z['wait']
        history = z['history']
        start = z['epoch'] + 1
        random.setstate(z['python_rng'])
        np.random.set_state(z['numpy_rng'])
        torch.set_rng_state(z['torch_rng'])
        restore_cuda_state(z['cuda_rng'])

    def predict(ids):
        model.eval()
        out = []
        with torch.inference_mode():
            for j in range(0, len(ids), 32):
                ii = ids[j:j + 32]
                im = batch_images(cache, ii)
                logits = image_eval_forward(model, im, None if cfg.get('image_only') else attr[ii])
                out.append(logits.float().sigmoid().cpu().numpy())
        return np.concatenate(out)
    put(folder / 'configuration.json', dict(cfg, initialization_sha256=sha(cfg['initialization']), normalization='Original ImageNet RGB normalization applied to exact engineered three-channel patch bytes', no_spatial_augmentation=True))
    for epoch in range(start, cfg['max_epochs'] + 1):
        for p in model.backbone.parameters():
            p.requires_grad = False
        if epoch >= 3:
            for part in ['layer3', 'layer4']:
                for p in getattr(model.backbone, part).parameters():
                    p.requires_grad = True
        model.train()
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        if cfg.get('freeze_direction_bn'):
            for (name, m) in model.named_modules():
                if name.startswith('direction_block') and isinstance(m, nn.BatchNorm2d):
                    m.eval()
        order = sp['train'][torch.randperm(len(sp['train'])).numpy()]
        total = 0.0
        seen = 0
        t = time.time()
        steps = (len(order) + 31) // 32
        progress('gpu', status='training', experiment=cfg['experiment'], candidate=cfg['name'], epoch=epoch, max_epochs=cfg['max_epochs'], step=0, steps=steps, best_validation_AP=best)
        for (step, j) in enumerate(range(0, len(order), 32), 1):
            ids = order[j:j + 32]
            if len(ids) < 2:
                continue
            im = batch_images(cache, ids, cfg.get('gain_aug', 0))
            yy = torch.tensor(y[ids], dtype=torch.float32, device=DEV)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_CUDA):
                logits = (model(im) if cfg.get('image_only') else model(im, attr[ids])).float()
                loss = nn.functional.binary_cross_entropy_with_logits(logits, yy, pos_weight=torch.tensor(cfg['pw'], device=DEV))
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(ids)
            seen += len(ids)
            if step % (20 if os.environ.get('PAPER_VISIBLE_CONSOLE') == '1' else 100) == 0:
                progress('gpu', status='training', experiment=cfg['experiment'], candidate=cfg['name'], epoch=epoch, max_epochs=cfg['max_epochs'], step=step, steps=steps, train_loss=total / seen, best_validation_AP=best)
                print('IMAGE_BATCH', cfg['name'], epoch, step, steps, 'seconds', round(time.time() - t, 1), flush=True)
        pp = predict(sp['val'])
        mm = metrics(y[sp['val']], pp)
        scheduler.step(mm['AP'])
        if mm['AP'] > best + 1e-06:
            best = mm['AP']
            wait = 0
            torchsave(dict(model_state_dict=model.state_dict(), configuration=cfg, epoch=epoch, validation=mm), folder / 'best.pt')
            writepred(folder / 'validation_predictions.npz', sp['val'], y, pp, mm['threshold'])
        else:
            wait += 1
        rec = dict(epoch=epoch, train_loss=total / seen, validation_AP=mm['AP'], validation_AUROC=mm['AUROC'], validation_MCC=mm['MCC'], best_validation_AP=best, wait=wait, seconds=time.time() - t)
        history.append(rec)
        pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
        torchsave(dict(model_state_dict=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch, best=best, wait=wait, history=history, python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), folder / 'last.pt')
        print('IMAGE_EPOCH', cfg['name'], json.dumps(rec), flush=True)
        if wait >= cfg['patience']:
            break
    ck = torch.load(folder / 'best.pt', map_location='cpu')
    record = dict(name=cfg['name'], family='ResNet18' if cfg.get('image_only') else 'C2', kind='image', fresh=True, path=str(folder / 'best.pt'), validation=ck['validation'], configuration=cfg, best_epoch=ck['epoch'], seconds=time.time() - tstart)
    put(folder / 'image_record.json', record)
    del model, opt, scaler, attr
    gc.collect()
    torch.cuda.empty_cache()
    return record


def embeddings(ns, cache, x, y, sp, folder, cfg, ids=None):
    folder = Path(folder)
    path = folder / 'embeddings.npy'
    wanted = np.r_[sp['train'], sp['val']] if ids is None else np.asarray(ids)
    if ids is None and path.exists():
        old = np.load(path)
        bad = ~np.isfinite(old[wanted]).all(1)
        if not bad.any():
            return path
        wanted = wanted[bad]
        del old
        print('REPAIR_NONFINITE_EMBEDDINGS', cfg['name'], len(wanted), flush=True)
    state = torch.load(folder / 'best.pt', map_location='cpu')
    net = make_image_model(ns, cfg)
    net.load_state_dict(state['model_state_dict'])
    net.to(DEV).eval()
    attr = torch.tensor(x, dtype=torch.float32, device=DEV)
    arr = np.load(path) if path.exists() else np.zeros((len(y), 768), 'float32')
    with torch.inference_mode():
        for j in range(0, len(wanted), 32):
            ii = wanted[j:j + 32]
            im = batch_images(cache, ii)
            (_, e) = image_eval_forward(net, im, attr[ii], return_embedding=True)
            assert torch.isfinite(e).all(), 'Non-finite embeddings remain in full precision'
            arr[ii] = e.float().cpu().numpy()
            if j % 3200 == 0:
                print('EMBEDDINGS', cfg['name'], j, len(wanted), flush=True)
    np.save(path, arr)
    del net, attr, arr
    gc.collect()
    torch.cuda.empty_cache()
    return path
