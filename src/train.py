import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch import nn
from .models import build_model, compute_macs
from .data import prepare, loaders


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(model, loader, device):
    model.eval()
    loss, correct, count, ys, preds, probs = 0., 0, 0, [], [], []
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            p = logits.argmax(1)
            correct += (p == y).sum().item()
            count += len(y)
            ys.extend(y.cpu().tolist()); preds.extend(p.cpu().tolist())
            probs.extend(logits.softmax(1).max(1).values.cpu().tolist())
    return {"loss":loss/count,"accuracy":correct/count,"count":count,"targets":ys,"predictions":preds,"confidence":probs}


def run(config, only=None, evaluate_only=False):
    torch.set_num_threads(6)
    device_name = config.get("device", "auto")
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    out = Path(config["output"])
    (out/"checkpoints").mkdir(parents=True, exist_ok=True)
    config = dict(config, device=device_name)
    save_json(out/"results/config.json", config)
    environment = {"python":platform.python_version(),"torch":torch.__version__,"torchvision":torchvision.__version__,
                   "numpy":np.__version__,"os":platform.platform(),"device":device_name,
                   "hardware":subprocess.getoutput("sysctl -n machdep.cpu.brand_string"),
                   "memory_bytes":subprocess.getoutput("sysctl -n hw.memsize"),
                   "determinism":"Seeds fixed; MPS kernels may not be bitwise deterministic",
                   "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').glob('*.py')}}
    save_json(out/"results/environment.json", environment)
    datasets, split, raw_test = prepare(config)
    for name in ([only] if only else config["models"]):
        random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
        if device.type == "mps":
            torch.mps.manual_seed(config["seed"])
        trainloader, valloader, testloader = loaders(datasets, config)
        model = build_model(name)
        params, macs = sum(p.numel() for p in model.parameters()), compute_macs(model)
        model.to(device)
        checkpoint = out/"checkpoints"/(name+".pt")
        history, best, best_epoch = [], -1., 0
        if not evaluate_only:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["epochs"])
            sync(device); start = time.perf_counter()
            for epoch in range(1, config["epochs"]+1):
                model.train(); total_loss, correct, count = 0., 0, 0
                tick = time.perf_counter()
                lr = optimizer.param_groups[0]['lr']
                for x, y in trainloader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite loss')
                    loss.backward(); optimizer.step()
                    total_loss += loss.item()*len(y)
                    correct += (logits.argmax(1) == y).sum().item(); count += len(y)
                val = evaluate(model, valloader, device)
                scheduler.step(); sync(device)
                row = {"epoch":epoch,"train_loss":total_loss/count,"train_accuracy":correct/count,
                       "val_loss":val["loss"],"val_accuracy":val["accuracy"],"learning_rate":lr,
                       "epoch_seconds":time.perf_counter()-tick}
                history.append(row)
                if val['accuracy'] > best:
                    best, best_epoch = val['accuracy'], epoch
                    torch.save({"model":model.state_dict(),"epoch":epoch,"config":config,"name":name}, checkpoint)
                save_json(out/"results"/(name+"_history.json"), history)
                print(json.dumps(dict(model=name, **row)), flush=True)
            sync(device); training_seconds = time.perf_counter()-start
        else:
            previous = json.loads((out/"results"/(name+"_results.json")).read_text())
            history, training_seconds = previous['history'], previous['training_seconds']
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state['model']); best_epoch = state['epoch']
        test = evaluate(model, testloader, device)
        cm = np.zeros((10, 10), dtype=int)
        np.add.at(cm, (test['targets'], test['predictions']), 1)
        x = torch.zeros(config['batch_size'], 3, 32, 32, device=device)
        with torch.inference_mode():
            for _ in range(10): model(x)
            timings = []
            for _ in range(30):
                sync(device); tick = time.perf_counter(); model(x); sync(device)
                timings.append((time.perf_counter()-tick)*1000)
        result = {"name":name,"config":config,"history":history,"parameters":params,"conv_linear_macs":macs,
                  "checkpoint_bytes":checkpoint.stat().st_size,"best_epoch":best_epoch,
                  "best_val_accuracy":history[best_epoch-1]['val_accuracy'],"training_seconds":training_seconds,
                  "inference_batch_ms_median":float(np.median(timings)),"inference_batch_ms_samples":timings,
                  "inference_ms_per_image":float(np.median(timings))/config['batch_size'],
                  "inference_protocol":"30 synchronized device-resident batches after 10 warmups; excludes preprocessing and transfer",
                  "test_accuracy":test['accuracy'],"test_loss":test['loss'],"test_count":test['count'],
                  "confusion_matrix":cm.tolist(),"class_recall":(cm.diagonal()/cm.sum(1)).tolist(),
                  "device":device_name,"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        save_json(out/"results"/(name+"_predictions.json"), test)
        save_json(out/"results"/(name+"_results.json"), result)
        print(f"FINISHED {name}: test={test['accuracy']:.4f}, training={training_seconds:.1f}s", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/full.json')
    parser.add_argument('--model', choices=['baseline','starnet','ablation'])
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.model, args.evaluate_only)
