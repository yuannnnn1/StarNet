"""Additional protocol checks without changing training or saved predictions."""
import json
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from src.models import build_model


def main():
    split=json.loads(Path('artifacts/results/split.json').read_text())
    train=CIFAR10('data',train=True,download=False)
    test=CIFAR10('data',train=False,download=False)
    targets=np.array(train.targets)
    counts={k:np.bincount(targets[split[k]],minlength=10).tolist() for k in ['train_indices','validation_indices']}
    assert counts['train_indices']==[4500]*10
    assert counts['validation_indices']==[500]*10
    assert np.bincount(test.targets,minlength=10).tolist()==[1000]*10
    torch.manual_seed(3024); star=build_model('starnet')
    torch.manual_seed(3024); additive=build_model('ablation')
    assert all(torch.equal(v,additive.state_dict()[k]) for k,v in star.state_dict().items())
    for name in ['baseline','starnet','ablation']:
        model=build_model(name)
        logits=model(torch.randn(2,3,32,32)); loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,9]))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    result={'passed':True,'per_class_counts':counts,'test_per_class':[1000]*10,
            'star_addition_initial_weights_identical_at_seed_3024':True,
            'all_models_finite_cpu_forward_backward_gradients':True,
            'pilot_test_exposure':'First 512 official test images, for pipeline sanity checks only'}
    Path('artifacts/results/protocol_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
