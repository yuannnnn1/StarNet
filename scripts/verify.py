import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.models import build_model


def verify(root):
    root=Path(root); checks=[]
    split=json.loads((root/'results/split.json').read_text())
    tr,va=set(split['train_indices']),set(split['validation_indices'])
    assert len(tr)==45000 and len(va)==5000 and not tr & va and tr|va==set(range(50000))
    checks.append('45000/5000 exhaustive nonoverlapping training/validation indices')
    a,b=build_model('starnet'),build_model('ablation')
    assert {k:tuple(v.shape) for k,v in a.state_dict().items()}=={k:tuple(v.shape) for k,v in b.state_dict().items()}
    b.load_state_dict(a.state_dict())
    a.eval(); b.eval()
    with torch.no_grad():
        x=torch.randn(2,3,32,32)
        assert a(x).shape==(2,10) and not torch.allclose(a(x),b(x))
    checks.append('Ablation has identical parameter structure and changes forward behavior')
    for name in ['baseline','starnet','ablation']:
        result=json.loads((root/f'results/{name}_results.json').read_text())
        pred=json.loads((root/f'results/{name}_predictions.json').read_text())
        cm=np.zeros((10,10),dtype=int)
        np.add.at(cm,(pred['targets'],pred['predictions']),1)
        assert np.array_equal(cm,result['confusion_matrix'])
        assert cm.sum()==result['test_count']
        assert abs(cm.trace()/cm.sum()-result['test_accuracy'])<1e-12
        assert len(result['history'])==result['config']['epochs']
        assert result['best_val_accuracy']==max(row['val_accuracy'] for row in result['history'])
        checkpoint=root/f'checkpoints/{name}.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==result['checkpoint_sha256']
        model=build_model(name)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
        model.eval()
        with torch.no_grad(): assert torch.isfinite(model(x)).all()
        assert sum(p.numel() for p in model.parameters())==result['parameters']
        checks.append(f'{name}: finite CPU forward; checkpoint hash/load; count, accuracy, confusion matrix and best epoch verified')
    dash=json.loads((root/'results/dashboard.json').read_text())
    for r in dash['runs']:
        assert r==json.loads((root/f'results/{r["name"]}_results.json').read_text())
    for f in root.glob('figures/*.png'): assert f.stat().st_size>1000
    checks.append('Dashboard equals raw results; generated figure files exist')
    (root/'results/verification.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2))
    print('\n'.join(checks))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',default='artifacts')
    verify(parser.parse_args().root)
