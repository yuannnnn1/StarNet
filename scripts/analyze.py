"""Produce all charts and dashboard data exclusively from saved runs."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10

NAMES = {'baseline':'Conventional CNN','starnet':'StarNet','ablation':'Additive ablation'}
COLORS = {'baseline':'#326ca8','starnet':'#19866d','ablation':'#bd5266'}


def analyze(root):
    root = Path(root); results = root/'results'; figures = root/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    runs = [json.loads((results/f'{n}_results.json').read_text()) for n in NAMES]
    split = json.loads((results/'split.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    for metric in ['loss','accuracy']:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout='constrained')
        for ax, phase in zip(axes, ['train','val']):
            for run in runs:
                ax.plot([r['epoch'] for r in run['history']], [r[f'{phase}_{metric}'] for r in run['history']],
                        label=NAMES[run['name']], color=COLORS[run['name']])
            ax.set(xlabel='Epoch',ylabel=metric.capitalize(),title={'train':'Training (augmented)','val':'Validation'}[phase])
            ax.grid(alpha=.18); ax.legend(fontsize=8)
        fig.savefig(figures/f'training_{metric}.png'); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,key,label in zip(axes,['test_accuracy','parameters','training_seconds'],['Test accuracy','Parameters','Training seconds']):
        ax.bar([NAMES[r['name']] for r in runs],[r[key] for r in runs],color=list(COLORS.values()))
        ax.set_title(label); ax.tick_params(axis='x',labelrotation=20,labelsize=8)
    fig.savefig(figures/'model_comparison.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,3.3),layout='constrained')
    ax.bar(['Star operation','Addition'],[runs[1]['test_accuracy']*100,runs[2]['test_accuracy']*100],color=[COLORS['starnet'],COLORS['ablation']])
    ax.set(ylabel='Test accuracy (%)',ylim=(0,100),title='Only branch combination changes')
    fig.savefig(figures/'ablation_comparison.png'); plt.close(fig)
    test = CIFAR10('data',train=False,download=False)
    examples = {}
    sample_dir = figures/'samples'; sample_dir.mkdir(exist_ok=True)
    for run in runs:
        name = run['name']; cm = np.array(run['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(7,6),layout='constrained')
        ax.imshow(cm,cmap='Blues')
        for i in range(10):
            for j in range(10):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set(xticks=range(10),yticks=range(10),xticklabels=split['classes'],yticklabels=split['classes'],xlabel='Predicted label',ylabel='True label',title=NAMES[name]+' test confusion matrix')
        plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
        fig.savefig(figures/f'confusion_matrix_{name}.png'); plt.close(fig)
        preds = json.loads((results/f'{name}_predictions.json').read_text())
        correct = [i for i,(a,b) in enumerate(zip(preds['targets'],preds['predictions'])) if a==b][:6]
        incorrect = [i for i,(a,b) in enumerate(zip(preds['targets'],preds['predictions'])) if a!=b][:6]
        chosen = correct+incorrect
        examples[name]=[]
        fig, axes = plt.subplots(2,6,figsize=(11,4),layout='constrained')
        for ax in axes.flat: ax.axis('off')
        for ax,i in zip(axes.flat,chosen):
            true,pred=preds['targets'][i],preds['predictions'][i]
            ax.imshow(test.data[i]); ax.set_title(f'T: {split["classes"][true]}\nP: {split["classes"][pred]}',fontsize=8)
            filename=f'samples/test_{i}.png'
            Image.fromarray(test.data[i]).save(figures/filename)
            examples[name].append({'index':i,'true':split['classes'][true],'predicted':split['classes'][pred],
                                   'correct':true==pred,'confidence':preds['confidence'][i],'image':filename})
        fig.savefig(figures/f'prediction_examples_{name}.png'); plt.close(fig)
    fig,axes=plt.subplots(2,5,figsize=(9,4),layout='constrained')
    train=CIFAR10('data',train=True,download=False)
    for c,ax in enumerate(axes.flat):
        idx=next(i for i in split['train_indices'] if train.targets[i]==c)
        ax.imshow(train.data[idx]); ax.set_title(split['classes'][c]); ax.axis('off')
    fig.savefig(figures/'dataset_samples.png'); plt.close(fig)
    fields=['name','test_accuracy','test_loss','best_val_accuracy','best_epoch','parameters','conv_linear_macs','training_seconds','inference_ms_per_image','checkpoint_bytes','device']
    with (results/'experiment_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in runs)
    dashboard={'runs':runs,'examples':examples,'dataset':{k:v for k,v in split.items() if 'indices' not in k},
               'environment':json.loads((results/'environment.json').read_text()),
               'pilot':'pilot' in str(root),'example_rule':'First six correct and first six incorrect in official test order'}
    (results/'dashboard.json').write_text(json.dumps(dashboard,indent=2))
    print(f'Generated {len(list(figures.glob("*.png")))} figures and dashboard data in {root}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',default='artifacts')
    analyze(parser.parse_args().root)
