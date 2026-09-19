import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


class Images(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images, self.labels, self.indices, self.transform = images, labels, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.transform(self.images[idx]), int(self.labels[idx])


def prepare(config):
    train = CIFAR10("data", train=True, download=True)
    test = CIFAR10("data", train=False, download=True)
    labels = np.array(train.targets)
    rng = np.random.default_rng(config["seed"])
    train_ids, val_ids = [], []
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(labels == c))
        val_ids.extend(ids[:500].tolist())
        train_ids.extend(ids[500:].tolist())
    train_ids = rng.permutation(train_ids).tolist()
    val_ids = rng.permutation(val_ids).tolist()
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(train_ids) == 45000 and len(val_ids) == 5000
    pixels = train.data[train_ids].astype(np.float32) / 255
    mean = pixels.mean(axis=(0, 1, 2), dtype=np.float64).tolist()
    std = pixels.std(axis=(0, 1, 2), dtype=np.float64).tolist()
    del pixels
    norm = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    augment = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), *norm])
    plain = transforms.Compose(norm)
    ids_test = list(range(len(test)))
    datasets = [Images(train.data, labels, train_ids[:config["train_limit"]], augment),
                Images(train.data, labels, val_ids[:config["val_limit"]], plain),
                Images(test.data, test.targets, ids_test[:config["test_limit"]], plain)]
    split = {"seed": config["seed"], "train_indices": train_ids, "validation_indices": val_ids,
             "train_count":len(datasets[0]),"validation_count":len(datasets[1]),"test_count":len(datasets[2]),
             "mean":mean,"std":std,"classes":train.classes,"normalization_source":"45000 training images only"}
    out = Path(config["output"])/"results"
    out.mkdir(parents=True, exist_ok=True)
    (out/"split.json").write_text(json.dumps(split))
    return datasets, split, test


def loaders(datasets, config):
    generator = torch.Generator().manual_seed(config["seed"])
    return [DataLoader(ds, batch_size=config["batch_size"], shuffle=i == 0,
                       num_workers=0, generator=generator if i == 0 else None) for i, ds in enumerate(datasets)]
