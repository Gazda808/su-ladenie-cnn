from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

DATASET_INFO = {
    "mnist": {
        "name": "MNIST",
        "in_channels": 1,
        "input_size": 28,
        "num_classes": 10,
        "classes": [str(i) for i in range(10)],
        "description": "Ručne písané číslice 0–9 (28×28 šedotónových).",
    },
    "fashion_mnist": {
        "name": "Fashion-MNIST",
        "in_channels": 1,
        "input_size": 28,
        "num_classes": 10,
        "classes": [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
        "description": "10 kategórií oblečenia (28×28 šedotónových).",
    },
    "cifar10": {
        "name": "CIFAR-10",
        "in_channels": 3,
        "input_size": 32,
        "num_classes": 10,
        "classes": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        "description": "Farebné prírodné obrázky 10 tried (32×32 RGB).",
    },
}

def _build_transform(name: str) -> transforms.Compose:
    if name in ("mnist", "fashion_mnist"):
        mean = (0.1307,) if name == "mnist" else (0.2860,)
        std = (0.3081,) if name == "mnist" else (0.3530,)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    if name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])
    raise ValueError(f"Neznáma dátová sada: {name}")

def _load_mnist_offline() -> Tuple[torch.Tensor, torch.Tensor]:
    from mlxtend.data import mnist_data
    X, y = mnist_data()
    X = X.astype(np.float32) / 255.0
    X = X.reshape(-1, 28, 28)
    X = (X - 0.1307) / 0.3081
    rng = np.random.RandomState(123)
    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]
    images = torch.from_numpy(X).unsqueeze(1)
    labels = torch.from_numpy(y.astype(np.int64))
    return images, labels

def _generate_synthetic_dataset(
    in_channels: int,
    input_size: int,
    num_classes: int,
    samples_per_class: int,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    total = num_classes * samples_per_class
    images = np.zeros((total, in_channels, input_size, input_size), dtype=np.float32)
    labels = np.zeros((total,), dtype=np.int64)

    for cls in range(num_classes):
        for j in range(samples_per_class):
            idx = cls * samples_per_class + j
            img = np.zeros((in_channels, input_size, input_size), dtype=np.float32)
            mode = cls
            for c in range(in_channels):
                base = rng.uniform(0.0, 0.05, size=(input_size, input_size))
                if mode % 5 == 0:
                    freq = (mode // 5) + 2
                    for r in range(input_size):
                        if (r // freq) % 2 == 0:
                            base[r, :] += 0.6
                elif mode % 5 == 1:
                    freq = (mode // 5) + 2
                    for col in range(input_size):
                        if (col // freq) % 2 == 0:
                            base[:, col] += 0.6
                elif mode % 5 == 2:
                    for r in range(input_size):
                        for col in range(input_size):
                            if abs(r - col) <= 2:
                                base[r, col] += 0.7
                elif mode % 5 == 3:
                    cy, cx = input_size // 2, input_size // 2
                    rad = 4 + (mode // 5) * 2
                    for r in range(input_size):
                        for col in range(input_size):
                            if (r - cy) ** 2 + (col - cx) ** 2 <= rad * rad:
                                base[r, col] += 0.7
                else:
                    for r in range(input_size):
                        base[r, :] += (r / input_size) * 0.7

                base = np.roll(base, c * 2, axis=0)
                base += rng.normal(0.0, 0.1, size=base.shape)
                img[c] = base

            images[idx] = img
            labels[idx] = cls

    perm = rng.permutation(total)
    return torch.from_numpy(images[perm]), torch.from_numpy(labels[perm])

def _load_offline(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if name == "mnist":
        return _load_mnist_offline()
    if name == "fashion_mnist":
        return _generate_synthetic_dataset(1, 28, 10, samples_per_class=400, seed=11)
    if name == "cifar10":
        return _generate_synthetic_dataset(3, 32, 10, samples_per_class=300, seed=22)
    raise ValueError(name)

def _split_train_test(
    images: torch.Tensor,
    labels: torch.Tensor,
    test_fraction: float = 0.2,
) -> Tuple[Dataset, Dataset]:
    n = images.size(0)
    n_test = int(n * test_fraction)
    return (
        TensorDataset(images[n_test:], labels[n_test:]),
        TensorDataset(images[:n_test], labels[:n_test]),
    )

def _try_torchvision(name: str):
    transform = _build_transform(name)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if name == "mnist":
            train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
            test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
        elif name == "fashion_mnist":
            train = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
            test = datasets.FashionMNIST(DATA_ROOT, train=False, download=True, transform=transform)
        elif name == "cifar10":
            train = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
            test = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
        else:
            return None
        return train, test
    except Exception as exc:
        print(f"  [WARN] torchvision download zlyhal ({name}): {exc}")
        return None

def load_dataset(
    name: str,
    batch_size: int = 64,
    subset_size: int = 0,
    test_subset_size: int = 0,
    num_workers: int = 0,
    prefer_offline: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_set: Dataset | None = None
    test_set: Dataset | None = None
    mode = "online"

    if not prefer_offline:
        result = _try_torchvision(name)
        if result is not None:
            train_set, test_set = result

    if train_set is None or test_set is None:
        print(f"  [INFO] OFFLINE režim pre {name}.")
        images, labels = _load_offline(name)
        train_set, test_set = _split_train_test(images, labels)
        mode = "offline"

    if subset_size > 0 and len(train_set) > subset_size:
        train_set = Subset(train_set, list(range(subset_size)))
    if test_subset_size > 0 and len(test_set) > test_subset_size:
        test_set = Subset(test_set, list(range(test_subset_size)))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    train_loader.dataset_mode = mode
    return train_loader, test_loader

def get_info(name: str) -> dict:
    if name not in DATASET_INFO:
        raise ValueError(f"Neznáma dátová sada: {name}")
    return DATASET_INFO[name]
