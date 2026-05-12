from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def get_optimizer(name: str, parameters, lr: float, weight_decay: float = 0.0) -> Optimizer:
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Neznámy optimalizátor: {name}")

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        for t, p in zip(y.view(-1).cpu(), preds.view(-1).cpu()):
            cm[t.item(), p.item()] += 1

    cm_np = cm.numpy()

    eps = 1e-12
    tp = cm_np.diagonal()
    fp = cm_np.sum(axis=0) - tp
    fn = cm_np.sum(axis=1) - tp
    support = cm_np.sum(axis=1)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    per_class_acc = tp / (support + eps)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "confusion_matrix": cm_np.tolist(),
        "per_class_accuracy": per_class_acc.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }

def collect_sample_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 16,
) -> List[Dict]:
    model.eval()
    samples: List[Dict] = []

    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            logits = model(x_dev)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            for i in range(x.size(0)):
                if len(samples) >= num_samples:
                    return samples

                img = x[i].cpu().numpy()
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min)

                samples.append(
                    {
                        "image": img.tolist(),
                        "true_label": int(y[i].item()),
                        "pred_label": int(preds[i].item()),
                        "confidence": float(probs[i, preds[i]].item()),
                    }
                )
            if len(samples) >= num_samples:
                break

    return samples
