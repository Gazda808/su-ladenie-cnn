from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from src import datasets, experiments, utils
from src.models import ConfigurableCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
WEB_DIR = PROJECT_ROOT / "web"

def run_single_experiment(
    cfg: Dict[str, Any],
    device: torch.device,
    quick: bool = False,
    epochs_override: int | None = None,
) -> Dict[str, Any]:
    dataset_name = cfg["dataset"]
    info = datasets.get_info(dataset_name)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    if quick:
        subset = 3000 if dataset_name != "cifar10" else 2400
        test_subset = 800
        epochs = 2
    else:
        subset = 0
        test_subset = 0
        epochs = train_cfg["epochs"]

    if epochs_override is not None:
        epochs = epochs_override

    print(f"\n[{cfg['id']}] {cfg['name']}  ({info['name']})")
    print(f"    model={model_cfg}")
    print(f"    train={train_cfg}  epochs={epochs}  quick={quick}")

    train_loader, test_loader = datasets.load_dataset(
        dataset_name,
        batch_size=train_cfg["batch_size"],
        subset_size=subset,
        test_subset_size=test_subset,
    )

    model = ConfigurableCNN(
        in_channels=info["in_channels"],
        input_size=info["input_size"],
        num_classes=info["num_classes"],
        **model_cfg,
    ).to(device)

    optimizer = utils.get_optimizer(
        train_cfg["optimizer"],
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, float]] = []
    t_start = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = utils.train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        eval_metrics = utils.evaluate(
            model, test_loader, criterion, device, info["num_classes"]
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": eval_metrics["loss"],
            "val_accuracy": eval_metrics["accuracy"],
        })
        print(
            f"    epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={eval_metrics['loss']:.4f} val_acc={eval_metrics['accuracy']:.4f}"
        )

    t_total = time.time() - t_start

    final_metrics = utils.evaluate(
        model, test_loader, criterion, device, info["num_classes"]
    )
    samples = utils.collect_sample_predictions(model, test_loader, device, num_samples=12)

    return {
        "id": cfg["id"],
        "name": cfg["name"],
        "category": cfg["category"],
        "description": cfg["description"],
        "dataset": dataset_name,
        "dataset_info": info,
        "model_config": model_cfg,
        "training_config": train_cfg,
        "epochs_run": epochs,
        "num_parameters": model.num_parameters(),
        "architecture": model.architecture_summary(),
        "training_time_sec": t_total,
        "history": history,
        "final": {
            "test_accuracy": final_metrics["accuracy"],
            "test_loss": final_metrics["loss"],
            "precision_macro": final_metrics["precision_macro"],
            "recall_macro": final_metrics["recall_macro"],
            "f1_macro": final_metrics["f1_macro"],
            "confusion_matrix": final_metrics["confusion_matrix"],
            "per_class_accuracy": final_metrics["per_class_accuracy"],
            "per_class_precision": final_metrics["per_class_precision"],
            "per_class_recall": final_metrics["per_class_recall"],
            "per_class_f1": final_metrics["per_class_f1"],
        },
        "samples": samples,
    }

def save_results(all_results: List[Dict[str, Any]], merge: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "results.json"

    if merge and json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_by_id = {e["id"]: e for e in existing.get("experiments", [])}
            for r in all_results:
                existing_by_id[r["id"]] = r
            all_results = list(existing_by_id.values())
            print(f"[MERGE] Spojené, celkovo {len(all_results)} experimentov.")
        except Exception as exc:
            print(f"[MERGE] Zlyhalo načítanie existujúcich výsledkov: {exc}")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "categories": experiments.get_categories(),
        "experiments": all_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Uložené: {json_path}")

    js_path = WEB_DIR / "results.js"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("// Automaticky generované – nemeniť ručne.\n")
        f.write("window.RESULTS = ")
        json.dump(payload, f, ensure_ascii=False)
        f.write(";\n")
    print(f"[OK] Uložené: {js_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ladenie CNN – tréning experimentov.")
    parser.add_argument("--quick", action="store_true",
                        help="Rýchly demo režim (menej dát a epoch).")
    parser.add_argument("--filter", type=str, default="",
                        help="Spustí iba experimenty, ktorých id začína týmto prefixom.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Prepíše počet epoch pre všetky experimenty.")
    parser.add_argument("--threads", type=int, default=4,
                        help="Počet PyTorch CPU threadov.")
    parser.add_argument("--merge", action="store_true",
                        help="Pripojí výsledky k existujúcim namiesto prepísania.")
    return parser.parse_args()

# Príklady spustenia:
#   python -m src.train
#   python -m src.train --quick
#   python -m src.train --filter topo_ --merge
#   python -m src.train --epochs 10 --merge

def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Zariadenie: {device}  (threads={torch.get_num_threads()})")

    all_exps = experiments.build_experiment_list()
    if args.filter:
        all_exps = [e for e in all_exps if e["id"].startswith(args.filter)]
        print(f"Filter '{args.filter}': vybraných {len(all_exps)} experimentov.")

    print(f"Plán: {len(all_exps)} experimentov.")
    results: List[Dict[str, Any]] = []
    try:
        for i, cfg in enumerate(all_exps, 1):
            print(f"\n--- {i}/{len(all_exps)} ---")
            try:
                res = run_single_experiment(
                    cfg, device, quick=args.quick, epochs_override=args.epochs
                )
                results.append(res)
            except Exception as exc:
                print(f"[CHYBA] {cfg['id']}: {exc}")
                results.append({
                    "id": cfg["id"],
                    "name": cfg["name"],
                    "error": str(exc),
                    "category": cfg["category"],
                    "dataset": cfg["dataset"],
                })
    except KeyboardInterrupt:
        print("\n\n[PRERUSENIE] Proces zastaveny pouzivatelom (Ctrl+C).")
        print(f"[INFO] Bolo spustene {len(results)}/{len(all_exps)} experimentov.")
        if results:
            print(f"[INFO] Ukladam {len(results)} rezultatov...")
            save_results(results, merge=True)
            print("[OK] Rezultaty ulozene. Pokračuj s --merge:")
            print(f"     python -m src.train {' '.join(f'--{k} {v}' for k, v in vars(args).items() if v and k != 'filter')} --filter {args.filter or 'prefix_'}")
        return

    save_results(results, merge=args.merge)
    print("\nHotovo.")

if __name__ == "__main__":
    main()
