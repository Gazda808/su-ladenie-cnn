from copy import deepcopy
from typing import Any, Dict, List

BASELINE: Dict[str, Any] = {
    "model": {
        "filters": [32, 64],
        "kernel_size": 3,
        "activation": "relu",
        "use_batch_norm": False,
        "pooling": "max",
        "dropout": 0.0,
        "fc_units": 128,
    },
    "training": {
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 4,
        "batch_size": 64,
        "weight_decay": 0.0,
    },
}

def _make(
    exp_id: str,
    name: str,
    category: str,
    description: str,
    dataset: str,
    model_overrides: Dict[str, Any] | None = None,
    training_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = deepcopy(BASELINE)
    if model_overrides:
        cfg["model"].update(model_overrides)
    if training_overrides:
        cfg["training"].update(training_overrides)
    return {
        "id": exp_id,
        "name": name,
        "category": category,
        "description": description,
        "dataset": dataset,
        "model": cfg["model"],
        "training": cfg["training"],
    }

def build_experiment_list() -> List[Dict[str, Any]]:
    experiments: List[Dict[str, Any]] = []

    experiments += [
        _make(
            "topo_tiny", "Malá sieť (1 blok)", "topology",
            "Iba jeden konvolučný blok s 16 filtrami – minimálna kapacita.",
            "mnist",
            model_overrides={"filters": [16]},
        ),
        _make(
            "topo_small", "Malá sieť (2 bloky, 16-32)", "topology",
            "Dva konvolučné bloky s nízkym počtom filtrov.",
            "mnist",
            model_overrides={"filters": [16, 32]},
        ),
        _make(
            "topo_baseline", "Stredná sieť (2 bloky, 32-64)", "topology",
            "Baseline topológia – dva bloky 32 a 64 filtrov.",
            "mnist",
        ),
        _make(
            "topo_wide", "Široká sieť (2 bloky, 64-128)", "topology",
            "Rovnaká hĺbka ako baseline, ale dvojnásobná šírka.",
            "mnist",
            model_overrides={"filters": [64, 128]},
        ),
        _make(
            "topo_deep", "Hlboká sieť (3 bloky, 32-64-128)", "topology",
            "Tri konvolučné bloky – väčšia hĺbka aj kapacita.",
            "mnist",
            model_overrides={"filters": [32, 64, 128]},
        ),
    ]

    for opt_name, opt_desc in [
        ("sgd", "Klasický stochastický gradientný zostup."),
        ("adam", "Adaptívny optimalizátor s momentom 1. a 2. rádu."),
        ("adamw", "Variant Adam s odpojeným weight decay."),
        ("rmsprop", "Adaptívny optimalizátor s pohyblivým priemerom druhých momentov."),
    ]:
        experiments.append(
            _make(
                f"opt_{opt_name}", f"Optimalizátor: {opt_name.upper()}",
                "optimizer",
                opt_desc,
                "mnist",
                training_overrides={"optimizer": opt_name},
            )
        )

    for lr in [0.0001, 0.001, 0.01, 0.1]:
        experiments.append(
            _make(
                f"lr_{lr}", f"Learning rate {lr}",
                "lr",
                f"Vplyv rýchlosti učenia ({lr}) na konvergenciu siete.",
                "mnist",
                training_overrides={"lr": lr},
            )
        )

    for act, desc in [
        ("relu", "Štandardná ReLU – ostrý prah v nule."),
        ("leaky_relu", "LeakyReLU – malý sklon pre záporné hodnoty (0.1)."),
        ("elu", "ELU – exponenciálna pre záporné hodnoty, hladší prechod."),
        ("gelu", "GELU – pravdepodobnostná aproximácia, používaná v Transformers."),
    ]:
        experiments.append(
            _make(
                f"act_{act}", f"Aktivácia: {act}",
                "activation",
                desc,
                "mnist",
                model_overrides={"activation": act},
            )
        )

    experiments += [
        _make(
            "reg_none", "Bez regularizácie", "regularization",
            "Baseline – žiadny dropout ani batch normalization.",
            "mnist",
        ),
        _make(
            "reg_dropout25", "Dropout 0.25", "regularization",
            "Dropout 25% v konvolučných aj FC vrstvách.",
            "mnist",
            model_overrides={"dropout": 0.25},
        ),
        _make(
            "reg_dropout50", "Dropout 0.50", "regularization",
            "Agresívnejší dropout 50% – silnejšia regularizácia.",
            "mnist",
            model_overrides={"dropout": 0.5},
        ),
        _make(
            "reg_bn", "Batch normalization", "regularization",
            "Batch normalization po každej konvolúcii.",
            "mnist",
            model_overrides={"use_batch_norm": True},
        ),
        _make(
            "reg_bn_dropout", "BN + Dropout 0.25", "regularization",
            "Kombinácia batch norm a dropoutu.",
            "mnist",
            model_overrides={"use_batch_norm": True, "dropout": 0.25},
        ),
        _make(
            "reg_wd", "Weight decay 1e-3", "regularization",
            "L2 regularizácia cez weight decay v optimalizátore.",
            "mnist",
            training_overrides={"weight_decay": 1e-3},
        ),
    ]

    for k in [3, 5, 7]:
        experiments.append(
            _make(
                f"kernel_{k}", f"Kernel {k}×{k}",
                "kernel",
                f"Veľkosť konvolučného jadra {k}×{k}.",
                "mnist",
                model_overrides={"kernel_size": k},
            )
        )

    for bs in [32, 64, 128, 256]:
        experiments.append(
            _make(
                f"bs_{bs}", f"Batch size {bs}",
                "batch_size",
                f"Vplyv veľkosti dávky ({bs}) na trénovanie.",
                "mnist",
                training_overrides={"batch_size": bs},
            )
        )

    experiments += [
        _make(
            "fashion_baseline", "Fashion-MNIST: baseline", "topology",
            "Baseline architektúra na Fashion-MNIST.",
            "fashion_mnist",
        ),
        _make(
            "fashion_bn", "Fashion-MNIST: BN + Dropout", "regularization",
            "Najlepšia regularizácia aplikovaná na Fashion-MNIST.",
            "fashion_mnist",
            model_overrides={"use_batch_norm": True, "dropout": 0.25},
        ),
        _make(
            "cifar_baseline", "CIFAR-10: baseline", "topology",
            "Baseline architektúra na CIFAR-10 (3 kanály, 32×32).",
            "cifar10",
        ),
        _make(
            "cifar_deep", "CIFAR-10: hlboká + BN", "topology",
            "Hlboká sieť s BN na CIFAR-10.",
            "cifar10",
            model_overrides={
                "filters": [32, 64, 128],
                "use_batch_norm": True,
                "dropout": 0.25,
            },
        ),
    ]

    return experiments

def get_categories() -> Dict[str, str]:
    return {
        "topology": "Topológia siete",
        "optimizer": "Optimalizátor",
        "lr": "Rýchlosť učenia",
        "activation": "Aktivačná funkcia",
        "regularization": "Regularizácia",
        "kernel": "Veľkosť jadra",
        "batch_size": "Veľkosť dávky",
    }
