# Ladenie neurónovej siete CNN

**Praktické zadanie zo Strojového učenia · 2026**

Interaktívny framework na systematické ladenie hyperparametrov konvolučnej neurónovej siete (CNN) pomocou ablačných experimentov. Výsledky sú vizualizované v prehľadnom webovom rozhraní.

---

## Obsah

- [Cieľ zadania](#cieľ-zadania)
- [Experimenty](#experimenty)
- [Architektúra modelu](#architektúra-modelu)
- [Mierky efektívnosti](#mierky-efektívnosti)
- [Inštalácia](#inštalácia)
- [Spustenie tréningu](#spustenie-tréningu)
- [Webové rozhranie](#webové-rozhranie)
- [Štruktúra projektu](#štruktúra-projektu)
- [Technologický stack](#technologický-stack)

---

## Cieľ zadania

Preskúmať vplyv jednotlivých hyperparametrov a architektonických rozhodnutí na kvalitu CNN. V každej skupine experimentov sa mení iba **jeden parameter** pri zachovaní ostatných (ablačná metóda).

---

## Experimenty

Celkovo **34 experimentov** rozdelených do 7 kategórií na troch dátových sadách (MNIST, Fashion-MNIST, CIFAR-10):

| Kategória | Variácie | Dátová sada |
|---|---|---|
| **Topológia siete** | 1–3 bloky, 16–128 filtrov | MNIST |
| **Optimalizátor** | SGD, Adam, AdamW, RMSprop | MNIST |
| **Rýchlosť učenia** | 0.0001, 0.001, 0.01, 0.1 | MNIST |
| **Aktivačná funkcia** | ReLU, LeakyReLU, ELU, GELU | MNIST |
| **Regularizácia** | Dropout (0.25/0.5), BN, Weight decay, kombinácie | MNIST |
| **Veľkosť jadra** | 3×3, 5×5, 7×7 | MNIST |
| **Veľkosť dávky** | 32, 64, 128, 256 | MNIST |
| **Multiset** | Baseline + najlepšia konfig | Fashion-MNIST, CIFAR-10 |

---

## Architektúra modelu

Flexibilná trieda `ConfigurableCNN` (`src/models.py`):

```
Vstup → [Conv2d → BN? → Aktivácia → Pool → Dropout?] × N
       → Flatten → Linear → Aktivácia → Dropout? → Linear → Logit
```

Konfigurovateľné parametre: počet a šírka blokov, veľkosť jadra, aktivačná funkcia, batch normalizácia, pooling, dropout, veľkosť FC vrstvy.

---

## Mierky efektívnosti

Pre každý experiment sú zaznamenané:

- **Accuracy** — podiel správnych predikcií na testovacej sade
- **Cross-entropy loss** — sledovaná na trénovacej aj validačnej sade po každej epoche
- **Macro Precision / Recall / F1** — priemer cez všetky triedy bez váhy počtom vzoriek
- **Matica zámen** — vizualizácia chybných klasifikácií pre každý pár tried
- **Per-class Accuracy a F1** — identifikácia problematických tried
- **Krivky učenia** — loss a accuracy v každej epoche
- **Počet parametrov** a **čas trénovania**

---

## Inštalácia

```bash
# Klon repozitára
git clone https://github.com/Gazda808/su-ladenie-cnn.git
cd ladenie-cnn

# Vytvorenie virtuálneho prostredia (odporúčané)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Inštalácia závislostí
pip install torch torchvision numpy
```

> Dátové sady (MNIST, Fashion-MNIST, CIFAR-10) sa stiahnu automaticky pri prvom spustení cez torchvision.

---

## Spustenie tréningu

```bash
# Rýchle demo — menšia podmnožina dát, 2 epochy
python -m src.train --quick

# Plný tréning — všetky experimenty, celé dáta
python -m src.train

# Iba experimenty z jednej kategórie (napr. optimalizátory)
python -m src.train --filter opt_ --merge

# Iba regularizačné experimenty
python -m src.train --filter reg_ --merge

# Vlastný počet epoch
python -m src.train --epochs 10 --merge

# Výber konkrétneho experimentu
python -m src.train --filter topo_deep --merge
```

Po dokončení sa výsledky uložia do:
- `results/results.json` — surové dáta
- `web/results.js` — pre webové rozhranie

---

## Webové rozhranie

Otvor súbor `web/index.html` priamo v prehliadači (nevyžaduje server).

Päť interaktívnych pohľadov:

| Pohľad | Popis |
|---|---|
| **Prehľad** | Filtrovateľná tabuľka a stĺpcový graf všetkých experimentov |
| **Porovnanie** | Krivky učenia pre až 6 experimentov naraz s rôznymi štýlmi čiar |
| **Detail** | Matica zámen, presnosť po triedach, ukážkové predikcie, hyperparametre |
| **Konfigurátor** | Odhad výkonu novej konfigurácie pomocou kNN interpolácie |
| **O zadaní** | Popis metodológie, architektúry a fungovania systému |

---

## Štruktúra projektu

```
ladenie-cnn/
├── src/
│   ├── __init__.py
│   ├── datasets.py       # načítanie MNIST / Fashion-MNIST / CIFAR-10
│   ├── experiments.py    # definícia všetkých 34 experimentov
│   ├── models.py         # ConfigurableCNN – konfigurovateľná CNN
│   ├── train.py          # hlavný trénovací skript s CLI
│   └── utils.py          # optimalizátory, evaluácia, výpočet metrík
├── web/
│   ├── index.html        # SPA webové rozhranie
│   ├── app.js            # interaktívna logika (5 pohľadov)
│   ├── style.css         # dizajn
│   └── results.js        # auto-generované výsledky (git-ignored voliteľne)
├── results/
│   └── results.json      # surové výsledky experimentov
├── .gitignore
└── README.md
```

---

## Technologický stack

| Vrstva | Technológia |
|---|---|
| Modely a tréning | PyTorch 2.x |
| Dátové sady | torchvision |
| Metriky | NumPy |
| Vizualizácia | HTML5 / CSS3 / Vanilla JS |
| Grafy | Chart.js 4 |

---

*SU · Ladenie neurónovej siete CNN · 2026*
