# StarNet for CIFAR-10

CISC3024 Pattern Recognition, AI Assignment 1. An AI-implemented adaptation of
**Rewrite the Stars (CVPR 2024)** with a comparable conventional CNN and a controlled
addition ablation. This is not a reproduction of the original ImageNet experiments.

Repository webpage supplied by the student: **https://github.com/yuannnnn1/StarNet**.
Uploading this local project is deferred at the student's request. The webpage is
reachable, but local results are not claimed to have been published there.

## Deliverables

- `artifacts/report/StarNet_Report.docx` and `StarNet_Report.md`: six-section report.
- `website/`: data-driven dashboard with all eight required topic sections.
- `artifacts/results/`: configurations, environment, split indices, raw histories,
  per-image predictions, model metrics, summary CSV and verification evidence.
- `artifacts/figures/`: curves, comparisons, confusion matrices and prediction examples.
- `artifacts/checkpoints/`: local best-validation model states, excluded from Git.
- `artifacts/pilot/`: separate short pipeline checks, never final experiment results.
- `docs/research.md`, `docs/pilot.md`, `docs/final_audit.md`: provenance and audit.

## Environment setup

Run all commands from the project root. The executed experiments use Python 3.14.6,
PyTorch 2.14.0, torchvision 0.29.0 and an Apple M5 with 16 GiB unified memory.
Exact package versions are in `requirements.txt`; actual experiment environment
and training source hashes are in `artifacts/results/environment.json`.

For a new machine, use a Python version supported by the pinned packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On the original Mac, the ML dependencies were already present in Anaconda. The
local environment was created with `/opt/anaconda3/bin/python3 -m venv
--system-site-packages .venv` and the missing report packages installed there.
Activate that existing environment with `source .venv/bin/activate`.

Device selection is MPS, then CUDA, then CPU. The original Mac's sandbox hid MPS;
authorized terminal execution outside it exposed the GPU. Confirm availability:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

To force CPU, set `device` to `cpu` in a **copy** of a config and use a different
`output` directory. CPU execution will take longer. Fixed seeds do not guarantee
bitwise equality across devices, library releases or MPS kernels.

## Dataset preparation

Official source: https://www.cs.toronto.edu/~kriz/cifar.html. Dataset files are
downloaded locally and excluded from Git; no full dataset redistribution is claimed.
Torchvision checks the official archive's integrity. Download without training:

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data', train=True, download=True); CIFAR10('data', train=False, download=True)"
```

The seed-3024 stratified split uses 45,000 training and 5,000 validation images;
all 10,000 official test images are retained. Saved indices establish disjointness.
Normalization mean/std are computed only on the 45,000 training images with float64
reductions. Training uses random 32x32 crop with 4-pixel padding and horizontal
flip; evaluation uses only tensor conversion and normalization.

## Pilot before full training

```bash
python -m src.train --config configs/pilot.json
python -m scripts.analyze --root artifacts/pilot
python -m scripts.verify --root artifacts/pilot
```

The pilot uses two epochs, 2,048 training, 512 validation and 512 test examples.
It tests training, checkpoint reload, evaluation, metric persistence and charts.
Inspect its curves and `artifacts/pilot/results/verification.json`. The executed
pilot and corrections are documented in `docs/pilot.md`.

## Full training

```bash
python -m src.train --config configs/full.json
```

All three models train for 20 epochs, batch 128, AdamW lr 0.003, weight decay
0.0005, cosine scheduling, cross-entropy and seed 3024. The full runs take about
25 minutes total on the original M5, with variation from device load. Run one model:

```bash
python -m src.train --config configs/full.json --model starnet
```

The baseline has 289,194 parameters. StarNet and additive ablation each have
280,522. Only `ReLU6(a) * b` versus `ReLU6(a) + b` changes between the two StarNet
variants. They share initialization and data seeds as well as training settings.
Best validation accuracy selects the checkpoint, with the first epoch retained
on ties. All models train from scratch. Test results do not select epochs.

**Rerunning a configuration replaces that output directory's run files.** To retain
the delivered evidence, copy the config and change `output` before a new run.
The saved checkpoints contain model state, not optimizer state; interrupted
training must restart that model, rather than pretending to resume optimization.

## Evaluate saved checkpoints

```bash
python -m src.train --config configs/full.json --evaluate-only
```

This reloads all best checkpoints and recomputes test predictions and inference
timings. It retains recorded training histories and training duration but updates
evaluation files. Add `--model starnet` to evaluate one model. Checkpoints are kept
locally but excluded from the repository; a fresh clone must train first.

## Figures and verification

```bash
python -m scripts.analyze
python -m scripts.verify
python -m compileall -q src scripts
```

Analysis generates eleven main PNG figures, sample image assets, the summary CSV
and `dashboard.json`. Verification reconstructs confusion matrices and accuracy,
checks splits and selected epochs, verifies checkpoint hashes/loads and confirms
dashboard equality. If font cache permissions fail, use a writable cache:

```bash
MPLCONFIGDIR=/private/tmp/starnet-mpl XDG_CACHE_HOME=/private/tmp/starnet-cache python -m scripts.analyze
```

MACs count convolution/linear layers only, not a complete FLOPs estimate. Inference
time is median synchronized device-resident batch time after warmup divided by
128, not single-image latency; data transfer and preprocessing are excluded.
Per-class recall, test loss, checkpoint sizes and raw timing samples are saved.

## Visualization website

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Open **http://127.0.0.1:8000/website/**. Serve the **project root**, not just the
website directory, because it reads `../artifacts/` and `../docs/repository.json`.
Use another port if 8000 is occupied. HTTP serving is necessary for JSON fetches.
The website provides overview, algorithm, dataset, training, model comparison,
ablation, confusion matrices and filtered prediction examples. It rejects pilot
data in place of final results. No result is a fabricated example.

## Report generation and opening

Read `requirement.txt` before editing the report. Generate the Word report after
full training and analysis:

```bash
python -m scripts.report
```

Open `artifacts/report/StarNet_Report.docx` in Word or Pages. The report has exactly
six top-level required sections, includes the student-supplied repository webpage,
and draws numerical results from saved JSON. `report_content.json` and
`report_evidence.json` preserve generated content and numerical traceability.

The same command also generates `artifacts/report/StarNet_Report.md` with relative
links to the actual figures. Open it in a Markdown preview such as VS Code.
PDF generation was replaced with Markdown at the student's request after the
bundled renderer lacked LibreOffice and native Pages export failed. No PDF is
claimed. The DOCX passed structural and numerical checks, but its rendered page
layout could not be verified in this environment. Review it in Word before submission.
`python -m scripts.audit_report` checks the six headings, repository hyperlink,
required figures and report values against the raw experiment files.

Optional website UI checks use Playwright (`python -m pip install playwright==1.63.0`)
and the original Mac's Microsoft Edge executable: `python -m scripts.test_website`.
For other systems, adjust that executable path. `python -m scripts.check_site`
checks all linked local resources while the local server is running.

## Research, attribution and limitations

Ma et al., [Rewrite the Stars, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Rewrite_the_Stars_CVPR_2024_paper.html).
Official code: https://github.com/ma-xu/Rewrite-the-Stars, inspected at commit
`c999eb50840a44f9f1d92e8f7d2c22cd645a6d5e`.
The official block source and Apache-2.0 license are retained in `docs/sources/`.
Our adaptation changes input/output size, stem stride/width, number of stages,
stage depths, expansion and training budget. See the report for exact differences.

This is one seed, one dataset and a limited epoch budget, with no significance
test. Close parameter counts do not equal identical architectural or compute
budgets. Addition also changes activation scale. Results support only the stated
local comparison, not universal superiority or the original ImageNet claims.

## Source publication later

The local Git repository excludes data, virtual environments, caches and model
checkpoints. Inspect files before publishing. The configured destination is:

```text
https://github.com/yuannnnn1/StarNet.git
```

No remote push is performed automatically. Publication is the remaining external
step explicitly deferred by the student.
