# Software — MoA Prediction

## 1. Project description

This folder contains the full, runnable solution for the Kaggle **Mechanisms of
Action (MoA) Prediction** challenge, delivered as a single Jupyter notebook:
[`moa_prediction.ipynb`](moa_prediction.ipynb), with a landscape PDF export,
[`moa_notebook_landscape.pdf`](moa_notebook_landscape.pdf).

The notebook builds and cross-validates three base models (TabularResNet,
FT-Transformer, CatBoost) on gene-expression / cell-viability profiles, blends them
with a nested-CV-validated weighted ensemble, and produces a Kaggle-format
`submission.csv`. See the top-level [`README.md`](../README.md) for objectives,
methodology and results; the slides are in [`../doc/`](../doc/). The first cell of
the notebook contains a numbered workflow overview, and every major code section is
commented in English and tagged with the matching step number (`# Step N.k`).

**Kaggle challenge reference:** <https://www.kaggle.com/competitions/lish-moa>

## 2. Environment the notebook was developed and run in

The notebook was authored and executed **on Kaggle**, inside a Kaggle Notebook
(kernel), not on a local machine. A few things in the code and in the results are
tied to that platform:

- **Accelerator:** GPU T4 x2 (two NVIDIA Tesla T4 GPUs). This is why the code
  contains GPU-count-aware logic (`N_GPUS`, round-robin device assignment across
  workers) and comments referencing the two T4s and their ~16 GB VRAM limits.
- **Python / image:** Python 3.12, Kaggle's standard `python` Docker image (with
  `numpy`, `pandas`, `torch`, `scikit-learn`, etc. pre-installed).
- **Data mount:** the competition data is attached as a Kaggle competition source
  and appears already extracted at `/kaggle/input/competitions/lish-moa`. This path
  depends on the platform version: if it differs, check `/kaggle/input` in a cell
  and edit `DATA_DIR`.
- **Working directory:** intermediate artefacts (model checkpoints, fold
  preprocessing pipelines, experiment logs, the final `submission.csv`) are written
  to `/kaggle/working`, which Kaggle persists as the notebook's output.
- **Runtime:** a full run (5-fold × 2 networks × 3 seeds, plus the CatBoost CV) took
  about **129 minutes** wall-clock on the two T4s (≈96 min neural-network CV,
  ≈30 min CatBoost CV, <1 min test-set inference). Timing varies with the platform
  machine: expect roughly ±20%.
- **Internet access:** the only network call is `!pip install -q
  iterative-stratification` in the first code cell, so the "Internet" toggle must be
  **on** for that Kaggle session.

Running **on Kaggle** (Option A below) is the path closest to how the notebook was
built and validated. It can also be run **locally** (Option B): only `DATA_DIR` and
`WORK_DIR` need to point elsewhere, and a GPU must be provided by the local machine.

## 3. Data

The raw competition data is **not included in this repository** (over 100 MB, and
redistribution outside Kaggle's platform is against the competition rules).

- Data page (permanent link): <https://www.kaggle.com/competitions/lish-moa/data>
- Expected files: `train_features.csv`, `train_targets_scored.csv`,
  `test_features.csv`, `sample_submission.csv`

## 4. Prerequisites (both options)

- Python 3.12 (the version the notebook was developed and run on)
- A CUDA-capable GPU is strongly recommended. Without one, training time for the
  full multi-seed, multi-model cross-validation would be prohibitively long.
- Packages (see [`requirements.txt`](requirements.txt)): `numpy`, `pandas`, `scipy`,
  `joblib`, `scikit-learn`, `torch`, `iterative-stratification`, `catboost`.
  Versions are not pinned; to record the exact environment of a working run, execute
  `pip freeze > requirements-frozen.txt` in it.

---

## 5. Option A — Run on Kaggle (environment the notebook was built in)

1. Go to the competition page, <https://www.kaggle.com/competitions/lish-moa>, and
   join the competition if required (accepting its rules is needed to access the
   data, even though it no longer accepts new leaderboard submissions).
2. On Kaggle, click **Create → New Notebook**, then **File → Import Notebook** and
   upload [`moa_prediction.ipynb`](moa_prediction.ipynb).
3. Attach the competition data: in the right-hand panel, **Add Input** → search
   `lish-moa` → **Mechanisms of Action (MoA) Prediction** → **Add**. This mounts the
   four CSV files under `/kaggle/input/competitions/lish-moa`, matching the default
   `DATA_DIR`.
4. Set the accelerator: **Notebook Settings** → **Accelerator** → **GPU T4 x2**
   (any CUDA GPU works, but this reproduces the original setup and runtime; a
   single-GPU accelerator runs with `N_GPUS = 1`, so fewer jobs run in parallel).
5. Set **Internet: On** in the same settings panel (needed only for the first code
   cell's `pip install`).
6. Run the notebook top to bottom: **Run → Run All**, or **Restart & Run All**.
   `submission.csv` is written to `/kaggle/working` and appears in the notebook's
   **Output** tab once the run finishes.

## 6. Option B — Run fully locally (not on Kaggle)

### 6.1 Get the code and data

```bash
# from this repository
cd merlonghi-moa-prediction/software

# competition data: download via the Kaggle API (requires a Kaggle account and
# an API token — see https://www.kaggle.com/docs/api#authentication — saved as
# ~/.kaggle/kaggle.json beforehand)
pip install --user kaggle
kaggle competitions download -c lish-moa -p ./data
unzip ./data/lish-moa.zip -d ./data
```

You should now have, under `./data/`: `train_features.csv`,
`train_targets_scored.csv`, `test_features.csv`, `sample_submission.csv`.

### 6.2 Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyterlab             # or `notebook`, to open/run the .ipynb locally
```

`iterative-stratification` is also installed by the notebook's own first code cell,
so having it in `requirements.txt` is redundant but harmless.

**GPU users (recommended):** install a `torch` build matching your CUDA version
instead of the plain `pip install torch` from `requirements.txt` — see
<https://pytorch.org/get-started/locally/> — and install it *before* the rest of
`requirements.txt` (or reinstall `torch` afterwards). Without a CUDA-visible GPU the
notebook still runs on CPU (`N_GPUS` is detected automatically), just far slower.

### 6.3 Point the notebook at your local paths

Open the notebook (`jupyter lab moa_prediction.ipynb`) and, in the first
configuration code cell (Step 1.3), change:

```python
DATA_DIR = "/kaggle/input/competitions/lish-moa"   # -> "./data" (or wherever you unzipped the CSVs)
WORK_DIR = "/kaggle/working"                       # -> "./working" (or any writable local folder)
```

All paths, including the output `submission.csv`, are derived from these two
constants. Also remove or comment out the `!pip install` line if you installed the
packages already (it is harmless to keep).

### 6.4 Run

Run all cells top to bottom (`Run → Run All Cells` in Jupyter Lab, or
`Kernel → Restart & Run All`). `submission.csv` is written to the `WORK_DIR` you
configured above.

**Expected runtime:** on a machine with a single modern CUDA GPU, expect roughly
2–4× the ~129-minute reference runtime (the code parallelises across GPUs, so a
single-GPU machine loses that 2-way overlap). On CPU only, plan for several hours to
a day depending on core count; CatBoost falls back to CPU automatically.

---

## 7. Expected results (to validate a rerun)

A rerun should give values close to the following (out-of-fold mean log loss):

| Model | OOF log loss |
|---|---|
| TabularResNet | 0.01632 |
| FT-Transformer | 0.01643 |
| CatBoost | 0.01700 |
| Ensemble (nested) | 0.01606 ± 0.00019 |

Mixed precision, `cudnn.benchmark` and parallel GPU jobs make runs non-deterministic:
expect differences of the order of 1e-5 in the OOF scores and a few hundredths in
the blend weights (final weights of the reference run: ResNet 0.447, FT-Transformer
0.375, CatBoost 0.178), not an exact match.

## 8. Notes

- The delivered notebook and its PDF export come from a single clean run
  (*Restart & Run All*) on Kaggle, so cells are numbered from 1.
- The notebook reproduces the cross-validated scores above. The leaderboard score
  was not measured, and the CV is not grouped by compound (the data has no compound
  identifier); see the top-level README for the limitations.
- Model checkpoints and fold-level preprocessing artefacts (`*.pth`, `*.joblib`,
  `*.cbm`) are written to `WORK_DIR` during the run; they are intermediate
  artefacts, not part of this delivery, and are safe to delete between runs. The
  notebook clears stale fold pipelines at the start of each run.
