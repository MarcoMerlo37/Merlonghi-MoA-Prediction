# Software — MoA Prediction

## Project description

This folder contains the full, runnable solution for the Kaggle **Mechanisms of
Action (MoA) Prediction** challenge, delivered as a single Jupyter notebook:
[`moa_prediction.ipynb`](moa_prediction.ipynb).

The notebook builds and cross-validates four base models (MLP, TabularResNet,
FT-Transformer, CatBoost) on gene-expression / cell-viability profiles, blends them
with a nested-CV-validated weighted ensemble, and produces a Kaggle-format
`submission.csv`. See the top-level [`README.md`](../README.md) for objectives,
methodology and results; the notebook itself contains a numbered workflow overview
in its first cell, and every major code section is commented in English and tagged
with the matching step number.

**Kaggle challenge reference:** <https://www.kaggle.com/competitions/lish-moa>

## Environment the notebook was developed and run in

The notebook was authored and executed **on Kaggle**, inside a Kaggle Notebook
(kernel), not on a local machine. This is relevant because a few things in the code
and in the results are tied to that platform:

- **Accelerator:** GPU T4 x2 (two NVIDIA Tesla T4 GPUs). This is why the code
  contains GPU-count-aware logic (`N_GPUS`, round-robin device assignment across
  workers) and comments referencing "the two T4s" and their ~16 GB VRAM limits.
- **Python / image:** Python 3.12, Kaggle's standard `python` Docker image (the
  one pre-installed with `numpy`, `pandas`, `torch`, `scikit-learn`, etc.).
- **Data mount:** the competition data is attached as a *Kaggle dataset/competition
  source* and appears already extracted at `/kaggle/input/competitions/lish-moa`;
  it is never downloaded manually inside the notebook.
- **Working directory:** intermediate artefacts (model checkpoints, fold
  preprocessing pipelines, experiment logs, the final `submission.csv`) are written
  to `/kaggle/working`, which Kaggle persists as the notebook's output.
- **Runtime:** a full run (5-fold × 3 models × 3 seeds for the neural nets, plus the
  CatBoost CV) took **≈ 2 hours** wall-clock on the two T4s (~86 min neural-net CV,
  ~29 min CatBoost CV, < 1 min test-set inference).
- **Internet access:** the only network call the notebook makes is
  `!pip install iterative-stratification` in the first cell, so the "Internet"
  toggle must be **on** for that Kaggle session (the competition dataset itself does
  not require it, since it is attached as a local input, not downloaded).

Reproducing the notebook **on Kaggle** (Option A below) is the path closest to how it
was actually built and validated. It can also be run **fully locally**, off Kaggle
(Option B below); the code itself does not depend on Kaggle in any hard way — only
`DATA_DIR`/`WORK_DIR` need to point somewhere else, and a GPU needs to be provided
by the local machine instead.

## 3. Data

The raw competition data is **not included in this repository** (several hundred MB,
and redistribution outside Kaggle's platform is against the competition rules).

- Data page (permanent link): <https://www.kaggle.com/competitions/lish-moa/data>
- Expected files: `train_features.csv`, `train_targets_scored.csv`,
  `test_features.csv`, `sample_submission.csv`

## 4. Prerequisites (both options)

- Python 3.12 (the version the notebook was developed and run on)
- A CUDA-capable GPU is strongly recommended, on Kaggle or locally. Without one,
  training time for the full multi-seed, multi-model cross-validation would be
  prohibitively long (hours instead of minutes per fold on CPU).
- Packages (see [`requirements.txt`](requirements.txt) for the full list):
  `numpy`, `pandas`, `scipy`, `joblib`, `scikit-learn`, `torch`,
  `iterative-stratification`, `catboost`.

---

## 5. Option A — Run on Kaggle (environment the notebook was built and tested in)

1. Go to the competition page, <https://www.kaggle.com/competitions/lish-moa>, and
   join the competition if required (accepting its rules is needed to access the
   data, even though it is no longer accepting new leaderboard submissions).
2. On Kaggle, click **Create → New Notebook**, then **File → Import Notebook** and
   upload [`moa_prediction.ipynb`](moa_prediction.ipynb) — or
   copy/paste its cells into a fresh notebook.
3. Attach the competition data: in the right-hand panel, **Add Input** → search
   `lish-moa` → **Mechanisms of Action (MoA) Prediction** → **Add**. This mounts the
   four CSV files under `/kaggle/input/competitions/lish-moa`, matching the
   notebook's default `DATA_DIR` — no path edits needed.
4. Set the accelerator: **Notebook Settings** (right panel, or ⋮ menu) → **Accelerator**
   → **GPU T4 x2** (any CUDA GPU works, but this reproduces the original setup and
   the reported ~2h runtime; a single-GPU accelerator will run, just slower and with
   `N_GPUS = 1`, so fewer NN jobs will run in parallel).
5. Set **Internet: On** in the same settings panel (needed only for the first cell's
   `pip install`).
6. Run the notebook top to bottom: **Run → Run All**, or `Kernel → Restart & Run All`
   from the classic editor. `submission.csv` is written to `/kaggle/working` and
   will appear in the notebook's **Output** tab once the run finishes.

## 6. Option B — Run fully locally (not on Kaggle)

### 6.1 Get the code and data

```bash
# from this repository
cd surname-projecttitle/software

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

`iterative-stratification` is also installed by the notebook's own first cell
(`!pip install iterative-stratification`), so having it in `requirements.txt` is
redundant but harmless — it is kept there so the environment is fully specified even
if that cell is skipped.

**GPU users (recommended):** install a `torch` build matching your CUDA version
instead of the plain `pip install torch` from `requirements.txt` — see
<https://pytorch.org/get-started/locally/> for the correct command for your OS/CUDA
combination, and install it *before* the rest of `requirements.txt` (or reinstall
`torch` afterwards). Without a CUDA-visible GPU, the notebook still runs on CPU
(`N_GPUS` is detected automatically), just far slower.

### 6.3 Point the notebook at your local paths

Open the notebook (`jupyter lab moa_prediction.ipynb`) and, in the first
configuration code cell (Step 1), change:

```python
DATA_DIR = "/kaggle/input/competitions/lish-moa"   # -> "./data" (or wherever you unzipped the CSVs)
WORK_DIR = "/kaggle/working"                       # -> "./working" (or any writable local folder)
```

Nothing else in the notebook is Kaggle-specific: all paths are derived from these
two constants.

### 6.4 Run

Run all cells top to bottom (`Run → Run All Cells` in Jupyter Lab, or
`Kernel → Restart & Run All` in the classic Notebook interface). `submission.csv`
will be written to the `WORK_DIR` you configured above.

**Expected runtime:** on a machine with a single modern CUDA GPU, expect roughly
2–4× the ~2h reference runtime (the code parallelises across GPUs, so a
single-GPU machine loses that 2-way overlap); on CPU only, plan for several hours
to a day depending on core count, since neither the neural networks nor CatBoost's
`task_type="GPU"` path will be used (CatBoost falls back to CPU automatically, the
PyTorch models will use whatever device `torch.cuda.is_available()` reports).

---

## 7. Notes

- The notebook was reset before delivery (cell execution numbers cleared) so that a
  fresh full run produces cleanly numbered cells `[1]…[n]` and up-to-date output. It
  must be executed once, end to end, using Option A or Option B above, before final
  submission — this cannot be done from within this delivery environment.
- Model checkpoints and fold-level preprocessing artefacts (`*.pth`, `*.joblib`,
  `*.cbm`) are written by the notebook to `WORK_DIR` during the run; they are
  intermediate artefacts, not part of this delivery, and are safe to delete between
  runs.
