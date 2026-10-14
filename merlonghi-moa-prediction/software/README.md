# Software — MoA Prediction

## 1. Project description

This folder contains the full, runnable solution for the Kaggle **Mechanisms of Action (MoA) Prediction** challenge, delivered as two Jupyter notebooks:

| Notebook | Purpose | PDF export |
|---|---|---|
| [`moa_prediction.ipynb`](moa_prediction.ipynb) | **Training**: exploratory analysis, cross-validation of three base models, nested-CV blending, saving of all trained artifacts | [`moa_notebook_landscape.pdf`](moa_notebook_landscape.pdf) |
| [`moa_inference.ipynb`](moa_inference.ipynb) | **Inference**: loads the saved artifacts and writes the Kaggle `submission.csv`, with no training | [`moa_inference_landscape.pdf`](moa_inference_landscape.pdf) |

The training notebook builds and cross-validates three base models (TabularResNet, FT-Transformer, CatBoost) on gene-expression / cell-viability profiles and blends them with a nested-CV-validated weighted ensemble. The inference notebook applies the trained models to the test set. The two are separate because the challenge is a **code competition**: a notebook is submitted and re-run by Kaggle on the hidden test set, without internet access. Retraining (about 130 minutes on 2 GPUs) is not needed for that. See the top-level [`README.md`](../README.md) for objectives, methodology and results; the slides are in [`../doc/`](../doc/).

The first cell of each notebook contains a numbered workflow overview, and every major code section is commented in English and tagged with the matching step number (`# Step N.k`). The two notebooks have independent step numbering.

**Kaggle challenge reference:** <https://www.kaggle.com/competitions/lish-moa>

## 2. Environment the notebooks were developed and run in

Both notebooks were authored and executed **on Kaggle**, inside Kaggle Notebooks (kernels), not on a local machine. A few things in the code and in the results are tied to that platform:

- **Accelerator:** GPU T4 x2 (two NVIDIA Tesla T4 GPUs) for the training notebook. This is why the code contains GPU-count-aware logic (`N_GPUS`, round-robin device assignment across workers) and comments about the GPUs' VRAM limits.
- **Python / image:** Kaggle's standard `python` Docker image (with `numpy`, `pandas`, `torch`, `scikit-learn`, etc. pre-installed). The training run used Python 3.12.
- **Data mount:** the competition data is attached as a Kaggle competition source and appears already extracted under `/kaggle/input/competitions/lish-moa`. This path depends on the platform version: if it differs, check `/kaggle/input` in a cell and edit `DATA_DIR` in the training notebook. The inference notebook finds its input folders by file name and needs no path edits.
- **Working directory:** intermediate artefacts (model checkpoints, fold preprocessing pipelines, experiment logs, the final `submission.csv`) are written to `/kaggle/working`, which Kaggle persists as the notebook's output.
- **Runtime (training):** a full run (5-fold × 2 networks × 3 seeds, plus the CatBoost CV) took about **131 minutes** wall-clock on the two T4s (≈98 min neural-network CV, ≈29 min CatBoost CV, <1 min test-set inference). Timing varies with the platform machine: expect roughly ±20%.
- **Internet access:** the only network call in the training notebook is `!pip install -q iterative-stratification` in the first code cell, so the "Internet" toggle must be **on** for that session. The inference notebook makes no network call and runs with internet **off**.

Running **on Kaggle** (Option A below) is the path closest to how the notebooks were built and validated. The training notebook can also be run **locally** (Option B): only `DATA_DIR` and `WORK_DIR` need to point elsewhere, and a GPU must be provided by the local machine.

## 3. Data

The raw competition data is **not included in this repository** (over 100 MB, and redistribution outside Kaggle's platform is against the competition rules).

- Data page (permanent link): <https://www.kaggle.com/competitions/lish-moa/data>
- Expected files: `train_features.csv`, `train_targets_scored.csv`, `test_features.csv`, `sample_submission.csv`

The trained artifacts used by the inference notebook are not included either; they are produced by the training notebook (see section 8.2).

## 4. Prerequisites (both options)

- Python 3.12 (the version the training notebook was developed and run on)
- A CUDA-capable GPU is strongly recommended. Without one, training time for the full multi-seed, multi-model cross-validation would be prohibitively long.
- Packages (see [`requirements.txt`](requirements.txt)): `numpy`, `pandas`, `scipy`, `joblib`, `scikit-learn`, `torch`, `iterative-stratification`, `catboost`, `matplotlib` (exploratory plots). Versions are not pinned; to record the exact environment of a working run, execute `pip freeze > requirements-frozen.txt` in it.
- The inference notebook needs the same libraries, minus `iterative-stratification`. The fold preprocessing pipelines are `scikit-learn` pickles, so the installed `scikit-learn` version should match the one used for training. Step 9.1 of the inference notebook prints the installed versions.

---

## 5. Option A — Run the training notebook on Kaggle (environment it was built in)

1. Go to the competition page, <https://www.kaggle.com/competitions/lish-moa>, and join the competition if required (accepting its rules is needed to access the data).
2. On Kaggle, click **Create → New Notebook**, then **File → Import Notebook** and upload [`moa_prediction.ipynb`](moa_prediction.ipynb).
3. Attach the competition data: in the right-hand panel, **Add Input** → search `lish-moa` → **Mechanisms of Action (MoA) Prediction** → **Add**. This mounts the four CSV files under `/kaggle/input/competitions/lish-moa`, matching the default `DATA_DIR`.
4. Set the accelerator: **Notebook Settings** → **Accelerator** → **GPU T4 x2** (any CUDA GPU works, but this reproduces the original setup and runtime; a single-GPU accelerator runs with `N_GPUS = 1`, so fewer jobs run in parallel).
5. Set **Internet: On** in the same settings panel (needed only for the first code cell's `pip install`).
6. Run the notebook top to bottom: **Run → Run All**, or **Restart & Run All**. `submission.csv` is written to `/kaggle/working` and appears in the notebook's **Output** tab once the run finishes.

Besides `submission.csv`, the run writes to `/kaggle/working` everything the inference notebook needs: the 30 network checkpoints, the 10 fold pipelines, the 5 CatBoost models with their metadata, and `metadata.json` (Step 14.9). To use these artifacts, continue with section 8.

## 6. Option B — Run the training notebook fully locally (not on Kaggle)

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

You should now have, under `./data/`: `train_features.csv`, `train_targets_scored.csv`, `test_features.csv`, `sample_submission.csv`.

### 6.2 Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install jupyterlab             # or `notebook`, to open/run the .ipynb locally
```

`iterative-stratification` is also installed by the notebook's own first code cell, so having it in `requirements.txt` is redundant but harmless.

**GPU users (recommended):** install a `torch` build matching your CUDA version instead of the plain `pip install torch` from `requirements.txt` — see <https://pytorch.org/get-started/locally/> — and install it *before* the rest of `requirements.txt` (or reinstall `torch` afterwards). Without a CUDA-visible GPU the notebook still runs on CPU (`N_GPUS` is detected automatically), just far slower.

### 6.3 Point the notebook at your local paths

Open the notebook (`jupyter lab moa_prediction.ipynb`) and, in the first configuration code cell (Step 1.3), change:

```python
DATA_DIR = "/kaggle/input/competitions/lish-moa"   # -> "./data" (or wherever you unzipped the CSVs)
WORK_DIR = "/kaggle/working"                       # -> "./working" (or any writable local folder)
```

All paths, including the output `submission.csv`, are derived from these two constants. Also remove or comment out the `!pip install` line if you installed the packages already (it is harmless to keep).

### 6.4 Run

Run all cells top to bottom (`Run → Run All Cells` in Jupyter Lab, or `Kernel → Restart & Run All`). `submission.csv` is written to the `WORK_DIR` you configured above.

**Expected runtime:** on a machine with a single modern CUDA GPU, expect roughly 2–4× the ~131-minute reference runtime (the code parallelises across GPUs, so a single-GPU machine loses that 2-way overlap; this factor is an estimate, not a measurement). On CPU only, plan for several hours to a day depending on core count; CatBoost falls back to CPU automatically.

---

## 7. Expected results of the training notebook (to validate a rerun)

A rerun should give values close to the following (out-of-fold mean log loss, treated rows only):

| Model | OOF log loss |
|---|---|
| TabularResNet | 0.01632 |
| FT-Transformer | 0.01642 |
| CatBoost | 0.01699 |
| Ensemble (nested) | 0.01605 ± 0.00019 |

The exploratory analysis (Step 2.3) should report 23,814 rows (21,948 treated, 1,866 control), a positive rate of 0.00373 on treated rows, and 53 of 206 targets with fewer than 20 positives.

Mixed precision and parallel GPU jobs make runs non-deterministic: expect differences of the order of 1e-5 in the OOF scores and a few hundredths in the blend weights (final weights of the reference run: ResNet 0.436, FT-Transformer 0.387, CatBoost 0.177), not an exact match.

---

## 8. Inference notebook and Kaggle submission

### 8.1 What it does

[`moa_inference.ipynb`](moa_inference.ipynb) rebuilds the networks, loads the saved weights, fitted preprocessing pipelines and blend weights, and predicts the test set. It reuses the training code for the feature engineering, the architectures, the pipelines (transform only), the inference and the submission checks, so the predictions are identical. It takes about a minute on a GPU, needs no internet, and writes `/kaggle/working/submission.csv`. It never writes to the artifacts folder, which is read-only on Kaggle.

### 8.2 Create the Dataset of artifacts

1. In the training notebook, run all cells so that Step 14.9 writes `metadata.json` (feature order, column groups, target names, blend weights, folds and seeds).
2. Save the notebook with **Save Version → Save & Run All (Commit)**.
3. Open the saved version, tab **Output**, and click **New Dataset**.
4. Select these 51 files, name the Dataset `moa-artifacts` and create it:
   - `best_resnet_fold_*_seed*.pth` and `best_ft_transformer_fold_*_seed*.pth` (30 files);
   - `fold_pipeline_*.joblib` and `fold_pipeline_raw_*.joblib` (10 files);
   - `catboost_fold_*.cbm` and `catboost_fold_*_meta.joblib` (10 files);
   - `metadata.json`.

The Dataset used for the reported submission is private to the author's account: **[moa-artifacts](https://www.kaggle.com/datasets/marcomerlo37/moa-artifacts)**. It can be regenerated at any time by rerunning the training notebook.

### 8.3 Run the inference notebook

1. Create a new Kaggle notebook and import [`moa_inference.ipynb`](moa_inference.ipynb).
2. **Add Input**: the competition `lish-moa` and the Dataset `moa-artifacts`.
3. Set **Internet: Off**. Select a GPU accelerator (a CPU-only run works but the FT-Transformer is slow).
4. Run all cells. The notebook locates the two input folders by file name (`metadata.json` and `test_features.csv`), so no path needs editing. If it does not find exactly one folder for a file, it stops with an explicit message.
5. Step 9.1 checks that all 51 artifact files exist and prints the library versions; Step 9.5 runs the sanity checks and writes `submission.csv`.

### 8.4 Consistency check

On the public test set, the `submission.csv` of the inference notebook was compared with the one produced by the training notebook: the maximum absolute difference is 1.1e-16. This confirms that the artifacts, the feature order and the blend weights are loaded correctly.

### 8.5 Late submission

The challenge is closed, but a notebook can still be submitted as a **late submission** from the competition page (*Late Submission*), selecting a saved version of the inference notebook. Kaggle re-runs it on the hidden test set and scores the resulting predictions. Late submissions do not appear on the leaderboard.

Save the notebook version first (**Save Version → Save & Run All**): the submission uses the saved version, not the open session. The submission was made **once**, after all design choices were fixed, and was not used for tuning.

### 8.6 Result

| Estimate | Log loss |
|---|---|
| Nested-CV blend, treated rows only | 0.01605 ± 0.00019 |
| Nested-CV blend, rescaled to all test rows (× 0.910) | 0.01461 |
| Kaggle private score | 0.01634 |
| Kaggle public score | 0.01864 |

The private score is 11.8% above the rescaled CV estimate, so the cross-validated estimate is optimistic. See the top-level [`README.md`](../README.md) for the rescaling and the discussion of the gap.

---

## 9. Notes

- The delivered notebooks and their PDF exports come from single clean runs (*Restart & Run All*) on Kaggle, so cells are numbered from 1.
- The training notebook reproduces the cross-validated scores above. The Kaggle test score was measured once with the inference notebook (section 8.6). The CV is not grouped by compound (the data has no compound identifier); see the top-level README for the limitations.
- Model checkpoints and fold-level preprocessing artefacts (`*.pth`, `*.joblib`, `*.cbm`) are written to `WORK_DIR` during the training run. They are not part of this repository, are the input of the inference notebook (section 8), and are safe to delete between training runs. The training notebook clears stale fold pipelines at the start of each run.