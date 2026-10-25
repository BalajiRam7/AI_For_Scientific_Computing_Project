# Continual Learning for Autoregressive PDE Surrogates under Evolving Physical Regimes

This repository contains the code for generating the datasets and running the experiments and additional analysis in the paper *Continual Learning for Autoregressive PDE Surrogates under Evolving Physical Regimes*, published in the Journal of Machine Learning for Computational Science and Engineering.

All configs and scripts for data generation and training are based on [Hydra](https://hydra.cc) and stored under `configs/` and `scripts/` accordingly.

---

## Installation

The project uses [uv](https://docs.astral.sh/uv/). Python 3.10 and all dependencies are pinned in `pyproject.toml` / `uv.lock`.

```bash
# 1. install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create the environment from the lockfile (requires Python 3.10)
uv sync

# 3. activate it (the scripts call `python` directly)
source .venv/bin/activate
```

Single commands can also be run without activating: `uv run python train_cl.py ...`.

Do not install `pytorch` from PyPI; the package name for PyTorch is `torch`. The
notebooks use the same package name. If an environment was created with Python
3.14, recreate it with Python 3.10 before running `uv sync`, because the
project and lockfile intentionally require Python 3.10.

Training runs on CPU by default; pass `device=cuda` or `device=mps` (or set `DEVICE` in the script) to use a GPU.

---



## Repository layout

```
configs/            Hydra configs: equation/, stream/, model/, cl_algorithm/, iid_trainer/
src/simulation/     PDE solvers and dataset generation
src/common/         datasets, streams, models
src/iid/            joint (i.i.d.) trainer
src/cl/             continual-learning algorithms
scripts/            scripts for data generation, training and analysis
notebooks/          notebooks for visualization of data
generate_dataset.py / train_iid.py / train_cl.py    entry points
```

**Before running any script, open it and set the paths (if needed) in its header block**, in particular:


| Variable                | Meaning                                                                          |
| ----------------------- | -------------------------------------------------------------------------------- |
| `DATA_ROOT`             | directory containing the generated datasets (default `./data`)                   |
| `CHECKPOINT_PATH`       | pretrained checkpoint to start from (placeholder `"PRETRAINED_CHECKPOINT_PATH"`) |
| `DEVICE`, `NUM_WORKERS` | compute resources                                                                |


Run all scripts from the repository root, e.g. `bash scripts/cl/cl_advdiff1d.sh`.

---



## Stage 1 — Generate the data

Each script solves one equation family over a grid of physical parameters and writes the dataset to
`data/<equation>_stream/`.

```bash
uv run bash scripts/simulation/generate_advdiff1d.sh
```

The result is one `.npz` (snapshots) plus one `.yaml` (parameters) per simulation, and dataset-level
statistics used for normalization:

```
data/advdiff1d_stream/
├── simulations/sim_0.npz, sim_0.yaml, ...
└── normalization.yaml
```

The notebooks in `notebooks/simulation/` generate and plot single random simulations to inspect the data.

---

## Baseline experiment notebook

`notebooks/experiments/continual_learning_baselines.ipynb` runs a configurable
matrix across the available single and mixed PDE streams, continual-learning
algorithms, and random seeds. It uses the same `train_cl.py` Hydra entry point
and algorithm hyperparameters as the shell scripts, then aggregates the final
Aim metrics (`test_loss_stream` and `test_loss_rell2_stream`) into a CSV and
comparison plot.

Open the notebook after generating the selected datasets. Set `CHECKPOINTS` for
pretrained initialization, adjust `STREAMS`, `ALGORITHMS`, `SEEDS`, and
`EPOCHS`, inspect the dry-run commands, and then run the launch cell with
`dry_run=False`.

### Colab compatibility

Do not install `protobuf==3.20.3` into Colab's global Python environment: it
produces dependency-conflict warnings for Colab's Google Cloud, TensorFlow, and
Weights & Biases packages. Create the project environment separately and run
the training subprocesses with it:

```python
!pip install -q uv
!uv python install 3.10
!uv venv /content/pde-env --python 3.10 --system-site-packages
!uv pip install --python /content/pde-env/bin/python \
    "aim==3.29.1" "protobuf==3.20.3" "hydra-core==1.3.2" \
    h5py tables tqdm joblib rich seaborn scikit-learn pyyaml
```

The notebook automatically uses `/content/pde-env/bin/python` when it exists.
If the notebook kernel itself imports Aim under an incompatible protobuf
runtime, its first cell enables the slower pure-Python protobuf implementation
as a compatibility fallback. Verify that the dry-run command starts with
`/content/pde-env/bin/python train_cl.py`.

---


## Stage 2 — Pretrain

The continual-learning experiments start from a pretrained model. There is one script per architecture,
each pretraining on every PDE stream of the corresponding dimensionality:

```bash
uv run bash scripts/iid/pretrain_fno1d.sh
```

The final weights are written to

```
outputs/outputs_local/runs/<run_hash>/checkpoints/model_weights_final.pt
```

Use that path as `CHECKPOINT_PATH` in the scripts of the next stages.

---



## Stage 3 — Joint and continual-learning experiments

**Joint (i.i.d.) upper bound** — trains on the whole stream at once:

```bash
uv run bash scripts/iid/joint_fno1d.sh
```

**Continual learning** — trains sequentially over the experiences of a stream and evaluates on the full
stream after every experience. Each script runs all algorithms (Naive, Replay, Replay-TS, GEM, A-GEM, EWC):

```bash
uv run bash scripts/cl/cl_advdiff1d.sh                         # single PDE
uv run bash scripts/cl_mixed/cl_mixed_advdiff1d_burgers1d.sh   # mixed
```

Note: `--multirun` together with comma-separated values (e.g. `stream.seed=123,456,789,111,222`) sweeps over seeds, which is how the scripts obtain their five repetitions.

---



## Stage 4 — Additional analysis

Ablations of the replay mechanism and a loss-landscape study of out-of-distribution experiences:

```bash
uv run bash scripts/analysis/analysis_replayimp_slicefactor.sh
```

---



## Results and tracking

Every run creates a directory under `logger.outputs_dir` (default `outputs/`, configurable in
`configs/train_iid.yaml` and `configs/train_cl.yaml`) containing the resolved `config.yaml`, `plots/` and
`checkpoints/`. Metrics and figures are tracked with [Aim](https://aimstack.io):

```bash
uv run aim up --repo outputs/.aim
```
