# DeepONet and FNO surrogates for steady heat conduction in a plate with a cavity

![Plate with a cooled cavity: geometry, finite element mesh, signed-distance input and temperature output, with six further cases of the dataset](docs/figures/banner.png)

This repository compares two neural operators that learn the map from cavity geometry to temperature field:

- the **DeepONet** of Lu et al. (2021)
- the **Fourier Neural Operator (FNO)** of Li et al. (2021)

Both are trained on finite element solutions computed with NGSolve.

The repository contains everything needed to:

1. generate the finite element dataset;
2. tune both operators on validation data only;
3. measure how their test error falls as the number of training cases grows (the data-size study);
4. produce the comparison figures and tables.

One program, `run.py`, does all of it. It runs on a Windows computer (Anaconda Prompt), on Linux or macOS, and on a Slurm cluster (Sunbird, Supercomputing Wales).

---

## The problem

<p align="center"><i>−∇·(k ∇T) = 0 in a 36 mm × 21 mm plate with one cooling cavity</i></p>

**Plate and loads**

| Item | Value |
|---|---|
| Top edge | heat flux q = 0.5 MW/m² enters |
| Cavity wall | convection −k ∂T/∂n = h (T − T∞), h = 4000 W/(m² K), T∞ = 300 K |
| Other edges | adiabatic |
| Material | k = 28.3 W/(m K) |
| Cavity | circle (diameter 8–14 mm) or rectangle (8–14 mm × 6–12 mm), anywhere in the plate |

**Reference solutions and network data**

- Reference solutions use H1 finite elements of order 2 (NGSolve), sampled on a 128 × 96 grid.
- Network input: the signed distance field (SDF) of the cavity on that grid.
- Network output: the temperature on the same grid.

**The data-size study** uses a single dataset of 3800 finite element cases:

| Part | Cases | Sampling | Role |
|---|---|---|---|
| Training pool | 3200 | Halton sequence | the training set of size N is the first N cases, so the sets are nested |
| Validation | 200 | Latin hypercube | the same for every N; used for early stopping and tuning |
| Test | 400 | Latin hypercube | the same for every N; used only for final errors |

Both operators are trained at N = 200, 400, 800, 1600 and 3200, with five seeds each. A power law e = C·N^(−α) is fitted to the test error.

---

## What is in the repository

```
Feature_Extraction/
├── run.py                     the only entry point:  python run.py STAGE OPTIONS
├── requirements.txt           numpy, matplotlib, ngsolve, torch
├── src/
│   ├── config.py              every fixed setting (physics, geometry ranges, grid), each with a comment
│   ├── fem/                   geometry and SDF, mesh, weak form, sampling of one FE case on the grid
│   ├── data/                  Latin-hypercube and Halton datasets, frozen split, scaling, rounded-rectangle test set
│   ├── models/                DeepONet, FNO, named variants
│   ├── training/              training loop, checkpoints and resume, tuning stages, data-size runs
│   └── analysis/              error metrics, timing, mesh check, all figures and reports
├── jobs/sunbird/
│   ├── gpu.sh                 one run.py command on one A100  (sbatch jobs/sunbird/gpu.sh STAGE ...)
│   └── cpu.sh                 one run.py command on a CPU node
├── tests/                     numerical regression tests (python tests/run_all.py)
├── docs/
│   ├── manual/
│   │   ├── commands_windows.txt          every command for Windows, in order (Anaconda Prompt)
│   │   ├── commands_sunbird.txt          every command for Sunbird, in order
│   │   └── Operator_Learning_Manual.tex  the full manual (pdflatex, twice)
│   └── figures/               the banner above and the script that draws it
├── outputs/slurm/             output files of the cluster jobs (empty in the repository)
└── archive/                   earlier scripts (POD-DeepONet, depth study, old figures); not used
```

The code creates two more folders, `data/` (datasets) and `outputs/` (models, logs, tables, figures). Neither is part of the repository.

The two command files are meant to be run from top to bottom. Every command line can be copied as it is; the note lines above it explain it. Notes start with `rem` (Windows) or `#` (Linux), so pasting them does nothing.

---

## 1. Windows (Anaconda Prompt)

This is the complete route on a Windows 10/11 computer. Blocks 0–6 of `docs/manual/commands_windows.txt` contain the same lines with an explanation of each.

### 1.1 Install Miniconda and Git (once)

Open the ordinary **Command Prompt** (Start menu → type `cmd`; not PowerShell) and type:

```bat
cd /d %TEMP%
curl -L -o Miniconda3.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
start /wait "" Miniconda3.exe /S /InstallationType=JustMe /RegisterPython=0 /D=%UserProfile%\miniconda3
winget install --id Git.Git -e --source winget
```

- The installer runs silently for one to three minutes. It installs for your user only (no administrator rights) and leaves the Windows `PATH` unchanged.
- Skip a line if the program is already there: `where git` prints a path when Git is installed.
- By hand instead: install *Miniconda3 Windows 64-bit* from [anaconda.com/download/success](https://www.anaconda.com/download/success) (choose *Just Me*, keep *Create shortcuts*, leave *Add to PATH* unticked) and Git from [git-scm.com/download/win](https://git-scm.com/download/win) (default choices).

Close the window. From now on, open **Anaconda Prompt (miniconda3)** from the Start menu; its prompt starts with `(base)`. If that entry is missing, open Command Prompt and type `%UserProfile%\miniconda3\Scripts\activate.bat`.

### 1.2 Environment, PyTorch and the project (once, in Anaconda Prompt)

```bat
nvidia-smi
conda create -n opl python=3.11 -y --override-channels -c conda-forge
conda activate opl
pip install numpy matplotlib ngsolve
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

mkdir C:\PhD
cd /d C:\PhD
git clone https://github.com/prajwal1798/Feature_Extraction.git
cd /d C:\PhD\Feature_Extraction
python tests\run_all.py
python run.py smoke
```

- `nvidia-smi` shows the newest CUDA version the driver supports (top right). Only the driver is needed, not the CUDA Toolkit. Use `cu126` instead of `cu130` if it shows 12.6–12.9.
- `--override-channels -c conda-forge` avoids the *Terms of Service have not been accepted* error.
- The `python -c` line must print a version ending in `+cu130` and `True`.
- `mkdir` may report that `C:\PhD` already exists; that is fine. Any other folder works too.
- `run_all.py` must end with *All tests passed*, `smoke` with *Smoke test passed*.
- **No NVIDIA GPU:** use `pip install torch` and add `--device cpu` to every training, tuning, evaluation and plotting command.

### 1.3 Every new window

```bat
conda activate opl
cd /d C:\PhD\Feature_Extraction
```

To fetch later changes of the repository: `git pull` inside that folder. Your `data` and `outputs` folders are not touched.

### 1.4 The full study (Anaconda Prompt)

Each step needs the results of the one before. Every command can be typed again after a stop; finished work is kept.

```bat
rem steps 1-2: data (CPU)
python run.py generate --family scaling
python run.py generate-rounded --family scaling --rounded-cases 50
python run.py plot-sampling --family scaling
python run.py preview
python run.py mesh-check

rem step 3 (optional): one quick model
python run.py train    --family scaling --train-size 800 --operator fno --seed 321 --run-name try_fno --epochs 300
python run.py evaluate --family scaling --operator fno --run-name try_fno
python run.py plot     --family scaling --operator fno --run-name try_fno

rem step 4: tuning at N = 800 (keep the order)
python run.py tune        --family scaling --train-size 800 --operator deeponet --stage-name grid --epochs 2000
python run.py tune-select --family scaling --train-size 800 --operator deeponet --stage-name grid
python run.py tune        --family scaling --train-size 800 --operator deeponet --stage-name grid_gelu --epochs 2000
python run.py tune-select --family scaling --train-size 800 --operator deeponet --stage-name grid_gelu
python run.py tune        --family scaling --train-size 800 --operator deeponet --stage-name lr --epochs 2000
python run.py tune-select --family scaling --train-size 800 --operator deeponet --stage-name lr
python run.py tune        --family scaling --train-size 800 --operator fno --stage-name modes_width --epochs 2000
python run.py tune-select --family scaling --train-size 800 --operator fno --stage-name modes_width
python run.py tune        --family scaling --train-size 800 --operator fno --stage-name depth --epochs 2000
python run.py tune-select --family scaling --train-size 800 --operator fno --stage-name depth
python run.py tune        --family scaling --train-size 800 --operator fno --stage-name lr --epochs 2000
python run.py tune-select --family scaling --train-size 800 --operator fno --stage-name lr

rem step 5: the data-size runs (5 sizes x 5 seeds per operator)
python run.py train-scaling --operator deeponet
python run.py train-scaling --operator fno

rem steps 6-8: results
python run.py scaling-report
for %n in (200 400 800 1600 3200) do python run.py compare --family scaling --train-size %n
python run.py plot             --family scaling --operator fno --run-name scale_fno_N3200_seed321
python run.py evaluate-rounded --family scaling --operator fno --run-name scale_fno_N3200_seed321
python run.py benchmark        --family scaling --operator fno --run-name scale_fno_N3200_seed321
explorer outputs
```

---

## 2. Tuning stages

| Operator | Stage | What changes | Models |
|---|---|---|---|
| DeepONet | `grid` | layers L ∈ {2,4,6,8,10} × neurons N ∈ {10,20,40,80,160}, branch and trunk alike; tanh | 25 |
| DeepONet | `grid_gelu` | the same 25 with GELU | 25 |
| DeepONet | `lr` | learning rate ∈ {3e-4, 1e-3, 3e-3} at the better grid winner | 3 |
| FNO | `modes_width` | Fourier modes ∈ {4,8,12,16} × width ∈ {16,32,64}; 4 blocks | 12 |
| FNO | `depth` | Fourier blocks ∈ {2,3,4,6} | 4 |
| FNO | `lr` | learning rate ∈ {3e-4, 1e-3, 3e-3} | 3 |

- **Fair comparison:** all models of a stage share the split, the epoch budget, early stopping and the scheduler. `tune-select` refuses to compare models where these differ.
- **Selection:** the winner has the lowest mean validation relative L2 error. RMSE and R² are reported alongside.
- **Order:** each stage starts from the winner of the one before; `train-scaling` stops with an error if the `lr` stage of that operator has not been selected.

---

## 3. Linux, macOS, or Ubuntu inside Windows (WSL)

```bash
# Miniforge, if conda is not installed yet (Linux; macOS installers on the same page)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh

# the project, in your home folder
cd ~
git clone https://github.com/prajwal1798/Feature_Extraction.git
cd ~/Feature_Extraction

conda create -n opl python=3.11 -y --override-channels -c conda-forge
conda activate opl
python -m pip install -r requirements.txt
python tests/run_all.py
python run.py smoke
```

The study commands are those of section 1.4 with `/` in paths and a bash loop:

```bash
for n in 200 400 800 1600 3200; do python run.py compare --family scaling --train-size $n; done
```

On macOS, add `--device cpu`.

---

## 4. Sunbird (Slurm)

Every command, in order, with notes: `docs/manual/commands_sunbird.txt`. Set-up (once, on the login node):

```bash
cd /scratch/$USER
git clone https://github.com/prajwal1798/Feature_Extraction.git ML
cd ML
mkdir -p outputs/slurm
module load anaconda/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --prefix /scratch/$USER/ML/env python=3.11 -y --override-channels -c conda-forge
conda activate /scratch/$USER/ML/env
/scratch/$USER/ML/env/bin/python -m pip install --no-cache-dir --only-binary=:all: numpy matplotlib ngsolve torch
sbatch jobs/sunbird/gpu.sh gpu-info
sbatch jobs/sunbird/cpu.sh smoke
```

The job files expect the project in `/scratch/$USER/ML` and the environment in `/scratch/$USER/ML/env`. Submit every job from inside `ML` (`cd /scratch/$USER/ML`). The words after the job file are those you would type after `python run.py`:

```bash
sbatch jobs/sunbird/cpu.sh generate --family scaling
sbatch jobs/sunbird/gpu.sh train --operator fno --seed 321
```

A tuning stage or the data-size study runs as a **job array**; every task trains one model on its own GPU:

```bash
sbatch --array=0-24 --output=outputs/slurm/gpu_%A_%a.out jobs/sunbird/gpu.sh tune --family scaling --train-size 800 --operator deeponet --stage-name grid --epochs 2000
sbatch jobs/sunbird/cpu.sh tune-select --family scaling --train-size 800 --operator deeponet --stage-name grid
sbatch --array=0-24 --output=outputs/slurm/gpu_%A_%a.out jobs/sunbird/gpu.sh train-scaling --operator deeponet
```

- **Tuning:** `run.py` reads `SLURM_ARRAY_TASK_ID` and trains cell k of the stage.
- **`train-scaling`:** task k trains size k ÷ 5 with seed k mod 5.

| Command | Shows |
|---|---|
| `squeue -u $USER` | waiting (`PD`) and running (`R`) jobs |
| `tail -F outputs/slurm/gpu_ARRAY_TASK.out` | one task's output, live (Ctrl+C stops watching, not the job) |
| `sacct -j NUMBER` | whether a job `COMPLETED`, `FAILED` or hit `TIMEOUT` |

**Other Slurm clusters:** change the `#SBATCH --account`, `--partition` and `--gres` lines at the top of `jobs/sunbird/gpu.sh` and `cpu.sh`, and the `module load` line, to those of your cluster.

---

## 5. Documentation

| File | For |
|---|---|
| `docs/manual/commands_windows.txt` | a Windows computer, Anaconda Prompt, blocks 0–18 |
| `docs/manual/commands_sunbird.txt` | Sunbird, blocks 0–14 |
| `docs/manual/Operator_Learning_Manual.tex` | the full manual: set-up, every step, every option, troubleshooting. Compile with `pdflatex` twice |

---

## 6. Where the results go

`TAG` is `scaling_3200` for the data-size study, or `combined_0500` for the 500-case Latin-hypercube dataset (the default without `--family`). `RUN` is the run name.

| What | Where |
|---|---|
| dataset | `data/processed/TAG/protocol_v2/dataset.npz` |
| trained model | `outputs/checkpoints/TAG/RUN/best_model.pt` |
| loss history, training summary | `outputs/logs/TAG/RUN/` |
| test (or validation) errors | `outputs/study_results/TAG/RUN/` |
| figures of one run | `outputs/figures/TAG/RUN/` |
| tuning tables and winners | `outputs/study_results/TAG/tuning/` |
| tuning heatmaps | `outputs/figures/TAG/report/` (data-size study: `report/N0800/`) |
| learning curves, fitted α | `outputs/figures/scaling_3200/report/scaling.pdf`, `scaling.csv`, `scaling_rates.json` |
| DeepONet-FNO report at size N | `outputs/figures/scaling_3200/report/N0200/` … `N3200/` |

Figures are saved as PDF and 600 dpi PNG. The numbers behind every figure are saved next to it as CSV or JSON.

---

## 7. Options and settings

Every option of `run.py` has a default; type an option only to change it. `python run.py --help` lists them. The manual (section "All options") explains each one.

Settings that define the problem have no option on purpose. These are the physics, the cavity ranges, the mesh, the grid and the split seeds. They live in `src/config.py`, each with a comment. Changing one means generating the data again (`--force`).

Stopping and continuing:
- **`train` and `train-final`:** a recovery point is saved every 50 epochs. Type the same command again with `--resume`.
- **`tune`, `train-scaling` and `generate`:** type the same command again; finished work is kept.

---

## 8. Reproducibility

- **Numerics:** FP32 throughout, TF32 off, deterministic algorithms on. The same seed on the same machine gives the same numbers.
- **Data checks:** the split, the temperature scaling (from training cases only) and a SHA-256 fingerprint of the dataset are stored with the data. Every checkpoint and tuning table is checked against them.
- **One process per stage:** NGSolve and PyTorch are never imported in the same Python process (together they crash). `run.py` keeps them in separate stages, and `tests/run_all.py` runs each test file in its own process.

---

## Citation and licence

If you use this code, please cite the accompanying publication (to be added). The code is released under the GNU General Public License v3.0 (`LICENSE`).
