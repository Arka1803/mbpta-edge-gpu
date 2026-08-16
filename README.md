# DNN GPU Inference Profiler (MBPTA-Edge-GPU)

A Python-based pipeline to profile inference execution times of Deep Neural Networks (DNNs) on an edge GPU and perform **Measurement-Based Probabilistic Timing Analysis (MBPTA)** for pWCET estimation.

The profiler runs DNN inference frame-by-frame over your video scenes, records per-frame latency using CUDA events, and feeds those timings into EVT-based statistical tests and plots.

---

## Directory Structure

```text
DNN_pWCET/
├── DNN_models/            # Pre-trained weights + ONNX files (auto-managed)
├── scenes/                # Your .mp4 / .avi video scenes (one subfolder per scene)
│   ├── day_foggy/
│   ├── day_sunny/
│   └── night_foggy/
└── mbpta-edge-gpu/
    ├── run_pipeline.sh        # One-command automated pipeline runner (Linux/macOS)
    ├── environment.yml        # Conda environment definition (Python 3.8)
    ├── requirements.txt       # pip dependency list
    ├── export_onnx.py         # (Optional) Exports PyTorch models to ONNX
    ├── profiler.py            # Step 1 — GPU inference profiler (produces CSVs + MAT files)
    ├── ks_test.py             # Step 2 — KS identical-distribution test
    ├── crps_test.py           # Step 3 — CRPS EVT convergence test
    ├── plot_evt.py            # Step 4 — EVT PDF/CDF plots (block maxima + Gumbel fit)
    ├── extract_pwcet.py       # Step 5 — Extracts discrete pWCET PMF
    └── results/
        ├── csv_files/         # Raw per-frame timing CSVs  (<model>_<scene>_raw.csv)
        ├── mat_files/         # MATLAB-compatible files     (<model>_<scene>.mat)
        └── plots_png/         # EVT PDF/CDF plots           (<model>_<scene>_evt.png)
```

> **Scene structure**: Place each video scene in its own subfolder under `../scenes/`.  
> All `.mp4` and `.avi` files inside that folder are profiled together as one scene.

---

## Environment Setup (Python 3.8)

All code is compatible with **Python 3.8+**.  
Python 3.8 is recommended for maximum edge-device compatibility.

### Option A — Conda (recommended)
```bash
conda env create -f environment.yml
conda activate mbpta-edge-gpu
```

### Option B — pip + venv
```bash
# Windows
python3.8 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Edge GPU note**: Install the CUDA-enabled PyTorch wheel matching your device's CUDA version.  
> See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to get the exact `pip install` command.

---

## Automated End-to-End Run

> **Recommended for Linux / edge device.**  
> The shell script runs the entire pipeline — profiling → tests → plots — in one command.

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

This single command executes the following steps in order:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `export_onnx.py --agent all` | Downloads weights and exports all models to ONNX |
| 2 | `profiler.py --agent all` | Profiles all models over all scenes → CSVs + MAT files |
| 3 | `ks_test.py --all` | KS test on all generated CSVs |
| 4 | `crps_test.py --all` | CRPS convergence test on all generated CSVs |
| 5 | `plot_evt.py --all` | Generates and saves EVT PDF/CDF plots for all CSVs |

All outputs land in `results/`.

---

## Manual Step-by-Step Run

### Step 0 — (Optional) Export Models to ONNX

Only needed if you plan to use TensorRT separately. The profiler loads PyTorch weights directly.

```bash
# Single model
python export_onnx.py --agent resnet18

# All models
python export_onnx.py --agent all
```

Supported models: `lenet5`, `alexnet`, `vgg16`, `googlenet`, `resnet18`, `resnet50`  
ONNX files are saved to `../DNN_models/`.

---

### Step 1 — Profile DNN Inference

Scans all subfolders in `../scenes/`, runs inference on every `.mp4` / `.avi` video frame-by-frame using PyTorch CUDA events (falls back to `time.perf_counter()` on CPU-only machines), and saves outputs.

```bash
# Profile one model across all scenes
python profiler.py --agent resnet18

# Profile all supported models across all scenes
python profiler.py --agent all

# Custom deadline miss probability
python profiler.py --agent resnet18 --dmp 1e-9
```

**Outputs per model–scene pair:**
- `results/csv_files/<model>_<scene>_raw.csv` — raw per-frame inference times (ms)
- `results/mat_files/<model>_<scene>.mat` — MATLAB file with times, block maxima, Gumbel params

---

### Step 2 — KS Test (Identical Distribution Check)

Verifies the **identical distribution** property required by MBPTA by randomly drawing two blocks of size *m* and applying a two-sample KS test.

```bash
# Single CSV file
python ks_test.py --input "results/csv_files/resnet18_day_foggy_raw.csv"

# All CSV files in results/csv_files/ at once
python ks_test.py --all

# Custom significance threshold (default: 0.05)
python ks_test.py --all --threshold 0.05
```

Tested block sizes: **m = 50** and **m = 100**.  
A p-value > threshold → **PASS** (identical distribution verified).

---

### Step 3 — CRPS Convergence Test (Minimum Number of Runs)

Evaluates whether the fitted EVT (Gumbel) distribution has **stabilised** as more observations are added, giving the Minimum Number of Runs (MNR).

```bash
# Single CSV file
python crps_test.py --input "results/csv_files/resnet18_day_foggy_raw.csv"

# All CSV files at once
python crps_test.py --all

# Custom CRPS threshold (default: 0.001)
python crps_test.py --all --threshold 0.001
```

Convergence is declared when the sum of squared PDF differences between consecutive fits falls below the threshold.

---

### Step 4 — EVT PDF/CDF Plots

Applies **block maxima** (default block size *m = 50*) to the raw timings, fits a Gumbel extreme value distribution, and produces side-by-side PDF and CDF plots.  
The X-axis is automatically clipped to remove warm-up outliers.

```bash
# Single file — show interactive plot
python plot_evt.py --input "results/csv_files/resnet18_day_foggy_raw.csv"

# Single file — save PNG
python plot_evt.py \
    --input  "results/csv_files/resnet18_day_foggy_raw.csv" \
    --output "results/plots_png/resnet18_day_foggy_evt.png"

# All CSVs at once — auto-saves to results/plots_png/
python plot_evt.py --all

# Custom block size
python plot_evt.py --all --block_size 100
```

**Fitted parameters** (loc μ and scale σ) are printed to the terminal and are the inputs to Step 5.

---

### Step 5 — pWCET PMF Extraction

Takes the Gumbel parameters from Step 4 and produces a discrete **Probability Mass Function (PMF)** for schedulability analysis.

```bash
python extract_pwcet.py \
    --mu     5.57 \
    --sigma  1.59 \
    --dmp    1e-9 \
    --points 15.0 20.0 25.0 30.0 \
    --output results/pwcet_dist.txt
```

| Argument | Description |
|----------|-------------|
| `--mu` | Gumbel location parameter (from `plot_evt.py` output) |
| `--sigma` | Gumbel scale parameter (from `plot_evt.py` output) |
| `--dmp` | Target Deadline Miss Probability (e.g. `1e-9`) |
| `--points` | Upper-bound time values for each PMF bucket (ms) |
| `--output` | Path to save the PMF text file |

A warning is printed if the last evaluation point does not fully cover the target DMP.

---

## Outputs Summary

| File pattern | Script | Description |
|---|---|---|
| `results/csv_files/<model>_<scene>_raw.csv` | `profiler.py` | Raw per-frame inference times (ms) |
| `results/mat_files/<model>_<scene>.mat` | `profiler.py` | MATLAB file: times, maxima, Gumbel params |
| `results/plots_png/<model>_<scene>_evt.png` | `plot_evt.py` | EVT PDF + CDF side-by-side plot |
| `results/pwcet_dist.txt` | `extract_pwcet.py` | Discrete pWCET PMF for schedulability |

---

## Supported Models

| Name | Architecture |
|------|-------------|
| `lenet5` | Custom LeNet-5 (224×224 input, no pretrained weights) |
| `alexnet` | AlexNet (ImageNet pretrained) |
| `vgg16` | VGG-16 (ImageNet pretrained) |
| `googlenet` | GoogLeNet / Inception v1 (ImageNet pretrained) |
| `resnet18` | ResNet-18 (ImageNet pretrained) |
| `resnet50` | ResNet-50 (ImageNet pretrained) |

Pretrained weights are downloaded automatically to `../DNN_models/` on first run.
