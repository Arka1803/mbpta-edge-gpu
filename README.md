# DNN GPU Inference Profiler

This tool profiles the inference execution time of various Deep Neural Networks (DNNs) on an edge GPU for Worst-Case Execution Time (WCET) analysis. It uses highly accurate TensorRT profiling via a `trtexec` wrapper written in C, running across all scenes automatically.

## Features
- **Accurate Profiling**: Uses TensorRT's native `trtexec` to capture extremely accurate, iteration-by-iteration inference execution times on the device.
- **Scene-Aware**: Automatically discovers and runs profiling across every video scene in the `scenes/` folder.
- **ONNX Export**: Provides a utility to export standard PyTorch models to ONNX for ingestion by TensorRT.
- **Dynamic Model Loading**: Supports standard `torchvision` models seamlessly.
- **One-Command Pipeline**: A single shell script runs the entire pipeline end-to-end.

## Directory Structure
Ensure your workspace resembles the following structure:
```text
DNN_pWCET/
├── DNN_models/        # Pre-trained models cache + ONNX files (auto-managed)
├── scenes/            # Place your .mp4 / .avi video scenes here
└── mbpta-edge-gpu/
    ├── run_pipeline.sh    # One-command end-to-end pipeline runner
    ├── export_onnx.py     # Exports PyTorch models to ONNX
    ├── profiler.c         # C wrapper for trtexec profiling (scene-aware)
    ├── Makefile           # Builds the C profiler
    ├── extract_pwcet.py   # Extracts pWCET discrete PMF
    ├── ks_test.py         # Performs the Kolmogorov-Smirnov (KS) Test
    ├── crps_test.py       # Computes Continuous Ranked Probability Score (CRPS)
    ├── plot_evt.py        # Plots PDF and CDF using EVT block maxima method
    └── results/           # Contains organized outputs
        ├── csv_files/     # Raw timing CSV files (one per model × scene)
        └── plots_png/     # Generated EVT PDF/CDF plots
```

## Quick Start (Recommended)

The easiest way to run the entire pipeline is with the provided shell script:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```
This single command will:
1. Export all models to ONNX (into `../DNN_models/`)
2. Compile the C profiler
3. Profile every model against every scene in `../scenes/`
4. Run the KS test on all output CSVs
5. Run the CRPS convergence test on all output CSVs
6. Generate and save EVT PDF/CDF plots for all output CSVs

---

## Step-by-Step Manual Run

### 1. Export Models to ONNX
```bash
# Single model
python export_onnx.py --agent resnet18

# All models
python export_onnx.py --agent all
```
Supported models: `lenet5`, `alexnet`, `vgg16`, `googlenet`, `resnet18`, `resnet50`.
ONNX files are stored alongside the pre-trained model cache in `../DNN_models/`.

### 2. Compile the C Profiler
```bash
make
```

### 3. Profile the Model(s)
The profiler automatically scans `../scenes/` for all `.mp4` and `.avi` files and
runs TensorRT inference for each model–scene pair. No iteration count is needed.

```bash
# Profile a single model across all scenes
./profiler resnet18

# Profile all models across all scenes
./profiler all
```
Output files are saved as `results/csv_files/<model>_<scene>_raw.csv`.

---

## Outputs

For each model–scene pair the pipeline generates:
1. **Raw Timing Data**: `results/csv_files/<model>_<scene>_raw.csv` — per-inference latency in ms.
2. **Engine Files**: Cached TensorRT engine files in `trt_engines/`.
3. **EVT Plots**: PDF and CDF plots in `results/plots_png/<model>_<scene>_evt.png`.

---

## pWCET Distribution Extraction (`extract_pwcet.py`)

This utility takes the Gumbel parameters (loc $\mu$, scale $\sigma$) obtained from `plot_evt.py` and generates a discrete Probability Mass Function (PMF) formatted for schedulability analysis.

```bash
python extract_pwcet.py --mu 5.57 --sigma 1.59 --dmp 1e-9 --points 15.0 20.0 25.0 --output results/pwcet_dist.txt
```
If the final evaluation point does not cover the target Deadline Miss Probability, the tool will issue a warning.

---

## MBPTA Statistical Tests

Tools to evaluate statistical properties of the execution time traces per MBPTA requirements.

### 1. Kolmogorov-Smirnov (KS) Test (`ks_test.py`)
Verifies the identical distribution property by randomly selecting two sequential blocks of size $m=50$, and another pair of blocks of size $m=100$ elements, and applying a two-sample KS test on each pair independently.

```bash
# Single file
python ks_test.py --input "results/csv_files/resnet18_day_foggy_raw.csv"

# All CSVs at once
python ks_test.py --all
```

### 2. Continuous Ranked Probability Score (CRPS) Test (`crps_test.py`)
Determines if the EVT distribution has stabilized to find the Minimum Number of Runs (MNR). Incrementally evaluates blocks of 50 observations.

```bash
# Single file
python crps_test.py --input "results/csv_files/resnet18_day_foggy_raw.csv"

# All CSVs at once
python crps_test.py --all
```

### 3. EVT Distribution Plotting (`plot_evt.py`)
Reads CSVs, applies block maxima (default block size $m=50$), fits a Gumbel extreme value distribution, and produces PDF and CDF plots. The X-axis is automatically bounded to exclude extreme warm-up outliers.

```bash
# Single file (interactive window)
python plot_evt.py --input "results/csv_files/resnet18_day_foggy_raw.csv"

# Single file (save to PNG)
python plot_evt.py --input "results/csv_files/resnet18_day_foggy_raw.csv" --output "results/plots_png/resnet18_day_foggy_evt.png"

# All CSVs at once (auto-saves to results/plots_png/)
python plot_evt.py --all
```
