#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full end-to-end MBPTA profiling pipeline
#
# Steps:
#   1. Export all DNN models to ONNX (stored in ../DNN_models/)
#   2. Compile the C profiler (make)
#   3. Run the C profiler for all models across all scenes
#   4. Run KS test on all generated CSVs
#   5. Run CRPS convergence test on all generated CSVs
#   6. Generate EVT PDF/CDF plots for all generated CSVs
# =============================================================================

set -e  # Exit immediately if any command fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  MBPTA Edge GPU — Full Profiling Pipeline"
echo "============================================================"

# --------------------------------------------------------------------------
# Step 1: Export all models to ONNX
# --------------------------------------------------------------------------
echo ""
echo "--- Step 1: Exporting all models to ONNX ---"
python export_onnx.py --agent all
echo "ONNX export complete."

# --------------------------------------------------------------------------
# Step 2: Profile all models across all scenes (Python)
# --------------------------------------------------------------------------
echo ""
echo "--- Step 2: Profiling all models across all scenes (Python) ---"
python profiler.py --agent all
echo "Profiling complete."

# --------------------------------------------------------------------------
# Step 4: KS Test on all CSVs
# --------------------------------------------------------------------------
echo ""
echo "--- Step 4: Running KS Test on all CSV files ---"
python ks_test.py --all
echo "KS Test complete."

# --------------------------------------------------------------------------
# Step 5: CRPS Test on all CSVs
# --------------------------------------------------------------------------
echo ""
echo "--- Step 5: Running CRPS Test on all CSV files ---"
python crps_test.py --all
echo "CRPS Test complete."

# --------------------------------------------------------------------------
# Step 6: Generate EVT plots for all CSVs
# --------------------------------------------------------------------------
echo ""
echo "--- Step 6: Generating EVT PDF/CDF plots for all CSV files ---"
python plot_evt.py --all
echo "Plotting complete."

echo ""
echo "============================================================"
echo "  Pipeline finished successfully!"
echo "  Results are in: results/csv_files/"
echo "  Plots are in:   results/plots_png/"
echo "============================================================"
