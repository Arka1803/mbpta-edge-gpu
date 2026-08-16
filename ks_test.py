import argparse
import pandas as pd
import numpy as np
from scipy import stats

def run_ks_test_for_m(data, m, threshold):
    n = len(data)
    if n < 2 * m:
        print(f"Not enough data points ({n}) to sample two blocks of {m}.")
        return None, False

    idx1 = np.random.randint(0, n - m + 1)
    sample1 = data[idx1 : idx1 + m]
    
    idx2 = np.random.randint(0, n - m + 1)
    sample2 = data[idx2 : idx2 + m]
    
    statistic, p_value = stats.ks_2samp(sample1, sample2)
    passed = p_value > threshold
    
    print(f"--- KS Test for m={m} ---")
    print(f"Sample 1 idx: {idx1} to {idx1+m-1}")
    print(f"Sample 2 idx: {idx2} to {idx2+m-1}")
    print(f"KS Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    if passed:
        print("Result: PASS (Identical distribution property verified)")
    else:
        print("Result: FAIL (Identical distribution property not verified)")
    return p_value, passed

def run_ks_test(csv_path: str, threshold: float = 0.05):
    """
    Reads execution times from a CSV and performs a two-sample KS test
    on two randomly selected blocks of size m=50, and again for m=100.
    """
    df = pd.read_csv(csv_path)
    if 'inference_time_ms' not in df.columns:
        raise ValueError("CSV must contain an 'inference_time_ms' column.")
        
    data = df['inference_time_ms'].values
    print(f"Data file: {csv_path}")
    print(f"Threshold: {threshold}")
    
    run_ks_test_for_m(data, 50, threshold)
    print("")
    run_ks_test_for_m(data, 100, threshold)

import os
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KS Test for identical distribution property.")
    parser.add_argument("--input", type=str, help="Path to raw CSV file.")
    parser.add_argument("--all", action="store_true", help="Run test on all *_raw.csv files in results/csv_files/.")
    parser.add_argument("--threshold", type=float, default=0.05, help="P-value threshold (default: 0.05).")
    args = parser.parse_args()
    
    if args.all:
        csv_files = glob.glob(os.path.join("results", "csv_files", "*_raw.csv"))
        if not csv_files:
            print("No CSV files found in results/csv_files/")
        for f in csv_files:
            print(f"\n{'='*50}\nTesting {f}\n{'='*50}")
            run_ks_test(f, args.threshold)
    elif args.input:
        run_ks_test(args.input, args.threshold)
    else:
        parser.print_help()
        print("\nError: You must specify either --input or --all.")
