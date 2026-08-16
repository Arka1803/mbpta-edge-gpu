import argparse
import pandas as pd
import numpy as np
from scipy import stats

def run_ks_test(csv_path: str, threshold: float = 0.05):
    """
    Reads execution times from a CSV and performs a two-sample KS test
    on two randomly selected consecutive blocks of sizes 50 and 100.
    """
    df = pd.read_csv(csv_path)
    if 'inference_time_ms' not in df.columns:
        raise ValueError("CSV must contain an 'inference_time_ms' column.")
        
    data = df['inference_time_ms'].values
    n = len(data)
    
    m1 = 50
    m2 = 100
    
    if n < m1 + m2:
        raise ValueError(f"Not enough data points ({n}) to sample blocks of {m1} and {m2}.")

    # Pick a random start index for the first block (m=50)
    idx1 = np.random.randint(0, n - m1 + 1)
    sample1 = data[idx1 : idx1 + m1]
    
    # Pick a random start index for the second block (m=100)
    # Ensure it doesn't overlap completely, or just randomly pick anywhere
    idx2 = np.random.randint(0, n - m2 + 1)
    sample2 = data[idx2 : idx2 + m2]
    
    # Perform two-sample KS test
    statistic, p_value = stats.ks_2samp(sample1, sample2)
    
    passed = p_value > threshold
    
    print(f"--- Kolmogorov-Smirnov (KS) Test ---")
    print(f"Data file: {csv_path}")
    print(f"Sample 1 size: {m1} (idx {idx1} to {idx1+m1-1})")
    print(f"Sample 2 size: {m2} (idx {idx2} to {idx2+m2-1})")
    print(f"KS Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Threshold: {threshold}")
    if passed:
        print("Result: PASS (Identical distribution property verified)")
    else:
        print("Result: FAIL (Identical distribution property not verified)")
        
    return p_value, passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KS Test for identical distribution property.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV file.")
    parser.add_argument("--threshold", type=float, default=0.05, help="P-value threshold (default: 0.05).")
    args = parser.parse_args()
    
    run_ks_test(args.input, args.threshold)
