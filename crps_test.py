import argparse
import pandas as pd
import numpy as np
from scipy.stats import gumbel_r

def apply_block_maxima(data: np.ndarray, block_size: int = 50) -> np.ndarray:
    n_blocks = len(data) // block_size
    if n_blocks == 0:
        return np.array([])
    blocks = data[:n_blocks * block_size].reshape((n_blocks, block_size))
    return np.max(blocks, axis=1)

def run_crps_test(csv_path: str, block_size: int = 50, delta_n: int = 50, threshold: float = 0.001):
    """
    Evaluates the CRPS convergence metric incrementally.
    """
    df = pd.read_csv(csv_path)
    if 'inference_time_ms' not in df.columns:
        raise ValueError("CSV must contain an 'inference_time_ms' column.")
        
    data = df['inference_time_ms'].values
    
    if len(data) < 2 * block_size:
        print("Not enough data to start the test.")
        return
        
    print(f"--- Continuous Ranked Probability Score (CRPS) Convergence Test ---")
    print(f"Data file: {csv_path}")
    print(f"Total available observations: {len(data)}")
    
    # Start with enough observations to get at least 2 block maxima
    current_n = max(delta_n, 2 * block_size)
    prev_loc = None
    prev_scale = None
    
    converged = False
    
    while current_n <= len(data):
        current_data = data[:current_n]
        maxima = apply_block_maxima(current_data, block_size)
        
        if len(maxima) < 2:
            current_n += delta_n
            continue
            
        loc, scale = gumbel_r.fit(maxima)
        
        if prev_loc is not None and prev_scale is not None:
            # Calculate sum of squared differences of PDFs
            # Evaluate over a wide enough range of discrete integer execution times
            # Let's say from 0 to global max time + 10 * scale
            max_val = int(np.ceil(np.max(maxima) + 10 * scale))
            
            x_vals = np.arange(0, max_val + 1)
            pdf_X = gumbel_r.pdf(x_vals, loc=loc, scale=scale)
            pdf_Y = gumbel_r.pdf(x_vals, loc=prev_loc, scale=prev_scale)
            
            crps_metric = np.sum((pdf_X - pdf_Y) ** 2)
            
            print(f"Observations: {current_n:4d} | Maxima count: {len(maxima):3d} | CRPS: {crps_metric:.6f}")
            
            if crps_metric < threshold:
                print(f"Convergence reached at N={current_n} (CRPS = {crps_metric:.6f} < {threshold})")
                converged = True
                return crps_metric, converged, current_n
        else:
            print(f"Observations: {current_n:4d} | Maxima count: {len(maxima):3d} | CRPS: N/A (first valid block)")
            
        prev_loc = loc
        prev_scale = scale
        current_n += delta_n
        
    print(f"Test finished without reaching convergence. Lowest CRPS not below {threshold} or ran out of data.")
    return None, False, current_n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRPS Test for EVT Stability.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV file.")
    parser.add_argument("--threshold", type=float, default=0.001, help="CRPS Threshold (default: 0.001).")
    args = parser.parse_args()
    
    run_crps_test(args.input, threshold=args.threshold)
