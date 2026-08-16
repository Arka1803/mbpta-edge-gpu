import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
import os

def apply_block_maxima(data: np.ndarray, block_size: int = 50) -> np.ndarray:
    n_blocks = len(data) // block_size
    if n_blocks == 0:
        return np.array([])
    blocks = data[:n_blocks * block_size].reshape((n_blocks, block_size))
    return np.max(blocks, axis=1)

def plot_evt(csv_path: str, block_size: int = 50, output_path: str = None):
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    if 'inference_time_ms' not in df.columns:
        raise ValueError("CSV must contain an 'inference_time_ms' column.")
        
    data = df['inference_time_ms'].values
    
    print(f"Applying block maxima with block size m={block_size}...")
    maxima = apply_block_maxima(data, block_size)
    if len(maxima) < 2:
        print("Not enough data to fit Gumbel distribution.")
        return
        
    loc, scale = gumbel_r.fit(maxima)
    print(f"Fitted Gumbel parameters: loc (mu) = {loc:.4f}, scale (sigma) = {scale:.4f}")
    
    # Determine plotting limits to ignore extreme outliers (like warm-up times)
    x_min_plot = max(0, min(maxima) - scale)
    x_max_plot = min(np.max(maxima), loc + 15 * scale)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ------------------ PDF Plot ------------------
    # Histogram of empirical data (limited range to avoid flattening from outliers)
    count, bins, _ = ax1.hist(maxima, bins=30, range=(x_min_plot, x_max_plot), density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Empirical Block Maxima')
    
    # Fitted Gumbel PDF
    x = np.linspace(x_min_plot, x_max_plot, 200)
    pdf = gumbel_r.pdf(x, loc, scale)
    ax1.plot(x, pdf, 'r-', lw=2, label=f'Gumbel PDF\n$\mu={loc:.2f}, \sigma={scale:.2f}$')
    
    ax1.set_xlim(x_min_plot, x_max_plot)
    ax1.set_title('Probability Density Function (PDF)')
    ax1.set_xlabel('Inference Time (ms)')
    ax1.set_ylabel('Density')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # ------------------ CDF Plot ------------------
    # Empirical CDF
    sorted_maxima = np.sort(maxima)
    y_ecdf = np.arange(1, len(sorted_maxima) + 1) / len(sorted_maxima)
    ax2.step(sorted_maxima, y_ecdf, where='post', color='blue', lw=2, label='Empirical CDF')
    
    # Fitted Gumbel CDF
    cdf = gumbel_r.cdf(x, loc, scale)
    ax2.plot(x, cdf, 'r--', lw=2, label='Gumbel CDF')
    
    ax2.set_xlim(x_min_plot, x_max_plot)
    ax2.set_title('Cumulative Distribution Function (CDF)')
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    plt.suptitle(f'EVT Block Maxima Analysis\nFile: {os.path.basename(csv_path)}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot EVT (PDF/CDF) using Block Maxima.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV file.")
    parser.add_argument("--block_size", type=int, default=50, help="Block size for maxima (default: 50).")
    parser.add_argument("--output", type=str, default=None, help="Path to save the plot image (optional).")
    args = parser.parse_args()
    
    plot_evt(args.input, args.block_size, args.output)
