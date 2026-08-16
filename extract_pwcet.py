import argparse
import os
import numpy as np
from scipy.stats import gumbel_r

def generate_pmf(mu: float, sigma: float, points: list, dmp: float) -> list:
    """
    Generates a Probability Mass Function (PMF) from a Gumbel distribution
    across specific time bucket boundaries.
    """
    # Ensure points are sorted
    points = sorted(points)
    
    pmf = []
    prev_cdf = 0.0
    
    for pt in points:
        current_cdf = gumbel_r.cdf(pt, loc=mu, scale=sigma)
        
        # The probability mass falling into this bucket
        mass = current_cdf - prev_cdf
        pmf.append((pt, mass))
        
        prev_cdf = current_cdf
        
    # Check if the final bucket covers the required Deadline Miss Probability (DMP)
    final_exceedance = 1.0 - prev_cdf
    if final_exceedance > dmp:
        print(f"Warning: The maximum evaluation point ({points[-1]}) has an exceedance "
              f"probability of {final_exceedance:.2e}, which is greater than the target DMP ({dmp:.2e}). "
              f"You may need to add larger evaluation points.")
              
    return pmf

def main():
    parser = argparse.ArgumentParser(description="Extract pWCET discrete PMF from Gumbel parameters.")
    parser.add_argument("--mu", type=float, required=True, help="Gumbel location parameter")
    parser.add_argument("--sigma", type=float, required=True, help="Gumbel scale parameter")
    parser.add_argument("--dmp", type=float, required=True, help="Target Deadline Miss Probability")
    parser.add_argument("--points", type=float, nargs='+', required=True, help="List of manual time evaluation points")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    
    args = parser.parse_args()
    
    # Generate the PMF
    pmf = generate_pmf(args.mu, args.sigma, args.points, args.dmp)
    
    # Write to output file
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    with open(args.output, "w") as f:
        for pt, mass in pmf:
            f.write(f"{pt:.1f} {mass:.5f}\n")
            
    print(f"Successfully exported PMF to {args.output}")

if __name__ == "__main__":
    main()
