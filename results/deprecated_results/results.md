# MBPTA Statistical Tests Execution Results

The following table summarizes the execution of the KS and CRPS convergence tests on the `googlenet_day_foggy_raw.csv` dataset.

### Execution Results
| Execution Phase | Statistical Test | Target Property | Computed Metric | Threshold Condition | Outcome / Next Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Collecting | KS Test (Two-Sample) | Identical Distribution | 0.0000 | p-value > 0.05 | Fail |
| Convergence | CRPS | EVT Stability | 0.000183 | Score < 0.001 | Converged |
