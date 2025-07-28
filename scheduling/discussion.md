# Discussion of Simulation Results

This section discusses the results obtained from the carbon-intelligent workload scheduling simulation. We analyze the performance of different scheduling policies across two datasets, Alibaba and Bitbrains (BB), focusing on energy consumption and green energy utilization. The implementation details underpinning these results are first summarized, followed by a detailed analysis of the averaged metrics, with particular emphasis on the impact of CPU utilization prediction accuracy.

## 1. Simulation Implementation Overview

The simulation framework is designed to evaluate the efficacy of leveraging CPU utilization predictions for optimizing workload scheduling in terms of green energy use. Key components of the simulation include:

*   **Workload and Server Configuration**: Workloads are defined by their CPU core requirements and duration. Servers have defined capacities and a power model (`power_model`) that estimates power consumption based on CPU load (idle power, max power, power per core).
*   **Data Handling**:
    *   CPU utilization data (actual and predicted) and solar photovoltaic (PV) generation data are used.
    *   Solar PV data is upsampled to a 5-minute resolution and normalized to a [0,1] range for the entire day before being aligned with CPU data for each machine. This `solar_pv_aligned` represents the relative availability of green energy.
    *   CPU utilization data is used as percentages (0-100%).
*   **Scheduling Policies**:
    *   **`GREEN_OBLIVIOUS`**: Schedules workloads as soon as resources are available, without considering carbon intensity or solar PV availability. This serves as a baseline.
    *   **`NAIVE_CARBON_AWARE`**: An oracle policy that uses *actual* future CPU utilization and solar PV availability to find the optimal scheduling window. It aims to maximize green energy use by deferring workloads to periods with high solar output, assuming perfect foresight of CPU load.
    *   **`PREDICTIVE_CARBON_AWARE`**: Uses *predicted* CPU utilization and actual solar PV availability to make scheduling decisions. The goal is to emulate a real-world scenario where future CPU load is uncertain.
*   **Scoring Mechanism**:
    *   The `calculate_energy_score` function evaluates potential scheduling windows.
    *   It considers both normalized solar availability (`normalized_solar` in [0,1]) and normalized estimated power consumption (`normalized_estimated_power` in [0,1]). The estimated power is derived from the `power_model` using either actual CPU (for `NAIVE_CARBON_AWARE`) or predicted CPU (for `PREDICTIVE_CARBON_AWARE`).
    *   The scoring formula is: `score = normalized_solar * (1 - normalized_estimated_power)`. A higher score indicates a more favorable window (high solar, low anticipated power consumption).
    *   The simulation engine (`simulate`) uses this score within `find_best_scheduling_window` to select the start time for each workload.

## 2. Analysis of Averaged Metrics

The simulation was run using CPU utilization prediction data from two algorithms: CEDL and TempoSight. The following tables summarize the machine-averaged results for the Alibaba and Bitbrains (BB) datasets. We focus on `energy_consumed_wh` (total energy) and `geuf_percentage` (Green Energy Utilization Factor).

### 2.1. Alibaba Dataset Results

| Algorithm   | Policy                    | avg_cpu_actual | avg_cpu_predicted | avg_solar_pv | energy_consumed_wh | green_energy_wh | grid_energy_wh | geuf_percentage |
|-------------|---------------------------|----------------|-------------------|--------------|--------------------|-----------------|----------------|-----------------|
| CEDL        | GREEN_OBLIVIOUS           | 30.43          | 30.33             | 0.11         | 260.85             | 98.77           | 162.09         | 37.85           |
| CEDL        | NAIVE_CARBON_AWARE        | 30.84          | 30.70             | 0.17         | 261.68             | 111.66          | 150.02         | 42.79           |
| CEDL        | PREDICTIVE_CARBON_AWARE   | 30.84          | 30.72             | 0.16         | 261.68             | 109.77          | 151.91         | 42.01           |
| TempoSight  | GREEN_OBLIVIOUS           | 30.19          | 30.58             | 0.09         | 260.39             | 57.93           | 202.46         | 22.33           |
| TempoSight  | NAIVE_CARBON_AWARE        | 32.07          | 32.03             | 0.17         | 264.14             | 73.60           | 190.53         | 28.12           |
| TempoSight  | PREDICTIVE_CARBON_AWARE   | 31.95          | 32.34             | 0.18         | 263.91             | 74.52           | 189.38         | 28.57           |

*(Data based on `Alibaba_machine_averaged_results.csv` from run `run_20250511_195836`)*

**Observations for Alibaba Dataset:**

*   **Baseline Performance (`GREEN_OBLIVIOUS`)**: This policy results in the lowest `geuf_percentage` for both CEDL (37.85%) and TempoSight (22.33%), as expected, since it does not optimize for green energy. The `avg_solar_pv` is also generally lower, indicating that workloads are often scheduled during periods of lower solar availability.
*   **Oracle Performance (`NAIVE_CARBON_AWARE`)**: This policy significantly improves `geuf_percentage` compared to the oblivious baseline for both CEDL (42.79%) and TempoSight (28.12%). This demonstrates the potential benefit of carbon-aware scheduling if perfect future knowledge of CPU load were available. The higher `avg_solar_pv` reflects successful shifting of workloads to sunnier periods.
*   **Predictive Performance (`PREDICTIVE_CARBON_AWARE`)**:
    *   For **CEDL**, the `PREDICTIVE_CARBON_AWARE` policy achieves a `geuf_percentage` of 42.01%, which is very close to the oracle policy's 42.79%. This suggests that CEDL's CPU predictions are accurate enough to enable effective green energy utilization, nearly matching the performance with perfect foresight. The `avg_cpu_actual` (30.84) and `avg_cpu_predicted` (30.72) for the scheduled workloads under this policy are very close, indicating high prediction accuracy during the selected windows.
    *   For **TempoSight**, the `PREDICTIVE_CARBON_AWARE` policy achieves a `geuf_percentage` of 28.57%, which is slightly *higher* than the `NAIVE_CARBON_AWARE` policy (28.12%). While counter-intuitive for an oracle, this can occur due to the heuristic nature of the scoring function and how predictions interact with it. If predictions consistently overestimate load during low solar periods or underestimate during high solar, it might lead to slightly different (and in this case, marginally better for GEUF) scheduling choices than the oracle which uses actuals. The `avg_cpu_actual` (31.95) and `avg_cpu_predicted` (32.34) for scheduled workloads are reasonably close, but the slight overprediction by TempoSight might play a role here.
*   **Energy Consumption**: Total energy consumption (`energy_consumed_wh`) is broadly similar across policies for a given algorithm, with minor variations. This is expected as the same workload is being scheduled, and the primary optimization target is the *source* of energy (green vs. grid) rather than total consumption, although shifting to periods of lower CPU load (if correlated with solar) could also slightly reduce consumption.

### 2.2. Bitbrains (BB) Dataset Results

| Algorithm   | Policy                    | avg_cpu_actual | avg_cpu_predicted | avg_solar_pv | energy_consumed_wh | green_energy_wh | grid_energy_wh | geuf_percentage |
|-------------|---------------------------|----------------|-------------------|--------------|--------------------|-----------------|----------------|-----------------|
| CEDL        | GREEN_OBLIVIOUS           | 9.35           | 30.20             | 0.00         | 218.71             | 0.00            | 218.71         | 0.00            |
| CEDL        | NAIVE_CARBON_AWARE        | 44.23          | 41.51             | 0.32         | 288.46             | 116.54          | 171.92         | 35.59           |
| CEDL        | PREDICTIVE_CARBON_AWARE   | 15.00          | 64.90             | 0.29         | 229.99             | 104.80          | 125.20         | 44.90           |
| TempoSight  | GREEN_OBLIVIOUS           | 9.34           | 30.13             | 0.00         | 218.67             | 0.00            | 218.67         | 0.00            |
| TempoSight  | NAIVE_CARBON_AWARE        | 44.23          | 42.78             | 0.35         | 288.45             | 120.95          | 167.50         | 37.28           |
| TempoSight  | PREDICTIVE_CARBON_AWARE   | 14.75          | 62.94             | 0.29         | 229.50             | 90.15           | 139.35         | 37.48           |

*(Data based on `BB_machine_averaged_results.csv` from run `run_20250511_195836`)*

**Observations for Bitbrains (BB) Dataset:**

*   **Baseline Performance (`GREEN_OBLIVIOUS`)**: This policy results in 0% `geuf_percentage` for both algorithms. The `avg_solar_pv` is 0.0, indicating that the default scheduling (immediate) happens to fall entirely outside solar generation periods for this dataset and workload configuration. This starkly contrasts with the Alibaba dataset, highlighting dataset-specific characteristics. The `avg_cpu_actual` for scheduled workloads is very low (around 9.3%).
*   **Oracle Performance (`NAIVE_CARBON_AWARE`)**: This policy dramatically improves `geuf_percentage` to 35.59% for CEDL and 37.28% for TempoSight. This shows a significant potential for green energy utilization by shifting workloads. The `avg_cpu_actual` for scheduled workloads is much higher (around 44%) compared to the oblivious policy, suggesting workloads are shifted to periods that might also have higher baseline CPU activity but coincide with solar availability. Energy consumption also increases, likely due to running workloads at higher average CPU utilization.
*   **Predictive Performance (`PREDICTIVE_CARBON_AWARE`)**:
    *   For **CEDL**, the `PREDICTIVE_CARBON_AWARE` policy achieves a `geuf_percentage` of 44.90%, which is notably *higher* than the `NAIVE_CARBON_AWARE` policy (35.59%). This is a significant outperformance. Looking at the CPU figures for scheduled workloads, `avg_cpu_actual` (15.00%) is much lower than what the oracle scheduled at (44.23%), and `avg_cpu_predicted` (64.90%) is substantially higher than both. This large discrepancy between predicted and actual CPU (significant overprediction) seems to have led the predictive policy to choose different, ultimately more effective, scheduling windows for green energy utilization. The policy, anticipating very high power draw based on predictions, likely became highly selective for periods of maximum solar, which also happened to have low actual CPU load. The total energy consumed is also lower than the naive policy.
    *   For **TempoSight**, `PREDICTIVE_CARBON_AWARE` (37.48% GEUF) also slightly outperforms `NAIVE_CARBON_AWARE` (37.28% GEUF). Similar to CEDL, `avg_cpu_actual` (14.75%) for scheduled workloads is low, while `avg_cpu_predicted` (62.94%) is very high. Again, significant overprediction of CPU load appears to guide the scheduler towards choices that, in this instance, yield better GEUF than the oracle, along with lower total energy consumption.

## 3. The Critical Role of CPU Prediction Accuracy

The results highlight the profound impact of CPU utilization prediction accuracy on the effectiveness of the `PREDICTIVE_CARBON_AWARE` scheduling policy.

*   **When Predictions are Accurate (Alibaba - CEDL)**: As seen with the CEDL algorithm on the Alibaba dataset (`avg_cpu_actual` 30.84 vs. `avg_cpu_predicted` 30.72 for the scheduled workloads under the `PREDICTIVE_CARBON_AWARE` policy), the predictive policy's performance in terms of `geuf_percentage` (42.01%) closely mirrors that of the `NAIVE_CARBON_AWARE` oracle (42.79%). This demonstrates that with accurate CPU forecasts, the predictive scheduler can make decisions that are nearly as good as those made with perfect foresight, effectively maximizing green energy use.

*   **When Predictions are Less Accurate or Biased (BB Dataset - CEDL & TempoSight)**: On the BB dataset, both CEDL and TempoSight significantly *overpredicted* CPU utilization for the `PREDICTIVE_CARBON_AWARE` policy (e.g., CEDL: actual 15.00% vs. predicted 64.90% for scheduled workloads). Paradoxically, this led to the predictive policy *outperforming* the naive (oracle) policy in terms of `geuf_percentage`.
    This counter-intuitive result arises because the scoring function `score = normalized_solar * (1 - normalized_estimated_power)` is sensitive to `normalized_estimated_power`. A high predicted CPU leads to a high `normalized_estimated_power`, which heavily penalizes the score for a given solar level. Consequently, the scheduler becomes extremely selective, only choosing windows with very high solar output to compensate for the anticipated high power draw. If these high-solar windows also happen to coincide with periods where the *actual* CPU load turns out to be low (contrary to predictions), the outcome can be favorable in terms of GEUF and even total energy consumed.
    However, this "beneficial misprediction" is not a reliable strategy. If the overprediction led the scheduler to avoid windows that would have been good with accurate (lower) CPU forecasts, or if it led to excessive deferral and SLA violations (not measured here), the overall performance would suffer.

*   **General Implications**:
    *   The primary goal of CPU prediction in this context is to accurately estimate future power draw to make informed trade-offs against solar availability.
    *   Systematic biases in predictions (e.g., consistent overestimation or underestimation) can lead to suboptimal or unpredictable scheduling behavior. While overestimation might accidentally improve GEUF in some scenarios (as seen with the BB dataset), it could also lead to underutilization of viable green energy windows or unnecessary delays.
    *   Underestimation of CPU load could lead the `PREDICTIVE_CARBON_AWARE` policy to be too aggressive, scheduling workloads during periods it believes will have low energy consumption, only for the actual consumption to be higher, thereby reducing the actual green energy fraction.
    *   The closer the `avg_cpu_predicted` is to `avg_cpu_actual` for the chosen schedule, the more reliably the `PREDICTIVE_CARBON_AWARE` policy can approach the optimal performance suggested by `NAIVE_CARBON_AWARE`.

## 4. Conclusion and Future Work

The simulation results demonstrate that carbon-aware scheduling policies can significantly increase the utilization of green energy compared to oblivious approaches. The `PREDICTIVE_CARBON_AWARE` policy shows considerable promise, with its effectiveness being intrinsically linked to the accuracy of the underlying CPU utilization predictions.

On the Alibaba dataset, where CEDL provided accurate predictions, the predictive policy nearly matched the oracle's performance. On the BB dataset, significant overpredictions by both algorithms led to the predictive policy unexpectedly surpassing the oracle in GEUF. This highlights the complex interplay between prediction accuracy, the heuristic nature of the scheduling score, and dataset characteristics.

Future work should focus on:
*   Investigating the impact of different types and magnitudes of prediction errors (e.g., bias, variance) on scheduling performance.
*   Exploring more sophisticated power models that might capture nuanced hardware behavior.
*   Incorporating other factors into the scheduling decision, such as workload deadlines or Service Level Agreements (SLAs), to provide a more holistic evaluation.
*   Evaluating a wider range of CPU prediction algorithms and their performance characteristics across diverse datasets.

The development of highly accurate CPU prediction models is paramount for realizing the full potential of predictive carbon-aware workload scheduling, ensuring that decisions are both green and efficient in a reliable and predictable manner.
