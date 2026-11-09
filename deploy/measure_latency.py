"""
Latency measurement on the Snapdragon platform.

Follows Step 4 of the deployment pipeline in Section 3.5.5 of the thesis:
    - 10 warm-up runs followed by 100 timed runs
    - the per-run median is recorded
    - the measurement is repeated across 10 independent runs
    - the mean, standard deviation and 95% confidence interval are
      reported

This script parses the SNPE log for the per-run median inference time
and aggregates the ten runs.
"""
import subprocess

import numpy as np
import yaml


def measure(cfg_path="configs/default.yaml",
            dlc_path="sdvit_int8.dlc",
            input_list="input_list.txt",
            n_runs=10,
            warm_up=10,
            timed_runs=100):
    """
    Run the SNPE runtime on the target platform and report the latency
    statistics.

    Args:
        cfg_path:    path to the YAML config file (reads the SNPE root).
        dlc_path:    path to the quantized DLC file.
        input_list:  text file listing the input tensors.
        n_runs:      number of independent runs.
        warm_up:     number of warm-up iterations per run.
        timed_runs:  number of timed iterations per run.

    Returns:
        A dict with keys mean, std, ci95, per_run.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    snpe_root = cfg["deploy"]["snapdragon_snpe"]
    runner = snpe_root + "/bin/snpe-net-run"

    per_run_medians = []
    for run_idx in range(n_runs):
        out = subprocess.run(
            [
                runner,
                "--container", dlc_path,
                "--input_list", input_list,
                "--iterations", str(warm_up + timed_runs),
            ],
            capture_output=True,
            text=True,
        )
        latencies = []
        for line in out.stdout.splitlines():
            if "Total Inference Time" in line:
                try:
                    latencies.append(float(line.split(":")[-1].strip().split()[0]))
                except (ValueError, IndexError):
                    continue
        if len(latencies) >= warm_up + 1:
            timed = latencies[warm_up:]
            per_run_medians.append(float(np.median(timed)))
        else:
            per_run_medians.append(float("nan"))

    arr = np.array(per_run_medians, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    ci95 = (mean - 1.96 * std / np.sqrt(len(arr)),
            mean + 1.96 * std / np.sqrt(len(arr)))
    max_median = float(arr.max())

    print(f"Mean latency    : {mean:.2f} ms")
    print(f"Max run median  : {max_median:.2f} ms")
    print(f"Std dev         : {std:.2f} ms")
    print(f"95% CI          : ({ci95[0]:.2f}, {ci95[1]:.2f}) ms")

    return {
        "mean": mean,
        "std": std,
        "max_run_median": max_median,
        "ci95": ci95,
        "per_run": per_run_medians,
    }


if __name__ == "__main__":
    measure()