"""
Statistical validation of deployment metrics.

Follows Step 6 of the deployment pipeline in Section 3.5.5 of the thesis:
latency and power measurements are repeated across independent runs and
the mean, standard deviation and 95% confidence interval are reported.
Pairwise comparisons between the proposed model and baseline models use
the same statistical protocol defined in Section 3.5.3 (paired t-test
when normality holds, Wilcoxon signed-rank test otherwise, with
Holm-Bonferroni correction within the family of comparisons).
"""
import numpy as np
from scipy import stats


def validate(samples):
    """
    Report the mean, standard deviation and 95% confidence interval of a
    one-dimensional list of measurements.

    Args:
        samples: list or array of numeric measurements.
    """
    arr = np.asarray(samples, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    ci95 = stats.t.interval(
        0.95,
        len(arr) - 1,
        loc=mean,
        scale=std / np.sqrt(len(arr)),
    )
    print(f"Mean : {mean:.4f}")
    print(f"Std  : {std:.4f}")
    print(f"95%CI: ({ci95[0]:.4f}, {ci95[1]:.4f})")


def compare(a, b):
    """
    Pairwise comparison between two measurement series using the same
    test selection rule as Section 4.1.4.

    Args:
        a: measurements of model A.
        b: measurements of model B.

    Returns:
        (test_name, p_value) tuple.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    if stats.shapiro(diff).pvalue > 0.05:
        p = float(stats.ttest_rel(a, b).pvalue)
        return "paired-t", p
    p = float(stats.wilcoxon(a, b).pvalue)
    return "wilcoxon", p


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        validate([float(x) for x in sys.argv[1:]])