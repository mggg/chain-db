import numpy as np
from typing import Dict


def summary_stats(hist: Dict[float, float]) -> Dict[str, float]:
    items = sorted(hist.items(), key=lambda kv: kv[0])
    vals = np.array([kv[0] for kv in items])
    weights = np.array([kv[1] for kv in items])
    size = len(items)
    # see https://stackoverflow.com/a/22639392
    percentiles = 100 * np.cumsum(weights) / np.sum(weights)

    def percentile(p: float) -> float:
        idx = max(min(len(percentiles[percentiles <= p]) - 1, size - 1), 0)
        return vals[idx]

    # TODO (mean, median, q1, q3, configurable tails, mean, mode,
    #       stddev, min, max)
    mean = np.average(vals, weights=weights)
    # NumPy does not include a weighted stddev function. See
    # https://stackoverflow.com/a/2415343
    stddev = np.sqrt(np.average((vals - mean)**2, weights=weights))

    # TODO: what percentiles should be included here?
    # (e.g. 68-95-99.7?) Can we compute them more efficiently?
    return {
        'mean': mean,
        'stddev': stddev,
        'p0.1': percentile(0.1),
        'p1': percentile(1),
        'p5': percentile(5),
        'p10': percentile(10),
        'q1': percentile(25),
        'median': percentile(50),
        'q3': percentile(75),
        'p90': percentile(90),
        'p95': percentile(95),
        'p99': percentile(99),
        'p99.9': percentile(99.9),
        'min': np.min(vals),
        'max': np.max(vals),
        'modes': list(vals[weights == np.max(weights)])
    }
