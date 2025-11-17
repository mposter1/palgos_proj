import numpy as np


def runtime_stats(time: list) -> str:
    times = np.array(time)
    med = np.median(times) * 1000
    low = np.min(times) * 1000
    high = np.max(times) * 1000
    std_dev = np.std(times) * 1000

    return f"{med}, {low}, {high}, {std_dev}"
