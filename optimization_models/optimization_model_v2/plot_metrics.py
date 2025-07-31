import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, shapiro, boxcox
from sklearn.preprocessing import QuantileTransformer

from optimization_models.optimization_model_v2.utils import STATS_FILENAME_V2

# Load the stats
with open(STATS_FILENAME_V2, "r") as f:
    stats = json.load(f)
    values = np.array(stats["values"])

# Metrics: [loss, latency, power, bandwidth]
metric_names = ["Loss", "Latency (s)", "Power (pW)", "Bandwidth (bps)"]

for j in range(4):  # metric index
    metric_name = metric_names[j]
    v = values[:, j]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))

    for mode, ax in enumerate(axes.flatten()):
        if mode == 0:
            norm_data = (v - v.mean()) / v.std()
            norm_type = "Regular normalization"
            label = "Normal"
        elif mode == 1:
            log_data = np.log1p(v)
            norm_data = (log_data - log_data.mean()) / log_data.std()
            norm_type = "Log normalization"
            label = "Lognormal"
        elif mode == 2:
            boxcox_data, _ = boxcox(v + 1)
            norm_data = (boxcox_data - np.mean(boxcox_data)) / np.std(boxcox_data)
            norm_type = "Box-Cox normalization"
            label = "Box-Cox"
        else:
            qt = QuantileTransformer(output_distribution="normal")
            norm_data = qt.fit_transform(v.reshape(-1, 1)).flatten()
            norm_type = "Quantile Gaussianization"
            label = "Quantile Gaussian"

        ax.hist(norm_data, bins=20, density=True, alpha=0.7, color='skyblue')
        x = np.linspace(-3, 3, 1000)
        gaussian_density = norm.pdf(x, loc=0, scale=1)
        ax.plot(x, gaussian_density, label=label, linestyle="--")

        ax.legend()
        ax.set_title(f"{metric_name} ({label})")
        ax.set_xlabel(f"{metric_name} - {norm_type}")
        ax.set_ylabel("Density")

        sk = skew(norm_data)
        kurt = kurtosis(norm_data)
        try:
            stat, pvalue = shapiro(norm_data)
        except Exception as e:
            pvalue = float('nan')

        print(f"{metric_name} - {norm_type}:")
        print(f"  Skewness: {sk:.4f}")
        print(f"  Kurtosis: {kurt:.4f}")
        print(f"  Shapiro-Wilk p-value: {pvalue:.10e}")
        print("-" * 40)

    plt.tight_layout()
    plt.show()