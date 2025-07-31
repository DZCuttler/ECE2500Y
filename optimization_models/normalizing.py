from enum import Enum
import json
import numpy as np
from scipy.special import boxcox1p
from scipy.stats import boxcox
from sklearn.preprocessing import QuantileTransformer

class NormalizationType(Enum):
    STANDARD = "standard"
    LOG_STANDARD = "log_standard"
    BOX_COX = "box-cox"
    QUANTILE_GAUSSIAN = "quantile_gaussian"

# base class
class Normalizer:
    def is_ready(self) -> bool:
        raise NotImplementedError

    def fit(self, values: np.ndarray) -> None:
        raise NotImplementedError
    
    def normalize(self, metrics: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# standard normalizer
class StandardNormalizer(Normalizer):
    def __init__(self):
        self.means = None
        self.stds = None

    def is_ready(self):
        return self.means is not None and self.stds is not None

    def fit(self, values):
        self.means = values.mean(axis=0)
        self.stds = values.std(axis=0)

    def normalize(self, metrics):
        return (metrics - self.means) / self.stds

# log-standard normalizer
class LogStandardNormalizer(StandardNormalizer):
    def fit(self, values):
        values = np.log1p(values)
        super().fit(values)

    def normalize(self, metrics):
        metrics = np.log1p(metrics)
        return super().normalize(metrics)

# box-cox normalizer
class BoxCoxNormalizer(Normalizer):
    def __init__(self):
        self.means = None
        self.stds = None
        self.lambdas = None

    def is_ready(self):
        return self.means is not None and self.stds is not None and self.lambdas is not None

    def fit(self, values):
        transformed = []
        self.lambdas = []

        for i in range(values.shape[1]):
            v_boxcox, lam = boxcox(values[:, i] + 1)
            transformed.append(v_boxcox)
            self.lambdas.append(lam)

        transformed = np.stack(transformed, axis=1)
        self.means = transformed.mean(axis=0)
        self.stds = transformed.std(axis=0)

    def normalize(self, metrics):
        transformed = []
        for m, l in zip(metrics, self.lambdas):
            if l == 0:
                transformed.append(np.log1p(m))
            else:
                transformed.append(boxcox1p(m, l))

        transformed = np.array(transformed)
        return (transformed - self.means) / self.stds

# quantile gaussian normalizer
class QuantileGaussianNormalizer(Normalizer):
    def __init__(self):
        self.transformer = None

    def is_ready(self):
        return self.transformer is not None

    def fit(self, values):
        self.transformer = QuantileTransformer(output_distribution="normal")
        self.transformer.fit(values)

    def normalize(self, metrics):
        metrics = metrics.reshape(1, -1)
        return self.transformer.transform(metrics).flatten()

class MetricNormalizer:
    def __init__(self, normalize_type: NormalizationType, filename):
        self.filename = filename
        self.normalize_type = normalize_type
        self.normalizer = self._get_normalizer(normalize_type)

    def _get_normalizer(self, normalize_type):
        match normalize_type:
            case NormalizationType.STANDARD:
                return StandardNormalizer()
            case NormalizationType.LOG_STANDARD:
                return LogStandardNormalizer()
            case NormalizationType.BOX_COX:
                return BoxCoxNormalizer()
            case NormalizationType.QUANTILE_GAUSSIAN:
                return QuantileGaussianNormalizer()
    
    def _is_ready(self):
        return self.normalizer.is_ready()

    def normalize(self, metrics: np.ndarray) -> np.ndarray:
        if not self.normalizer.is_ready():
            raise ValueError("Cannot normalize before fitting")
        
        return self.normalizer.normalize(metrics)

    def read_mean_std(self):
        with open(self.filename, "r") as f:
            stats = json.load(f)
            values = np.array(stats["values"])

        self.normalizer.fit(values)

    def write_mean_std(self, values: np.ndarray, update: bool = True):
        if update:
            with open(self.filename, "r") as f:
                old_stats = json.load(f)
                old_values = np.array(old_stats["values"])
            values = np.vstack([old_values, values])
        
        stats = {"values": values.tolist()}

        with open(self.filename, "w") as f:
            json.dump(stats, f)