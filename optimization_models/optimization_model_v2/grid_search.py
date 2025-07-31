import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import optuna
import numpy as np
import torch
from model_stats import ModelStats
from mnistfnn import MNISTFNN
from typing import List, Tuple, TypeAlias, Optional

from optimization_models.model_consts import Hardware
from optimization_models.normalizing import MetricNormalizer, NormalizationType
from optimization_models.optimization_model_v2.utils import STATS_FILENAME_V2, TrialParam
from sampling_architecture import ArchitectureConstraints, ArchitectureSampler


QuadTuple: TypeAlias = Tuple[float, float, float, float]
class OptunaStudy:
    def __init__(
            self, 
            arch_sampler: ArchitectureSampler, 
            normalizer: MetricNormalizer, 
            hardware: Hardware,
            weights: QuadTuple=(0.2, 0.25, 0.35, 0.2),
        ):
        # weights are in the order of [loss, latency, power, bandwidth]

        self.cost_weights = np.array(weights)
        self.arch_sampler = arch_sampler
        self.normalizer = normalizer
        self.hardware = hardware
        
        # If True, the study will optimize for a single cost function
        # If False, it will do random sampling to get means/stds
        self.optimizing = False
        self.study: Optional[optuna.Study] = None
        
    def read_mean_std(self) -> None:
        self.normalizer.read_mean_std()
        
    def write_mean_std(self, update: bool=True) -> None:
        if self.study is None:
            raise ValueError("No study has been run yet.")
        
        values_array = np.array([t.values for t in self.study.trials if t.values is not None])

        self.normalizer.write_mean_std(values_array, update)

    def _get_model_stats(self, model: MNISTFNN) -> Tuple[float, float, float]:
        stats = ModelStats(
            model,
            self.hardware
        )
        return stats.latency_s, stats.power_pW, stats.bandwidth_bps

    def build_objective(self):
        def objective(trial: optuna.Trial):
            model = self.arch_sampler.build_model(trial)
            dw = trial.params[TrialParam.DATAWIDTH.value]

            print(f"Trial {trial.number+1}:\t datawidth = {dw},\t model = {model},")

            self.hardware.datawidth = dw #Update datawidth
            lat_s, pow_pW, bw_bps = self._get_model_stats(model)
            model.train_model()
            acc, loss = model.test_model()

            trial.set_user_attr("accuracy", acc)
            trial.set_user_attr("loss", loss)
            trial.set_user_attr("latency", lat_s)
            trial.set_user_attr("power", pow_pW)
            trial.set_user_attr("bandwidth", bw_bps)

            if not self.optimizing:
                return loss, lat_s, pow_pW, bw_bps
            else:
                metrics = np.array([loss, lat_s, pow_pW, bw_bps])
                normalized = self.normalizer.normalize(metrics)

                cost = float(normalized @ self.cost_weights)

                return float(cost)
        
        return objective

    def __call__(self, n_trials: int):

        if self.optimizing:
            sampler = optuna.samplers.TPESampler(multivariate=True, group=True)
            self.study = optuna.create_study(direction="minimize", sampler=sampler)
        else:
            sampler = optuna.samplers.RandomSampler()
            self.study = optuna.create_study(directions=["minimize"]*len(self.cost_weights), sampler=sampler)

        objective = self.build_objective()
        self.study.optimize(objective, n_trials=n_trials, callbacks=[display_trial_results])
        
        if self.optimizing:
            return self.study.best_trial
        
### Logging Functions

def display_trial_results(study, trial) -> None:
    print(f"Trial {trial.number+1} finished with "
            f"Loss={trial.user_attrs.get('loss', 'N/A'):.4f}, "
            f"Accuracy={trial.user_attrs.get('accuracy', 'N/A'):.2f}%, "
            f"Latency={trial.user_attrs.get('latency', 'N/A') * 1000:.4f}ms, "
            f"Power={trial.user_attrs.get('power', 'N/A') / 1e9:.4f}mW, "
            f"Bandwidth={trial.user_attrs.get('bandwidth', 'N/A') / 1e6:.4f}Mbps"
        )

def display_trial_params(trial):
    for name, value in trial.params.items():
        print(f"{name}: {value}")

##########


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.backends.quantized.engine = 'qnnpack'

    # params go here
    arch_constraints = ArchitectureConstraints()
    a_s = ArchitectureSampler(arch_constraints)
    n = MetricNormalizer(NormalizationType.QUANTILE_GAUSSIAN, STATS_FILENAME_V2)
    h = Hardware() # datawidth gets set in OptunaStudy

    opt_study = OptunaStudy(arch_sampler=a_s, normalizer=n, hardware=h)

    # Sample to get mean/std
    opt_study.optimizing = False
    opt_study(100)
    opt_study.write_mean_std(update=True)

    # Optimize using cost function
    opt_study.read_mean_std()
    opt_study.optimizing = True
    best_trial = opt_study(150)

    if best_trial is not None:
        best_model = opt_study.arch_sampler.recover_model(best_trial)
        print(f"Best trial found: Trial {best_trial.number+1}")
        print(best_model)
        best_model.display_model()
        display_trial_results(None, best_trial)

        print(f"Cost: {best_trial.value:.4f}")
        


# Best trial found:
# Trial 26 finished with Loss=0.9133, Latency=0.7115ms, Power=0.7673mW, Bandwidth=0.2400MBps
# Trial 26:        Head: [8], Tail: [128, 256], datawidth: 8 bits
# Cost: -2.5009

# Best trial found:
# Trial 43 finished with Loss=0.9587, Latency=0.8244ms, Power=0.7673mW, Bandwidth=0.2400MBps
# Trial 43:        Head: [8], Tail: [128, 32, 64], datawidth: 8 bits
# Cost: -2.4815

# Best trial found:
# Trial 28 finished with Loss=0.9711, Latency=0.8313ms, Power=0.7673mW, Bandwidth=0.2400MBps
# Trial 28:        Head: [8], Tail: [32, 128, 64], datawidth: 8 bits
# Cost: -2.5214

# Best trial found:
# Trial 25 finished with Accuracy=93.58%, Latency=0.7691ms, Power=0.7673mW, Bandwidth=1.9200Mbps
# Trial 25:        Head: [8], Tail: [128, 16, 32, 16], datawidth: 8 bits
# Cost: -2.4921

# Best trial found:
# Trial 42 finished with Loss=1.9198, Accuracy=94.05%, Latency=0.9834ms, Power=0.4406mW, Bandwidth=1.9200Mbps
# Layer 0: Quantize
# Layer 1: Flatten()
# Layer 2: Linear
# Layer 3: ReLU()
# Layer 4: DeQuantize
# SPLIT
# Layer 5: Linear(7, 128)
# Layer 6: ReLU()
# Layer 7: Linear(128, 128)
# Layer 8: ReLU()
# Layer 9: Linear(128, 16)
# Layer 10: ReLU()
# Layer 11: Linear(16, 10)
# None
# Cost: -0.6267

# Best trial found:
# 784 -> 16 -> 64 -> 10
# Trial 50 finished with Loss=1.3261, Accuracy=95.95%, Latency=1.2787ms, Power=1.0841mW, Bandwidth=0.0000Mbps
# Cost: -0.5831

# Best trial found:
# TX -> 8 -> 128 -> 64 -> 10
# Layer 0: Quantize
# Layer 1: Flatten()
# Layer 2: Linear
# Layer 3: ReLU()
# Layer 4: DeQuantize
# SPLIT
# Layer 5: Linear(8, 128)
# Layer 6: ReLU()
# Layer 7: Linear(128, 64)
# Layer 8: ReLU()
# Layer 9: Linear(64, 10)
# Trial 72 finished with Loss=1.8325, Accuracy=94.69%, Latency=0.9834ms, Power=0.4406mW, Bandwidth=1.9200Mbps
# Cost: -0.6387

# Best trial found: Trial 23
# 784 -> 32 -> 128 -> 10
# Trial 23 finished with Loss=0.9639, Accuracy=97.12%, Latency=2.7418ms, Power=1.5320mW, Bandwidth=0.0000Mbps
# Layer 0: Quantize
# Layer 1: Flatten()
# Layer 2: Linear(784, 32)
# Layer 3: ReLU()
# Layer 4: Linear(32, 128)
# Layer 5: ReLU()
# Layer 6: Linear(128, 10)
# Layer 7: DeQuantize
# Cost: 0.2442

# Best trial found: Trial 89
# 784 -> 16 -> 64 -> 10
# Trial 89 finished with Loss=1.3981, Accuracy=95.70%, Latency=1.2787ms, Power=1.0841mW, Bandwidth=0.0000Mbps
# Layer 1: Flatten()
# Layer 2: Linear(784, 16)
# Layer 3: ReLU()
# Layer 4: Linear(16, 64)
# Layer 5: ReLU()
# Layer 6: Linear(64, 10)
# Cost: -0.3913

# Best trial found: Trial 73
# 784 -> 32 -> 128 -> 10
# Trial 73 finished with Loss=0.9147, Accuracy=97.28%, Latency=2.7418ms, Power=1.5320mW, Bandwidth=0.0000Mbps
# Layer 1: Flatten()
# Layer 2: Linear(784, 32)
# Layer 3: ReLU()
# Layer 4: Linear(32, 128)
# Layer 5: ReLU()
# Layer 6: Linear(128, 10)
# Cost: 0.4488

# Best trial found: Trial 82
# 784 -> 16 -> 32 -> 10
# Layer 0: Flatten()
# Layer 1: Linear(784, 16)
# Layer 2: ReLU()
# Layer 3: Linear(16, 32)
# Layer 4: ReLU()
# Layer 5: Linear(32, 10)
# Trial 82 finished with Loss=1.4320, Accuracy=95.56%, Latency=1.2038ms, Power=1.0841mW, Bandwidth=0.0000Mbps
# Cost: -0.8065