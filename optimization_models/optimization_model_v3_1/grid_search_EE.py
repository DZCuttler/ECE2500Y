from enum import Enum
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import optuna
import numpy as np
import torch
from typing import List, Tuple, TypeAlias, Optional
from optimization_models.optimization_model_v3.mnistfnn_EE import MNISTFNN_EE
from optimization_models.optimization_model_v3.model_stats_EE import ModelStats_EE

from optimization_models.model_consts import Hardware
from optimization_models.normalizing import MetricNormalizer, NormalizationType
from optimization_models.optimization_model_v3_1.sampling_architecture import ArchitectureConstraints, ArchitectureSampler
from optimization_models.optimization_model_v3_1.utils import STATS_FILENAME_V3_1, TrialParam

class TrialAttr:
    EE_ACCURACY = "early_exit_accuracy"
    EE_LOSS = "early_exit_loss"
    EE_LATENCY = "early_exit_latency"
    FE_ACCURACY = "final_exit_accuracy"
    FE_LOSS = "final_exit_loss"
    FE_LATENCY = "final_exit_latency"
    POWER = "power"
    BANDWIDTH = "bandwidth"

SexTuple: TypeAlias = Tuple[float, float, float, float, float, float]
class OptunaStudy_EE:
    def __init__(
            self,
            arch_sampler: ArchitectureSampler, 
            normalizer: MetricNormalizer, 
            hardware: Hardware,
            weights: SexTuple = (0.05, 0.15, 0.1, 0.2, 0.35, 0.15),
            constraints: SexTuple = (80, 90, None, None, None, None, None)
        ):
        # weights are in the order of [branch_loss, tail_loss, branch_latency, tail_latency, power, bandwidth]

        self.arch_sampler = arch_sampler
        self.normalizer = normalizer
        self.hardware = hardware
        self.cost_weights = np.array(weights)
        self.constraints = np.array(constraints)

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

    def _get_model_stats(self, model: MNISTFNN_EE) -> Tuple[float, float, float, float]:
        stats = ModelStats_EE(
            model,
            self.hardware
        )

        return stats.branch_latency_s, stats.tail_latency_s, stats.power_pW, stats.bandwidth_bps
    
    def build_objective(self):
        def objective(trial: optuna.Trial):
            model = self.arch_sampler.build_model(trial)
            datawidth = trial.params[TrialParam.DATAWIDTH]
            loss_weight = trial.params[TrialParam.LOSS_WEIGHT]

            print(f"Trial {trial.number+1}: datawidth={datawidth}, loss weight={loss_weight:.2f}")
            print(model)

            self.hardware.datawidth = datawidth
            branch_lat_s, tail_lat_s, pow_pW, bw_bps = self._get_model_stats(model)
            model.train_model(loss_weight)
            branch_acc, branch_loss, tail_acc, tail_loss = model.test_model()

            trial.set_user_attr(TrialAttr.EE_ACCURACY, branch_acc)
            trial.set_user_attr(TrialAttr.FE_ACCURACY, tail_acc)
            trial.set_user_attr(TrialAttr.EE_LOSS, branch_loss)
            trial.set_user_attr(TrialAttr.FE_LOSS, tail_loss)
            trial.set_user_attr(TrialAttr.EE_LATENCY, branch_lat_s)
            trial.set_user_attr(TrialAttr.FE_LATENCY, tail_lat_s)
            trial.set_user_attr(TrialAttr.POWER, pow_pW)
            trial.set_user_attr(TrialAttr.BANDWIDTH, bw_bps)

            if not self.optimizing:
                return branch_loss, tail_loss, branch_lat_s, tail_lat_s, pow_pW, bw_bps
            else:
                metrics = np.array([branch_loss, tail_loss, branch_lat_s, tail_lat_s, pow_pW, bw_bps])
                normalized = self.normalizer.normalize(metrics)

                cost = float(normalized @ self.cost_weights)
                print(f"Cost: {cost}")
                return float(cost)
         
        return objective
    
    # # SOFT constraints. Applies penalty if violated
    # def build_constraints(self):
    #     def constraints(trial:optuna.Trial):
    #         # Contrain accuracy not loss for interpretability
    #         metrics = [
    #             trial.user_attrs[TrialAttr.EE_ACCURACY],
    #             trial.user_attrs[TrialAttr.FE_ACCURACY],
    #             trial.user_attrs[TrialAttr.EE_LATENCY],
    #             trial.user_attrs[TrialAttr.FE_LATENCY],
    #             trial.user_attrs[TrialAttr.POWER],
    #             trial.user_attrs[TrialAttr.BANDWIDTH]
    #         ]

    #         is_min = [False, False, True, True, True, True]

    #         c = []
    #         for i in range(len(metrics)):
    #             if self.constraints[i] is None:
    #                 continue
    #             elif is_min:
    #                 c.append(metrics[i] - self.constraints[i])
    #             else:
    #                 c.append(self.constraints[i] - metrics[i])
            
    #         return tuple(c)
            
    #     return constraints

    def __call__(self, n_trials: int):

        if self.optimizing:            
            sampler = optuna.samplers.TPESampler(multivariate=True, group=True) #, constraints_func=self.build_constraints())
            self.study = optuna.create_study(direction="minimize", sampler=sampler)
        else:
            sampler = optuna.samplers.RandomSampler()
            self.study = optuna.create_study(directions=["minimize"]*len(self.cost_weights), sampler=sampler)

        objective = self.build_objective()
        self.study.optimize(objective, n_trials=n_trials, callbacks=[display_trial_results])
        
        if self.optimizing:
            return self.study.best_trial

### Logging Functions

def display_trial_results(study, trial:optuna.Trial) -> None:
    print(f"Trial {trial.number+1} finished with \n"
        f"\tBranch Loss={trial.user_attrs[TrialAttr.FE_LOSS]:.4f}, "
        f"Branch Accuracy={trial.user_attrs[TrialAttr.EE_ACCURACY]:.2f}%, "
        f"Full Loss={trial.user_attrs[TrialAttr.FE_LOSS]:.4f}, "
        f"Full Accuracy={trial.user_attrs[TrialAttr.FE_ACCURACY]:.2f}%,\n"
        f"\tBranch Latency={trial.user_attrs[TrialAttr.EE_LATENCY] * 1000:.4f}ms, "
        f"Tail Latency={trial.user_attrs[TrialAttr.FE_LATENCY] * 1000:.4f}ms, "
        f"Power={trial.user_attrs[TrialAttr.POWER] / 1e9:.4f}mW, "
        f"Bandwidth={trial.user_attrs[TrialAttr.BANDWIDTH] / 1e6:.4f}Mbps"
    )

def display_trial_params(trial):
    for name, value in trial.params.items():
        print(f"{name}: {value}")
##########

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.backends.quantized.engine = 'qnnpack'

    arch_constraints = ArchitectureConstraints()
    a_s = ArchitectureSampler(arch_constraints)
    n = MetricNormalizer(NormalizationType.QUANTILE_GAUSSIAN, STATS_FILENAME_V3_1)
    h = Hardware() # datawidth gets set in OptunaStudy

    opt_study = OptunaStudy_EE(arch_sampler=a_s, normalizer=n, hardware=h)

    # Sample to get mean/std
    opt_study.optimizing = False
    opt_study(100)
    opt_study.write_mean_std(update=False)

    # Optimize using cost function
    opt_study.read_mean_std()
    opt_study.optimizing = True
    best_trial = opt_study(200)
    
    if best_trial is not None:
        best_model = opt_study.arch_sampler.recover_model(best_trial)
        print(f"Best trial found:")
        print(f"Trial {best_trial.number+1}, datawidth={best_trial.params[TrialParam.DATAWIDTH]}, loss weight={best_trial.params[TrialParam.LOSS_WEIGHT]}")
        print(best_model)
        best_model.display_model()
        display_trial_results(None, best_trial)
        
        print(f"Cost: {best_trial.value:.4f}")


# Best trial found:
# Trial 62 finished with Branch Loss=2.2944, Branch Accuracy=93.24%, Full Loss=2.2253, Full Accuracy=93.34%,
#         Latency=0.7262ms, Power=0.7673mW, Bandwidth=1.6800Mbps
# Trial 62:        Head: [7], Branch: [64] (1 layers before split), Tail: [128],  datawidth: 8 bits, loss weight: 0.65
# Layer 0: Flatten()
# Layer 1: Linear(784, 7)
# Layer 2: ReLU()
# BRANCH
#         Layer 3: Linear(7, 64)
#         Layer 4: ReLU()
#         Layer 5: Linear(64, 10)
# SPLIT
# Layer 6: Linear(7, 128)
# Layer 7: ReLU()
# Layer 8: Linear(128, 10)
# Cost: -0.7080

# Best trial found:
# Trial 39 finished with Branch Loss=2.3585, Branch Accuracy=92.94%, Full Loss=2.4399, Full Accuracy=92.89%,
#         Latency=0.6802ms, Power=0.7673mW, Bandwidth=1.6800Mbps
# Trial 39:        Head: [7], Branch: [64] (0 layers before split), Tail: [32, 16],  datawidth: 8 bits, loss weight: 0.35
# Layer 0: Flatten()
# Layer 1: Linear(784, 7)
# Layer 2: ReLU()
# BRANCH
#         Layer 3: Linear(7, 64)
#         Layer 4: ReLU()
#         Layer 5: Linear(64, 10)
# SPLIT
# Layer 6: Linear(7, 32)
# Layer 7: ReLU()
# Layer 8: Linear(32, 16)
# Layer 9: ReLU()
# Layer 10: Linear(16, 10)
# Cost: -0.7667

# Best trial found:
# Trial 11 finished with 
#         Branch Loss=2.6706, Branch Accuracy=92.14%, Full Loss=2.3200, Full Accuracy=93.04%,
#         Branch Latency=0.4694ms, Tail Latency=1.8665ms, Power=0.1808mW, Bandwidth=1.4400Mbps
# Trial 11:        Head: [6], Branch: [32] (0 layers before split), Tail: [16, 128, 256],  datawidth: 8 bits, loss weight: 0.25
# Layer 0: Flatten()
# Layer 1: Linear(784, 6)
# Layer 2: ReLU()
# BRANCH
#         Layer 3: Linear(6, 32)
#         Layer 4: ReLU()
#         Layer 5: Linear(32, 10)
# SPLIT
# Layer 6: Linear(6, 16)
# Layer 7: ReLU()
# Layer 8: Linear(16, 128)
# Layer 9: ReLU()
# Layer 10: Linear(128, 256)
# Layer 11: ReLU()
# Layer 12: Linear(256, 10)
# Cost: -0.6838


# Best trial found:
# Trial 72 finished with 
#         Branch Loss=2.7620, Branch Accuracy=92.52%, Full Loss=1.9756, Full Accuracy=93.92%,
#         Branch Latency=0.5717ms, Tail Latency=1.4214ms, Power=0.3053mW, Bandwidth=1.9200Mbps
# 784 -> 8 -> BRANCH -> TX -> 8 -> 128 -> 128 -> 32 -> 10
#                \--> 8 -> 10
# Layer 1: Flatten()
# Layer 2: Linear(784, 8)
# Layer 3: ReLU()
# BRANCH
#         Layer 4: Linear(8, 10)
#         SPLIT
# Layer 7: Linear(8, 128)
# Layer 8: ReLU()
# Layer 9: Linear(128, 128)
# Layer 10: ReLU()
# Layer 11: Linear(128, 32)
# Layer 12: ReLU()
# Layer 13: Linear(32, 10)
# Cost: -0.6406

# Best trial found:
# Trial 68 finished with 
#         Branch Loss=1.8096, Branch Accuracy=94.72%, Full Loss=1.8235, Full Accuracy=94.71%,
#         Branch Latency=1.1434ms, Tail Latency=1.1578ms, Power=1.0605mW, Bandwidth=0.0000Mbps
# 784 -> 16 -> BRANCH -> 16 -> 10
#                 \--> 16 -> 10
# Layer 1: Flatten()
# Layer 2: Linear(784, 16)
# Layer 3: ReLU()
# BRANCH
#         Layer 4: Linear(16, 10)
#         Layer 6: Linear(16, 10)
# Cost: -0.6783

# Best trial found: Trial {best_trial.number+1}
# 784 -> 1 -> BRANCH -> TX -> 1 -> 256 -> 128 -> 128 -> 256 -> 10
#                \--> 1 -> 10
# Layer 0: Flatten()
# Layer 1: Linear(784, 1)
# Layer 2: ReLU()
# BRANCH
#         Layer 3: Linear(1, 10)
# SPLIT
# Layer 4: Linear(1, 256)
# Layer 5: ReLU()
# Layer 6: Linear(256, 128)
# Layer 7: ReLU()
# Layer 8: Linear(128, 128)
# Layer 9: ReLU()
# Layer 10: Linear(128, 256)
# Layer 11: ReLU()
# Layer 12: Linear(256, 10)
# Trial 83 finished with 
#         Branch Loss=16.6873, Branch Accuracy=34.14%, Full Loss=15.2836, Full Accuracy=38.33%,
#         Branch Latency=0.0715ms, Tail Latency=3.1300ms, Power=0.0087mW, Bandwidth=0.2400Mbps
# Cost: -0.9153

# Best trial found:
# Trial 108, datawidth=8, loss weight=0.25
# 784 -> 16 -> BRANCH -> 16 -> 256 -> 10
#                 \--> 16 -> 10
# Layer 0: Flatten()
# Layer 1: Linear(784, 16)
# Layer 2: ReLU()
# BRANCH
#         Layer 3: Linear(16, 10)
# Layer 4: Linear(16, 256)
# Layer 5: ReLU()
# Layer 6: Linear(256, 10)
# Trial 108 finished with 
#         Branch Loss=1.7215, Branch Accuracy=95.02%, Full Loss=1.1654, Full Accuracy=96.64%,
#         Branch Latency=1.1434ms, Tail Latency=1.7424ms, Power=0.9674mW, Bandwidth=0.0000Mbps
# Cost: -0.6198

# Best trial found:
# Trial 110, datawidth=8, loss weight=0.15000000000000002
# 784 -> 16 -> BRANCH -> 16 -> 128 -> 10
#                 \--> 16 -> 10
# Layer 0: Flatten()
# Layer 1: Linear(784, 16)
# Layer 2: ReLU()
# BRANCH
#         Layer 3: Linear(16, 10)
# Layer 4: Linear(16, 128)
# Layer 5: ReLU()
# Layer 6: Linear(128, 10)
# Trial 110 finished with 
#         Branch Loss=1.2817, Branch Accuracy=93.92%, Full Loss=1.2817, Full Accuracy=96.00%,
#         Branch Latency=1.1434ms, Tail Latency=1.4429ms, Power=0.9624mW, Bandwidth=0.0000Mbps
# Cost: -0.9942

# Best trial found:
# Trial 118, datawidth=8, loss weight=0.15000000000000002
# 784 -> 16 -> BRANCH -> 16 -> 128 -> 10
#                 \--> 16 -> 10
# Layer 0: Flatten()
# Layer 1: Linear(784, 16)
# Layer 2: ReLU()
# BRANCH
#         Layer 3: Linear(16, 10)
# Layer 4: Linear(16, 128)
# Layer 5: ReLU()
# Layer 6: Linear(128, 10)
# Trial 118 finished with 
#         Branch Loss=1.2178, Branch Accuracy=94.24%, Full Loss=1.2178, Full Accuracy=96.11%,
#         Branch Latency=1.1434ms, Tail Latency=1.4429ms, Power=0.9624mW, Bandwidth=0.0000Mbps
# Cost: -1.0949