import numpy as np
import optuna
from dataclasses import dataclass, field
from typing import List, Tuple
from optimization_models.model_consts import RunType
from optimization_models.optimization_model_v3_1.utils import TrialParam
from optimization_models.optimization_model_v3.mnistfnn_EE import MNISTFNN_EE

@dataclass
class ArchitectureConstraints:
    widths: List = field(default_factory=list)
    max_width: int = 256
    max_bn_width: int = 8
    max_head_depth: int = 3
    max_branch_depth: int = 3
    max_tail_depth: int = 5

    def __post_init__(self):
        self.max_head_depth = self.max_head_depth - 1
        self.max_branch_depth = self.max_branch_depth - 1
        self.max_tail_depth = self.max_tail_depth - 1

        min_log_width = int(np.log2(self.max_bn_width) + 1)
        max_log_width = int(np.log2(self.max_width) + 1)
        self.widths = [2**i for i in range(min_log_width, max_log_width)]

class ArchitectureSampler:
    def __init__(self, architecture: ArchitectureConstraints = ArchitectureConstraints()):
        self.architecture = architecture

    def _suggest_params(self, trial:optuna.Trial) -> None:
        trial.suggest_categorical(TrialParam.DATAWIDTH, [8, 16])
        trial.suggest_categorical(TrialParam.BRANCHPOINT, [1, 0])
        trial.suggest_float(TrialParam.LOSS_WEIGHT, 0.0, 1.0, step=0.05)

        trial.suggest_int(TrialParam.HEAD_DEPTH, 1, self.architecture.max_head_depth)
        trial.suggest_int(TrialParam.BRANCH_DEPTH, 0, self.architecture.max_branch_depth)
        trial.suggest_int(TrialParam.TAIL_DEPTH, -1, self.architecture.max_tail_depth)

        for i in range(self.architecture.max_head_depth):
            trial.suggest_categorical(TrialParam.HEAD_WIDTH.with_index(i), self.architecture.widths)
        # trial.suggest_int(TrialParam.BOTTLENECK_WIDTH, 1, self.architecture.max_bn_width)
        trial.suggest_categorical(TrialParam.BOTTLENECK_WIDTH, self.architecture.widths)
        for i in range(self.architecture.max_branch_depth):
            trial.suggest_categorical(TrialParam.BRANCH_WIDTH.with_index(i), self.architecture.widths)
        for i in range(self.architecture.max_tail_depth):
            trial.suggest_categorical(TrialParam.TAIL_WIDTH.with_index(i), self.architecture.widths)

    def _build_trial_layers(self, trial:optuna.Trial) -> Tuple[List[int], List[int], List[int]]:
        head_depth = trial.params[TrialParam.HEAD_DEPTH]
        branch_depth = trial.params[TrialParam.BRANCH_DEPTH]
        tail_depth = trial.params[TrialParam.TAIL_DEPTH]

        head_layers = []
        for i in range(head_depth):
            head_layers.append(trial.params[TrialParam.HEAD_WIDTH.with_index(i)])
        
        if head_depth > 0 and tail_depth > 0:
            head_layers[-1] = trial.params[TrialParam.BOTTLENECK_WIDTH]
        
        branch_layers = []
        for i in range(branch_depth):
            branch_layers.append(trial.params[TrialParam.BRANCH_WIDTH.with_index(i)])

        tail_layers = []
        for i in range(tail_depth):
            tail_layers.append(trial.params[TrialParam.TAIL_WIDTH.with_index(i)])

        return head_layers, branch_layers, tail_layers
    
    def _get_trial_runType(self, trial:optuna.Trial) -> RunType:
        head_depth = trial.params[TrialParam.HEAD_DEPTH]
        tail_depth = trial.params[TrialParam.TAIL_DEPTH]

        if head_depth == 0:
            runType = RunType.CLOUD_COMPUTING
        elif tail_depth == -1:
            runType = RunType.EDGE_COMPUTING
        else:
            runType = RunType.SPLIT_COMPUTING
        
        return runType

    def build_model(self, trial:optuna.Trial) -> MNISTFNN_EE:
        print("Building model...")
        self._suggest_params(trial)
        datawidth = trial.params[TrialParam.DATAWIDTH]
        branchpoint = trial.params[TrialParam.BRANCHPOINT]
        head_layers, branch_layers, tail_layers = self._build_trial_layers(trial)
        runType = self._get_trial_runType(trial)

        return MNISTFNN_EE(head_layers, branch_layers, tail_layers, datawidth, runType, branchpoint)
    
    def recover_model(self, trial:optuna.Trial) -> MNISTFNN_EE:
        datawidth = trial.params[TrialParam.DATAWIDTH]
        branchpoint = trial.params[TrialParam.BRANCHPOINT]
        head_layers, branch_layers, tail_layers = self._build_trial_layers(trial)
        runType = self._get_trial_runType(trial)

        return MNISTFNN_EE(head_layers, branch_layers, tail_layers, datawidth, runType, branchpoint)