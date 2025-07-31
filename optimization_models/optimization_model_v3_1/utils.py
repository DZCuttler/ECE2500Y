from enum import Enum
from optimization_models.model_consts import STATS_FILENAME

STATS_FILENAME_V3_1 = "optimization_models/optimization_model_v3_1/" + STATS_FILENAME

class TrialParam(str, Enum):
    DATAWIDTH = "dw"
    HEAD_DEPTH = "hd"
    BRANCH_DEPTH = "bd"
    TAIL_DEPTH = "td"
    HEAD_WIDTH = "hw"
    BRANCH_WIDTH = "bw"
    TAIL_WIDTH = "tw"
    BOTTLENECK_WIDTH = "bnw"
    BRANCHPOINT = "bp"
    LOSS_WEIGHT = "lw"

    def with_index(self, idx:int) -> str:
        if self not in [TrialParam.HEAD_WIDTH, TrialParam.TAIL_WIDTH, TrialParam.BRANCH_WIDTH]:
            raise ValueError(f"with_index is only valid for HEAD_WIDTH, BRANCH_WIDTH or TAIL_WIDTH, not {self.name}") 

        return f"{self.value}_layer_{idx}"