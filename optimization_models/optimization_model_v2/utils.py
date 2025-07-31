from enum import Enum
from optimization_models.model_consts import STATS_FILENAME

STATS_FILENAME_V2 = "optimization_models/optimization_model_v2/" + STATS_FILENAME

class TrialParam(Enum):
    DATAWIDTH = "dw"
    HEAD_DEPTH = "hd"
    TAIL_DEPTH = "td"
    HEAD_WIDTH = "hw"
    TAIL_WIDTH = "tw"
    BOTTLENECK_WIDTH = "bnw"

    def with_index(self, idx:int) -> str:
        if self != TrialParam.HEAD_WIDTH and self != TrialParam.TAIL_WIDTH:
            raise ValueError(f"with_index is only valid for HEAD_WIDTH or TAIL_WIDTH, not {self.name}") 

        return f"{self.value}_layer_{idx}"

