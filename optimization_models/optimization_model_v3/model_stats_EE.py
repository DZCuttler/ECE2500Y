import numpy as np
import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis
from typing import Tuple
from mnistfnn_EE import MNISTFNN_EE

from optimization_models.model_consts import Hardware, RunType, shapeType

class PrelimModuleStats():
    def __init__(self, module: nn.Sequential, input_shape: shapeType, hardware:Hardware) -> None:
        self.module = module
        self.input_shape = input_shape
        self.hw = hardware

        self.L = len(list(module.children()))

        self.macs = self._get_macs()
        self.reads, self.writes = self._get_read_write_count()
        self.cycles = self._get_cycles()
        self.memory_size_bit = self._get_memory_size()
        self.energy_pJ = self._get_energy()

    def _get_macs(self) -> int:
        if self.L == 0: return 0

        dummy_input = torch.randn(*self.input_shape).to('cpu')
        macs_head = FlopCountAnalysis(self.module, dummy_input).by_module()
        macs_list = [macs_head[str(i)] for i in range(self.L + 1)]
        return sum(macs_list)
    
    def _get_read_write_count(self) -> Tuple[int, int]:
        reads = self.macs * self.hw.reads_per_mac
        writes = self.macs * self.hw.writes_per_mac

        return reads, writes
    
    def _get_cycles(self) -> int:
        return self.hw.cycles_per_mac * self.macs + \
            self.hw.cycles_per_read * self.reads + \
            self.hw.cycles_per_write * self.writes
    
    def _get_memory_size(self) -> int:
        # Assuming the memory just needs to store all weights and inputs
        if self.input_shape is None:
            return 0
        
        memory_size_bit = np.prod(self.input_shape) * self.hw.datawidth
        for layer in list(self.module.children()):
            if hasattr(layer, 'weight') and layer.weight is not None:
                size = layer.weight.nelement() * self.hw.datawidth
                memory_size_bit += size 

        return memory_size_bit
    
    def _get_energy(self) -> float:
        # Assuming reads and writes cost the same energy
        
        # Energy for MAC operations
        energy_pJ = self.macs * self.hw.energy_per_mac_pJ[self.hw.datawidth]
        
        # Energy for reads and writes
        rounded_log_mem_size = np.ceil(np.log2(self.memory_size_bit)) if self.memory_size_bit > 0 else 0
        energy_pJ += (self.reads + self.writes) * self.hw.energy_per_read_write_k * self.hw.datawidth * np.sqrt(2**rounded_log_mem_size)

        return energy_pJ

class ModelStats_EE():
    def __init__(self, model: MNISTFNN_EE, hardware:Hardware):
        self.model = model
        self.hw = hardware

        self.prehead_stats = PrelimModuleStats(model.head_prebranch, model.head_shape, self.hw)
        self.branch_stats = PrelimModuleStats(model.branch, model.branch_shape, self.hw)
        self.posthead_stats = PrelimModuleStats(model.head_postbranch, model.branch_shape, self.hw)
        self.tail_stats = PrelimModuleStats(model.tail, model.tail_shape, self.hw)

        self.transmission_size_bit = self._get_transmission_size_bit()
        self.branch_latency_s, self.tail_latency_s = self._get_latency()
        self.power_pW = self._get_power()
        self.bandwidth_bps = self._get_bandwidth()

    def _get_transmission_size_bit(self) -> int:
        if self.model.tail_shape is None:
            return 0
        return np.prod(self.model.tail_shape) * self.hw.datawidth

    def _get_latency(self) -> Tuple[float, float]:
        prehead_latency_s = self.prehead_stats.cycles / self.hw.edge_clock_Hz
        branch_latency_s = self.branch_stats.cycles / self.hw.edge_clock_Hz
        
        # Only include tx/rx if we are doing it
        if self.model.runType == RunType.EDGE_COMPUTING:
            posthead_latency_s = self.posthead_stats.cycles / self.hw.edge_clock_Hz
            tail_latency_s = self.tail_stats.cycles  / self.hw.cloud_clock_Hz
        else:
            posthead_latency_s = (self.posthead_stats.cycles + self.hw.cycles_per_tx) / self.hw.edge_clock_Hz
            tail_latency_s = (self.tail_stats.cycles + self.hw.cycles_per_rx) /self.hw.cloud_clock_Hz

        wait_latency_s = self.transmission_size_bit / self.hw.bandwidth_bps

        early_exit_latency = prehead_latency_s + branch_latency_s
        full_exit_latency = early_exit_latency + posthead_latency_s + wait_latency_s + tail_latency_s

        return early_exit_latency, full_exit_latency

    def _get_power(self) -> float:
        edge_energy_pJ = self.prehead_stats.energy_pJ + self.branch_stats.energy_pJ + self.posthead_stats.energy_pJ
        power_pW = (edge_energy_pJ + self.hw.tx_energy_per_bit_pJ * self.transmission_size_bit) / self.tail_latency_s
        return power_pW
    
    def _get_bandwidth(self) -> float:
        return  self.transmission_size_bit / self.hw.max_tx_latency_s