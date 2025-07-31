import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.ao.quantization as tq
from mnist_cnn import MNIST
from fvcore.nn import FlopCountAnalysis
from skcriteria import mkdm
from skcriteria.agg import similarity
from skcriteria.pipeline import mkpipe
from skcriteria.preprocessing import scalers, invert_objectives
from model_splitting import split_model
import matplotlib.pyplot as plt
from typing import Dict, Tuple

class SplitOptimizer():
    def __init__(self, model:nn.Module, input_shape:Tuple[int, int, int, int]):
        self.model = model
        self.input_shape = input_shape

        self.datawidths = [8, 16, 32]
        self.datawidth = 8

        self.L = len(list(model.children()))
        def_array_shape = (self.L+1, len(self.datawidths)) #[split, datawidth]

        self.macs_edge = np.zeros(def_array_shape)
        self.macs_cloud = np.zeros(def_array_shape)

        self.reads_edge = np.zeros(def_array_shape)
        self.reads_cloud = np.zeros(def_array_shape)

        self.writes_edge = np.zeros(def_array_shape)
        self.writes_cloud = np.zeros(def_array_shape)

        self.memory_size_bit = np.zeros(def_array_shape) #in bytes

        self.cycles_edge = np.zeros(def_array_shape)
        self.cycles_cloud = np.zeros(def_array_shape)

        self.transmission_size_bit = np.zeros(def_array_shape) # in bytes
        self.bandwidth_bps = np.zeros(def_array_shape)
        self.latency_edge_s = np.zeros(def_array_shape)
        self.latency_s = np.zeros(def_array_shape)

        self.energy_read_write_pJ = np.zeros(def_array_shape)
        self.energy_mac_pJ = np.zeros(def_array_shape)
        self.power_pW = np.zeros(def_array_shape)

        self.area_mm2 = np.zeros(def_array_shape)

        self.accuracy = np.zeros(def_array_shape)

    def get_size_bits(self, x:torch.Tensor) -> int:
        return x.nelement() * x.element_size() * 8
    
    def get_macs(self) -> Tuple[np.ndarray, np.ndarray]:

        for dw in range(len(self.datawidths)):
            dummy_input = torch.randn(self.input_shape).to('mps')
            macs = FlopCountAnalysis(self.model, dummy_input).by_module()

            macs_list = [macs[str(i)] for i in range(self.L + 1)]

            for l in range(self.L + 1):
                self.macs_edge[l, dw] = sum(macs_list[:l])
                self.macs_cloud[l, dw] = sum(macs_list[l:])

        return self.macs_edge, self.macs_cloud


    def get_memory_size(self, input_size: Tuple[int, int, int, int]) -> np.ndarray:
        # Assuming the memory just needs to store all weights and inputs
        memory_list = []
        for layer in self.model.children():
            if hasattr(layer, 'weight') and layer.weight is not None:
                size = layer.weight.nelement()
                memory_list.append(size)
            else:
                memory_list.append(0)

        for l in range(self.L + 1):
            for i, dw in enumerate(self.datawidths):
                self.memory_size_bit[l, i] = np.prod(input_size) + sum(memory_list[:l]) * dw

        return self.memory_size_bit

    def get_read_write_count(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Assuming each MAC operation requires 3 reads and 1 write
        self.reads_edge = 3 * self.macs_edge
        self.reads_cloud = 3 * self.macs_cloud

        self.writes_edge = self.macs_edge
        self.writes_cloud = self.macs_cloud

        return self.reads_edge, self.reads_cloud, self.writes_edge, self.writes_cloud


    def get_cycles(self, cycles_per_mac:int = 5, cycles_per_read:int = 1, cycles_per_write:int = 1, cycles_per_tx:int = 5, cycles_per_rx:int = 5) -> Tuple[np.ndarray, np.ndarray]:
        # Assuming number of reads and write are equal and multipy and adds are equal (1 MAC = 1 add + 1 mult)
        # Assuming read and write sizes are equal to datawidth and they are read in parallel
        self.cycles_edge = cycles_per_mac * self.macs_edge + \
            cycles_per_read * self.reads_edge + \
            cycles_per_write * self.writes_edge + \
            cycles_per_tx
        
        self.cycles_cloud = cycles_per_mac * self.macs_cloud + \
            cycles_per_read * self.reads_cloud + \
            cycles_per_write * self.writes_cloud + \
            cycles_per_rx
        
        return self.cycles_edge, self.cycles_cloud
    
    def get_transmission_size(self) -> np.ndarray:
        input = torch.randn(self.input_shape).to('mps')

        for i, dw in enumerate(self.datawidths):
            self.transmission_size_bit[0, i] = self.get_size_bits(input)
            x = input
            for j, layer in enumerate(self.model.children()):
                x = layer(x)
                self.transmission_size_bit[j+1, i] = x.nelement() * dw

        return self.transmission_size_bit

    def get_latency(self, edge_clock_Hz:int=100e6, cloud_clock_Hz:int=250e6, bandwidth_bps:int=1e6) -> np.ndarray:
        self.latency_edge_s = self.cycles_edge / edge_clock_Hz

        self.latency_s = self.latency_edge_s + \
            self.cycles_cloud / cloud_clock_Hz + \
            self.transmission_size_bit / bandwidth_bps
        
        return self.latency_s
    
    def get_energy(self, energy_per_mac_pJ:Dict[int, float] = {8: 0.25, 16: 1, 32: 4}, energy_per_read_write_k:float = 0.0084) -> Tuple[np.ndarray, np.ndarray]:
        # Assuming reads and writes cost the same energy

        # 32bit datawidth x 32kB memory is 4.3pJ per read/write from https://www.researchgate.net/publication/220904916_A_65_nm_850_MHz_256_kbit_43_pJaccess_Ultra_Low_Leakage_Power_Memory_Using_Dynamic_Cell_Stability_and_a_Dual_Swing_Data_Link
        # Assume energy increases linearly with datawidth and sqrt with memory size ~number of columns (backed by https://ieeexplore.ieee.org/document/6757323)
        # assume E = k * D * sqrt(M) D in bits, M in Kbits
        # k = E / (D * sqrt(M)) = 4.3 / (32 * sqrt(256)) =~ 0.0084

        for i, dw in enumerate(self.datawidths):
            self.energy_mac_pJ[:, i] = self.macs_edge[:, i] * energy_per_mac_pJ[dw]


        for i, dw in enumerate(self.datawidths):
            rounded_log_mem_size = np.ceil(np.log2(self.memory_size_bit[:, i]))
            self.energy_read_write_pJ[:, i] = (self.reads_edge[:, i] + self.writes_edge[:, i])  * energy_per_read_write_k * dw * np.sqrt(2**rounded_log_mem_size)

        return self.energy_mac_pJ, self.energy_read_write_pJ

    def get_area(self, mac_reuse_factor:int=500000, area_per_mac_mm2:float=0.055) -> np.ndarray:
        # area_per_mac from https://ieeexplore.ieee.org/document/7738524

        self.area_mm2 = self.macs_edge / mac_reuse_factor * area_per_mac_mm2

        return self.area_mm2

    def quantize_model(self, model:nn.Module, images:torch.Tensor) -> nn.Module:
        # https://docs.pytorch.org/docs/main/quantization.html

        # model = tq.fuse_modules(model, [["conv", "relu"]], inplace=False)
        model = nn.Sequential(tq.QuantStub(), *model.children(), tq.DeQuantStub())
        model.qconfig = tq.get_default_qconfig("qnnpack")

        prepared_model = tq.prepare(model)
        prepared_model(images)
        return tq.convert(prepared_model.to('cpu')).to('cpu')

    def get_accuracy(self, testloader:DataLoader) -> np.ndarray:
        torch.backends.quantized.engine = 'qnnpack'

        self.model = self.model.to('cpu')
        self.model.eval()

        int8_model = copy.deepcopy(self.model)
        int8_model.eval()

        for l in range(self.L+1):
            for i, dw in enumerate(self.datawidths):
                head, tail = split_model(model, l)

                if dw == 8:
                    ihead, _ = split_model(int8_model, l)
                    images, _ = next(iter(testloader))
                    qhead = self.quantize_model(ihead, images[0:10, :, :, :])
                elif dw == 16:
                    head.half()
                else:
                    head.float()

                correct = 0
                total = 0

                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to('cpu'), labels.to('cpu')

                        if dw == 8:
                            x = qhead(inputs)
                            # print(list(qhead.children()))
                            outputs = tail(x)
                        elif dw == 16:
                            inputs = inputs.half()
                            x = head(inputs)
                            outputs = tail(x.float())
                        else:
                            x = head(inputs)
                            outputs = tail(x)

                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                self.accuracy[l, i] = 100 * correct / total

        return self.accuracy

    def get_power(self, tx_power_per_bit_pJ:float=1.2) -> np.ndarray:
        # https://ieeexplore.ieee.org/document/6894245
        self.power_pW = (self.energy_mac_pJ + self.energy_read_write_pJ) / self.latency_edge_s + tx_power_per_bit_pJ * self.transmission_size_bit

        return self.power_pW
    
    def get_bandwidth(self, max_tx_latency_s:float=1/30000) -> np.ndarray:
        self.bandwidth = self.transmission_size_bit * 8 / max_tx_latency_s
        return self.bandwidth

    def optimize(self, max=[torch.inf]*8, weights=[1/8]*8):
        # See allF for order of objectives

        allF = np.array([
            self.macs_edge,
            self.latency_s,
            self.memory_size_bit,
            self.transmission_size_bit,
            self.energy_mac_pJ,
            self.energy_read_write_pJ,
            -self.accuracy,
            self.power_pW
        ])

        F = []
        alts = []
        idxs = []

        for i in range(self.L + 1):
            for j, dw in enumerate(self.datawidths):
                if np.all(allF[:, i, j] < max):
                    F.append(allF[:, i, j].tolist())
                    alts.append(f"{i} - {dw} dw")
                    idxs.append((i, j))

        # print(np.array(F))

        if len(F) == 0:
            print("No valid splits under constraints")
            return None, None
        elif len(F) == 1:
            print("Only one valid option:", alts[0], F[0])
            return alts[0], F[0]

        dm = mkdm(F, objectives=["min"]*8, weights=weights, alternatives=alts)

        pipe = mkpipe(
            invert_objectives.NegateMinimize(),
            scalers.VectorScaler(target="matrix"),  # this scaler transform the matrix
            scalers.SumScaler(target="weights"),  # and this transform the weights
            similarity.TOPSIS(),
        )
        
        res_topsis = pipe.evaluate(dm)
        dm.plot()
        plt.show()
        best_index = res_topsis.rank_.argmin()

        print(res_topsis)
        print("Best index by TOPSIS:", idxs[best_index])
        print("Optimal Values:", F[best_index])

        return idxs[best_index], res_topsis
    
    
    def get_split_stats(self, index) -> Dict[str, float]:
        stats = {
            "macs_edge": self.macs_edge[index].item(),
            "macs_cloud": self.macs_cloud[index].item(),
            "memory_size_edge": self.memory_size_bit[index].item(),
            "reads_edge": self.reads_edge[index].item(),
            "reads_cloud": self.reads_cloud[index].item(),
            "writes_edge": self.writes_edge[index].item(),
            "writes_cloud": self.writes_cloud[index].item(),
            "cycles_edge": self.cycles_edge[index].item(),
            "cycles_cloud": self.cycles_cloud[index].item(),
            "transmission_size": self.transmission_size_bit[index].item(),
            "tx_bandwidth_bps": self.bandwidth_bps[index].item(),
            "latency_s": self.latency_s[index].item(),
            "energy_mac_pJ": self.energy_mac_pJ[index].item(),
            "energy_read_write_pJ": self.energy_read_write_pJ[index].item(),
            "power_pW": self.power_pW[index].item(),
            "area_mm2": self.area_mm2[index].item(),
            "accuracy": self.accuracy[index].item()
        }

        return stats

def display_top_alternatives(res_topsis, k=10) -> None:

    df = pd.DataFrame({
        "Alternative": res_topsis.alternatives,
        "Rank": res_topsis.rank_
    })

    df = df.sort_values("Rank").head(k)
    print(df.to_string(index=False))

if __name__ == "__main__":

        
    trainset, _, testset = MNIST()
    image = trainset[0][0]
    input_size = image.nelement() * image.element_size()

    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    model = torch.load('models/MNIST_CNN.pt', weights_only=False)
    
    opt = SplitOptimizer(model.model, input_shape=(1, 1, 28, 28))
    macs = opt.get_macs()
    # print("Macs", macs)
    mem = opt.get_memory_size(input_size)
    # print("Memory", mem)
    rw = opt.get_read_write_count()
    # print("Read/write count", rw)
    cyc = opt.get_cycles()
    # print("Cycles", cyc)
    tx = opt.get_transmission_size()
    # print("TX size", tx)
    lat = opt.get_latency()
    # print("Latency", lat)
    energy = opt.get_energy()
    # print("Energy", energy)
    area = opt.get_area()
    # print("Area", area)
    acc = opt.get_accuracy(testloader)
    pow = opt.get_power()
    bw = opt.get_bandwidth()

    idx, res = opt.optimize()

    if idx and res:
        display_top_alternatives(res)
        stats = opt.get_split_stats(idx)
        print(stats)

    # Minimize
    # Latency
    # MACs
    # Memory size
    # transmission size

# Q:
# Where is the datawidth used? Cycles? Does it relate to cycles per read/write?
# If the # of reads/writes is proportional to the number of MACs, should it be used for minimiztion or just as info to be outputed? 
# How do I get the area from the MACs? I am calculating the number of MAC operations, not the MAC-based PEs. Wouldn't the are come from the latter?
# Does the energy per read/write depend on the total size of memory?





# power (add power to transmit (find per bit?)) max 10 mW
# latency < 20ms (not as important)
# model accuracy/loss
# bandwidth
# weight all equally for now

# FC -> ReLU -> etc
# Add conv later

# max 256 width
# max depth 8
# INT8, FP16, INT16


# 7 bp for presentation
# literature search
# optimization model

# implement on prev


# add references