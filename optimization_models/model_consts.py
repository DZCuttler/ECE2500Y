from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, TypeAlias
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def MNIST() -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    return trainloader, testloader

MODEL_IN = 28*28
MODEL_OUT = 10
MODEL_IN_SHAPE = (1,1,28,28)
LOADERS = MNIST()
STATS_FILENAME = "objective_stats.json"

shapeType: TypeAlias = Tuple[int, int, int, int]

class RunType(Enum):
    EDGE_COMPUTING = "edge"
    SPLIT_COMPUTING = "split"
    CLOUD_COMPUTING = "cloud"

@dataclass
class Hardware():
    _datawidth: int = 8
    # Assuming each MAC operation requires 3 reads and 1 write
    reads_per_mac: int = 3
    writes_per_mac: int = 1

    cycles_per_mac: int = 5
    cycles_per_read: int = 1
    cycles_per_write: int = 1
    cycles_per_tx: int = 5
    cycles_per_rx: int = 5

    edge_clock_Hz: float = 100e6
    cloud_clock_Hz: float = 250e6
    bandwidth_bps: float = 1e6

    # 32bit datawidth x 32kB memory is 4.3pJ per read/write from https://www.researchgate.net/publication/220904916_A_65_nm_850_MHz_256_kbit_43_pJaccess_Ultra_Low_Leakage_Power_Memory_Using_Dynamic_Cell_Stability_and_a_Dual_Swing_Data_Link
    # Assume energy increases linearly with datawidth and sqrt with memory size ~number of columns (backed by https://ieeexplore.ieee.org/document/6757323)
    # assume E = k * D * sqrt(M) D in bits, M in Kbits
    # k = E / (D * sqrt(M)) = 4.3 / (32 * sqrt(256)) =~ 0.0084
    energy_per_mac_pJ: Dict[int, float] = field(default_factory=lambda: {8: 0.25, 16: 1, 32: 4})
    energy_per_read_write_k: float = 0.0084

    tx_energy_per_bit_pJ: float = 1.2 # https://ieeexplore.ieee.org/document/6894245
    max_tx_latency_s: float = 1 / 30000

    @property
    def datawidth(self) -> int:
        return self._datawidth

    @datawidth.setter
    def datawidth(self, value: int) -> None:
        if value not in [8,16]:
            raise ValueError(f"Unsupported datawidth: {value}. Supported: [8, 16]")
        self._datawidth = value