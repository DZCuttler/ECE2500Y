from torch import nn
import torch.ao.quantization as tq
from torch.utils.data import random_split
from torchvision import datasets, transforms

def MNIST():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    validationset, testset = random_split(testset, [5000, 5000])

    return trainset, validationset, testset

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # (N, 1, 28, 28) → (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # (N, 32, 28, 28) → (N, 32, 14, 14)
            nn.Conv2d(32, 64, 3, padding=1), # (N, 32, 14, 14) → (N, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # (N, 64, 14, 14) → (N, 64, 7, 7)
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),      # (N, 3136) → (N, 128)
            nn.ReLU(),
            nn.Linear(128, 10)               # (N, 128) → (N, 10)
        )

    def children(self):
        return self.model.children()
    
    def forward(self, x):
        x = self.model(x)
        return x
    

class MNISTCNN_Complex(nn.Module):
    def __init__(self):
        super(MNISTCNN_Complex, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),    # (N, 1, 28, 28) → (N, 32, 28, 28)
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),   # (N, 32, 28, 28) → (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # (N, 32, 28, 28) → (N, 32, 14, 14)

            nn.Conv2d(32, 64, 3, padding=1),   # (N, 32, 14, 14) → (N, 64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),   # (N, 64, 14, 14) → (N, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                # (N, 64, 14, 14) → (N, 64, 7, 7)

            nn.Conv2d(64, 128, 3, padding=1),  # (N, 64, 7, 7) → (N, 128, 7, 7)
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), # (N, 128, 7, 7) → (N, 128, 7, 7)
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),       # (N, 6272) → (N, 512)
            nn.ReLU(),
            nn.Linear(512, 256),               # (N, 512) → (N, 256)
            nn.ReLU(),
            nn.Linear(256, 128),               # (N, 256) → (N, 128)
            nn.ReLU(),
            nn.Linear(128, 10)                 # (N, 128) → (N, 10)
        )

    def children(self):
        return self.model.children()
    
    def forward(self, x):
        x = self.model(x)
        return x

class MNISTFNN(nn.Module):
    def __init__(self):
        super(MNISTFNN, self).__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def children(self):
        return self.model.children()
    
    def forward(self, x):
        x = self.model(x)
        return x