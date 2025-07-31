import torch
from torch import nn
import torch.ao.quantization as tq
import torch.nn.quantized
from typing import List, Tuple
from optimization_models.model_consts import LOADERS, MODEL_IN, MODEL_IN_SHAPE, MODEL_OUT, RunType, shapeType

def build_mlp(in_dim: int, layers: List[int]) -> nn.Sequential:
    modules = []
    for out_dim in layers:
        modules.append(nn.Linear(in_dim, out_dim))
        modules.append(nn.ReLU())
        in_dim = out_dim
    return nn.Sequential(*modules)


class MNISTFNN(nn.Module):
    def __init__(self, head_layers:List[int], tail_layers:List[int], datawidth:int, runType:RunType):
        super(MNISTFNN, self).__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.device = 'cpu'
        self.quantized = False
        self.runType = runType

        self.trainloader, self.testloader = LOADERS

        self.datawidth = datawidth

        self.head, self.head_shape, self.tail, self.tail_shape = self._build_modules(head_layers, tail_layers)

    def _build_modules(self, head_layers:List[int], tail_layers:List[int]) -> Tuple[nn.Sequential, shapeType, nn.Sequential, shapeType]:
        head_in_dim = MODEL_IN
        head_out_dim = head_layers[-1] if len(head_layers) > 0 else head_in_dim
        tail_in_dim = head_out_dim
        tail_out_dim = tail_layers[-1] if len(tail_layers) > 0 else tail_in_dim

        match self.runType:
            case RunType.EDGE_COMPUTING:
                head =  nn.Sequential(
                    nn.Flatten(),
                    *build_mlp(head_in_dim, head_layers),
                    nn.Linear(head_out_dim, MODEL_OUT)
                )
                tail = nn.Sequential()

                head_shape = MODEL_IN_SHAPE
                tail_shape = None
            case RunType.SPLIT_COMPUTING:
                head =  nn.Sequential(
                    nn.Flatten(),
                    *build_mlp(head_in_dim, head_layers),
                )
                tail = nn.Sequential(
                    *build_mlp(tail_in_dim, tail_layers),
                    nn.Linear(tail_out_dim, MODEL_OUT)
                )

                head_shape = MODEL_IN_SHAPE
                tail_shape = (1, 1, 1, tail_in_dim)
            case RunType.CLOUD_COMPUTING:
                head = nn.Sequential()
                tail = nn.Sequential(
                    nn.Flatten(),
                    *build_mlp(tail_in_dim, tail_layers),
                    nn.Linear(tail_out_dim, MODEL_OUT)
                )

                head_shape = None
                tail_shape = MODEL_IN_SHAPE
        
        return head, head_shape, tail, tail_shape

        
    def children(self):
        return [*self.head.children(), *self.tail.children()]
    
    def forward(self, x):
        if self.quantized and self.datawidth == 16:
            x = self.head(x.half())
            x = self.tail(x.float())
        else:
            x = self.head(x)
            x = self.tail(x)
        return x
    
    def prepare_model(self) -> None:

        if self.datawidth == 8:
            self.head = nn.Sequential(self.quant, *self.head, self.dequant)
            self.head.qconfig = tq.get_default_qat_qconfig("qnnpack")
            tq.prepare_qat(self.head, inplace=True)
        else:
            pass
            # Can use this to prepare, but not quantize (I think)
            # # Define custom INT16 Quantizer
            # int16_fakequant = tq.FakeQuantize.with_args(
            #     observer=tq.observer.MinMaxObserver.with_args(
            #         dtype=torch.qint32,
            #         quant_min=-32768,
            #         quant_max=32767,
            #         qscheme=torch.per_tensor_symmetric,
            #         reduce_range=False
            #     ),
            #     quant_min=-32768,
            #     quant_max=32767,
            #     dtype=torch.qint32,
            #     qscheme=torch.per_tensor_symmetric,
            #     reduce_range=False
            # )

            # self.head.qconfig = tq.QConfig(activation=int16_fakequant, weight=int16_fakequant)
            # tq.prepare_qat(self.head, inplace=True)


    def quantize_model(self) -> None:
        self.quantized = True

        if self.datawidth == 8:
            tq.convert(self.head, inplace=True)
        else:
            self.head.half()
    
    def train_model(self, quiet=True) -> None:
        self.to(self.device)
        self.train()

        self.prepare_model()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            running_loss = 0.0
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            if not quiet:
                print(f"Epoch {epoch+1} - Loss: {running_loss / len(self.trainloader):.4f}")
        
    def test_model(self) -> Tuple[float, float]:
        self.eval()
        self.quantize_model()

        correct = 0
        total = 0
        loss = 0

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self(inputs)

                _, predicted = torch.max(outputs, 1)
                loss += criterion(outputs, labels).item()
                correct += (predicted == labels).sum().item()

                total += labels.size(0)

        return 100 * correct / total, loss
    
    def display_model(self) -> None:
        for i, layer in enumerate(self.children()):
            if i == len(list(self.head.children())):
                print("SPLIT")
            if isinstance(layer, (nn.Linear, torch.nn.quantized.Linear)):
                print(f"Layer {i}: Linear({layer.in_features}, {layer.out_features})")
            elif isinstance(layer, nn.ReLU):
                print(f"Layer {i}: ReLU()")
            elif isinstance(layer, nn.Flatten):
                print(f"Layer {i}: Flatten()")
            # else:
            #     print(f"Layer {i}: {layer.__class__.__name__}")

    def _module_to_str(self, module:nn.Sequential) -> str:
        lin_layers = [layer for layer in module if isinstance(layer, (nn.Linear, torch.nn.quantized.Linear))]
        l = len(lin_layers)

        out = ""
        for i, layer in enumerate(lin_layers):
            out += f"{layer.in_features} -> "
            if i == l-1:
                out += f"{layer.out_features}"

        return out


    def __str__(self):
        head_out = self._module_to_str(self.head)
        tail_out = self._module_to_str(self.tail)

        out = ""
        out += head_out

        if len(head_out) > 0 and len(tail_out) > 0:
            out += " -> "
        if len(tail_out) > 0:
            out += "TX -> "

        out += tail_out

        return out
            
