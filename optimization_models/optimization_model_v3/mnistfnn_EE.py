import torch
from torch import nn
import torch.ao.quantization as tq
from typing import Tuple, List
from optimization_models.model_consts import LOADERS, MODEL_IN, MODEL_IN_SHAPE, MODEL_OUT, RunType, shapeType

def build_mlp(in_dim: int, layers: List[int]) -> nn.Sequential:
    modules = []
    for out_dim in layers:
        modules.append(nn.Linear(in_dim, out_dim))
        modules.append(nn.ReLU())
        in_dim = out_dim
    return nn.Sequential(*modules)

def prepare_module(module: nn.Sequential) -> None:
    module.qconfig = tq.get_default_qconfig("qnnpack")
    tq.prepare(module, inplace=True)

def build_loss_fn(w:float):
    def loss(outputs: Tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        # Assuming outputs is a tuple of (y_branch, y_tail)
        y_branch, y_tail = outputs
        loss_branch = nn.CrossEntropyLoss()(y_branch, labels)
        loss_tail = nn.CrossEntropyLoss()(y_tail, labels)
        return w * loss_branch + (1-w) * loss_tail
    
    return loss

class MNISTFNN_EE(nn.Module):
    def __init__(self, head_layers: List[int], branch_layers: List[int], tail_layers: List[int], datawidth: int, runType:RunType, branchpoint: int):
        super(MNISTFNN_EE, self).__init__()
        self.quant = tq.QuantStub()
        self.head_dequant = tq.DeQuantStub()
        self.branch_dequant = tq.DeQuantStub()
        self.device = 'cpu'
        self.quantized = False
        self.runType = runType

        self.trainloader, self.testloader = LOADERS

        self.datawidth = datawidth

        self.branchpoint = self._set_branchpoint(branchpoint, len(head_layers))

        (
            self.head_prebranch, self.head_postbranch, self.head_shape, 
            self.branch, self.branch_shape, 
            self.tail, self.tail_shape
        ) = self._build_modules(head_layers, branch_layers, tail_layers)
    
    def _set_branchpoint(self, branchpoint, head_depth):
        if head_depth == 0:
            return 0
        elif branchpoint < 0:
            return head_depth-1
        elif branchpoint >= head_depth:
            return head_depth-1
        else:
            return branchpoint
        
    def _build_modules(self, head_layers:List[int], branch_layers:List[int], tail_layers:List[int]) -> Tuple[nn.Sequential, nn.Sequential, shapeType, nn.Sequential, shapeType, nn.Sequential, shapeType]:
        head_in_dim = MODEL_IN
        prehead_out_dim = head_layers[self.branchpoint] if len(head_layers) > 0 else head_in_dim

        fork_in_dim = prehead_out_dim
        posthead_out_dim = head_layers[-1] if len(head_layers) > 0 else fork_in_dim
        branch_out_dim = branch_layers[-1] if len(branch_layers) > 0 else fork_in_dim

        tail_in_dim = posthead_out_dim
        tail_out_dim = tail_layers[-1] if len(tail_layers) > 0 else tail_in_dim

        prehead_layers = head_layers[:self.branchpoint+1]
        posthead_layers = head_layers[self.branchpoint+1:]

        match self.runType:
            case RunType.EDGE_COMPUTING:
                head_prebranch = nn.Sequential(
                    nn.Flatten(),
                    *build_mlp(head_in_dim, prehead_layers)
                )
                branch = nn.Sequential(
                    *build_mlp(fork_in_dim, branch_layers), 
                    nn.Linear(branch_out_dim, MODEL_OUT)
                )
                head_postbranch = nn.Sequential(
                    *build_mlp(fork_in_dim, posthead_layers),
                    nn.Linear(posthead_out_dim, MODEL_OUT)
                )
                tail = nn.Sequential()

                head_shape = MODEL_IN_SHAPE
                branch_shape = (1, 1, 1, fork_in_dim)
                tail_shape = None
            case RunType.SPLIT_COMPUTING:
                head_prebranch = nn.Sequential(
                    nn.Flatten(),
                    *build_mlp(head_in_dim, prehead_layers)
                )
                branch = nn.Sequential(
                    *build_mlp(fork_in_dim, branch_layers), 
                    nn.Linear(branch_out_dim, MODEL_OUT)
                )
                head_postbranch = build_mlp(fork_in_dim, posthead_layers)
                tail = nn.Sequential(
                    *build_mlp(tail_in_dim, tail_layers),
                    nn.Linear(tail_out_dim, MODEL_OUT)
                )

                head_shape = MODEL_IN_SHAPE
                branch_shape = (1, 1, 1, fork_in_dim)
                tail_shape = (1, 1, 1, tail_in_dim)
            case RunType.CLOUD_COMPUTING:
                head_prebranch = nn.Sequential()
                branch = nn.Sequential(
                    nn.Flatten(),
                    *build_mlp(fork_in_dim, branch_layers), 
                    nn.Linear(branch_out_dim, MODEL_OUT)
                )
                head_postbranch = nn.Sequential()
                tail = nn.Sequential(
                    nn.Flatten(),
                    *build_mlp(tail_in_dim, tail_layers),
                    nn.Linear(tail_out_dim, MODEL_OUT)
                )

                head_shape = None
                branch_shape = MODEL_IN_SHAPE
                tail_shape = MODEL_IN_SHAPE
            
        return head_prebranch, head_postbranch, head_shape, branch, branch_shape, tail, tail_shape

    def children(self):
        return [*self.head_prebranch.children(), *self.branch.children(), *self.head_postbranch.children(), *self.tail.children()]
    
    def get_full_head(self) -> nn.Sequential:
        return nn.Sequential(*self.head_prebranch, *self.head_postbranch)
    
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.datawidth == 16 and self.quantized:
            x = self.head_prebranch(x.half())
            y_branch = self.branch(x).float()

            x = self.head_postbranch(x).float()
            y_tail = self.tail(x)

        else:
            x = self.head_prebranch(x)
            y_branch = self.branch(x)

            x = self.head_postbranch(x)
            y_tail = self.tail(x)

        return y_branch, y_tail
    
    
    def prepare_model(self) -> None:

        if self.datawidth == 8:
            self.head_prebranch = nn.Sequential(self.quant, *self.head_prebranch)
            self.branch = nn.Sequential(*self.branch, self.branch_dequant)
            self.head_postbranch = nn.Sequential(*self.head_postbranch, self.head_dequant)

            prepare_module(self.head_prebranch)
            prepare_module(self.branch)
            prepare_module(self.head_postbranch)
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
            tq.convert(self.head_prebranch, inplace=True)
            tq.convert(self.branch, inplace=True)
            tq.convert(self.head_postbranch, inplace=True)
        else:
            self.head_prebranch.half()
            self.branch.half()
            self.head_postbranch.half()
    
    def train_model(self, w:float, quiet=True) -> None:
        self.to(self.device)
        self.train()

        self.prepare_model()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = build_loss_fn(w)

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
        
    def test_model(self) -> Tuple[float, float, float, float]:
        self.eval()
        self.quantize_model()

        branch_correct = 0
        branch_loss = 0
        tail_correct = 0
        tail_loss = 0
        total = 0

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                branch_out, tail_out = self(inputs)

                # Eval EE
                _, branch_predicted = torch.max(branch_out, 1)
                branch_loss += criterion(branch_out, labels).item()
                branch_correct += (branch_predicted == labels).sum().item()

                # Eval Tail
                _, tail_predicted = torch.max(tail_out, 1)
                tail_loss += criterion(tail_out, labels).item()
                tail_correct += (tail_predicted == labels).sum().item()

                total += labels.size(0)

        return 100 * branch_correct / total, branch_loss, 100 * tail_correct / total, tail_loss
    
    def display_model(self) -> None:
        branchStart = len(list(self.head_prebranch.children()))
        branchEnd = branchStart + len(list(self.branch.children()))
        splitIndex = len(list(self.head_prebranch.children()) + list(self.branch.children()) + list(self.head_postbranch.children()))

        for i, layer in enumerate(self.children()):
            if i == branchStart:
                print("BRANCH")
            if i >= branchStart and i < branchEnd:
                print("\t", end="")
            
            if i == splitIndex:
                print("SPLIT")

            if isinstance(layer, (nn.Linear, nn.quantized.Linear)):
                print(f"Layer {i}: Linear({layer.in_features}, {layer.out_features})")
            elif isinstance(layer, nn.ReLU):
                print(f"Layer {i}: ReLU()")
            elif isinstance(layer, nn.Flatten):
                print(f"Layer {i}: Flatten()")
            else:
                print(f"Layer {i}: {layer.__class__.__name__}")

    def _module_to_str(self, module:nn.Sequential) -> str:
        lin_layers = [layer for layer in module if isinstance(layer, (nn.Linear, nn.quantized.Linear))]
        l = len(lin_layers)

        out = ""
        for i, layer in enumerate(lin_layers):
            out += f"{layer.in_features} -> "
            if i == l-1:
                out += f"{layer.out_features}"

        return out
    
    def __str__(self):
        prehead_out = self._module_to_str(self.head_prebranch)
        branch_out = self._module_to_str(self.branch)
        posthead_out = self._module_to_str(self.head_postbranch)
        tail_out = self._module_to_str(self.tail)

        out = ""
        out += prehead_out
        if len(prehead_out) > 0 and (len(branch_out) > 0 or len(posthead_out) > 0 or len(tail_out) > 0):
            out += " -> "
        if len(branch_out) > 0:
            out += "BRANCH"
        if len(branch_out) > 0 and (len(posthead_out) > 0 or len(tail_out) > 0):
            out += " -> "
        out += posthead_out
        if len(posthead_out) > 0 and len(tail_out) > 0:
            out += " -> "
        if len(tail_out) > 0:
            out += "TX -> "
        out += tail_out
        
        
        if len(branch_out) > 0:
            idx = out.find("BRANCH")
            out += "\n" + " " * (idx + 3) + "\--> " + self._module_to_str(self.branch)

        return out