import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from mnist_cnn import MNIST
from model_splitting import make_split_inference, split_model
from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt

EDGE = 'cpu'
CLOUD = 'mps'

def test_model_latency_accuracy(model, device, dataloader):
    model = model.to(device)

    # Warm-up runs to ensure the model is ready
    for _ in range(3):
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        _ = model(dummy_input)

    
    total = 0
    correct = 0

    torch.mps.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    torch.mps.synchronize()
    end_time = time.time()

    accuracy = 100 * correct / total
    latency = (end_time - start_time) / len(dataloader)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Latency per batch: {latency:.4f} seconds")
    return accuracy, latency


def eval_accuracy(model, testloader, head_device='cpu', quiet=True):
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(head_device), labels.to(head_device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    accuracy = 100 * correct / total

    if not quiet:
        print(f"Accuracy: {accuracy:.2f}%")

    return accuracy

def test_split_latency_accuracy(head, tail, dataloader, head_device='cpu', tail_device='mps', quiet=True):
    head = head.to(head_device)
    tail = tail.to(tail_device)

    split_model = make_split_inference(head, tail, head_device=head_device, tail_device=tail_device)

    # Warm-up runs to ensure the model is ready
    for _ in range(3):
        dummy_input = torch.randn(1, 1, 28, 28).to(head_device)
        _ = split_model(dummy_input)

    
    total = 0
    correct = 0
    times = []
    sizes = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(head_device), labels.to(head_device)

            time.sleep(0.001)  # Add delay between batches to simulate real-world conditions (also prevents crash with small batches)

            torch.mps.synchronize()
            start_time = time.time()
            outputs, size = split_model(inputs)
            torch.mps.synchronize()
            end_time = time.time()

            sizes.append(size)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            total += labels.size(0)
            times.append(end_time - start_time)



    accuracy = 100 * correct / total
    latency =  sum(times) / len(times)
    avg_size = sum(sizes) / len(sizes)

    if not quiet:
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Latency per batch: {latency:.4f} seconds")
        print(f"Avg Transmission Size: {avg_size} bytes")

    return accuracy, latency


def latencies_with_bw(head, tail, bandwidth_bps, testloader, head_device='cpu', tail_device='mps'):
    head = head.to(head_device)
    tail = tail.to(tail_device)

    head.eval()
    tail.eval()

    times = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(head_device), labels.to(head_device)

            time.sleep(0.001)  # Add delay between batches to simulate real-world conditions (also prevents crash with small batches)

            torch.mps.synchronize()
            start_time = time.time()
            intermediate = head(inputs)
            outputs = tail(intermediate.to(tail_device))
            torch.mps.synchronize()
            end_time = time.time()

            transfer_size_bits = intermediate.numel() * intermediate.element_size() * 8

            times.append(end_time - start_time + transfer_size_bits / bandwidth_bps)

    return sum(times)/len(times)



def get_flops(model, input_size=(1, 1, 28, 28)):
    dummy_input = torch.randn(input_size)
    flops = FlopCountAnalysis(model, dummy_input).by_module()

    print(flops)

    del flops['']

    F =  {int(k): v for k, v in flops.items()}
    return F

def get_MACs(model, input_size=(1, 1, 28, 28)):
    dummy_input = torch.randn(input_size)
    macs, params = get_model_complexity_info(model, (1, 28, 28), as_strings=True)
    print(f"MACs: {macs} | Parameters: {params}")

    # del flops['']

    # F =  {int(k): v for k, v in flops.items()}
    return 

def get_read_writes(model, input_size=(1, 1, 28, 28)):
    x = torch.randn(input_size)
    reads = []
    writes = []
    for i, layer in enumerate(model.children()):
        reads.append(x.element_size() * x.nelement())

        if hasattr(layer, 'weight') and layer.weight is not None:
            reads[i] += layer.weight.nelement() * layer.weight.element_size()

        x = layer(x)

        writes.append(x.element_size() * x.nelement())
    
    return reads, writes




# For energy use powermetrics
# For FLOPs or MACs use fvcore

if __name__ == "__main__":
    _, _, testset = MNIST()
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    model = torch.load('models/MNIST_CNN.pt', weights_only=False).to('cpu')
    reads, writes = get_read_writes(model.model, input_size=(1, 1, 28, 28))

    print(f"FLOPs by layer: {reads, writes}")

    # head, tail = split_model(model, 3)

    # print("EDGE")
    # test_split_latency_accuracy(model, nn.Sequential(), testloader, quiet=False)
    # print("\nEDGE-EDGE")
    # test_split_latency_accuracy(head, tail, EDGE, EDGE, testloader)
    # print("\nEDGE-CLOUD")
    # test_split_latency_accuracy(head, tail, EDGE, CLOUD, testloader)
    # print("\nCLOUD-CLOUD")
    # test_split_latency_accuracy(head, tail, CLOUD, CLOUD, testloader)
    # print("\nCLOUD")
    # test_split_latency_accuracy(nn.Sequential(), model, EDGE, CLOUD, testloader)


    # print("EDGE")
    # test_split_latency_accuracy(model, nn.Sequential(), EDGE, CLOUD, testloader)
    # print("\nEDGE-EDGE")
    # test_split_latency_accuracy(head, tail, EDGE, EDGE, testloader)
    # print("\nEDGE-CLOUD")
    # test_split_latency_accuracy(head, tail, EDGE, CLOUD, testloader)
    # print("\nCLOUD-CLOUD")
    # test_split_latency_accuracy(head, tail, CLOUD, CLOUD, testloader)
    # print("\nCLOUD")
    # test_split_latency_accuracy(nn.Sequential(), model, EDGE, CLOUD, testloader)

    # lats = []
    # n = len(list(model.model.children()))
    # for i in range(n):
    #     print(f"\nSplitting at layer {i}")
    #     head, tail = split_model(model, i)
    #     _, lat = test_split_latency_accuracy(head, tail, EDGE, CLOUD, testloader)
    #     lats.append(lat)

    # plt.plot(lats)
    # plt.xlabel('Split Layer')
    # plt.ylabel('Latency (seconds per batch)')
    # plt.title('Latency vs Split Layer')
    # plt.show()
