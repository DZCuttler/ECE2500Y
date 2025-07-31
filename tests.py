import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mnist_cnn import MNIST
from CUI import cui
from model_splitting import split_model
from lat_acc_test_funcs import test_split_latency_accuracy
import matplotlib.pyplot as plt


def test_split_improvement(model, split_layer, bn_head, bn_tail, testloader):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # model = torch.load('MNIST_CNN.pt', weights_only=False).to(device)
    t_head, t_tail = split_model(model, split_layer)

    print("EDGE")
    acc, lat = test_split_latency_accuracy(model, nn.Sequential(), testloader, quiet=False)
    # print(acc, lat)

    print("\nCLOUD")
    acc, lat = test_split_latency_accuracy(nn.Sequential(), model, testloader, quiet=False)
    # print(acc, lat)

    print("\nSPLIT")
    acc, lat = test_split_latency_accuracy(t_head.to('cpu'), t_tail.to('mps'), testloader, quiet=False)
    # print(acc, lat)

    print("\nSPLIT (BOTTLEFIT)")
    acc, lat = test_split_latency_accuracy(bn_head, bn_tail, testloader, quiet=False)
    # print(acc, lat)

def test_layers(model, testset):

    conv_types = (nn.Conv2d, nn.ConvTranspose2d, nn.AvgPool2d, nn.MaxPool2d)

    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    layers = list(model.model.children())

    accs = []
    lats = []
    cuis = []

    for i, layer in enumerate(layers):
        print(f"\nSplitting at layer {i}")
        head, tail = split_model(model, i)
        head = head.to('cpu')
        tail = tail.to('mps')
        acc, lat = test_split_latency_accuracy(head, tail, testloader)
        accs.append(acc)
        lats.append(lat)

        if isinstance(layer, conv_types):
            c = cui(model.to('mps'), [layer], testset)
            print(c)
            cuis.extend(c.values())
        else:
            cuis.append(None)

        
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # axes[0].plot(accs, label='Accuracy')
    # axes[0].set_ylabel('Accuracy')

    axes[0].plot(lats, label='Latency')
    axes[0].set_ylabel('Latency (seconds per batch)')

    axes[1].plot(cuis, 'o-', label='CUI')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('CUI')

    axes[1].set_xlabel('Split Layer')

    plt.savefig('split_analysis-64.png')
    plt.show()


if __name__ == "__main__":
    # test_split_improvement()
    _, _, testset = MNIST()
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    model = torch.load('models/MNIST_CNN_Complex.pt', weights_only=False).to('mps')
    head = torch.load('models/MNIST_HEAD_BN_Complex.pt', weights_only=False).to('cpu')
    tail = torch.load('models/MNIST_TAIL_BN_Complex.pt', weights_only=False).to('mps')

    # test_layers(model, testset)
    test_split_improvement(model, 5, head, tail, testloader)