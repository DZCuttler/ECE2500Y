import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from mnist_cnn import  MNIST
from model_splitting import split_model
import torch.ao.quantization as tq
from torch.ao.quantization import fuse_modules


def get_acc(model, trainset, testset):
    torch.backends.quantized.engine = 'qnnpack'

    datawidths = [8,16,32]

    trainset, _, testset = MNIST()

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    model = torch.load('models/MNIST_CNN.pt', weights_only=False).to('cpu')
    model.eval()


    int8_model = torch.load('models/MNIST_CNN.pt', weights_only=False).to('cpu')
    int8_model.eval()


    L = len(list(model.children()))

    accs = np.zeros((L+1, len(datawidths)))

    for l in range(L+1):
        for i, dw in enumerate(datawidths):
            head, tail = split_model(model, l)

            if dw == 8:
                ihead, _ = split_model(int8_model, l)
                # fused_model = fuse_modules(int8_model, [["conv", "relu"]], inplace=False)

                ihead = nn.Sequential(tq.QuantStub(), *ihead.children(), tq.DeQuantStub())
                ihead.qconfig = tq.get_default_qconfig("qnnpack")

                prepared_head = tq.prepare(ihead)

                images = torch.stack([trainset[i][0] for i in range(4)]).to('cpu')
                prepared_head(images)
                qhead = tq.convert(prepared_head.to('cpu')).to('cpu')
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

            print(f"Test Accuracy: {100 * correct / total:.4f}%")
            accs[l, i] = 100 * correct / total

    np.save('datawidths_accs.npy', accs)
    print(accs)

        

    

