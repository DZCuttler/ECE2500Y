import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from compressai.entropy_models import EntropyBottleneck
from model_splitting import split_model
from mnist_cnn import MNIST
from Tx_Rx import transfer


class hecs_bottleneck(nn.Module):
    def __init__(self, encoder, bottleneck, decoder, tail):
        super(hecs_bottleneck, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.tail = tail

        self.precompressed_sizes = 0
        self.compressed_sizes = 0
        self.inference_count = 0
        self.times = 0

    def get_avg_transmission_sizes(self):
        return self.precompressed_sizes / self.inference_count, self.compressed_sizes / self.inference_count
    
    def get_avg_inference_time(self):
        return self.times / self.inference_count
    
    def clear_stats(self):
        self.precompressed_sizes = 0
        self.compressed_sizes = 0
        self.inference_count = 0
        self.times = 0

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def update_bottleneck(self):
        self.bottleneck.update()

    def forward_training(self, x):
        y = self.encoder(x)
        z, p = self.bottleneck(y, training=True)
        y_hat = self.decoder(z)
        pred = self.tail(y_hat)

        return pred, p

    def forward(self, x, head_device='cpu', tail_device='mps'):
        self.encoder = self.encoder.to(head_device)
        self.decoder = self.decoder.to(tail_device)
        self.tail = self.tail.to(tail_device)

        time.sleep(0.001) #add delay to stop tx/rx errors

        torch.mps.synchronize()
        start_time = time.time()

        # Evaluate the head model
        with torch.no_grad():
            y = self.encoder(x)
            strings = self.bottleneck.to(head_device).compress(y)
            shape = y.size()[2:]

        # EDGE -> CLOUD
        channel_out = transfer(strings)

        # Evaluate tail model
        with torch.no_grad():
            y_hat = self.bottleneck.to(tail_device).decompress(channel_out, shape)
            x_hat = self.decoder(y_hat)
            pred = self.tail(x_hat)

        # CLOUD -> EDGE
        pred = transfer(pred)
        torch.mps.synchronize()
        end_time = time.time()


        self.precompressed_sizes += y.element_size() * y.nelement()
        self.compressed_sizes += sum(len(s) for s in strings)
        self.times += end_time - start_time
        self.inference_count +=1

        return pred.to(head_device)



def hecs_loss(prediction, target, likelihood, beta=1.0):
    ce = nn.CrossEntropyLoss()

    distortion = ce(prediction, target)
    compression = -torch.log2(likelihood).sum(dim=(1,2,3)).mean()

    return distortion + beta * compression

def training_stage1(teacher, student, trainloader, epochs=10, lr=0.001, T=2, a=0.5, beta=0.01, quiet=True):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    teacher = teacher.to(device)
    student = student.to(device)
    
    teacher.eval()
    student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    kl = nn.KLDivLoss(reduction='batchmean')
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)


    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_out = teacher(inputs)

            student_out, p = student.forward_training(inputs)
            

            student_log_prob = logsoftmax(student_out / T)
            teacher_prob = softmax(teacher_out / T)

            loss = a * hecs_loss(student_out, labels, p, beta) + (1 - a) * T**2 * kl(student_log_prob, teacher_prob)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if not quiet:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")
    
    student.unfreeze_encoder()
    student.update_bottleneck()

def training_stage2(student, trainloader, epochs=10, lr=0.001, beta=0.01, quiet=True):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    student.to(device)
    student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            pred, p = student.forward_training(inputs)

            loss_value = hecs_loss(pred, labels, p, beta)
            loss_value.backward()
            optimizer.step()

            running_loss += loss_value.item()

        if not quiet:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(trainloader):.4f}")

    student.update_bottleneck()


if __name__ == "__main__":
    trainset, _, testset = MNIST()
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
        
    encoder = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), 
        nn.Conv2d(64, 4, 3, padding=1),
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(4, 32, 3, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )

    bottleneck = EntropyBottleneck(4)

    model = torch.load('models/MNIST_CNN_Complex.pt', weights_only=False).to('mps')
    head, tail = split_model(model, 10)

    student = hecs_bottleneck(encoder, bottleneck, decoder, copy.deepcopy(tail))

    training_stage1(model, student, trainloader, quiet=False)

    student.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')

            outputs = student(inputs)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    accuracy = 100 * correct / total
    latency =  student.get_avg_inference_time()

    pre, post = student.get_avg_transmission_sizes()

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Latency per batch: {latency:.4f} seconds")
    print(f"Avg. Pre-compressed Size: {pre} bytes")
    print(f"Avg. Compressed Size: {post} bytes")

    training_stage2(student, trainloader, beta=0.005, quiet=False)

    student.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to('cpu'), labels.to('cpu')

            outputs = student(inputs)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

            total += labels.size(0)

    accuracy = 100 * correct / total

    latency =  student.get_avg_inference_time()
    pre, post = student.get_avg_transmission_sizes()

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Latency per batch: {latency:.4f} seconds")
    print(f"Avg. Pre-compressed Size: {pre} bytes")
    print(f"Avg. Compressed Size: {post} bytes")



