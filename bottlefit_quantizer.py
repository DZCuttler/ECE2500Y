import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from mnist_cnn import MNIST
from model_splitting import split_model
from lat_acc_test_funcs import eval_accuracy
from Tx_Rx import transfer


def to_bitstring(x, bits):
    return format(x, f'0{bits}b')

class Quantizer:
    def __init__(self, n):
        self.n = n

    def quantize(self, tensor):
        if self.n == 0: return tensor

        # print("Tensor:")
        # print(tensor)

        tensor = torch.round(tensor * (2**self.n - 1))

        bitstring = ""
        for x in tensor.flatten():
            bitstring += to_bitstring(x.item(), self.n)

        padding = (8 - len(bitstring) % 8) % 8
        bitstring = bitstring + '0' * padding

        bit_array = np.array(list(bitstring), dtype=np.uint8)
        byte_array = np.packbits(bit_array)
        return byte_array

    
    def dequantize(self, byte_array, shape):
        if self.n == 0: return byte_array

        # print("Bitstring:")
        # print(bitstring)
        # print(shape)

        num_vals = np.prod(shape)
        bit_array = np.unpackbits(byte_array.cpu()[0:num_vals], axis=1)
        bit_array = bit_array.reshape(num_vals, self.n).cpu()



        padded_bits = np.pad(bit_array, ((0, 0), (8 - self.n, 0)), mode='constant')
        bytes_ = np.packbits(padded_bits, axis=1)
        values = torch.tensor(bytes_.flatten().reshape(shape))
    
        return values / (2**self.n - 1)
    
    def training(self, x):
        if self.n == 0: return x
        noise = torch.empty_like(x).uniform_(-1/self.n, 1/self.n)
        return x + noise
    
    # def tensor_to_bitstring(tensor, n_bits):
    #     flat = tensor.flatten().cpu().numpy().astype(np.uint8)
    #     bitstring = np.unpackbits(flat[:, None], axis=1)[:, -n_bits:]  # take only n_bits from each byte
    #     return bitstring.flatten()

class bottlefit_quantizer(nn.Module):
    def __init__(self, encoder, decoder, tail, quantizer=0):
        super(bottlefit_quantizer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tail = tail

        self.quantizer = Quantizer(quantizer)

        self.transmission_sizes = 0
        self.inference_count = 0
        self.times = 0

    def children(self):
        return nn.Sequential(*[layer for seq in super().children() for layer in seq])

    def get_avg_transmission_sizes(self):
        return self.transmission_sizes / self.inference_count
    
    def get_avg_inference_time(self):
        return self.times / self.inference_count
    
    def clear_stats(self):
        self.transmission_sizes = 0
        self.inference_count = 0
        self.times = 0

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_tail(self):
        for param in self.tail.parameters():
            param.requires_grad = False

    def unfreeze_tail(self):
        for param in self.tail.parameters():
            param.requires_grad = True

    def forward_head(self, x):
        y = self.encoder(x)

        y = self.quantizer.training(y)

        x_hat = self.decoder(y)
        self.tail(x_hat)  # run tail since it has hooks
        return x_hat

    def forward_training(self, x):
        y = self.encoder(x)

        y = self.quantizer.training(y)

        x_hat = self.decoder(y)
        pred = self.tail(x_hat)
        return pred

    def forward(self, x, head_device='cpu', tail_device='mps'):
        self.encoder = self.encoder.to(head_device)
        self.decoder = self.decoder.to(tail_device)
        self.tail = self.tail.to(tail_device)

        time.sleep(0.001)  # add delay to stop tx/rx errors

        torch.mps.synchronize()
        start_time = time.time()

        # Evaluate the head model
        with torch.no_grad():
            y = self.encoder(x)
            shape = y.shape
        
        y = self.quantizer.quantize(y)

        # EDGE -> CLOUD
        y_hat = transfer(y).to(tail_device)

        y_hat = self.quantizer.dequantize(y_hat, shape).to(tail_device)

        # Evaluate tail model
        with torch.no_grad():
            x_hat = self.decoder(y_hat)
            pred = self.tail(x_hat)

        # CLOUD -> EDGE
        pred = transfer(pred)

        torch.mps.synchronize()
        end_time = time.time()

        self.transmission_sizes += y.element_size() * y.nelement()
        self.times += end_time - start_time
        self.inference_count += 1

        return pred.to(head_device)


def apply_hooks(model):
    outputs = []

    def hook_fn(module, input, output):
        outputs.append(output)

    handles = []
    for layer in model:
        handle = layer.register_forward_hook(hook_fn)
        handles.append(handle)

    return outputs, handles


def ghnd_loss_fn(t, s):
    mse = nn.MSELoss()

    loss = 0
    for t_output, s_output in zip(t, s):
        loss += mse(t_output, s_output)
    return loss


def stage1_training(t_head, t_tail, student, trainloader, epochs=10, lr=0.001, quiet=True):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    t_head = t_head.to(device)
    t_tail = t_tail.to(device)
    student = student.to(device)

    t_head.eval()
    t_tail.eval()
    student.encoder.train()
    student.decoder.train()
    student.tail.eval()

    student.freeze_tail()

    teacher_outputs, teacher_handles = apply_hooks(t_tail)
    student_outputs, student_handles = apply_hooks(student.tail)

    # filter out frozen params
    params = filter(lambda p: p.requires_grad, student.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):  # Try 10 for better results
        running_loss = 0.0
        for inputs, _ in trainloader:
            inputs = inputs.to(device)

            teacher_outputs.clear()
            student_outputs.clear()

            optimizer.zero_grad()

            with torch.no_grad():
                teach_head_out = t_head(inputs)
                t_tail(teach_head_out)

            stud_head_out = student.forward_head(inputs)

            teacher_outputs.append(teach_head_out)
            student_outputs.append(stud_head_out)

            loss = ghnd_loss_fn(teacher_outputs, student_outputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if not quiet:
            print(
                f"Epoch {epoch+1} - Loss: {running_loss / len(trainloader):.4f}")

    # unfreeze tail parameters
    student.unfreeze_tail()

    # remove hooks
    for handle in teacher_handles + student_handles:
        handle.remove()


def stage2_training(teacher, student, trainloader, epochs=10, lr=0.001, T=2, a=0.5, freeze_encoder=False, quiet=True):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    teacher = teacher.to(device)
    student = student.to(device)

    teacher.eval()
    student.train()

    if freeze_encoder:
        student.freeze_encoder()
        student.encoder.eval()

    # filter out unfrozen params
    params = filter(lambda p: p.requires_grad, student.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction='batchmean')
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)

    for epoch in range(epochs):  # Try 10 for better results
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_out = teacher(inputs)

            student_out = student.forward_training(inputs)

            student_log_prob = logsoftmax(student_out / T)
            teacher_prob = softmax(teacher_out / T)

            loss = a * ce(student_out, labels) + (1 - a) * T**2 * kl(student_log_prob, teacher_prob)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if not quiet:
            print(
                f"Epoch {epoch+1} - Loss: {running_loss / len(trainloader):.4f}")

    student.unfreeze_encoder()


if __name__ == "__main__":
    pass
    # trainset, _, testset = MNIST()
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    # testloader = DataLoader(testset, batch_size=8, shuffle=False)

    # model = torch.load('models/MNIST_CNN.pt', weights_only=False)
    # head, tail = split_model(model, 3)

    # encoder = nn.Sequential(
    #     *list(copy.deepcopy(head)),
    #     nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1),
    #     nn.ReLU(),
    # )

    # decoder = nn.Sequential(
    #     nn.Upsample(scale_factor=2, mode='nearest'),
    #     nn.Conv2d(8, 32, kernel_size=3, padding=1),
    #     nn.ReLU()
    # )

    # student = bottlefit_bottleneck(encoder, decoder, copy.deepcopy(tail))

    # stage1_training(head, tail, student, trainloader, quiet=False)
    # stage2_training(model, student, trainloader, freeze_encoder=False, quiet=False)



    # accuracy = eval_accuracy(student, testloader)
    # latency =  student.get_avg_inference_time()
    # size = student.get_avg_transmission_sizes()

    # print(f"Accuracy: {accuracy:.2f}%")
    # print(f"Latency per batch: {latency:.4f} seconds")
    # print(f"Avg. Transmission Size: {size} bytes")
