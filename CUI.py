import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from mnist_cnn import MNIST

def cui(model, layers, validset):

    # Split the dataset into 10 classes based on labels
    validset_class = {i: [] for i in range(10)}
    for img, label in validset:
        validset_class[label].append(img)

    layer_cuis = {}
    for l, layer in enumerate(layers):
        cam = GradCAM(model=model, target_layers=[layer])

        all_image_cuis = []
        for c in range(10):

            dataloader = DataLoader(validset_class[c], batch_size=50, shuffle=False)
            for inputs in dataloader:
                inputs = inputs.to('mps')
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                targets = [ClassifierOutputTarget(label) for label in predicted]
                grayscale_cam = cam(input_tensor=inputs, targets=targets)

                cuis = grayscale_cam.sum(axis=(1,2))
                all_image_cuis.extend(cuis)
        
        layer_cuis[l] = torch.tensor(all_image_cuis).mean().item()

    return layer_cuis


if __name__ == "__main__":
    _, validset, _ = MNIST()
    model = torch.load('MNIST_CNN.pt', weights_only=False).to('mps')

    cuis = cui(model, validset)

    print(cuis)

    plt.plot(list(cuis.values()))
    plt.yscale('log')
    plt.show()



# _, validset, _ = MNIST()

# image = validset[0][0].unsqueeze(0).to('mps')
# label = validset[0][1]

# mean = 0.1307
# std = 0.3081
# image = (image * std) + mean  # back to 0–1 range

# # Clip to ensure valid image range
# image = torch.clip(image, 0, 1)

# print(image)

# model = torch.load('MNIST_CNN.pt', weights_only=False).to('mps')

# target_layer = model.model[0]

# cam = GradCAM(model=model, target_layers=[target_layer])

# output = model(image)
# pred = output.argmax(dim=1).item()

# grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget(pred)])

# # Overlay CAM on the input image
# input_image_np = image.squeeze().cpu().numpy()
# input_image_rgb = np.stack([input_image_np]*3, axis=-1)  # Convert to 3-channel for display

# print(input_image_rgb)
# visualization = show_cam_on_image(input_image_rgb, grayscale_cam[0], use_rgb=True)

# # Show the heatmap
# plt.imshow(visualization)
# plt.title(f"Grad-CAM for class {pred}")
# plt.axis('off')
# plt.show()