# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

<img width="655" height="325" alt="image" src="https://github.com/user-attachments/assets/b496381f-8494-4ffb-9e30-d647dfd050cc" />



Problem Statement:

Images often contain "noise" (random graininess or static) due to poor lighting or camera limitations, which can hide important details. The goal is to build a Convolutional Autoencoder that takes a noisy version of an MNIST digit as input and reconstructs a clean, denoised version of that same digit.

Theory
1. What is an Autoencoder?
    An Autoencoder is a type of neural network that learns to compress data into a short "summary" and then reconstructs the original data from that summary. It consists of two main parts:
   
    Encoder: Compresses the input image into a lower-dimensional representation (latent space).
    Decoder: Takes that compressed representation and expands it back into a full-sized image.

3. Convolutional Layers:
    Instead of using standard flat layers, we use Convolutional layers because they are much better at recognizing spatial patterns in images (like edges and curves).
   
     Encoder: Uses Conv2d layers to reduce the height and width of the image while increasing the number of filters.
     Decoder: Uses ConvTranspose2d (often called Deconvolution) to "upsample" or grow the image back to its original $28 \times 28$ size.

5. Image Denoising
In a Denoising Autoencoder, we intentionally corrupt the input images by adding Gaussian Noise. The model is then forced to learn the underlying shapes of the digits so it can "ignore" the noise and output only the clean pixels.

## DESIGN STEPS

### Step 1: Load Dataset

* Import required libraries.
* Load MNIST handwritten digit dataset.
* Convert images into tensor format.

### Step 2: Add Noise to Images

* Add random Gaussian noise to input images.
* Clip pixel values between 0 and 1.

### Step 3: Create DataLoader

* Divide dataset into training and testing sets.
* Use DataLoader to create batches for training.

### Step 4: Build Autoencoder Model

* Design encoder using convolution layers to compress image features.
* Design decoder using transpose convolution layers to reconstruct images.

### Step 5: Train the Model

* Pass noisy images as input and original images as target.
* Compute reconstruction loss using Mean Squared Error.
* Update model weights using Adam optimizer.

### Step 6: Test and Visualize Output

* Feed noisy test images to trained model.
* Generate denoised images.
* Compare original, noisy and reconstructed images visually.

---

## PROGRAM

### Name: Surya Prakash B

### Register Number: 212224230281

```python
# Autoencoder for Image Denoising using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform: Normalize and convert to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load MNIST dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)


from torch.nn.modules.conv import ConvTranspose2d
# Define Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
      super(DenoisingAutoencoder, self).__init__()
      self.encoder = nn.Sequential(
          nn.Conv2d(1, 16, kernel_size = 3, stride = 2, padding = 1),
          nn.ReLU(),
          nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding =1),
          nn.ReLU()
      )

      self.decoder = nn.Sequential(
          nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2),
          nn.ReLU(),
          nn.ConvTranspose2d(16, 1, kernel_size = 2, stride = 2)
      )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Re-initialize model with the fixed architecture, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

# Print model summary
print('Name: Surya Prakash B')
print('Register Number: 212224230281')
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
  model.train()
  print('Name: Surya Prakash B')
  print('Register Number: 212224230281')
  for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print('Name: Surya Prakash B')
    print('Register Number: 212224230281')
    plt.figure(figsize=(18, 6))

    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noise
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoise
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run training and visualization with the fixed model
train(model, train_loader, criterion, optimizer, epochs = 5)
visualize_denoising(model, test_loader)

```

### OUTPUT

### Model Summary

<img width="663" height="391" alt="image" src="https://github.com/user-attachments/assets/5b2e4711-69ea-4153-8dc9-4618a690c86f" />


### Training loss

<img width="493" height="125" alt="image" src="https://github.com/user-attachments/assets/752021bd-5c8b-4c27-870e-27ae7a4f34bf" />


## Original vs Noisy Vs Reconstructed Image

<img width="1723" height="31" alt="image" src="https://github.com/user-attachments/assets/ee0250d1-9c65-411a-bac7-c65a56a625f4" />
<img width="1789" height="589" alt="image" src="https://github.com/user-attachments/assets/87d1186e-179c-4d79-8182-89c9b19dd4bd" />



## Result

The autoencoder model was successfully trained to remove noise from images and reconstruct clear handwritten digit images.
