import torch  # Imports the PyTorch library
import torch.nn as nn  # Imports neural network modules from PyTorch

class MNISTModel(nn.Module):  # Defines a custom neural network model class that inherits from nn.Module
    def __init__(self):  # Constructor method for initializing the model
        super(MNISTModel, self).__init__()  # Calls the constructor of the parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 16, 3)  # First convolutional layer: 1 input channel, 16 output channels 3x3 kernel + 16 biases = 160 params
        self.conv2 = nn.Conv2d(16, 32, 3)  # Second convolutional layer: 16 input channels, 32 output channels, 3x3 kernel = 16*3*3*32+32 = 4640
        self.fc1 = nn.Linear(32 * 5 * 5, 16)  # First fully connected layer: 32*5*5 input features, 16 output features = 32 * 5 * 5* 16+16 =12816
        self.fc2 = nn.Linear(16, 10)  # Second fully connected layer: 16 input features, 10 output features (for 10 digits) = 170
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer with 2x2 kernel and stride 2
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):  # Defines the forward pass of the model
        x = self.pool(self.relu(self.conv1(x)))  # Apply conv1, ReLU, and pooling
        x = self.pool(self.relu(self.conv2(x)))  # Apply conv2, ReLU, and pooling
        x = x.view(-1, 32 * 5 * 5)  # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))  # Apply fc1 and ReLU
        x = self.fc2(x)  # Apply fc2 (final layer)
        return x  # Return the output
