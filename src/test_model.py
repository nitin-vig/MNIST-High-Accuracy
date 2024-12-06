import torch
from torchvision import datasets, transforms
from model import MNISTModel
import numpy as np
import pytest
import sys


def test_model_architecture():
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy location: {np.__file__}")
    
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of model parameters: {total_params:,}")

    assert total_params < 20000, "Model has too many parameters"

def test_model_performance():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    # Load latest model
    import glob
    import os
    model_files = glob.glob('models/mnist_model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    
    # Test on validation set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy >= 99.4, f"Model accuracy ({accuracy:.2f}%) is below 99.4%"

    def test_model_on_noisy_images():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MNISTModel().to(device)
        
        # Load latest model
        import glob
        import os
        model_files = glob.glob('models/mnist_model_*.pth')
        latest_model = max(model_files, key=os.path.getctime)
        model.load_state_dict(torch.load(latest_model))
        
        # Test on noisy images
        transform_noisy = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset_noisy = datasets.MNIST('./data', train=False, download=True, transform=transform_noisy)
        test_loader_noisy = torch.utils.data.DataLoader(test_dataset_noisy, batch_size=1000)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader_noisy:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        noisy_accuracy = 100 * correct / total
        assert noisy_accuracy > 85, f"Model accuracy on noisy images ({noisy_accuracy:.2f}%) is below 85%"

def test_model_on_rotated_images():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
        
    # Load latest model
    import glob
    import os
    model_files = glob.glob('models/mnist_model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
        
    # Test on scaled images
    transform_scaled = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(20),  # roate
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset_scaled = datasets.MNIST('./data', train=False, download=True, transform=transform_scaled)
    test_loader_scaled = torch.utils.data.DataLoader(test_dataset_scaled, batch_size=1000)
        
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader_scaled:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
    scaled_accuracy = 100 * correct / total
    assert scaled_accuracy > 90, f"Model accuracy on rotated images ({scaled_accuracy:.2f}%) is below 90%"

def test_model_on_blurred_images():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
        
    # Load latest model
    import glob
    import os
    model_files = glob.glob('models/mnist_model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
        
    # Test on blurred images
    transform_blurred = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=5),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset_blurred = datasets.MNIST('./data', train=False, download=True, transform=transform_blurred)
    test_loader_blurred = torch.utils.data.DataLoader(test_dataset_blurred, batch_size=1000)
        
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader_blurred:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
    blurred_accuracy = 100 * correct / total
    assert blurred_accuracy > 90, f"Model accuracy on blurred images ({blurred_accuracy:.2f}%) is below 90%"

def test_model_on_combined_augmentations():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
        
    # Load latest model
    import glob
    import os
    model_files = glob.glob('models/mnist_model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
        
    # Test on images with multiple augmentations
    transform_combined = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(20),
        transforms.GaussianBlur(kernel_size=5),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset_combined = datasets.MNIST('./data', train=False, download=True, transform=transform_combined)
    test_loader_combined = torch.utils.data.DataLoader(test_dataset_combined, batch_size=1000)
        
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader_combined:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
    combined_accuracy = 100 * correct / total
    assert combined_accuracy > 85, f"Model accuracy on combined augmented images ({combined_accuracy:.2f}%) is below 85%"
