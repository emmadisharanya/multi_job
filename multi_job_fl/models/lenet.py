"""
LeNet-5 for MNIST
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 architecture for MNIST
    
    Original paper: LeCun et al., 1998
    
    Architecture:
    - Conv1: 1->6 channels, 5x5 kernel
    - Conv2: 6->16 channels, 5x5 kernel
    - FC1: 120 units
    - FC2: 84 units
    - FC3: 10 classes (output)
    """
    
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Pooling
        self.pool = nn.AvgPool2d(2, 2)
    
    def forward(self, x):
        # Conv block 1: 28x28 -> 24x24 -> 12x12
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2: 12x12 -> 8x8 -> 4x4
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 16 * 4 * 4)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# Test
if __name__ == "__main__":
    print("Testing LeNet5\n")
    
    model = LeNet5(num_classes=10)
    
    # Test with random input (batch of 4 MNIST images)
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nLeNet model working correctly!")