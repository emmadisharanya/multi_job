"""
AlexNet Implementation for Federated Learning
Adapted for small image datasets (28x28 MNIST, 32x32 CIFAR-10)

Based on paper: "Efficient Device Scheduling with Multi-Job Federated Learning"
Group B: AlexNet achieves 98.9-99.0% on MNIST (non-IID)
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet adapted for small images (MNIST 28x28 or CIFAR-10 32x32)
    
    Paper uses this for MNIST in Group B with 3,275K parameters.
    Achieves 98.9-99.0% accuracy on MNIST with non-IID data.
    """
    
    def __init__(self, num_classes=10, input_channels=1, input_size=28):
        """
        Args:
            num_classes: Number of output classes (10 for MNIST/CIFAR-10)
            input_channels: 1 for MNIST (grayscale), 3 for CIFAR-10 (RGB)
            input_size: 28 for MNIST, 32 for CIFAR-10
        """
        super(AlexNet, self).__init__()
        
        # Feature extraction layers (convolutional)
        self.features = nn.Sequential(
            # Conv1: input -> 64 channels
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 64 -> 192 channels
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3: 192 -> 384 channels
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 384 -> 256 channels
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 256 -> 256 channels
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate feature map size after convolutions
        if input_size == 28:
            feature_size = 256 * 3 * 3  # MNIST: 28->14->7->3
        elif input_size == 32:
            feature_size = 256 * 4 * 4  # CIFAR-10: 32->16->8->4
        else:
            raise ValueError(f"Unsupported input size: {input_size}")
        
        # Fully connected classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
        
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    print("Testing AlexNet...")
    model = AlexNet(num_classes=10, input_channels=1, input_size=28)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
    print(f"Paper target: 3,275K")
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    print(f"Output shape: {output.shape}")
    print("✓ Model ready!")
