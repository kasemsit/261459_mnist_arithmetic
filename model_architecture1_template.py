import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTArithmeticNet(nn.Module):
    """
    Architecture 1 - TODO 3: Complete this end-to-end CNN architecture

    This model:
    1. Processes the full 28x56 image through CNN layers
    2. Combines image features with operation information
    3. Predicts the arithmetic result directly
    """
    def __init__(self, max_result=90):
        super(MNISTArithmeticNet, self).__init__()

        # TODO 3a: Design CNN layers for 28x56 input
        self.conv1 = # TODO: nn.Conv2d
        self.conv2 = # TODO: nn.Conv2d
        self.conv3 = # TODO: nn.Conv2d

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # TODO 3b: Calculate flattened size after convolutions
        # Hint: 28x56 -> 14x28 -> 7x14 -> 3x7, with 128 channels = 128 * 3 * 7 = 2688 features
        self.fc1 = nn.Linear(2688, 512)  # TODO: Calculate size based on conv output
        self.fc2 = nn.Linear(512, 256)

        # TODO 3c: Combine image features with operation info
        # Image features (256) + Operation info (1) = 257 features
        self.classifier = nn.Linear(256 + 1, max_result + 1)  # 91 classes (0-90)

    def forward(self, x, operation_info=None):
        batch_size = x.size(0)

        # TODO 3d: Implement CNN forward pass
        x = # TODO: self.pool(F.relu(self.conv1()))
        x = # TODO: self.pool(F.relu(self.conv2()))
        x = # TODO: self.pool(F.relu(self.conv3()))

        # Flatten image features
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        image_features = F.relu(self.fc2(x))
        image_features = self.dropout(image_features)

        # Extract operation information (0=add, 1=multiply)
        if operation_info is not None:
            op_features = operation_info[:, 2:3].float()  # Operation code
        else:
            op_features = torch.zeros(batch_size, 1).to(x.device)  # Default to addition

        # Combine image features with operation info
        combined_features = torch.cat([image_features, op_features], dim=1)

        # Final classification
        result = self.classifier(combined_features)
        return result

if __name__ == "__main__":
    # Test the model
    model = MNISTArithmeticNet()
    dummy_input = torch.randn(4, 1, 28, 56)

    try:
        output = model(dummy_input)
        print(f"✅ Model works! Input: {dummy_input.shape} -> Output: {output.shape}")
        print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your layer dimensions!")