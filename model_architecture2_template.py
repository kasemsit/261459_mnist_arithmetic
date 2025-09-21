import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskArithmeticNet(nn.Module):
    """
    Architecture 2 - TODO 4: Multi-task CNN architecture

    This model:
    1. Splits 28x56 image into two 28x28 digit images
    2. Recognizes each digit using shared CNN
    3. Combines digit predictions with operation info
    4. Predicts arithmetic result
    """
    def __init__(self, max_result=90):
        super(MultiTaskArithmeticNet, self).__init__()

        # TODO 4a: Shared CNN for digit recognition (28x28 -> 10 classes)
        self.digit_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14

            # TODO 4a: Add second conv layer (32 -> 64 channels)
            # TODO: nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d

            # TODO 4a: Add third conv layer (64 -> 128 channels)
            # TODO: nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d

            nn.Flatten(),
            nn.Linear(1152, 256),  # TODO: Calculate size (128 * 3 * 3 = 1152)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)  # 10 digit classes
        )

        # TODO 4b: Operation classifier
        # Input: 10 + 10 + 1 = 21 features (digit1 + digit2 + operation)
        self.op_classifier = nn.Sequential(
            nn.Linear(21, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_result + 1)
        )

    def forward(self, x, operation_info=None):
        batch_size = x.size(0)

        # TODO 4c: Split the 28x56 combined image into two 28x28 digit images
        # TODO: Use tensor slicing
        img1 = # TODO: x[:, :, :, :28]
        img2 = # TODO: x[:, :, :, 28:]

        # TODO 4d: Run both digit images through the shared CNN
        digit1_logits = # TODO: self.digit_cnn()
        digit2_logits = # TODO: self.digit_cnn()

        # TODO 4e: Extract operation information from operation_info tensor
        if operation_info is not None:
            op_features = # TODO: operation_info[:, 2:3].float()
        else:
            op_features = torch.zeros(batch_size, 1).to(x.device)

        # TODO 4f: Combine all features and get final result
        combined = # TODO: torch.cat()
        result = # TODO: self.op_classifier()

        return result, digit1_logits, digit2_logits

# Utility functions
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model):
    """Get model size in MB"""
    param_size = sum(param.numel() * param.element_size() for param in model.parameters())
    buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())
    return (param_size + buffer_size) / 1024**2

if __name__ == "__main__":

    print("Testing Multi-task CNN...")
    multitask = MultiTaskArithmeticNet()

    dummy_input = torch.randn(4, 1, 28, 56)
    dummy_op_info = torch.randint(0, 10, (4, 3))

    try:
        result, d1, d2 = multitask(dummy_input, dummy_op_info)
        
        print(f"✅ Multi-task: {result.shape}, {d1.shape}, {d2.shape}")        
        print(f"✅ Multi-task params: {count_parameters(multitask):,}")
    except Exception as e:
        print(f"❌ Error: {e}")