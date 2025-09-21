import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import random

def set_seed(seed=42):
    """
    Set seeds for reproducible data generation across all random number generators.

    This ensures that:
    - Random sampling from MNIST dataset is consistent
    - Data augmentation produces the same transformations
    - Results are reproducible across different runs

    Args:
        seed (int): Random seed value (default: 42)
    """
    torch.manual_seed(seed)                    # PyTorch random number generator
    torch.cuda.manual_seed(seed)               # PyTorch CUDA random number generator
    torch.cuda.manual_seed_all(seed)           # All CUDA devices
    np.random.seed(seed)                       # NumPy random number generator
    random.seed(seed)                          # Python's random module
    torch.backends.cudnn.deterministic = True # Make CUDA operations deterministic
    torch.backends.cudnn.benchmark = False    # Disable auto-tuning for reproducibility

class MNISTArithmeticDataset(Dataset):
    """
    Custom PyTorch Dataset for MNIST arithmetic operations.

    This dataset creates arithmetic problems using MNIST digit images:
    - Combines two digit images side-by-side (28x28 → 28x56)
    - Performs addition or multiplication on the digit labels
    - Returns: combined image, operation info, and arithmetic result

    Example:
        Input: digit "3" and digit "7" with operation "add"
        Output: [3|7] image, [3, 7, 0], result=10
        (operation code: 0=add, 1=multiply)
    """

    def __init__(self, train=True, operations=['add', 'multiply'], transform=None, augment=False, seed=42):
        """
        Initialize the MNIST arithmetic dataset.

        Args:
            train (bool): If True, use training split; if False, use test split
            operations (list): List of operations to perform ['add', 'multiply']
            transform (callable, optional): Custom transform to apply to images
            augment (bool): If True, apply data augmentation (rotation, translation)
            seed (int): Random seed for reproducible data generation
        """
        # Set seed for reproducible data generation
        set_seed(seed)

        # Configure data transformations
        if transform is None:
            if train and augment:
                # Training with data augmentation for better generalization
                self.transform = transforms.Compose([
                    transforms.RandomRotation(degrees=10),      # Rotate ±10 degrees
                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translate ±10%
                    transforms.ToTensor(),                      # Convert PIL to tensor
                    transforms.Normalize((0.1307,), (0.3081,)) # MNIST normalization
                ])
            else:
                # Default transform (training without augmentation or testing)
                self.transform = transforms.Compose([
                    transforms.ToTensor(),                      # Convert PIL to tensor
                    transforms.Normalize((0.1307,), (0.3081,)) # MNIST normalization
                ])
        else:
            self.transform = transform

        # Load the original MNIST dataset
        self.mnist_dataset = datasets.MNIST(
            root='./data',          # Download/cache directory
            train=train,            # Training or test split
            download=True,          # Download if not present
            transform=self.transform # Apply transforms to individual digits
        )

        self.operations = operations
        # Double the dataset size to create more variety in digit combinations
        self.length = len(self.mnist_dataset) * 2
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx):
        """
        Generate a single arithmetic problem.

        Args:
            idx (int): Index (not used directly, we randomly sample digits)

        Returns:
            tuple: (combined_image, operation_info, result)
                - combined_image: Tensor of shape (1, 28, 56) - two digits side-by-side
                - operation_info: Tensor [digit1_label, digit2_label, operation_code]
                - result: Tensor with arithmetic result (0-90 for add, 0-81 for multiply)
        """
        # Randomly sample two MNIST digits (with replacement for variety)
        idx1 = random.randint(0, len(self.mnist_dataset) - 1)
        idx2 = random.randint(0, len(self.mnist_dataset) - 1)

        # Get the digit images and their true labels
        img1, label1 = self.mnist_dataset[idx1]  # First digit (0-9)
        img2, label2 = self.mnist_dataset[idx2]  # Second digit (0-9)

        # Randomly choose which arithmetic operation to perform
        operation = random.choice(self.operations)

        # Perform the arithmetic operation and encode it
        if operation == 'add':
            result = label1 + label2    # Range: 0-18
            op_code = 0                 # Addition operation code
        elif operation == 'multiply':
            result = label1 * label2    # Range: 0-81
            op_code = 1                 # Multiplication operation code
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Combine the two digit images horizontally (side-by-side)
        # Shape: (1, 28, 28) + (1, 28, 28) → (1, 28, 56)
        combined_img = torch.cat([img1, img2], dim=2)

        # Return the data triplet needed for training
        return (
            combined_img,                                    # Input image (1, 28, 56)
            torch.tensor([label1, label2, op_code]),        # Operation info (3,)
            torch.tensor(result, dtype=torch.long)          # Target result (1,)
        )

def get_dataloaders(batch_size=64, train_ops=['add', 'multiply'], test_ops=['add', 'multiply'], augment=False, seed=42):
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        batch_size (int): Number of samples per batch
        train_ops (list): Operations for training data ['add', 'multiply']
        test_ops (list): Operations for test data ['add', 'multiply']
        augment (bool): Whether to apply data augmentation to training data
        seed (int): Random seed for reproducible data generation

    Returns:
        tuple: (train_loader, test_loader)
            - train_loader: DataLoader for training with shuffling
            - test_loader: DataLoader for testing without shuffling

    Example:
        >>> train_loader, test_loader = get_dataloaders(batch_size=32, augment=True)
        >>> for batch_idx, (data, operation_info, target) in enumerate(train_loader):
        ...     # data.shape: (32, 1, 28, 56)
        ...     # operation_info.shape: (32, 3)
        ...     # target.shape: (32,)
        ...     break
    """
    # Create training dataset with optional augmentation
    train_dataset = MNISTArithmeticDataset(
        train=True,
        operations=train_ops,
        augment=augment,     # Apply data augmentation only if requested
        seed=seed
    )

    # Create test dataset (never augment test data for fair evaluation)
    test_dataset = MNISTArithmeticDataset(
        train=False,
        operations=test_ops,
        augment=False,       # Never augment test data
        seed=seed
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # Shuffle training data for better learning
        num_workers=0        # Use 0 for compatibility across platforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,       # Don't shuffle test data for consistent evaluation
        num_workers=0        # Use 0 for compatibility across platforms
    )

    return train_loader, test_loader


# Test the dataset if run directly
if __name__ == "__main__":
    """
    Test the dataset functionality and show example data.
    Run with: python dataset.py
    """
    print("Testing MNIST Arithmetic Dataset...")
    print("=" * 50)

    # Create a small dataset for testing
    dataset = MNISTArithmeticDataset(train=True, augment=False, seed=42)
    print(f"Dataset length: {len(dataset)}")

    # Test a few samples
    for i in range(3):
        img, op_info, result = dataset[i]
        digit1, digit2, op_code = op_info.tolist()
        operation = 'add' if op_code == 0 else 'multiply'

        print(f"\nSample {i+1}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Digit 1: {digit1}, Digit 2: {digit2}")
        print(f"  Operation: {operation} (code: {op_code})")
        print(f"  Expected result: {digit1} {'+' if op_code == 0 else '×'} {digit2} = {result.item()}")

    # Test dataloader
    print(f"\nTesting DataLoader...")
    train_loader, test_loader = get_dataloaders(batch_size=4, augment=False, seed=42)

    for batch_idx, (data, operation_info, target) in enumerate(train_loader):
        print(f"Batch shape: {data.shape}")
        print(f"Operation info shape: {operation_info.shape}")
        print(f"Target shape: {target.shape}")
        break

    print("\n✅ Dataset test completed successfully!")