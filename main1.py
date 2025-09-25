import torch
from torch.utils.data import DataLoader
import time
from dataset import MNISTArithmeticDataset, set_seed

def main():
    """
    Architecture 1 - Training script
    """
    # Set global seed for reproducibility
    set_seed(42)

    print("MNIST Arithmetic Training - Architecture 1")
    print("=" * 50)

    # Set up data loaders (with data augmentation for better training)
    train_dataset = MNISTArithmeticDataset(train=True, operations=['add', 'multiply'], augment=True, seed=42)
    test_dataset = MNISTArithmeticDataset(train=False, operations=['add', 'multiply'], augment=False, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Import your implementations
    from model1 import MNISTArithmeticNet
    from trainer1 import ArithmeticTrainer

    # Create model and trainer
    model = MNISTArithmeticNet()
    trainer = ArithmeticTrainer(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train the model
    print("\nStarting training...")
    start_time = time.time()
    history = trainer.train(train_loader, test_loader, epochs=10)
    training_time = time.time() - start_time

    # Save training results for analysis
    torch.save({
        'history': history,
        'training_time': training_time,
        'model_params': sum(p.numel() for p in model.parameters())
    }, 'training_results_arch1.pth')

    print(f"\nTraining time: {training_time:.1f} seconds")
    print("\n✅ Training completed!")
    print("✅ Best model saved as 'best_model_arch1.pth'")
    print("✅ Training results saved as 'training_results_arch1.pth'")


if __name__ == '__main__':
    main()