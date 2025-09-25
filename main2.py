import torch
from torch.utils.data import DataLoader
import time
from dataset import MNISTArithmeticDataset, set_seed

def main():
    """
    Architecture 2 - Multi-task CNN training script

    This script trains Architecture 2 (Multi-task CNN) independently.
    Architecture 2 is a standalone approach, not dependent on Architecture 1.
    """
    # Set global seed for reproducibility
    set_seed(42)

    print("MNIST Arithmetic Training - Architecture 2 (Multi-task CNN)")
    print("=" * 60)

    # Set up datasets (with data augmentation for better training)
    train_dataset = MNISTArithmeticDataset(train=True, operations=['add', 'multiply'], augment=True, seed=42)
    test_dataset = MNISTArithmeticDataset(train=False, operations=['add', 'multiply'], augment=False, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Import Architecture 2 components
    from model2 import MultiTaskArithmeticNet, count_parameters, get_model_size
    from trainer2 import MultiTaskTrainer

    # Train Multi-Task CNN (Architecture 2)
    print("\nStarting Architecture 2 training...")

    # Create and train multi-task model
    multitask_model = MultiTaskArithmeticNet()
    print(f"Model parameters: {count_parameters(multitask_model):,}")

    multitask_trainer = MultiTaskTrainer(multitask_model)

    start_time = time.time()
    multitask_history = multitask_trainer.train(train_loader, test_loader, epochs=20)
    multitask_time = time.time() - start_time

    print(f"\nTraining time: {multitask_time:.1f} seconds")

    # Save training results for analysis
    torch.save({
        'history': multitask_history,
        'training_time': multitask_time,
        'model_params': count_parameters(multitask_model)
    }, 'training_results_arch2.pth')

    # Display results
    multitask_best = max(multitask_history[3])  # Best validation accuracy

    print(f"\n🎯 Architecture 2 Results:")
    print(f"   Best Accuracy: {multitask_best:.2f}%")
    print(f"   Model Parameters: {count_parameters(multitask_model):,}")
    print(f"   Training Time: {multitask_time:.1f} seconds")
    print(f"   Multi-task Learning: ✅ (Digit recognition + Arithmetic)")

    print("\n✅ Training completed!")
    print("✅ Training results saved as 'training_results_arch2.pth'")
    print("✅ Best model saved as 'best_model_arch2.pth'")

if __name__ == '__main__':
    main()