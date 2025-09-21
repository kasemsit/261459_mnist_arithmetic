import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ArithmeticTrainer:
    """
    Architecture 1: End-to-End CNN Trainer

    Handles training and validation for single-task learning:
    - Takes combined 28x56 images and operation codes
    - Predicts arithmetic results directly (0-90)
    - Uses CrossEntropyLoss for classification
    - Tracks training/validation loss and accuracy
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

        # TODO 5a: Set up training components
        self.criterion = # TODO: nn.CrossEntropyLoss
        self.optimizer = # TODO: optim.Adam

        # Learning rate scheduler: gets reduced by half every 5 epochs:
        #   - Epochs 1-5: lr = 0.001
        #   - Epochs 6-10: lr = 0.0005
        #   - Epochs 11-15: lr = 0.00025
        #   - Epochs 16-20: lr = 0.000125
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Track history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, train_loader):
        """TODO 5b: Implement one training epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, operation_info, target in tqdm(train_loader, desc="Training"):
            # TODO 5b: Move tensors to device
            data, target = # TODO: .to()
            operation_info = # TODO: .to()

            # TODO 5c: Training step
            self.optimizer.zero_grad()
            output = # TODO: self.model()
            loss = # TODO: self.criterion()
            # TODO: loss.backward()
            # TODO: self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, val_loader):
        """TODO 5d: Implement validation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, operation_info, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                operation_info = operation_info.to(self.device)

                output = self.model(data, operation_info)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return total_loss / len(val_loader), 100. * correct / total

    def train(self, train_loader, val_loader, epochs=20):
        """TODO 5e: Main training loop"""
        print(f"Training on {self.device}")
        best_val_acc = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # TODO 5f: Run training and validation
            train_loss, train_acc = # TODO: self.train_epoch()
            val_loss, val_acc = # TODO: self.validate()

            # Update scheduler and save metrics
            self.scheduler.step()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f"Train: {train_loss:.4f} loss, {train_acc:.2f}% acc")
            print(f"Val: {val_loss:.4f} loss, {val_acc:.2f}% acc")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model_arch1.pth')
                print(f"✅ New best: {best_val_acc:.2f}%")

        return self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies