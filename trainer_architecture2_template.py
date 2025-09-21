import torch
import torch.nn as nn
from tqdm import tqdm

# Import the basic trainer
from trainer1 import ArithmeticTrainer

class MultiTaskTrainer(ArithmeticTrainer):
    """
    Architecture 2: Multi-task CNN Trainer

    Extends the basic trainer to handle multi-task learning:
    - Multiple model outputs (result + digit1 + digit2)
    - Combined loss functions with weighting
    - Multi-task accuracy tracking
    - Explicit digit recognition evaluation
    - Inherit optimizer from trainer1.py

    Key Features:
    - Weighted loss: result_loss + 0.3 * (digit1_loss + digit2_loss)
    - Three separate accuracy metrics
    - Enhanced interpretability through digit predictions
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(model, device)

        # TODO 6a: Initialize digit classification loss function
        self.digit_criterion = # TODO: nn.CrossEntropyLoss()

        # Track digit accuracies
        self.digit1_accuracies = []
        self.digit2_accuracies = []

    def train_epoch(self, train_loader):
        """TODO 6b: Multi-task training epoch"""
        self.model.train()
        total_loss = 0
        correct_result = correct_digit1 = correct_digit2 = 0
        total = 0

        for data, operation_info, target in tqdm(train_loader, desc="Multi-task Training"):
            # TODO 6b: Move tensors to device
            data = # TODO: data.to(self.device)
            target = # TODO: target.to(self.device)
            operation_info = # TODO: operation_info.to(self.device)

            # True digit labels
            digit1_true = operation_info[:, 0]
            digit2_true = operation_info[:, 1]

            self.optimizer.zero_grad()

            # TODO 6c: Multi-task forward pass
            result_out, digit1_out, digit2_out = # TODO: self.model(data, operation_info)

            # TODO 6d: Calculate individual losses
            result_loss = # TODO: self.criterion(result_out, target)
            digit1_loss = # TODO: self.digit_criterion(digit1_out, digit1_true)
            digit2_loss = # TODO: self.digit_criterion(digit2_out, digit2_true)

            # TODO 6e: Combine losses with weighting
            total_loss_batch = # TODO: result_loss + 0.3 * (digit1_loss + digit2_loss)

            total_loss_batch.backward()
            self.optimizer.step()

            # Calculate accuracies
            total_loss += total_loss_batch.item()
            pred_result = result_out.argmax(dim=1)
            pred_digit1 = digit1_out.argmax(dim=1)
            pred_digit2 = digit2_out.argmax(dim=1)

            correct_result += pred_result.eq(target).sum().item()
            correct_digit1 += pred_digit1.eq(digit1_true).sum().item()
            correct_digit2 += pred_digit2.eq(digit2_true).sum().item()
            total += target.size(0)

        # Calculate accuracies
        result_acc = 100. * correct_result / total
        digit1_acc = 100. * correct_digit1 / total
        digit2_acc = 100. * correct_digit2 / total

        print(f"  Digit1: {digit1_acc:.1f}%, Digit2: {digit2_acc:.1f}%")
        self.digit1_accuracies.append(digit1_acc)
        self.digit2_accuracies.append(digit2_acc)

        return total_loss / len(train_loader), result_acc

    def validate(self, val_loader):
        """TODO 6f: Multi-task validation"""
        self.model.eval()
        total_loss = correct_result = correct_digit1 = correct_digit2 = total = 0

        with torch.no_grad():
            for data, operation_info, target in tqdm(val_loader, desc="Multi-task Validation"):
                # TODO 6f: Move tensors to device
                data = # TODO: data.to(self.device)
                target = # TODO: target.to(self.device)
                operation_info = # TODO: operation_info.to(self.device)

                digit1_true = operation_info[:, 0]
                digit2_true = operation_info[:, 1]

                # TODO 6g: Get model outputs
                result_out, digit1_out, digit2_out = # TODO: self.model(data, operation_info)

                # TODO 6h: Calculate individual losses
                result_loss = # TODO: self.criterion(result_out, target)
                digit1_loss = # TODO: self.digit_criterion(digit1_out, digit1_true)
                digit2_loss = # TODO: self.digit_criterion(digit2_out, digit2_true)

                # TODO 6i: Combine losses with weighting
                total_loss_batch = # TODO: result_loss + 0.3 * (digit1_loss + digit2_loss)
                total_loss += total_loss_batch.item()

                pred_result = result_out.argmax(dim=1)
                pred_digit1 = digit1_out.argmax(dim=1)
                pred_digit2 = digit2_out.argmax(dim=1)

                correct_result += pred_result.eq(target).sum().item()
                correct_digit1 += pred_digit1.eq(digit1_true).sum().item()
                correct_digit2 += pred_digit2.eq(digit2_true).sum().item()
                total += target.size(0)

        result_acc = 100. * correct_result / total
        print(f"  Val Digit1: {100. * correct_digit1 / total:.1f}%, Digit2: {100. * correct_digit2 / total:.1f}%")

        return total_loss / len(val_loader), result_acc