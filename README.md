# MNIST Arithmetic Assignment

Build neural networks that can perform arithmetic operations (addition and multiplication) on MNIST digit images.

In this assignment, there are 🎯 Two Architectures: **Architecture 1 (End-to-end CNN)** and  **Architecture 2 (Multi-task CNN)**.

## Policy

- **✅ This is an open-book assignment.**
- **✅ Feel free to discuss concepts and approaches with classmates.**
- **✅ Each student must submit their own implementation and report.**
- **✅ You may use online resources, documentation, and tutorials.**
- **✅ Generative AI such as ChatGPT are allowed.**
- **❌ Do not copy code directly from others or share complete solutions.**

## 🎓 Learning Objectives

- Implement CNN architectures in PyTorch
- Understand training loops and validation
- Practice model evaluation and analysis

## 🏗️ Architecture Diagrams

### **Architecture 1: End-to-End CNN**
```
Input: Combined Image (28x56)     Operation Info
        [digit1][digit2]              [op_code]
               │                         │
               ▼                         │
        ┌─────────────┐                  │
        │   Conv2D    │                  │
        │  (1→32→64→  │                  │
        │   →128)     │                  │
        │             │                  │
        │  MaxPool    │                  │
        │  Dropout    │                  │
        └─────────────┘                  │
               │                         │
               ▼                         │
        ┌─────────────┐                  │
        │   Flatten   │                  │
        │     FC      │                  │
        │ (2688→512→  │                  │
        │   →256)     │                  │
        └─────────────┘                  │
               │                         │
               ▼                         │
        [Image Features: 256]            │
               │                         │
               └────────┬────────────────┘
                        ▼
               ┌─────────────────┐
               │   Concatenate   │
               │ [256 + 1 = 257] │
               └─────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   Classifier    │
               │   (257→91)      │
               └─────────────────┘
                        │
                        ▼
              [Arithmetic Result: 0-90]

**Loss Function**: CrossEntropyLoss
• Single-task learning: Direct mapping from image to arithmetic result
• 91 output classes (0-90 for addition/multiplication results)
• End-to-end optimization of entire pipeline
**Model** has 1,624,246 parameters
```

### **Architecture 2: Multi-Task CNN**
```
Input: Combined Image (28x56)
        [digit1][digit2]
               │
               ▼
        ┌─────────────┐
        │    Split    │
        │   28x56 →   │
        │ 2×(28x28)   │
        └─────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
 ┌─────────────┐ ┌─────────────┐
 │ Shared CNN  │ │ Shared CNN  │
 │   (28x28)   │ │   (28x28)   │
 │    │ │ │    │ │    │ │ │    │
 │ Conv→Pool   │ │ Conv→Pool   │
 │ →BatchNorm  │ │ →BatchNorm  │
 │    FC       │ │    FC       │
 └─────────────┘ └─────────────┘
        │             │
        ▼             ▼
   [Digit1: 10]  [Digit2: 10]     Operation Info
        │             │              [op_code]
        └──────┬──────┘                  │
               │                         │
               ▼                         │
        ┌─────────────────┐             │
        │   Concatenate   │             │
        │ [10 + 10 + 1]   │◄────────────┘
        │    = 21         │
        └─────────────────┘
               │
               ▼
        ┌─────────────────┐
        │ Operation MLP   │
        │  (21→128→64)    │
        └─────────────────┘
               │
               ▼
        ┌─────────────────┐
        │   Classifier    │
        │    (64→91)      │
        └─────────────────┘
               │
               ▼
      [Arithmetic Result: 0-90]

**Loss Function**: Multi-task Weighted Loss
• Combined loss: `result_loss + 0.3 * (digit1_loss + digit2_loss)`
• Three CrossEntropyLoss functions for each output
• Multi-task learning with explicit digit recognition

**Multi-task Outputs**:
• Digit 1 prediction (10 classes)
• Digit 2 prediction (10 classes)
• Arithmetic result (91 classes)

**Model** has 407,845 parameters
```

## 🚀 Quick Start

1. **Create and activate virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv mnist_env

   # Activate virtual environment
   # On Linux/Mac:
   source mnist_env/bin/activate
   # On Windows:
   # mnist_env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the dataset**:
   ```bash
   python dataset.py
   ```

4. **Run training (after implementing templates)**:
   ```bash
   # For Architecture 1
   python main1.py

   # For Architecture 2
   python main2.py
   ```


## 📁 Files Overview

### **Provided (Ready to Use)**
- `dataset.py` - Complete MNIST arithmetic dataset with data augmentation support and reproducibility
- `requirements.txt` - All dependencies (works for both assignments)
- `main1.py` - Ready-to-run training script for Architecture 1
- `main2.py` - Ready-to-run training script for Architecture 2


### **Templates (Copy and Complete)**

**Architecture 1:**
- `model_architecture1_template.py` → copy to `model1.py`
- `trainer_architecture1_template.py` → copy to `trainer1.py`
- Run `main1.py` when completed (will run Architecture 1 only)

**Architecture 2:**
- `model_architecture2_template.py` → copy to `model2.py`
- `trainer_architecture2_template.py` → copy to `trainer2.py`
- Run `main2.py` when completed (will run Architecture 2 only)


## 📝 What to do?

1. Complete the **TODO** comments in each template file (4 files totally).
2. Train the models by `python main1.py` and `python main2.py`.
3. You can check you model implementation using `python model1.py` and `python model2.py`.
4. Success criteria ✅ is that the models train without errors and achieve >90% validation accuracy.
   
## 📝 What to submit?

1. Submit `model1.py`, `model2.py`, `trainer1.py`, and `trainer2.py`. 
2. Written report `report.pdf` that
   - shows the steps to compute **the number of parameters** for each architecture **in details** (i.e. how 1,624,246 and 407,845 parameters are derived).
   - **compares** and **discuss** about the two architectures in terms of, for example, model performance, memory efficiency, loss functions, interpretability, etc. 



Good luck! 🚀

---