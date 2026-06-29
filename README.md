
# CNN Hyperparameter Investigation on CIFAR-10

A PyTorch-based deep learning project that systematically investigates how architectural choices and hyperparameters affect the performance of Convolutional Neural Networks (CNNs) on the CIFAR-10 image classification dataset.

---

## Project Overview

This project explores how different CNN design choices influence model performance on image classification tasks. Instead of building a single model, we conduct **controlled experiments** to understand the effect of:

- Number of convolutional layers
- Activation functions
- Batch sizes
- Optimizers

The goal is to move beyond trial-and-error model building and adopt a **scientific experimental approach**.

---

## Dataset

### CIFAR-10
- 60,000 32×32 colour images
- 10 classes:
  - airplane 
  - automobile 
  - bird 
  - cat 
  - deer 
  - dog 
  - frog 
  - horse 
  - ship 
  - truck 

- 50,000 training images
- 10,000 test images

Source: torchvision CIFAR-10 dataset

---

## Objectives

- Build a baseline CNN using PyTorch
- Design controlled experiments
- Measure the impact of key hyperparameters
- Compare model performance systematically
- Identify the best-performing configuration

---

## Experimental Design

### Experiment A — CNN Depth
- 1 convolution block (shallow)
- 2 convolution blocks (baseline)
- 3 convolution blocks (deep)

---

### Experiment B — Activation Functions
- ReLU
- Tanh
- LeakyReLU

---

### Experiment C — Batch Size
- 16
- 32 (baseline)
- 64

---

### Experiment D — Optimizers
- Adam (baseline)
- SGD (momentum = 0.9)
- RMSprop

---

## Model Architecture

A flexible CNN architecture was implemented:

- Configurable convolutional blocks (1–3)
- MaxPooling after each convolution layer
- Fully connected layer with 128 neurons
- Output layer: 10 classes (softmax via logits)
- Activation functions: ReLU / Tanh / LeakyReLU

---

## Training Setup

- Loss Function: CrossEntropyLoss
- Epochs: 15
- Optimizers: Adam / SGD / RMSprop
- Framework: PyTorch
- Device: GPU (CUDA if available)
- Image Normalization: mean=0.5, std=0.5

---

## Results Summary

The experiments were evaluated using test accuracy.

> Final results vary depending on training runs.

| Experiment | Best Configuration | Insight |
|------------|-------------------|--------|
| Layers | 2–3 Conv Blocks | Deeper models improve feature extraction but may overfit |
| Activation | ReLU | Faster convergence and better gradient flow |
| Batch Size | 32 | Best balance between stability and performance |
| Optimizer | Adam | Most stable and highest accuracy |

---

## Key Findings

- Increasing CNN depth improves performance up to a point
- ReLU consistently outperforms tanh and LeakyReLU in convergence speed
- Batch size of 32 provides optimal training stability
- Adam optimizer achieves the best overall performance

---

## Visual Results

The project generates the following visualizations:

- Training accuracy curves (per experiment)
- Comparison bar charts
- Confusion matrix
- Sample predictions grid

All graphs are saved in the `/graphs` directory.

---

## Best Model (Final Outcome)

The best-performing configuration (based on experimental results) is typically:

- 2–3 convolution blocks
- ReLU activation
- Batch size = 32
- Adam optimizer

---

## Project Structure
CNN_Project/
│
├── app.py                  # Full implementation
├── graphs/                 # Training curves & visualizations
├── results/                # CSV + reports
├── models/                 # Saved models
├── checkpoints/            # Optional training checkpoints
├── data/                   # CIFAR-10 dataset

---

## How to Run

### 1. Install dependencies
```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn

python app.py

Streamlit Extension (Future Work)

This project can be extended into a Streamlit dashboard:

* Upload image → predict class
* Compare model performance interactively
* View training graphs dynamically
* Switch between trained models

⸻

🔬 Limitations

* Limited hyperparameter search space
* No data augmentation used
* Fixed CNN architecture family
* Training time constraints

⸻

📌 Future Improvements

* Add dropout and batch normalization
* Use learning rate scheduling
* Implement data augmentation
* Extend to ResNet-style architectures
* Hyperparameter tuning with Optuna

⸻

👤 Author

Dare Shonubi
