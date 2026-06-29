"""
===============================================================================
Project:
    CNN Hyperparameter Investigation on CIFAR-10 using PyTorch

Author:
    Dare Shonubi

Description:
    This project investigates the effects of CNN architecture and training
    hyperparameters on image classification performance using the CIFAR-10
    dataset.

Experiments:
    A. Number of Convolution Blocks
    B. Activation Functions
    C. Batch Size
    D. Optimizers

Framework:
    PyTorch

Dataset:
    CIFAR-10

Developed in:
    Google Colab

Compatible with:
    Google Colab
    Jupyter Notebook
    VS Code
    PyCharm
    Linux
    Windows
    macOS

===============================================================================
"""
# =============================================================================
# IMPORTS
# =============================================================================

import os
import random
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# =============================================================================
# MATPLOTLIB SETTINGS
# =============================================================================

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)
# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

SEED = 42

IMAGE_SIZE = 32

NUM_CLASSES = 10

EPOCHS = 15

LEARNING_RATE = 0.001

DEFAULT_BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

PROJECT_DIR = Path("CNN_Project")

MODEL_DIR = PROJECT_DIR / "models"

GRAPH_DIR = PROJECT_DIR / "graphs"

RESULT_DIR = PROJECT_DIR / "results"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

DATA_DIR = PROJECT_DIR / "data"

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    """
    Sets all random seeds to ensure reproducible experiments.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed()

logger.info("Random seed initialized.")
# =============================================================================
# PROJECT DIRECTORIES
# =============================================================================

def create_directories() -> None:
    """
    Create project directory structure if it does not exist.
    """

    directories = [
        PROJECT_DIR,
        MODEL_DIR,
        GRAPH_DIR,
        RESULT_DIR,
        CHECKPOINT_DIR,
        DATA_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("Project directories created.")


create_directories()
# =============================================================================
# DEVICE INFORMATION
# =============================================================================

logger.info("=" * 70)
logger.info("CNN Hyperparameter Investigation")
logger.info("=" * 70)

logger.info(f"PyTorch Version : {torch.__version__}")
logger.info(f"Torchvision     : {torchvision.__version__}")
logger.info(f"Device          : {DEVICE}")

if torch.cuda.is_available():
    logger.info(f"GPU             : {torch.cuda.get_device_name(0)}")

logger.info("=" * 70)

# =============================================================================
# DATASET LOADING
# =============================================================================

def load_datasets():
    """
    Downloads and loads CIFAR-10 dataset with normalization.

    Returns:
        train_dataset, test_dataset
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    logger.info("CIFAR-10 dataset loaded successfully.")

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Test size : {len(test_dataset)}")

    return train_dataset, test_dataset


train_dataset, test_dataset = load_datasets()

# =============================================================================
# DATA VISUALIZATION UTILITIES
# =============================================================================

def imshow(img):
    """
    Denormalizes and displays an image tensor.
    """

    img = img / 2 + 0.5
    np_img = img.numpy()

    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.axis("off")
    plt.show()
  # =============================================================================
# DATA EXPLORATION
# =============================================================================

def show_sample_images(dataset, n=8):
    """
    Displays sample images from dataset.
    """

    loader = DataLoader(dataset, batch_size=n, shuffle=True)

    images, labels = next(iter(loader))

    plt.figure(figsize=(12, 4))
    grid = make_grid(images, nrow=4)

    imshow(grid)

    print("Labels:")
    print([CLASS_NAMES[i] for i in labels])


show_sample_images(train_dataset)

# =============================================================================
# DATALOADERS
# =============================================================================

def create_dataloaders(batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Creates train and test DataLoaders.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    logger.info(f"DataLoaders created with batch size {batch_size}")

    return train_loader, test_loader


train_loader, test_loader = create_dataloaders()
# =============================================================================
# SANITY CHECK
# =============================================================================

images, labels = next(iter(train_loader))

logger.info(f"Batch shape: {images.shape}")
logger.info(f"Labels shape: {labels.shape}")

assert images.shape[1:] == (3, 32, 32), "Incorrect image shape!"

logger.info("Dataset pipeline verified successfully.")
# =============================================================================
# ACTIVATION FUNCTION SELECTOR
# =============================================================================

def get_activation(name: str):
    """
    Returns activation function based on string input.
    """

    name = name.lower()

    if name == "relu":
        return nn.ReLU()

    elif name == "tanh":
        return nn.Tanh()

    elif name == "leakyrelu":
        return nn.LeakyReLU(0.01)

    else:
        raise ValueError(f"Unsupported activation: {name}")
      # =============================================================================
# FLEXIBLE CNN MODEL
# =============================================================================

class FlexibleCNN(nn.Module):
    """
    CNN model with configurable:
    - number of convolution blocks
    - activation function
    - fully connected layer size
    """

    def __init__(
        self,
        conv_blocks: int = 2,
        activation: str = "relu",
        fc_units: int = 128
    ):

        super(FlexibleCNN, self).__init__()

        self.activation_name = activation
        act = get_activation(activation)

        layers = []

        in_channels = 3
        filters = [32, 64, 128]

        # =========================================================
        # CONVOLUTIONAL BLOCKS
        # =========================================================
        for i in range(conv_blocks):

            out_channels = filters[i]

            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )

            layers.append(act)
            layers.append(nn.MaxPool2d(2))

            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # =========================================================
        # FLATTEN SIZE CALCULATION
        # CIFAR-10: 32x32 images
        # Each pooling halves spatial dimensions
        # =========================================================

        final_size = 32 // (2 ** conv_blocks)
        flattened = in_channels * final_size * final_size

        # =========================================================
        # FULLY CONNECTED LAYERS
        # =========================================================

        self.fc = nn.Sequential(
            nn.Linear(flattened, fc_units),
            act,
            nn.Linear(fc_units, NUM_CLASSES)
        )

    def forward(self, x):

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
      # =============================================================================
# MODEL SANITY CHECK
# =============================================================================

model = FlexibleCNN(
    conv_blocks=2,
    activation="relu"
).to(DEVICE)

logger.info(model)

dummy_input = torch.randn(8, 3, 32, 32).to(DEVICE)

output = model(dummy_input)

logger.info(f"Output shape: {output.shape}")

assert output.shape == (8, NUM_CLASSES), "Model output shape incorrect"

logger.info("FlexibleCNN verified successfully.")
# =============================================================================
# ACCURACY FUNCTION
# =============================================================================

def compute_accuracy(outputs, labels):
    """
    Computes batch accuracy.
    """

    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)
  # =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, test_loader, optimizer, criterion, epochs=EPOCHS):
    """
    Trains the CNN model and tracks performance.

    Returns:
        history dict containing loss and accuracy
    """

    model.to(DEVICE)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):

        model.train()

        train_loss = 0
        train_acc = 0

        for images, labels in train_loader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += compute_accuracy(outputs, labels)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        # =========================================================
        # EVALUATION PHASE
        # =========================================================

        model.eval()

        test_loss = 0
        test_acc = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                test_acc += compute_accuracy(outputs, labels)

        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = test_acc / len(test_loader)

        # =========================================================
        # SAVE HISTORY
        # =========================================================

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["test_loss"].append(avg_test_loss)
        history["test_acc"].append(avg_test_acc)

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Acc: {avg_train_acc:.4f} | "
            f"Test Acc: {avg_test_acc:.4f}"
        )

    return history
  # =============================================================================
# OPTIMIZER SELECTOR
# =============================================================================

def get_optimizer(name, model, lr=LEARNING_RATE):
    """
    Returns optimizer based on string input.
    """

    name = name.lower()

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)

    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    elif name == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr)

    else:
        raise ValueError(f"Unsupported optimizer: {name}")
      # =============================================================================
# LOSS FUNCTION
# =============================================================================

criterion = nn.CrossEntropyLoss()
# =============================================================================
# QUICK TRAINING TEST
# =============================================================================

test_model = FlexibleCNN(
    conv_blocks=2,
    activation="relu"
).to(DEVICE)

optimizer = get_optimizer("adam", test_model)

logger.info("Starting quick test training...")

history = train_model(
    test_model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    epochs=1
)

logger.info("Quick training test completed successfully.")
# =============================================================================
# EXPERIMENT STORAGE
# =============================================================================

all_results = []
all_histories = {}

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    experiment_name: str,
    config_name: str,
    conv_blocks: int = 2,
    activation: str = "relu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    optimizer_name: str = "adam",
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE
):
    """
    Runs a single controlled experiment and logs results.
    """

    logger.info("=" * 70)
    logger.info(f"Experiment: {experiment_name} | {config_name}")
    logger.info("=" * 70)

    # =========================================================
    # DATA LOADERS (controlled variable: batch size)
    # =========================================================

    train_loader, test_loader = create_dataloaders(batch_size)

    # =========================================================
    # MODEL
    # =========================================================

    model = FlexibleCNN(
        conv_blocks=conv_blocks,
        activation=activation
    ).to(DEVICE)

    # =========================================================
    # OPTIMIZER
    # =========================================================

    optimizer = get_optimizer(
        optimizer_name,
        model,
        learning_rate
    )

    # =========================================================
    # TRAINING
    # =========================================================

    start_time = time.time()

    history = train_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        epochs
    )

    duration = time.time() - start_time

    # =========================================================
    # FINAL RESULT
    # =========================================================

    final_acc = history["test_acc"][-1]

    # Save results
    result_entry = {
        "Experiment": experiment_name,
        "Configuration": config_name,
        "ConvBlocks": conv_blocks,
        "Activation": activation,
        "BatchSize": batch_size,
        "Optimizer": optimizer_name,
        "FinalAccuracy": final_acc,
        "TrainingTime": duration
    }

    all_results.append(result_entry)

    key = f"{experiment_name}_{config_name}"
    all_histories[key] = history

    logger.info(f"Final Accuracy: {final_acc:.4f}")
    logger.info(f"Training Time : {duration:.2f} sec")

    return history, model
  # =============================================================================
# RESULTS TABLE
# =============================================================================

def get_results_table():
    """
    Returns all experiment results as a DataFrame.
    """

    df = pd.DataFrame(all_results)

    df = df.sort_values(
        by="FinalAccuracy",
        ascending=False
    ).reset_index(drop=True)

    return df
  # =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results():
    """
    Saves experiment results to CSV.
    """

    df = get_results_table()

    output_path = RESULT_DIR / "experiment_results.csv"

    df.to_csv(output_path, index=False)

    logger.info(f"Results saved to {output_path}")

    return df
  # =============================================================================
# PLOTTING FUNCTION
# =============================================================================

def plot_experiment(histories, title, filename=None):
    """
    Plots training accuracy curves for multiple runs.
    """

    plt.figure(figsize=(10, 6))

    for label, history in histories.items():
        plt.plot(
            history["train_acc"],
            label=label
        )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.legend()
    plt.grid(True)

    if filename:
        path = GRAPH_DIR / filename
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {path}")

    plt.show()
  # =============================================================================
# BEST MODEL TRACKER
# =============================================================================

def get_best_model():
    """
    Returns best performing experiment result.
    """

    df = get_results_table()

    best = df.iloc[0]

    logger.info("BEST MODEL FOUND:")
    logger.info(best)

    return best
  # =============================================================================
# EXPERIMENT A: NUMBER OF CONVOLUTION BLOCKS
# =============================================================================

history_A1, model_A1 = run_experiment(
    experiment_name="Layers",
    config_name="1_Block",
    conv_blocks=1
)

history_A2, model_A2 = run_experiment(
    experiment_name="Layers",
    config_name="2_Blocks",
    conv_blocks=2
)

history_A3, model_A3 = run_experiment(
    experiment_name="Layers",
    config_name="3_Blocks",
    conv_blocks=3
)
# =============================================================================
# EXPERIMENT B: ACTIVATION FUNCTIONS
# =============================================================================

history_B1, model_B1 = run_experiment(
    experiment_name="Activation",
    config_name="ReLU",
    activation="relu"
)

history_B2, model_B2 = run_experiment(
    experiment_name="Activation",
    config_name="Tanh",
    activation="tanh"
)

history_B3, model_B3 = run_experiment(
    experiment_name="Activation",
    config_name="LeakyReLU",
    activation="leakyrelu"
)
# =============================================================================
# EXPERIMENT C: BATCH SIZE
# =============================================================================

history_C1, model_C1 = run_experiment(
    experiment_name="BatchSize",
    config_name="16",
    batch_size=16
)

history_C2, model_C2 = run_experiment(
    experiment_name="BatchSize",
    config_name="32",
    batch_size=32
)

history_C3, model_C3 = run_experiment(
    experiment_name="BatchSize",
    config_name="64",
    batch_size=64
)
# =============================================================================
# EXPERIMENT D: OPTIMIZERS
# =============================================================================

history_D1, model_D1 = run_experiment(
    experiment_name="Optimizer",
    config_name="Adam",
    optimizer_name="adam"
)

history_D2, model_D2 = run_experiment(
    experiment_name="Optimizer",
    config_name="SGD",
    optimizer_name="sgd"
)

history_D3, model_D3 = run_experiment(
    experiment_name="Optimizer",
    config_name="RMSprop",
    optimizer_name="rmsprop"
)
# =============================================================================
# SAVE MODELS
# =============================================================================

def save_model(model, name):
    path = MODEL_DIR / f"{name}.pth"
    torch.save(model.state_dict(), path)
    logger.info(f"Saved model: {path}")


save_model(model_A2, "best_layers_model")
save_model(model_B1, "relu_model")
save_model(model_C2, "batch32_model")
save_model(model_D1, "adam_model")
# =============================================================================
# FINAL RESULTS TABLE
# =============================================================================

results_df = get_results_table()
results_df
# =============================================================================
# SAVE RESULTS TO CSV
# =============================================================================

results_df.to_csv(
    RESULT_DIR / "final_results.csv",
    index=False
)

logger.info("Final results saved successfully.")

  
