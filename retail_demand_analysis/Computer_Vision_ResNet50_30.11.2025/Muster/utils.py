"""
Utility functions for CIFAR-10 classification with TensorFlow / Keras.

This module provides:

- Reproducible setup
- CIFAR-10 loading and preprocessing
- Data augmentation pipeline
- A reasonably strong CNN architecture for CIFAR-10
- Model compilation, training, evaluation and prediction helpers
- Plotly figure export helpers (HTML + PNG)
- Image enhancement utilities for nicer visualisations
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

from plotly.graph_objects import Figure

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

# CIFAR-10 specific constants
INPUT_SHAPE: Final[Tuple[int, int, int]] = (32, 32, 3)
NUM_CLASSES: Final[int] = 10

CLASS_NAMES: List[str] = [
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

# Optional emoji representation for visualisations
CLASS_NAMES_EMOJI: List[str] = [
    "✈️",
    "🚗",
    "🦜",
    "🐱",
    "🫎",
    "🐶",
    "🐸",
    "🐴",
    "🛳️",
    "🚛",
]

# Base directories for exported figures and artifacts (from src/ to project root)
PLOTS_DIR: Final[Path] = Path("../plots")
DOCS_DIR: Final[Path] = Path("../docs")
MODELS_DIR: Final[Path] = Path("../models")
RESULTS_DIR: Final[Path] = Path("../results")

# Upscaling factor for CIFAR-10 images (for nicer plots)
UPSCALE_FACTOR: Final[int] = 4


# ---------------------------------------------------------------------------
# Image enhancement utilities
# ---------------------------------------------------------------------------

def upscale_and_super_sharpen(
    img: np.ndarray,
    scale: int = UPSCALE_FACTOR,
) -> np.ndarray:
    """
    Aggressively upscale and enhance a CIFAR-10 image for visualisation.

    This is intended ONLY for plots / presentations, not for training.

    The pipeline:
    - Upscale with a high-quality resampling filter (LANCZOS)
    - Apply auto-contrast to remove dullness
    - Boost contrast and sharpness
    - Apply a light unsharp mask on top

    Parameters
    ----------
    img : np.ndarray
        Input image, shape (H, W, 3), typically 32x32x3, dtype uint8 or float.
    scale : int, default UPSCALE_FACTOR (4)
        Upscaling factor (e.g. 4 -> 32x32 → 128x128).

    Returns
    -------
    np.ndarray
        Enhanced image as uint8, shape (H * scale, W * scale, 3).
    """
    pil_img = Image.fromarray(img.astype("uint8"))

    # High-quality upscaling
    new_size = (pil_img.width * scale, pil_img.height * scale)
    pil_up = pil_img.resize(new_size, resample=Image.Resampling.LANCZOS)

    # Remove dullness via auto-contrast
    pil_up = ImageOps.autocontrast(pil_up, cutoff=1)

    # Boost contrast & sharpness
    pil_up = ImageEnhance.Contrast(pil_up).enhance(1.15)
    pil_up = ImageEnhance.Sharpness(pil_up).enhance(3.0)

    # Additional unsharp mask (light but noticeable)
    pil_up = pil_up.filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=0),
    )

    return np.array(pil_up, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility.

    Parameters
    ----------
    seed : int, default 42
        Seed value used for NumPy and TensorFlow.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Plotly export helper
# ---------------------------------------------------------------------------

def save_fig(fig: Figure, name: str, scale: int = 2) -> None:
    """
    Save a Plotly figure as both HTML (interactive) and PNG (static).

    Parameters
    ----------
    fig : Figure
        The Plotly figure to be saved.
    name : str
        Base file name without extension (e.g. "class_distribution").
    scale : int, default 2
        Scale factor for the PNG export (higher = higher resolution).
        Requires the `kaleido` package to be installed.
    """
    # Ensure output directories exist
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    html_path = DOCS_DIR / f"{name}.html"
    png_path = PLOTS_DIR / f"{name}.png"

    # Save interactive HTML file
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    # Save static PNG image (requires kaleido)
    fig.write_image(str(png_path), scale=scale)

    print(f"Saved HTML to {html_path}")
    print(f"Saved PNG to {png_path}")


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

@dataclass
class Cifar10Data:
    """
    Container for CIFAR-10 data.

    Attributes
    ----------
    x_train : np.ndarray
        Training images, shape (N_train, 32, 32, 3), dtype float32 or uint8.
    y_train : np.ndarray
        Training labels as integers, shape (N_train,).
    x_test : np.ndarray
        Test images, shape (N_test, 32, 32, 3), dtype float32 or uint8.
    y_test : np.ndarray
        Test labels as integers, shape (N_test,).
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_cifar10(normalize: bool = True) -> Cifar10Data:
    """
    Load CIFAR-10 dataset and optionally normalize images to [0, 1].

    Note
    ----
    If you use the Rescaling layer inside the model (`layers.Rescaling(1./255)`),
    you may want to call this with `normalize=False` to avoid double-scaling.

    Parameters
    ----------
    normalize : bool, default True
        If True, images are scaled to the range [0, 1].

    Returns
    -------
    Cifar10Data
        Dataclass containing train and test splits.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Flatten labels from shape (N, 1) to (N,)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    return Cifar10Data(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def create_data_augmentation() -> keras.Sequential:
    """
    Create a data augmentation pipeline for CIFAR-10 images.

    Uses common and effective augmentations:
    - Random horizontal flips
    - Small random rotations
    - Small random zoom

    Returns
    -------
    keras.Sequential
        Keras Sequential model containing augmentation layers.
    """
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
        ],
        name="data_augmentation",
    )
    return data_augmentation


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_cifar10_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    data_augmentation: Optional[keras.Model] = None,
) -> keras.Model:
    """
    Build a CNN model for CIFAR-10 classification.

    Architecture:
    - Optional data augmentation block
    - Rescaling to [0, 1] (if you pass uint8 images)
    - 3 convolutional blocks with BatchNorm, ReLU, MaxPooling, Dropout
    - GlobalAveragePooling + Dense(256) + Dropout
    - Final Dense(num_classes, softmax)

    Parameters
    ----------
    input_shape : tuple[int, int, int], default (32, 32, 3)
        Shape of the input images.
    num_classes : int, default 10
        Number of classes in CIFAR-10.
    data_augmentation : keras.Model or None, default None
        If provided, this model is applied to the inputs for data augmentation.

    Returns
    -------
    keras.Model
        Uncompiled Keras model ready for training.
    """
    inputs = keras.Input(shape=input_shape)
    x: tf.Tensor = inputs

    # Optional augmentation block (only active during training)
    if data_augmentation is not None:
        x = data_augmentation(x)

    # If you pass raw uint8 images into the model, keep this.
    # If you normalize externally to [0, 1], consider removing it
    # or calling load_cifar10(normalize=False).
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.30)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.40)(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
    return model


# ---------------------------------------------------------------------------
# Compilation, training and evaluation
# ---------------------------------------------------------------------------

def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> None:
    """
    Compile the model with a sensible default optimizer and loss.

    Parameters
    ----------
    model : keras.Model
        Model to compile.
    learning_rate : float, default 1e-3
        Learning rate for the Adam optimizer.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def create_default_callbacks(
    patience_reduce_lr: int = 3,
    patience_early_stop: int = 8,
    min_lr: float = 1e-6,
) -> List[keras.callbacks.Callback]:
    """
    Create a list of default callbacks for training.

    Includes:
    - ReduceLROnPlateau to lower learning rate when validation accuracy plateaus
    - EarlyStopping to stop training when no more improvement is observed

    Parameters
    ----------
    patience_reduce_lr : int, default 3
        Number of epochs with no improvement after which learning rate is reduced.
    patience_early_stop : int, default 8
        Number of epochs with no improvement after which training is stopped.
    min_lr : float, default 1e-6
        Lower bound on the learning rate.

    Returns
    -------
    list[keras.callbacks.Callback]
        List of configured callbacks.
    """
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=patience_reduce_lr,
        min_lr=min_lr,
        verbose=1,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience_early_stop,
        restore_best_weights=True,
        verbose=1,
    )

    return [reduce_lr, early_stop]


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    batch_size: int = 64,
    epochs: int = 30,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
    validation_split: float = 0.1,
) -> keras.callbacks.History:
    """
    Train the model on CIFAR-10 training data.

    Parameters
    ----------
    model : keras.Model
        Compiled Keras model.
    x_train : np.ndarray
        Training images, shape (N_train, 32, 32, 3).
    y_train : np.ndarray
        Training labels as integers, shape (N_train,).
    x_val : np.ndarray or None, default None
        Optional validation images. If None, `validation_split` is used.
    y_val : np.ndarray or None, default None
        Optional validation labels.
    batch_size : int, default 64
        Training batch size.
    epochs : int, default 30
        Maximum number of training epochs.
    callbacks : list[keras.callbacks.Callback] or None, default None
        Additional callbacks. If None, default callbacks are created.
    validation_split : float, default 0.1
        Fraction of training data to use for validation if x_val is None.

    Returns
    -------
    keras.callbacks.History
        Keras History object containing training logs.
    """
    if callbacks is None:
        callbacks = create_default_callbacks()

    if x_val is not None and y_val is not None:
        validation_data: Any = (x_val, y_val)
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

    return history


def evaluate_model(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate the model on the test set.

    Parameters
    ----------
    model : keras.Model
        Trained Keras model.
    x_test : np.ndarray
        Test images, shape (N_test, 32, 32, 3).
    y_test : np.ndarray
        Test labels as integers, shape (N_test,).

    Returns
    -------
    dict
        Dictionary with keys 'loss' and 'accuracy'.
    """
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return {"loss": float(loss), "accuracy": float(acc)}


# ---------------------------------------------------------------------------
# Model + history persistence
# ---------------------------------------------------------------------------

def save_model_with_history(
    model: keras.Model,
    history: keras.callbacks.History,
    name: str,
) -> None:
    """
    Save a Keras model and its training history to disk.

    Files:
    - models/{name}.keras
    - results/history_{name}.json

    Parameters
    ----------
    model : keras.Model
        Trained Keras model to save.
    history : keras.callbacks.History
        History object returned from `model.fit`.
    name : str
        Base name for the saved files (without extension).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{name}.keras"
    history_path = RESULTS_DIR / f"history_{name}.json"

    model.save(str(model_path))

    with history_path.open("w") as f:
        json.dump(history.history, f)

    print(f"Saved model to {model_path}")
    print(f"Saved history to {history_path}")


def load_model(name: str) -> keras.Model:
    """
    Load a previously saved Keras model.

    Parameters
    ----------
    name : str
        Base name used when saving the model.

    Returns
    -------
    keras.Model
        Loaded Keras model.
    """
    model_path = MODELS_DIR / f"{name}.keras"
    return keras.models.load_model(str(model_path))


def load_history(name: str) -> keras.callbacks.History:
    """
    Load a previously saved training history.

    Parameters
    ----------
    name : str
        Base name used when saving the history JSON.

    Returns
    -------
    keras.callbacks.History
        History object whose `.history` attribute contains the metrics dict.
    """
    history_path = RESULTS_DIR / f"history_{name}.json"
    with history_path.open() as f:
        history_dict = json.load(f)

    history = keras.callbacks.History()
    history.history = history_dict
    return history


# ---------------------------------------------------------------------------
# Prediction and reporting helpers
# ---------------------------------------------------------------------------

def predict_classes(
    model: keras.Model,
    x: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Predict class indices for the given images.

    Parameters
    ----------
    model : keras.Model
        Trained Keras model.
    x : np.ndarray
        Input images, shape (N, 32, 32, 3).
    batch_size : int, default 128
        Batch size for prediction.

    Returns
    -------
    np.ndarray
        Predicted class indices, shape (N,).
    """
    probs = model.predict(x, batch_size=batch_size, verbose=0)
    preds = np.argmax(probs, axis=1)
    return preds


def classification_report_str(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """
    Generate a text classification report.

    Parameters
    ----------
    y_true : np.ndarray
        True labels as integers.
    y_pred : np.ndarray
        Predicted labels as integers.
    target_names : list[str] or None, default None
        Optional list of class names. If None, `CLASS_NAMES` is used.

    Returns
    -------
    str
        Multi-line text report.
    """
    if target_names is None:
        target_names = CLASS_NAMES

    return classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
    )


def confusion_matrix_array(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute the confusion matrix between true and predicted labels.

    Parameters
    ----------
    y_true : np.ndarray
        True labels as integers.
    y_pred : np.ndarray
        Predicted labels as integers.
    normalize : bool, default False
        If True, normalize the confusion matrix row-wise.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (NUM_CLASSES, NUM_CLASSES).
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

    if normalize:
        cm = cm.astype("float32")
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
        cm /= row_sums

    return cm