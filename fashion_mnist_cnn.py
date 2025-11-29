"""
Fashion MNIST – CNN Image Classification

This script trains a convolutional neural network (CNN) on the Fashion MNIST dataset
stored in CSV format (e.g. Kaggle "fashion-mnist_train.csv" / "fashion-mnist_test.csv").

Expected format:
- Train CSV: one column named 'label' (0–9) and 784 pixel columns.
- Test CSV: only 784 pixel columns (no 'label' column).

The script:
- loads and normalises the data
- trains a CNN with Keras
- evaluates on the training set
- generates a submission file with predicted labels for the test set
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, callbacks


# -------------------------------------------------------------------
# Configuration – adjust paths if needed
# -------------------------------------------------------------------
TRAIN_CSV_PATH = "data/fashion-mnist_train.csv"
TEST_CSV_PATH = "data/fashion-mnist_test.csv"
SUBMISSION_PATH = "submission.csv"


# -------------------------------------------------------------------
# Data loading and preprocessing
# -------------------------------------------------------------------
def load_fashion_mnist_from_csv(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "label" in test_df.columns:
        test_df = test_df.drop(columns=["label"])

    x_train = (
        train_df.drop(columns=["label"])
        .values.reshape(-1, 28, 28, 1)
        .astype("float32")
        / 255.0
    )
    y_train = train_df["label"].values.astype("int64")

    x_test = (
        test_df.values.reshape(-1, 28, 28, 1)
        .astype("float32")
        / 255.0
    )

    print("Data loaded:")
    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)

    return x_train, y_train, x_test


# -------------------------------------------------------------------
# Model definition
# -------------------------------------------------------------------
def build_cnn_model(input_shape=(28, 28, 1), num_classes: int = 10):
    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=0.0008)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# -------------------------------------------------------------------
# Training and evaluation
# -------------------------------------------------------------------
def train_model(x_train, y_train):
    model = build_cnn_model(input_shape=x_train.shape[1:], num_classes=10)

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
        callbacks=[early_stop, reduce_lr],
    )

    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=1)
    print(f"Training accuracy: {train_acc:.4f}")

    return model, history


# -------------------------------------------------------------------
# Prediction and submission file
# -------------------------------------------------------------------
def create_submission_file(model, x_test, output_path: str):
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)

    submission = np.column_stack(
        (np.arange(len(predicted_labels)), predicted_labels)
    )

    np.savetxt(
        output_path,
        submission,
        fmt="%d",
        delimiter=",",
        header="id,Category",
        comments="",
    )

    print(f"Saved submission to {output_path}")


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    x_train, y_train, x_test = load_fashion_mnist_from_csv(
        TRAIN_CSV_PATH,
        TEST_CSV_PATH,
    )

    model, history = train_model(x_train, y_train)

    create_submission_file(model, x_test, SUBMISSION_PATH)
