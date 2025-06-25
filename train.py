import argparse
import os
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import sys
import json
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import History
from augmentation import balance_and_augment_dataset
from utils import get_image_files, close_on_key
import tempfile
import zipfile

batch_size = 32
img_height = 180
img_width = 180
epochs = 10
AUTOTUNE = tf.data.AUTOTUNE


def set_training_validation_dataset(data_dir: pathlib.Path) -> tuple[Dataset, Dataset]:
    """
     Use 80% of the images for training and 20% for validation.
    :param data_dir: main leaves dataset directory
    :return: tuple containing training and validation dataset
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return train_ds, val_ds


def standardize_and_configure_data(train_ds: Dataset, val_ds: Dataset) -> tuple[Dataset, Dataset]:
    """
    Standardize and configure training and validation dataset
    :param train_ds: the training dataset
    :param val_ds: the validation dataset
    :return: the updated training and validation dataset
    """
    # Standardize data to be in [0, 1] range instead of [0, 255]
    normalization_layer = layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    # Configure dataset for better performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(f"Range of data: [{np.min(first_image)}, {np.max(first_image)}]")

    return train_ds, val_ds


def model_creation(num_classes: int) -> Sequential:
    """
    Create a Sequential model with different CNN layers
    :param num_classes: Number of categories for the classification
    :return: the created model
    """
    model = Sequential([
        keras.Input(shape=(img_height, img_width, 3)),
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True
                  ),
                  metrics=['accuracy'])

    model.summary()

    return model


def model_training(model: Sequential, train_ds: Dataset, val_ds: Dataset) -> tuple[Sequential, History]:
    """
    Training model based on training and validation sets
    :param model: model to be trained
    :param train_ds: the training dataset
    :param val_ds: the validation dataset
    :return: the model and history of training
    """
    print("Training the model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    return model, history


def model_accuracy(history: History) -> None:
    """
    Display the accuracy of the training and validation sets
    :param history: training history of model
    :return:
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    fig = plt.figure(figsize=(12, 12))
    fig.canvas.mpl_connect('key_press_event', close_on_key)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # For training and Validation accuracy
    final_train_acc = history.history['accuracy'][-1] * 100
    final_val_acc = history.history['val_accuracy'][-1] * 100

    print(f"Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")


def create_zip_with_learnings(data_dir, model, class_names):
    """
    Create .zip file to store learnings and the modified dataset
    :param data_dir: main dataset directory
    :param model: the trained model
    :param class_names: list of class names
    :return:
    """
    project_root = pathlib.Path().resolve()
    model_name = pathlib.Path(data_dir).name
    zip_path = project_root / f"learnings_{model_name}.zip"

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        temp_model_path = tmpdir / f"{model_name}_model.keras"
        model.save(temp_model_path)

        class_path = tmpdir / "class_names.json"
        with open(class_path, "w") as f:
            json.dump(class_names, f)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add the entire augmented data directory
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        file_path = pathlib.Path(root) / file
                        arcname = file_path.relative_to(data_dir.parent)
                        zipf.write(file_path, arcname)

                zipf.write(temp_model_path, temp_model_path.name)
                zipf.write(class_path, class_path.name)


def argparse_flags() -> argparse.Namespace:
    """
        Parse command line arguments
        :return: args passed in command line
        """
    parser = argparse.ArgumentParser(
        description="Train model for leaf disease classification."
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to the dataset and its subdirectories to be trained on",
    )

    parser.add_argument(
        "-a",
        "--accuracy",
        action="store_true",
        help="Information about the accuracy of the model",
    )

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    try:
        args = argparse_flags()
        data_path = args.path

        if not os.path.exists(data_path):
            print(f"Error: Directory {data_path} doesn't exist")
            sys.exit(1)

        image_paths = get_image_files([data_path])
        balance_and_augment_dataset(image_paths)
        data_dir = pathlib.Path(data_path)

        # Create and save training and validation datasets
        train_ds, val_ds = set_training_validation_dataset(data_dir)

        # Saving the training class names
        class_names = train_ds.class_names
        print(f"Training Class Names: {class_names}")

        train_ds, val_ds = standardize_and_configure_data(train_ds, val_ds)

        # Create the model
        print("Creating the model...")
        num_classes = len(class_names)
        model = model_creation(num_classes)

        # Training the model
        model, history = model_training(model, train_ds, val_ds)
        create_zip_with_learnings(data_dir, model, class_names)

        # Visualize training results
        if args.accuracy:
            model_accuracy(history)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
