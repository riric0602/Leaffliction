import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


batch_size = 32
img_height = 180
img_width = 180
AUTOTUNE = tf.data.AUTOTUNE


def set_training_validation_dataset(data_dir: pathlib.Path):
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


def standardize_and_configure_data(train_ds, val_ds):
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


if __name__ == "__main__":
    # Load leaves dataset
    data_dir = pathlib.Path("./leaves/images")

    image_count = len(list(data_dir.glob('*/*.JPG')))
    print(f"Image count in dataset: {image_count}")

    # Retrieve training and validation dataset
    train_ds, val_ds = set_training_validation_dataset(data_dir)
    class_names = train_ds.class_names
    print(f"Training Class Names: {class_names}")

    # Visualize some data from the training set
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.suptitle("Random 9 Images from the Training Set", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the big title
    plt.show()

    train_ds, val_ds = standardize_and_configure_data(train_ds, val_ds)

    