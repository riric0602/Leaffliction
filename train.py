import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib


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


if __name__ == "__main__":
    # Load leaves dataset
    data_dir = pathlib.Path("./leaves/images")

    image_count = len(list(data_dir.glob('*/*.JPG')))
    print(f"Image count in dataset: {image_count}")

    # Retrieve training and validation dataset
    train_ds, val_ds = set_training_validation_dataset(data_dir)
    class_names = train_ds.class_names
    print(f"Training Class Names: {class_names}")

    # Visualize data from the training set
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.title('Random 9 images from the training set')
    plt.show()

    # Configure dataset for better performance
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    