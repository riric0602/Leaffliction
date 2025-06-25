import argparse
import os
import numpy as np
import sys
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import cv2
from utils import close_on_key
import pathlib
import zipfile

batch_size = 32
img_height = 180
img_width = 180


def get_transformed_image(original_image: np.ndarray) -> np.ndarray:
    """
    Generate the transformed image from the original image
    :param original_image: leaf image to be predicted
    :return: transformed image
    """
    # Convert Original image to OpenCV format
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

    # Generate the Transformed Image
    hsv = cv2.cvtColor(original_cv, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    highlighted = cv2.bitwise_and(original_cv, original_cv, mask=mask)
    transformed_image = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)

    return transformed_image


def extract_zip(model_name: str) -> pathlib.Path:
    """
    Extract the zip file containing learnings
    :return: the extracted folder
    """
    zip_path = pathlib.Path(f"learnings_{model_name}.zip")
    extract_dir = pathlib.Path(f"learnings_{model_name}")

    if extract_dir.exists():
        return extract_dir

    print("Extracting learnings...")
    extract_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir


def argparse_flags() -> argparse.Namespace:
    """
    Parse command line arguments
    :return: args passed in command line
    """
    parser = argparse.ArgumentParser(
        description="Predict leaf disease"
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Model (Apple/Grape) to use for prediction",
    )

    parser.add_argument(
        "path",
        type=str,
        help="Image of leaf to be predicted",
    )

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    try:
        args = argparse_flags()
        image_path = args.path
        image_name = os.path.basename(image_path)

        if not os.path.exists(image_path):
            print("Error: Image path does not exist.")
            sys.exit(1)

        if args.model != "Apple" and args.model != "Grape":
            print("Error: Model must be either Apple or Grape.")
            sys.exit(1)

        model_name = args.model
        extract_dir = extract_zip(model_name)
        model_path = extract_dir / f"{model_name}_model.keras"
        classes_path = extract_dir / "class_names.json"

        if not os.path.exists(model_path) or not os.path.exists(classes_path):
            print("Error: Model not found. Train the model before prediction.")
            sys.exit(1)

        model = tf.keras.models.load_model(model_path)
        with open(classes_path, "r") as f:
            class_names = json.load(f)

        # Load and preprocess image for model
        img = tf.keras.utils.load_img(
            image_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)

        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        # Plot both images and prediction
        original_img = mpimg.imread(image_path)
        transformed_image = get_transformed_image(img)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.canvas.mpl_connect('key_press_event', close_on_key)

        axes[0].imshow(original_img)
        axes[0].set_title(f"Original Image {image_name}")
        axes[0].axis("off")

        axes[1].imshow(transformed_image)
        axes[1].set_title("Transformed Image")
        axes[1].axis("off")

        plt.suptitle(
            f"Predicted: {predicted_class} ({confidence:.2f}%)",
            fontsize=14
        )
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
