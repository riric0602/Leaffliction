import argparse
import os
import numpy as np
import sys
import json
import tensorflow as tf


batch_size = 32
img_height = 180
img_width = 180


def argparse_flags() -> argparse.Namespace:
    """
    Parse command line arguments
    :return: args passed in command line
    """
    parser = argparse.ArgumentParser(
        description="Predict leaf disease"
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

        if not os.path.exists(image_path):
            print("Error: Image path does not exist.")
            sys.exit(1)

        if not os.path.exists("leaffliction_model.keras") or not os.path.exists("class_names.json"):
            print(
                "Error: Model/Classes not found. Train the model before prediction."
            )
            sys.exit(1)

        model = tf.keras.models.load_model("leaffliction_model.keras")
        with open("class_names.json", "r") as f:
            class_names = json.load(f)

        img = tf.keras.utils.load_img(
            image_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "This image {} most likely belongs to {} with a {:.2f} percent confidence."
            .format(image_path, class_names[np.argmax(score)], 100 * np.max(score))
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
