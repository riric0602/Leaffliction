import argparse
import tensorflow as tf
import numpy as np


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
    args = argparse_flags()
    image_path = args.path
    class_names = ['Apple_Black_rot', 'Apple_healthy', 'Apple_rust', 'Apple_scab',
                   'Grape_Black_rot', 'Grape_Esca', 'Grape_healthy', 'Grape_spot']

    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    model = tf.keras.models.load_model("leaffliction_model.keras")

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
