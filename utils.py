from matplotlib import pyplot as plt
from matplotlib.backend_bases import KeyEvent
import os


def close_on_key(event: KeyEvent) -> None:
    """
    Close the window when the ESC key is pressed
    :param event: keyboard event
    :return:
    """
    if event.key == 'escape':
        plt.close(event.canvas.figure)


def get_image_files(paths: list) -> list:
    """
    Extract Image files from CLI arguments
    :param paths: paths passed in parameters
    :return: paths of the images in the parameters
    """
    image_paths = []

    for path in paths:
        if os.path.isfile(path):
            if path.lower().endswith((".jpg", ".jpeg", ".png")):
                # Check if element is a file and an image
                image_paths.append(path)
        elif os.path.isdir(path):
            # Check if element is a directory containing images
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_paths.append(os.path.join(root, file))
    return image_paths
