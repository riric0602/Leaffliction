import argparse
import os
import sys
import numpy as np
from plantcv import plantcv as pcv



pcv.params.debug = "plot"


def mask_transformation(binary_img: np.ndarray, img: np.ndarray):
    # Clean the mask (e.g. fill small holes)
    mask = pcv.fill(bin_img=binary_img, size=5)

    # Apply the mask to the original RGB image
    masked_image = pcv.apply_mask(img=img, mask=mask, mask_color='black')

    return masked_image

def gaussian_blur_transformation(gray: np.ndarray) -> np.ndarray:
    return pcv.gaussian_blur(img=gray, ksize=(21, 21), sigma_x=5, sigma_y=5)


def image_transformation(img_path: str) -> None:
    # Load image and convert it to gray scale
    img, path, filename = pcv.readimage(img_path)
    gray_img = pcv.rgb2gray_lab(rgb_img=img, channel='l')
    binary_img = pcv.threshold.binary(gray_img, threshold=120, object_type='light')

    # Apply Gaussian blur to reduce noise
    blurred = gaussian_blur_transformation(gray_img)

    # Apply Mask transformation
    mask = mask_transformation(binary_img, img)


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


def argparse_flags() -> argparse.Namespace:
    """
    Parse command line arguments
    :return: args passed in command line
    """
    parser = argparse.ArgumentParser(
        description="Display 6 types of data transformation for images"
    )

    parser.add_argument(
        "paths",
        nargs='+',
        type=str,
        help="One or more images or folders containing images"
    )

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    args = argparse_flags()
    images_path = get_image_files(args.paths)

    if not images_path:
        print("Error: No images found in passed parameters.")
        sys.exit(1)

    if len(images_path) == 1:
        # Display augmentations if only one image is processed
        transformed_images = image_transformation(images_path[0])
    else:
        for img_path in images_path:
            image_transformation(img_path)