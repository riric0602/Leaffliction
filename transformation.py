import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
import cv2
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)


def mask_transformation(labeled_mask: np.ndarray, img: np.ndarray):
    return pcv.apply_mask(img=img, mask=labeled_mask, mask_color="white")


def gaussian_blur_transformation(gray: np.ndarray) -> np.ndarray:
    return pcv.gaussian_blur(img=gray, ksize=(5, 5))


def object_analysis_transformation(labeled_mask: np.ndarray, img: np.ndarray):
    return pcv.analyze.size(img=img, labeled_mask=labeled_mask, n_labels=1)


def watershed_segmentation_transformation(labeled_mask: np.ndarray, img: np.ndarray):
    watershed = pcv.watershed_segmentation(rgb_img=img, mask=labeled_mask, distance=15, label="default")
    return pcv.visualize.colorize_label_img(label_img=watershed)


def pseudo_landmarks_transformation(labeled_mask: np.ndarray, img: np.ndarray):
    return pcv.homology.acute(img=img, mask=labeled_mask, win=25, threshold=90)


def color_histogram_transformation(labeled_mask: np.ndarray, img: np.ndarray):
    return pcv.visualize.histogram(img=img, mask=labeled_mask, hist_data=True)


def plot_transformations(img, blurred, mask, analysis, watershed, landmarks):
    images = [img, blurred, mask, analysis, watershed, img]
    titles = [
        "Original Image",
        "Gaussian Blur Transformation",
        "Mask Transformation",
        "Object Analysis Transformation",
        "Watershed Segmentation Transformation",
        "Acute Pseudo-Landmarks Transformation"
    ]
    cmaps = [None, 'gray', None, None, None, None]

    fig, ax = plt.subplots(3, 2, figsize=(12, 6))
    ax = ax.flatten()

    for i, (image, title, cmap) in enumerate(zip(images, titles, cmaps)):
        ax[i].imshow(image, cmap=cmap)
        ax[i].set_title(title)
        ax[i].axis('off')

    homolog_pts = landmarks.reshape(-1, 2)
    for pt in homolog_pts:
        ax[5].plot(pt[0], pt[1], 'ro', markersize=3)

    plt.tight_layout()
    plt.show()


def plot_histogram(hist_data):
    plt.figure(figsize=(10, 6))

    # Plot per color channel
    for color in ['blue', 'green', 'red']:
        subset = hist_data[hist_data['color channel'] == color]
        plt.plot(subset['pixel intensity'], subset['hist_count'], color=color, label=f'{color} channel')

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.title('Histogram of Pixel Intensities by Color Channel')
    plt.legend()
    plt.grid(True)
    plt.show()


def image_transformation(img_path: str) -> None:
    # Load image and convert it to gray scale
    img, path, filename = pcv.readimage(img_path)
    gray_img = pcv.rgb2gray_lab(rgb_img=img, channel='l')
    binary_img = pcv.threshold.binary(gray_img, threshold=120, object_type='dark')

    # Apply Gaussian blur transformation
    blurred = gaussian_blur_transformation(gray_img)

    # Apply Mask transformation
    labeled_mask = pcv.fill(bin_img=binary_img, size=50)
    mask = mask_transformation(labeled_mask, img)

    # Apply Object Analysis transformation
    object_analysis = object_analysis_transformation(labeled_mask, img)

    # Apply Watershed Segmentation transformation
    color_labels = watershed_segmentation_transformation(labeled_mask, img)

    # Apply Pseudo-Landmarks transformation
    homolog_pts, *_ = pseudo_landmarks_transformation(labeled_mask, img)

    # Apply Histogram of Color Repartition Transformation
    _, hist_data = color_histogram_transformation(labeled_mask, img)

    # Plot the 6 Image Transformations
    plot_transformations(img, blurred, mask, object_analysis, color_labels, homolog_pts)
    plot_histogram(hist_data)

    return img


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
    Parse the command-line arguments.
    :return: parsed commandline arguments
    """
    parser = argparse.ArgumentParser(description="Display 6 types of Image Transformation")
    parser.add_argument('-src', '--source', type=str, required=True, help="Input file or directory")
    parser.add_argument('-dst', '--destination', type=str, help="Output folder (optional if only one image)")

    return parser.parse_args()


if __name__ == "__main__":
    args = argparse_flags()
    images_path = get_image_files([args.source])

    if not images_path:
        print("Error: No images found in passed parameters.")
        sys.exit(1)

    if len(images_path) == 1 and not args.destination:
        # Set default destination folder if not provided
        args.destination = os.path.dirname(images_path[0])  # Use the source directory as destination

    # Ensure output folder exists, else create it
    if args.destination and not os.path.exists(args.destination):
        os.makedirs(args.destination)

    if len(images_path) == 1:
        # Set PlantCV debug output directory
        pcv.params.debug = None
        if args.destination:
            pcv.params.debug_outdir = args.destination

        # Process and plot the single image
        transformed_image = image_transformation(images_path[0])
        # Save the processed image
        output_path = os.path.join(args.destination, f"transformed_{os.path.basename(images_path[0])}")
        cv2.imwrite(output_path, transformed_image)  # Save the processed image
    else:
        # Set PlantCV debug output directory
        pcv.params.debug = "print"
        pcv.params.debug_outdir = args.destination

        for img_path in images_path:
            transformed_image = image_transformation(img_path)
            filename = os.path.basename(img_path)

            # Save the main transformed image
            output_path = os.path.join(args.destination, f"transformed_{filename}")
            cv2.imwrite(output_path, transformed_image)  # Save the processed image