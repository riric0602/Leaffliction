import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
import cv2
import warnings
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)
_original_acos = math.acos

def safe_acos(x):
    x = max(min(x, 1.0), -1.0)  # clamp to valid domain
    return _original_acos(x)


math.acos = safe_acos


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


def plot_transformations(transformed_images):
    img = transformed_images.get("Original_Image")

    images = [
        img,
        transformed_images.get("Blurred"),
        transformed_images.get("Mask"),
        transformed_images.get("Object_Analysis"),
        transformed_images.get("Watershed_Segmentation"),
        img
    ]
    titles = [
        "Original Image",
        "Gaussian Blur Transformation",
        "Mask Transformation",
        "Object Analysis Transformation",
        "Watershed Segmentation Transformation",
        "Acute Pseudo-Landmarks Transformation"
    ]
    cmaps = [None, 'gray', None, None, None, None]

    fig, ax = plt.subplots(3, 2, figsize=(10, 6))
    ax = ax.flatten()

    for i, (image, title, cmap) in enumerate(zip(images, titles, cmaps)):
        ax[i].imshow(image, cmap=cmap)
        ax[i].set_title(title)
        ax[i].axis('off')

    landmarks = transformed_images.get("Pseudo_Landmarks")
    if landmarks is not None:
        for pt in landmarks.reshape(-1, 2):
            ax[5].plot(pt[0], pt[1], 'ro', markersize=3)

    plt.tight_layout()
    plt.show()


def plot_histogram(hist_data):
    plt.figure(figsize=(10, 6))

    for color in ['blue', 'green', 'red']:
        subset = hist_data[hist_data['color channel'] == color]
        plt.plot(subset['pixel intensity'], subset['hist_count'], color=color, label=f'{color} channel')

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.title('Histogram of Pixel Intensities by Color Channel')
    plt.legend()
    plt.grid(True)
    plt.show()


def image_transformation(img_path: str) -> dict:
    # Load image and convert it to gray scale
    img, path, _ = pcv.readimage(img_path)
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

    transformed_images = {
        "Original_Image": img,
        "Blurred": blurred,
        "Mask": mask,
        "Object_Analysis": object_analysis,
        "Watershed_Segmentation": color_labels,
        "Pseudo_Landmarks": homolog_pts,
        "Histogram": hist_data,
    }
    return transformed_images


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


def save_image_with_landmarks(transformed_images, save_path):
    img = transformed_images.get("Original_Image")
    landmarks = transformed_images.get("Pseudo_Landmarks")

    landmarks = np.array(landmarks)  # Ensure it's a NumPy array
    for (x, y) in landmarks.reshape(-1, 2):
        cv2.circle(img, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(save_path, img)


def save_transformed_images(transformed_images, output_dir, img_path):
    for type, element in transformed_images.items():
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_filename = f"{base_name}_{type}.JPG"
        save_path = os.path.join(args.destination, new_filename)

        if type == 'Pseudo_Landmarks' and element is not None:
            save_image_with_landmarks(transformed_images, save_path)
        elif type == 'Histogram':
            pass
        elif isinstance(element, np.ndarray):
            if element.ndim == 2:  # grayscale or label
                path = os.path.join(output_dir, f"{new_filename}")
                cv2.imwrite(path, element.astype(np.uint8))

            elif element.ndim == 3 and element.shape[2] == 3:  # RGB image
                bgr = cv2.cvtColor(element, cv2.COLOR_RGB2BGR)
                path = os.path.join(output_dir, f"{new_filename}")
                cv2.imwrite(path, bgr)

            print(f"Saving {type} transformation for {img_path}")


if __name__ == "__main__":
    args = argparse_flags()
    images_path = get_image_files([args.source])
    output_dir = None

    if not images_path:
        print("Error: No images found in passed parameters.")
        sys.exit(1)

    # Ensure output folder exists, else create it
    if args.destination:
        if not os.path.exists(args.destination):
            os.makedirs(args.destination)
        output_dir = args.destination

    if len(images_path) == 1:
        transformed_images = image_transformation(images_path[0])

        if args.destination:
            save_transformed_images(transformed_images, output_dir, images_path[0])
        else:
            # Plot the 6 Image Transformations
            plot_transformations(transformed_images)
            plot_histogram(hist_data = transformed_images.get("Histogram"))
    else:
        if not output_dir:
            print("Error: You must specify a destination folder.")
            sys.exit(1)
        for img_path in images_path:
            print(f"Processing {img_path}")
            transformed_images = image_transformation(img_path)
            save_transformed_images(transformed_images, output_dir, img_path)
