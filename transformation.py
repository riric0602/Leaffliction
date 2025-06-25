import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
import cv2
import warnings
import math
from utils import close_on_key, get_image_files


warnings.filterwarnings("ignore", category=RuntimeWarning)
_original_acos = math.acos


def safe_acos(x):
    x = max(min(x, 1.0), -1.0)
    return _original_acos(x)


math.acos = safe_acos


def mask_transformation(labeled_mask: np.ndarray, img: np.ndarray):
    """
    Apply a mask transformation to an image
    :param labeled_mask: generated mask from image
    :param img: original image
    :return: Mask transformed image
    """
    return pcv.apply_mask(img=img, mask=labeled_mask, mask_color="white")


def gaussian_blur_transformation(gray: np.ndarray) -> np.ndarray:
    """
    Apply a gaussian blur to an image
    :param gray: Gray generated image from original image
    :return: Gaussian blur transformed image
    """
    return pcv.gaussian_blur(img=gray, ksize=(5, 5))


def object_analysis_transformation(labeled_mask: np.ndarray, img: np.ndarray):
    """
    Apply an object analysis to an image
    :param labeled_mask: generated mask from image
    :param img: original image
    :return: object analysis transformed image
    """
    return pcv.analyze.size(img=img, labeled_mask=labeled_mask, n_labels=1)


def watershed_segmentation_transformation(mask: np.ndarray, img: np.ndarray):
    """
    Apply a watershed segmentation transformation to an image
    :param mask: generated mask from image
    :param img: original image
    :return: Watershed segmentation transformed image
    """
    watershed = pcv.watershed_segmentation(
        rgb_img=img,
        mask=mask,
        distance=15,
        label="default"
    )
    return pcv.visualize.colorize_label_img(label_img=watershed)


def pseudo_landmarks_transformation(labeled_mask: np.ndarray, img: np.ndarray):
    """
    Apply a pseudo landmarks transformation to an image
    :param labeled_mask: generated mask from image
    :param img: original image
    :return: Landmarks transformed image
    """
    return pcv.homology.acute(img=img, mask=labeled_mask, win=25, threshold=90)


def color_histogram_transformation(labeled_mask: np.ndarray, img: np.ndarray):
    """
    Apply a color histogram transformation to an image
    :param labeled_mask: generated mask from image
    :param img: original image
    :return: Color Histogram on original image
    """
    return pcv.visualize.histogram(img=img, mask=labeled_mask, hist_data=True)


def plot_transformations(transformed_images: dict):
    """
    Plot the 5 image transformations
    :param transformed_images: dictionary of transformed images
    :return:
    """
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
    fig.canvas.mpl_connect('key_press_event', close_on_key)
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


def plot_histogram(hist_data, save_path):
    """
    Plot the color histogram of an image
    :param hist_data: histogram plantCV data
    :param save_path: path where to save the histogram
    :return:
    """
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.mpl_connect('key_press_event', close_on_key)

    for color in ['blue', 'green', 'red']:
        subset = hist_data[hist_data['color channel'] == color]
        plt.plot(
            subset['pixel intensity'],
            subset['hist_count'],
            color=color,
            label=f'{color} channel'
        )

    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    plt.title('Histogram of Pixel Intensities by Color Channel')
    plt.legend()
    plt.grid(True)

    if not save_path:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        plt.close()


def image_transformation(img_path: str) -> dict:
    """
    Apply 6 types of image transformations
    :param img_path: path of original image
    :return: dictionary of transformed images
    """
    # Load image and convert it to gray scale
    img, path, _ = pcv.readimage(img_path)
    gray_img = pcv.rgb2gray_lab(rgb_img=img, channel='l')
    binary_img = pcv.threshold.binary(
        gray_img,
        threshold=120,
        object_type='dark'
    )

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
    hist_figure, hist_data = color_histogram_transformation(labeled_mask, img)

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


def argparse_flags() -> argparse.Namespace:
    """
    Parse the command-line arguments.
    :return: parsed commandline arguments
    """
    parser = argparse.ArgumentParser(
        description="Display 6 types of Image Transformation"
    )
    parser.add_argument(
        '-src',
        '--source',
        type=str,
        required=True,
        help="Input file Image or directory"
    )
    parser.add_argument(
        '-dst',
        '--destination',
        type=str,
        help="Output folder (optional if only one image)"
    )

    return parser.parse_args()


def save_image_with_landmarks(transformed_images, save_path):
    """
    Save original image with pseudo-landmarks
    :param transformed_images: dictionary of transformed images
    :param save_path: path where to save the image
    :return:
    """
    img = transformed_images.get("Original_Image")
    landmarks = transformed_images.get("Pseudo_Landmarks")

    landmarks = np.array(landmarks)  # Ensure it's a NumPy array
    for (x, y) in landmarks.reshape(-1, 2):
        cv2.circle(
            img,
            (int(x), int(y)),
            radius=3,
            color=(0, 255, 0),
            thickness=-1
        )

    cv2.imwrite(save_path, img)


def save_images(transformed_images, output_dir, img_path):
    """
    Save the transformed images in output_dir
    :param transformed_images: dictionary with transformed images
    :param output_dir:
    :param img_path:
    :return:
    """
    for type, element in transformed_images.items():
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_filename = f"{base_name}_{type}.JPG"
        save_path = os.path.join(args.destination, new_filename)

        if type == 'Pseudo_Landmarks' and element is not None:
            save_image_with_landmarks(transformed_images, save_path)
        elif type == 'Histogram':
            plot_histogram(element, save_path)
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
    try:
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
                save_images(
                    transformed_images,
                    output_dir,
                    images_path[0]
                )
            else:
                # Plot the 6 Image Transformations
                plot_transformations(transformed_images)
                plot_histogram(
                    hist_data=transformed_images.get("Histogram"),
                    save_path=None
                )
        else:
            if not output_dir:
                print("Error: You must specify a destination folder.")
                sys.exit(1)
            for img_path in images_path:
                transformed_images = image_transformation(img_path)
                save_images(transformed_images, output_dir, img_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
