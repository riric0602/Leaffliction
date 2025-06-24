import argparse
import os
import sys
from collections import defaultdict
from matplotlib.backend_bases import KeyEvent
import matplotlib.pyplot as plt


def plot_distribution(plants: dict) -> None:
    """
    Plot the plants distribution according to their categories
    :param plants: dictionary of the plants
    :return:
    """
    count = len(plants)
    plants_list = list(plants.items())

    fig, axes = plt.subplots(nrows=count, ncols=2, figsize=(12, 5 * count))
    fig.suptitle("Plant Disease Distribution", fontsize=16)

    for idx in range(len(plants_list)):
        plant, diseases = plants_list[idx]
        print(f"{plant} {{state, count}} : {diseases}")

        labels = list(diseases.keys())
        counts = list(diseases.values())

        # Pie chart
        pie_ax = axes[idx][0] if count > 1 else axes[0]
        pie_ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        pie_ax.set_title(f"{plant} (Pie Chart)")
        pie_ax.axis('equal')

        # Bar chart
        bar_ax = axes[idx][1] if count > 1 else axes[1]
        bar_ax.bar(labels, counts)
        bar_ax.set_title(f"{plant} (Bar Chart)")
        bar_ax.set_ylabel("Elements Count")
        bar_ax.tick_params(axis='x', rotation=15)
        bar_ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.mpl_connect('key_press_event', close_on_key)
    plt.show()


def close_on_key(event: KeyEvent) -> None:
    if event.key == 'escape':
        plt.close(event.canvas.figure)


def plant_categories(dataset_path: str) -> defaultdict:
    """
    Organize the plants into their types and categories
    :param dataset_path: path of the dataset
    :return: dictionary with plants and categories
    """
    categories = {}

    for subdir in os.listdir(dataset_path):
        sub_path = os.path.join(dataset_path, subdir)
        if os.path.isdir(sub_path):
            image_count = len([
                f for f in os.listdir(sub_path)
            ])
            categories[subdir] = image_count

    plants = defaultdict(dict)
    for label, count in categories.items():
        plant, disease = label.split('_', 1)
        plants[plant][disease] = count

    return plants


def analyze_dataset(dataset_path: str) -> None:
    """
    Analyze dataset and extract elements to plot their distribution.
    :param dataset_path: path of the dataset
    :return:
    """
    # Extract the folders in the dataset path
    entries = os.listdir(dataset_path)

    if len(entries) == 1:
        # Subdirectories are located in inner folder
        dataset_path = os.path.join(dataset_path, entries[0])
        # Check inner folder is a directory
        if not os.path.isdir(dataset_path):
            print(f"Error: {dataset_path} is not a valid directory.")
            sys.exit(1)

    if len(os.listdir(dataset_path)) == 0:
        print(f"Error: {dataset_path} is empty.")
        sys.exit(1)

    plants = plant_categories(dataset_path)
    plot_distribution(plants)


def argparse_flags() -> argparse.Namespace:
    """
    Parse command line arguments
    :return: args passed in command line
    """
    parser = argparse.ArgumentParser(
        description="Extract, Analyze and Plot the dataset passed as argument"
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to the dataset and its subdirectories to be analyzed",
    )

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    try:
        args = argparse_flags()
        dataset_path = args.path

        if os.path.isdir(dataset_path):
            analyze_dataset(dataset_path)
        else:
            print(f"Error: {dataset_path} is not a valid directory.")
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
