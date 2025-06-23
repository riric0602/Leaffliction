import os
import sys
import wget
import shutil
import zipfile
import argparse
import subprocess

# Constants
URL = "https://cdn.intra.42.fr/document/document/17447/leaves.zip"
UNIT_TEST_URL = \
    "https://cdn.intra.42.fr/document/document/17546/test_images.zip"
ZIP_FILE = "leaves.zip"
EXTRACT_DIR = "extracted_data"
EXTRACTED_UNIT_TEST_DIR = "extracted_unit_test_data"
TEST_1 = "./test_images/Unit_test1/"
TEST_2 = "./test_images/Unit_test2/"
DATASETS = "datasets"
OUTPUT = "output"
SPLITTED = "splitted"
AUGMENTED = "augmented_directory"


def clean_up():
    """Remove existing 'dataset' directory if they exist."""
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
        print(f"Removed existing directory: {EXTRACT_DIR}")

    if os.path.exists(DATASETS):
        shutil.rmtree(DATASETS)
        print(f"Removed existing directory: {DATASETS}")

    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
        print(f"Removed existing file: {ZIP_FILE}")


def full_clean_up():
    """Remove all existing directories if they exist."""

    clean_up()

    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
        print(f"Removed existing directory: {OUTPUT}")
    if os.path.exists(SPLITTED):
        shutil.rmtree(SPLITTED)
        print(f"Removed existing directory: {SPLITTED}")
    if os.path.exists(EXTRACTED_UNIT_TEST_DIR):
        shutil.rmtree(EXTRACTED_UNIT_TEST_DIR)
        print(f"Removed existing directory: {EXTRACTED_UNIT_TEST_DIR}")
    if os.path.exists(AUGMENTED):
        shutil.rmtree(AUGMENTED)
        print(f"Removed existing directory: {AUGMENTED}")


def download_and_extract_zip(url, zip_file, extract_dir):
    """Download and extract the ZIP file."""
    if not os.path.exists(zip_file):
        print(f"Downloading {url}...")
        wget.download(url, zip_file)
        print("\nDownload complete.")

    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")


def no_test():
    """Just download and unzip the file without running any tests."""
    print("Running no test...")
    download_and_extract_zip(URL, ZIP_FILE, EXTRACT_DIR)
    print("Download and extraction completed.")


def move_and_split_dataset(extract_dir):
    subprocess.run(["sh", "split_plants.sh", extract_dir])


def rename_folders(extracted_unit_test_dir):
    """Rename the folders in the extracted unit test directory."""
    # Rename Unit_test1 to Apples and Unit_test2 to Grapes
    # They are in the format:
    # extracted_unit_test_data/test_images/Unit_test{1,2}
    for folder in os.listdir(extracted_unit_test_dir):
        print(folder)
        folder_path = os.path.join(extracted_unit_test_dir, folder)
        for subfolder in os.listdir(folder_path):
            print(subfolder)
            if subfolder == "Unit_test1":
                os.rename(os.path.join(folder_path, subfolder),
                          os.path.join(folder_path, "Apples"))
            elif subfolder == "Unit_test2":
                os.rename(os.path.join(folder_path, subfolder),
                          os.path.join(folder_path, "Grapes"))


def execute_test_single_image(test_folder):
    """Execute the test for all images in the test folder."""
    for image in os.listdir(test_folder):
        print("infor")
        image_path = os.path.join(test_folder, image)
        print(f"Executing test for image: {image_path}")
        subprocess.run(["python3", "predict.py", image_path])


def test_predict(test_type):
    training_data = f"splitted/datasets/{test_type}/training/{test_type}"
    validation_data = f"splitted/datasets/{test_type}/validation/{test_type}"
    cmd = f"python predict.py "
    subprocess.run(cmd, shell=True)


def main(test_type):
    # clean_up()
    #
    # if test_type == "notest":
    #     no_test()
    # elif test_type == "clean":
    #     full_clean_up()
    # elif test_type == "dns":
    #     print("Running DNS test...")
    #     download_and_extract_zip(URL, ZIP_FILE, EXTRACT_DIR)
    #     move_and_split_dataset(EXTRACT_DIR)
    #     print("Download and extraction completed.")
    # elif test_type == "unittest":
    #     download_and_extract_zip(UNIT_TEST_URL, ZIP_FILE,
    #                              EXTRACTED_UNIT_TEST_DIR)
    #     rename_folders(EXTRACTED_UNIT_TEST_DIR)
    # elif test_type == "test_apples":
    #     test_predict("Apples")
    # elif test_type == "test_grapes":
    #     test_predict("Grapes")
    if test_type == "test_1":
        print("Running test 1...")
        execute_test_single_image(TEST_1)
    elif test_type == "test_2":
        print("Running test 2...")
        execute_test_single_image(TEST_2)
    else:
        print("Invalid test_type. Use 'notest'.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='Tester for the Distribution.py script')
        parser.add_argument('test_type',
                            choices=['notest',  'clean', 'dns',
                                     "test_apples", "test_grapes",
                                     'unittest', "test_1", "test_2"],
                            help="Specify the type of test: 'dns'")

        args = parser.parse_args()
        main(args.test_type)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)