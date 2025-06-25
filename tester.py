import os
import sys
import argparse
import subprocess


TEST_1 = "./test_images/Unit_test1/"
TEST_2 = "./test_images/Unit_test2/"


def execute_test_single_image(test_folder, model_name):
    """Execute the test for all images in the test folder."""
    for image in os.listdir(test_folder):
        print("infor")
        image_path = os.path.join(test_folder, image)
        print(f"Executing test for image: {image_path}")
        subprocess.run(["python", "predict.py", image_path, "-m", model_name])


def main(test_type):
    if test_type == "apple_test":
        print("Running test 1...")
        execute_test_single_image(TEST_1, 'Apple')
    elif test_type == "grape_test":
        print("Running test 2...")
        execute_test_single_image(TEST_2, 'Grape')
    else:
        print("Invalid test_type.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='Tester for the leaf classification')
        parser.add_argument('test_type',
                            choices=["apple_test", "grape_test"],
                            help="Specify the type of test")

        args = parser.parse_args()
        main(args.test_type)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
