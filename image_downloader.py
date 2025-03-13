import os
import requests
import argparse
import logging
import glob
from urllib.parse import urlparse


def setup_logger(log_path):
    """Set up logging configuration"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def process_file(txt_file):
    """Process a single text file and download its images"""
    # Create output directory
    base_name = os.path.splitext(os.path.basename(txt_file))[0]
    output_dir = os.path.join(os.path.dirname(txt_file), f"{base_name}_images")
    os.makedirs(output_dir, exist_ok=True)

    # Set up logger
    log_file = os.path.join(output_dir, "bridge_pose_download.log")
    logger = setup_logger(log_file)

    logger.info(f"Starting processing for file: {txt_file}")

    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}")
        return

    # Added missing processing logic
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        try:
            parts = line.split('\t')
            if len(parts) < 2:
                raise ValueError("Invalid line format")

            filename_part, url = parts[0], parts[1].strip()
            filename = os.path.basename(filename_part)
            save_path = os.path.join(output_dir, filename)

            if os.path.exists(save_path):
                logger.warning(f"Skipping existing file: {filename}")
                continue

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type:
                raise ValueError(f"URL does not point to an image (Content-Type: {content_type})")

            with open(save_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Successfully downloaded: {filename}")

        except Exception as e:
            logger.error(f"Line {line_number}: Failed to process '{line}' - {str(e)}")


def process_directory(input_dir):
    """Process all .txt files in a directory"""
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    for txt_file in txt_files:
        print(f"Processing {txt_file}...")
        try:
            process_file(txt_file)
        except Exception as e:
            print(f"Failed to process {txt_file}: {str(e)}")


if __name__ == "__main__":
    # Fixed argparse configuration
    parser = argparse.ArgumentParser(description='Download images from all text files in a directory')
    parser.add_argument('--input_dir',
                        help='Path to directory containing .txt files',
                        default=r'D:\SHU\Applied ai\Assesment\archive\yoga_dataset_links',
                        type=str)

    args = parser.parse_args()

    # Normalize path for different OS
    input_dir = os.path.normpath(args.input_dir)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory {input_dir} not found!")
        exit(1)

    process_directory(input_dir)
    print("All files processed. Check individual folders for logs.")