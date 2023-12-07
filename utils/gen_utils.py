import re
import glob

def data_loader(folder_path, nums_of_pic):
    pattern = "*.jpg"

    image_paths = sorted(glob.glob(f"{folder_path}/{pattern}"))


    for image_path in image_paths:
        yield image_path

def read_text_file(txt_path):
    with open(txt_path, 'r') as file:
        for line in file:
            yield line.strip()

def extract_id(file_path):
    # Extract the base name of the file
    base_name = re.search(r'([^/]+)\.jpg$', file_path)
    if base_name:
        # Strip leading zeros from the file base name
        return base_name.group(1).lstrip('0')
    return None
