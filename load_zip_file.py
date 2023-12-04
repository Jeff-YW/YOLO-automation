import zipfile
import os

# assign zip directory and folder for processing...
# zip_file_path = "C:/Users/weiji/Downloads/val2017.zip"
# target_directory = "D:/Med_TA/"

zip_file_path = input("please input the target directory ...")
target_directory = input("please input where to unzip the file")

# extract the base folder
base_folder = os.path.basename(zip_file_path).split(".")[0]

# create the target directory if it does not exists
if not os.path.exists(target_directory):
    print("directory does not exists")
    os.makedirs(target_directory)

print("now extracting")

# extract zip files
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # use the base name as what to be extracted from
    # extract it to the target directory ...
    for file in zip_ref.namelist():
        if file.startswith(f"{base_folder}"):
            # print("yes")
            zip_ref.extract(file, target_directory)

print("extraction complete")