import zipfile
import os

# assign zip directory and folder for processing...
# zip_file_path = "C:/Users/weiji/Downloads/val2017.zip"
# target_directory = "D:/Med_TA/"

zip_file_path = input("please input the target directory ...")
target_directory = input("please input where to unzip the file")



# extract the base folder
base_folder = os.path.basename(zip_file_path).split(".")[0]

# 创建目标目录如果它不存在
if not os.path.exists(target_directory):
    print("directory does not exists")
    os.makedirs(target_directory)

print("now extracting")

# 提取 ZIP 文件
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # 这里假设 zip 文件中有一个名为 "coco" 的文件夹
    # 提取此文件夹及其内容到目标目录
    for file in zip_ref.namelist():
        if file.startswith(f"{base_folder}"):
            print("yes")
            # zip_ref.extract(file, target_directory)

print("提取完成")