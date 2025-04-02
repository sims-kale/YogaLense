# rename all the images in the dataset to a common format
# subfolder_name+index.jpg

import os


dataset_path = r"D:\SHU\Applied ai\Assesment\dataset"
image_files = []
for dirpath, subdir, filenames in os.walk(dataset_path):


    print(f"Found directory: {dirpath}")
    for i, filename in enumerate (filenames):
        basename = os.path.basename(filename)
        subfolder_name = os.path.basename(dirpath)
        # print(f"Found file: {basename}")
        # print(f"Subfolder name: {subfolder_name}")
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
        ):
            file_path = os.path.join(dirpath, filename)
            image_files.append(file_path)
            #rename the file to a common format
            new_file_name = os.path.join(dirpath, f"{subfolder_name}_{i}.jpg")
            os.rename(file_path, new_file_name)
            print(f"Renamed: {new_file_name}")
