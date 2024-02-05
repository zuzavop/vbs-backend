# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import re

import argparse


def get_total_files_in_folder(folder):
    return len(
        [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    )


def rename_files(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            # Construct the full path of the file
            file_path = os.path.join(foldername, filename)

            # Extract the base name and extension
            base_name, ext = os.path.splitext(filename)

            # Extract the number after the last underscore, if any
            match = re.search(r'(_|-)([0-9]+)$', base_name)
            if match:
                # Extract the number
                splitter = match.group(1)
                number = str(int(match.group(2)))

                # Determine the number of zeros needed based on the total files in the folder
                total_files = get_total_files_in_folder(foldername)
                num_zeros = len(str(total_files))

                # Pad the number with zeros
                padded_number = number.zfill(num_zeros)

                # Construct the new file name
                new_name = base_name.rsplit(splitter, 1)[0] + '_' + padded_number + ext

                # Construct the new full path
                new_path = os.path.join(foldername, new_name)

                # Rename the file
                os.rename(file_path, new_path)
                print(f'Renamed: {file_path} -> {new_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Process video keyframes to rename it to match features.'
    )
    parser.add_argument('-d', '--dir_kf', type=str, help='Path to the keyframe files.')

    args = parser.parse_args()
    dir_kf = args.dir_kf

    # Call the function to rename files
    rename_files(dir_kf)


if __name__ == "__main__":
    main()
