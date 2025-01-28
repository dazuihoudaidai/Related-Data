# -*- coding: utf-8 -*-

import os
import shutil

def move_folders_without_rdkit(source_path, destination_path):
    sub_directories = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]

    for sub_directory in sub_directories:
        sub_directory_path = os.path.join(source_path, sub_directory)

        rdkit_files = [f for f in os.listdir(sub_directory_path) if f.endswith('.rdkit')]

        if not rdkit_files:
            destination_directory = os.path.join(destination_path, sub_directory)
            shutil.move(sub_directory_path, destination_directory)
            print(f"Moved '{sub_directory}' to '{destination_directory}'")
        else:
            print(f"Skipped '{sub_directory}' due to existing .rdkit files")

source_path = r""
destination_path = r""

move_folders_without_rdkit(source_path, destination_path)
