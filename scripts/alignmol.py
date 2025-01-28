# -*- coding: latin-1 -*-
import os

def remove_extension_from_filenames_in_subdirectories(root_directory, extension):
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if extension in filename:
                old_filepath = os.path.join(dirpath, filename)
                
                new_filename = filename.replace(extension, "")
                new_filepath = os.path.join(dirpath, new_filename)
                
                os.rename(old_filepath, new_filepath)
                print(f"Renamed: {filename} -> {new_filename}")

root_directory_path = ""  
file_extension = ".mol2"

remove_extension_from_filenames_in_subdirectories(root_directory_path, file_extension)

