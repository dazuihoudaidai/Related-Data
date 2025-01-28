# -*- coding: latin-1 -*-

import os
import pandas as pd
from tqdm import tqdm

def rename_files_based_on_excel(excel_path):
   
    df = pd.read_excel(excel_path)

   
    names = df['name'].tolist()
    tcmbank_ids = df['TCMBank_ID'].tolist()
    target_directory = r''

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    files_to_rename = [filename for filename in os.listdir(target_directory) if os.path.isfile(os.path.join(target_directory, filename))]

    for filename in tqdm(files_to_rename, desc=""):
        if filename.startswith("TCMBANKIN"):
            print(f" {filename}  'TCMBANKIN' ")
            continue

        file_path = os.path.join(target_directory, filename)

        file_name, file_extension = os.path.splitext(filename)

        matching_names = [str(name) for name in names if str(name) in str(file_name)]

        if matching_names:
            tcmbank_id = tcmbank_ids[names.index(matching_names[0])]

            new_filename = f"{tcmbank_id}{file_extension}"

            new_file_path = os.path.join(target_directory, new_filename)

            if not os.path.exists(new_file_path):
                os.rename(file_path, new_file_path)

                print(f"{filename}{new_filename}")
            else:
                print(f"{new_filename} ")
        else:
            print(f" {filename} ")

excel_path = r''
rename_files_based_on_excel(excel_path)
