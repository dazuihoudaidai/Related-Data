# -*- coding: latin-1 -*-
import os
import pandas as pd

path_to_mol2_files = r"D:\TCMSP"


mol2_files = [f for f in os.listdir(path_to_mol2_files) if f.endswith('.mol2')]

df = pd.DataFrame(columns=['pdbid', '-logKd/Ki'])


df['pdbid'] = mol2_files
df['-logKd/Ki'] = 

df.to_csv('output.csv', index=False)
