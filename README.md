Data：The related data contain the molecular docking and MD simulation results of the selected compounds MOL011832，MOL011833 and reference inhibitor nikkomycin Z
Environment：
Create and activate the  environment.
conda env create -f Environment.yml
conda activate HIGAN
Dataset：
PDBbind database for taining and validation (http://www.pdbbind.org.cn/download.php)
Usage：
1. Dataset downloading
Firstly，download the PDBbind Version 2016 from http://www.pdbbind.org.cn/download.php, and the mol2 files of natural products in TCMSP database.
And orgnize them as trainingset and validationset
3. Environment building
conda env create -f Environment.yml
conda activate HIGAN
4. Model training
Use the train.py script to train the model.

