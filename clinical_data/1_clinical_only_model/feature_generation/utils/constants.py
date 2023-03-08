import os

DATA_HOME = os.path.join("/PATH_TO/data") #clinical data main dir
MRN_ACC_PATH = os.path.join(DATA_HOME, "IHD_8139_mrn_acc_ctdate.csv") #cohort data file listing mrn, accession, ct_date
SPLIT_PATH = os.path.join(DATA_HOME, 'splits') #dir containing folders 1-11, each containing relevant split files
SAVE_PATH = os.path.join(DATA_HOME, 'test_python_ftextract') #dir to write cached results/ft matrices

