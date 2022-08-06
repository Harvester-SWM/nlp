import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd


FILE = "./result_3.tsv"

LABEL_COLUMNS=["욕설","모욕","폭력위협/범죄조장","외설","성혐오","연령","인종/출신지","장애","종교","정치성향","직업혐오"]

chart = pd.read_csv(FILE, sep='\t')

train = chart.iloc[:400000,:]
valid = chart.iloc[400000:,:]

train.to_csv('train_300000.tsv', index=False, header=True, sep="\t")
valid.to_csv('valid_300000.tsv', index=False, header=True, sep="\t")



#print(multilabel_confusion_matrix(y_true, y_pred))