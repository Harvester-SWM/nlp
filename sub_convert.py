import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import pandas as pd


FILE = "./test.txt"

LABEL_COLUMNS=["욕설","모욕","폭력위협/범죄조장","외설","성혐오","연령","인종/출신지","장애","종교","정치성향","직업혐오"]

chart = pd.read_csv(FILE, sep='\t')


write_list = []

for index, row in chart.iterrows():
    #print(row)
    result = 0
    
    for i in LABEL_COLUMNS:
        if row[i] > 0:
            result = 1
            break
    #print(row['내용'], result)
    temp = [row['내용'], result]
    write_list.append(temp)

print(write_list)    

write_dataframe = pd.DataFrame(write_list ,columns=['문장', '악플']) # 쓸 새로운 dataframe 만들기

write_dataframe.to_csv('train_sub_500000.tsv', index=False, header=True, sep="\t")




#print(multilabel_confusion_matrix(y_true, y_pred))