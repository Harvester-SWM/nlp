import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import single_infer
import multi_infer
import inferTest_infer


PATH = "./test.txt"

RESULT_PATH="graph.txt"
DIFF_PATH="diff.txt"

df = pd.read_csv(PATH, sep='\t')



#print(type(df['악플']))

#print(df['악플'])
#print(df['악플'].tolist())


y_true = df['악플'].tolist()
multi_pred = []
single_pred = []
inferTest_pred = []

diff = open(DIFF_PATH, 'a')
result = open(RESULT_PATH, 'a')

for idx ,i in enumerate(df['내용']):
    mul = multi_infer.judge(i)
    sin = single_infer.judge(i)
    inf = inferTest_infer.judge(i)
    multi_pred.append(mul)
    single_pred.append(sin)
    inferTest_pred.append(inf)
    if mul != sin or mul != inf or sin != inf:
        print(f'mul : {mul}, sin : {sin}, inf : {inf}, pred : {y_true[idx]}, sentence : {i}', file=diff)
    print(idx)
    #pass



acc = accuracy_score(y_true, multi_pred)
prec = precision_score(y_true, multi_pred)
rec = recall_score(y_true, multi_pred)
f1 = f1_score(y_true, multi_pred)
print("MULTI: ",acc, prec, rec, f1, file=result)


acc = accuracy_score(y_true, single_pred)
prec = precision_score(y_true, single_pred)
rec = recall_score(y_true, single_pred)
f1 = f1_score(y_true, single_pred)
print("SINGLE: ",acc, prec, rec, f1, file=result)

acc = accuracy_score(y_true, inferTest_pred)
prec = precision_score(y_true, inferTest_pred)
rec = recall_score(y_true, inferTest_pred)
f1 = f1_score(y_true, inferTest_pred)
print("INFERTEST: ",acc, prec, rec, f1, file=result)


diff.close()
result.close()

#multi_infer.infer()