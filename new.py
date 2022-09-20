import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import infer.multi_infer


PATH = "./test.txt"

RESULT_PATH="graph.txt"
DIFF_PATH="diff.txt"
TEMP_PATH='temp.txt'
TEMP_WRITE='temp_write.txt'

f = open(TEMP_PATH, 'r')
f1 = open(TEMP_WRITE, 'w')

lines = f.readlines()

for line in lines:
    if infer.multi_infer.judge(line) == 1:
        print(line.strip(), file=f1)



f.close()
f1.close()

#multi_infer.infer()