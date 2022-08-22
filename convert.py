import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

from pytorch_lightning import LightningModule

from transformers import  AdamW, AutoTokenizer, AutoModel

from sklearn.metrics import multilabel_confusion_matrix

import re
import emoji
from soynlp.normalizer import repeat_normalize

#import multi_infer
#import single_infer

# 읽어 들일 파일
PATH="./valid.tsv"


def clean(x):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        x = str(x)
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

# data를 0 1 로 변환
def subDataframe(df):
    write_list = []
        
    for index, row in df.iterrows():
        result = 0
        for i in LABEL_COLUMNS:
            if row[i] > SENSITIVE:
                result = 1
                break
        
        temp = result
        write_list.append([row['내용'] ,temp])
    
    return pd.DataFrame(write_list, columns=['내용', '악플'])


df = pd.read_csv(PATH, sep='\t')

#print(df)
LABEL_COLUMNS = df.columns.tolist()[1:]
SENSITIVE = 0

df['내용'] = df['내용'].map(clean)

df = subDataframe(df)

#print(df)

df.to_csv('test.txt', index=False, header=True, sep="\t")