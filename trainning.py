from gc import callbacks
import os
import sys
import argparse

import pandas as pd
import numpy as np

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from transformers import BertForSequenceClassification, BertTokenizer, AdamW, AutoModelForSequenceClassification, AutoTokenizer, BertModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix

import re
import emoji
from soynlp.normalizer import repeat_normalize
from soynlp.normalizer import repeat_normalize

"""## 기본 학습 Arguments"""



tpu_cores: int = 0  # Enable TPU with 1 core or 8 cores

checkpoint_callback = ModelCheckpoint(
    save_top_k=2,
    monitor="val_loss",
    mode="min",
    dirpath="./mypath",
    filename="works"
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.1,
    patience=2,
    verbose=True,
    mode="min",
    check_on_train_epoch_end=True
)

#def loss_fn(outputs, targets):
#    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


"""## 기본값을 Override 하고싶은 경우 아래와 같이 수정"""

"""위에서 GPU가 V100/P100이면 아래 `batch_size`  를 32 이상으로 하셔도 됩니다."""

# args.tpu_cores = 8  # Enables TPU


"""# Model 만들기 with Pytorch Lightning"""

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() # 이 부분에서 self.hparams에 위 kwargs가 저장된다.
        
        self.bert = BertModel.from_pretrained(self.hparams.pretrained_model, return_dict=True)
        self.drop = torch.nn.Dropout(0.3) #forward에서 적용하셈 이거 있어야 할 듯 
        self.line = torch.nn.Linear(self.bert.config.hidden_size, self.hparams.n_classes) # (1025, x)에서 x 는 label의 개수다 ㅇㅇ kcbert large라서 1024개 ㅅㅂ..
        self.criterion = torch.nn.BCEWithLogitsLoss()#class 개수 많아지면 다른 loss 함수 써야한다.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
            if self.hparams.pretrained_tokenizer
            else self.hparams.pretrained_model
        )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        #forward에 인자 넘기고 싶으면 / self 있는 곳 들에서 인자 넘겨주면 된다.

        #print(self.tokenizer())
        output_2 = self.bert(input_ids, attention_mask=attention_mask)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #print(output_1)
        output_1 = self.line(output_2.pooler_output) # 모델의 결과 값
        output = torch.sigmoid(output_1)
        #print(labels)
        loss = 0
        if labels is not None:
            #print("label is not none")
            loss = self.criterion(output_1, labels)
        #print(loss)
        return loss, output

    def step(self, batch, batch_idx):
        data, labels, attention_mask = batch

        #print(data[0].shape)
        #print(labels[0].shape)
        #print(attention_mask[0].shape)

        loss ,output = self(input_ids=data, attention_mask=attention_mask, labels=labels)
        
        # Transformers 4.0.0+
        #self.log("train_loss", loss, prog_bar=True, logger=True)

        #loss = output.loss
        #logits = output.logits
        
        #preds = output.argmax(dim=-1)
        y_true = list(labels.detach().cpu().numpy())
        y_pred = list(output.detach().cpu().numpy())
        #print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}, Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')

        #print("y_true")
        #print(y_true)

        #print("y_pred")
        #print(y_pred)

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        
        loss = torch.tensor(0, dtype=torch.float)
        #print("epoch_end")
        #print(outputs)

        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)
        # loss 더하기
        
        y_true = []
        y_pred = []
        for i in outputs:
            #print(i)
            y_true += i['y_true']

            for x in i['y_pred']:
                 y_pred.append(np.array([1 if y > 0.5 else 0 for y in x]))
                 #print("for loop")
                 #print([1 if y > 0.5 else 0 for y in x])
        #print(multilabel_confusion_matrix(y_true, y_pred)) # confusion matrixx 구해져서 acc prc rec f1 구하면 된다,

        confusion_mat = multilabel_confusion_matrix(y_true, y_pred)

        for idx ,x in enumerate(confusion_mat):
          acc =  (x[0][0] + x[1][1]) / (x[0][0] + x[0][1] + x[1][0] + x[1][1]) if (x[0][0] + x[0][1] + x[1][0] + x[1][1]) > 0 else 0
          prec =  x[1][1] / (x[0][1] + x[1][1])  if (x[0][1] + x[1][1]) > 0 else 0
          rec = x[1][1] / (x[1][0] + x[1][1]) if (x[0][1] + x[1][1]) > 0 else 0
          f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
          print(f'class {idx : .5f} |  acc : {acc : .5f} | prec : {prec : .5f} | rec : {rec : .5f} | f1 : {f1 : .5f}')

        #print(y_true)
        #print(y_pred)
        #acc = accuracy_score(y_true, y_pred)
        #prec = precision_score(y_true, y_pred)
        #rec = recall_score(y_true, y_pred)
        #f1 = f1_score(y_true, y_pred)
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}', file=result)

        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        #self.log(state+'_acc', acc, on_epoch=True, prog_bar=True)
        #self.log(state+'_precision', prec, on_epoch=True, prog_bar=True)
        #self.log(state+'_recall', rec, on_epoch=True, prog_bar=True)
        #self.log(state+'_f1', f1, on_epoch=True, prog_bar=True)
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}')
        #print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}, Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, state='val')

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')[:100]
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def clean(self, x):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

    def convert(self, x, **kwargs):
        return self.tokenizer(
            self.clean(str(x)),
            add_special_tokens=True,
            max_length=self.hparams.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            **kwargs,
        )

    def encode(self, x, **kwargs):
        return self.tokenizer.encode(
                self.clean(str(x)),
                padding='max_length',
                max_length=self.hparams.max_length,
                truncation=True,
                **kwargs,
        )
    def preprocess_dataframe(self, df):
        temp = df['내용'].map(self.convert)
        #temp = temp.to_list()
        
        df['내용'] = temp.map(lambda x: x['input_ids'])
        attention_mask = temp.map(lambda x: x['attention_mask'])

        #print(df['문장'][:, 0])
        #print(df['문장'][:, 1])
        
        # 문장은 input_ids 로 return 해주고 
        #print("리턴 타입 텐서 아니라 list다!")
        
        #print(type(attention_mask))
        #print(df['문장'][0].shape)
        #print(attention_mask[0].shape)
        
        return attention_mask

    def dataloader(self, path, shuffle=False):
        df = self.read_data(path)
        attention_mask = self.preprocess_dataframe(df)
        LABEL_COLUMNS = df.columns.tolist()[1:]
        
        #일단 df에서 다 0 아니면 1로 만들어준다

        for i in LABEL_COLUMNS:
            df[i] = df[i].map(lambda x : 0 if x < 2 else 1)

        
        #print("temp ahead")
        #print(type(temp))
        #print(temp["문장"][0]['input_ids'])
        #print(temp["문장"][0]['attention_mask'])

        #print(df.columns)

        dataset = TensorDataset(
            torch.tensor(df['내용'].to_list(), dtype=torch.long),
            torch.tensor(df[LABEL_COLUMNS].values.tolist(), dtype=torch.float),
            torch.tensor(attention_mask, dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size * 1 if not self.hparams.tpu_cores else self.hparams.tpu_cores,
            shuffle=shuffle,
            num_workers=self.hparams.cpu_workers,
        )

    def train_dataloader(self):
        return self.dataloader(self.hparams.train_data_path, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.hparams.val_data_path, shuffle=False)
    

if __name__ == "__main__":
    ''' 이 파일의 전체 변수 '''
    args = {
    'random_seed': 42, # Random Seed
    'pretrained_model': 'beomi/kcbert-large',  # Transformers PLM name
    'pretrained_tokenizer': '',  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    'batch_size': 4,
    'lr': 5e-6,  # Starting Learning Rate
    'epochs': 5,  # Max Epochs
    'max_length': 150,  # Max Length input size
    'train_data_path': "",  # Train Dataset file 
    'val_data_path': "",  # Validation Dataset file 
    'test_mode': False,  # Test Mode enables `fast_dev_run`
    'optimizer': 'AdamW',  # AdamW vs AdamP
    'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
    'fp16': True,  # Enable train on FP16(if GPU)
    'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
    'cpu_workers': os.cpu_count(),
    'n_classes' : 11,
    }
    


    '''
    설명 : 모델 변수 설정을 유저로 부터 입력을 받는다. 
    형식 : python3 trainning.py --변수이름 변수 값
    
    예를들어 lr 과 epochs 를 변경하고 싶다면 다음과 같이 입력
    python3 trainning.py --lr 5e-10 --epochs 10 

    '''
    parser = argparse.ArgumentParser(description="usage")


    parser.add_argument('--batch_size', type=int, default=4, help='size of batch')
    parser.add_argument('--lr', type=float, default=5e-6, help='number of learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--train_data_path', type=str, default='./data/train.tsv', help='train file path')
    parser.add_argument('--val_data_path', type=str, default='./data/valid.tsv', help='validation file path')
    parser.add_argument('--result_file', type=str, default='result.txt', help='path and name of result file')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='type of optimizer')


    user_input = parser.parse_args()


    ''' user_input을 통해 받은 인자를 순회하면서 args에 넣어준다 '''
    for arg in vars(user_input):
        temp = getattr(user_input, arg)
        args[arg] = temp
    

    
    #argument value check
    #for key, value in args.items():
    #     print(f'key : {key} | value : {value}')
    

    '''모델 실행'''
    model = Model(**args)
    trainer = Trainer(
        max_epochs=args['epochs'],
        fast_dev_run=args['test_mode'],
        num_sanity_val_steps=None if args['test_mode'] else 0,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16 if args['fp16'] else 32,
        progress_bar_refresh_rate=30,
        callbacks = [early_stop_callback, checkpoint_callback]
        #callback?
        # For TPU Setup
        # tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    

    MODEL_TEST = False
    
    # 토크나이저나 그런거 테스트 하는 공간
    
    if MODEL_TEST == True:
        exit()
    
    result = open(args['result_file'], 'w')

    print(f"args.batch_size : {args['batch_size']} | lr : {args['lr']} | epoches : {args['epochs']} | optimizer : {args['optimizer']}", file=result)

    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])
    seed_everything(args['random_seed'])
    print(":: Start Training ::")
    trainer.fit(model)

    result.close()

    TEST = False

    if TEST == True:
        def infer(x):
            temp  = model.tokenizer(x, 
                                    add_special_tokens=True, 
                                    max_length=300, 
                                    return_token_type_ids=False, 
                                    padding="max_length",
                                    return_attention_mask=True, 
                                    return_tensors='pt'
                                    )
            return model(temp["input_ids"], temp["attention_mask"])

        def judge(sentence):
            if sentence == "":
                print("빈 문장")
            else:
                LABEL_COLUMNS=["욕설","모욕","폭력위협/범죄조장","외설","성혐오","연령","인종/출신지","장애","종교","정치성향","직업혐오"]
                _, test_prediction = infer(sentence)
                #print(type(test_prediction))
                #print(test_prediction)
                test_prediction = test_prediction.detach().flatten().numpy()
                for i in zip(LABEL_COLUMNS, test_prediction):
                    #if i[1] > 0.5:
                    print(i)
                    #print(f'probability : {prediction}')
        while True:
            sentence = input("문장을 입력하시오 : ")
            judge(sentence)
"""# 학습!

> 주의: 1epoch별로 GPU-P100기준 약 2~3시간, GPU V100기준 ~40분이 걸립니다.

> Update @ 2020.09.01
> 최근 Colab Pro에서 V100이 배정됩니다.

```python
# 1epoch 기준 아래 score가 나옵니다.
{'val_acc': 0.90522,
 'val_f1': 0.9049023739289227,
 'val_loss': 0.23429009318351746,
 'val_precision': 0.9143146796431468,
 'val_recall': 0.8956818813808446}
```
"""