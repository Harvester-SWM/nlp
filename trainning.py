from gc import callbacks
import os
from pickle import NONE
import sys
import argparse
import errno


import pandas as pd
import numpy as np

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts



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
        
        self.bert = BertModel.from_pretrained(self.hparams.pretrained_model)
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.main_classification = torch.nn.Linear(self.bert.config.hidden_size, 11) # classification label
        self.sub_classification = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
            if self.hparams.pretrained_tokenizer
            else self.hparams.pretrained_model
        )

    def forward(self, input_ids, main_labels=None, sub_labels=None,**kwargs):
        #forward에 인자 넘기고 싶으면 / self 있는 곳 들에서 인자 넘겨주면 된다.

        '''
        TO DO
        
        모델에 따라서 output layer 수정하기
                             /----- label 11개
        bert ---> dropout ---
                             \----- label 1개
        '''

        outputs = self.bert(input_ids)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        #main
        main_logits = self.main_classification(pooled_output)

        main_loss_fct = torch.nn.BCEWithLogitsLoss()
        main_loss = main_loss_fct(main_logits, main_labels)
        
        main_output = (main_logits,) + outputs[2:]
        #main

        #sub
        #if sub_labels is not None:
        sub_logits = self.sub_classification(pooled_output)
        
        sub_loss_fct = torch.nn.MSELoss()
        sub_loss = sub_loss_fct(sub_logits, sub_labels)

        sub_output = (sub_logits,) + outputs[2:]
        #sub

        return ((main_loss,) + main_output) ,((sub_loss, ) + sub_output)

    def step(self, batch, batch_idx):
        data, main_labels, sub_labels = batch


        #print(data[0].shape)
        #print(labels[0].shape)
        #print(attention_mask[0].shape)


        main_output, sub_output = self(input_ids=data, main_labels=main_labels, sub_labels=sub_labels)
        
        #print(main_output)


        main_loss, main_logits = main_output
        #main_logits = main_output.main_logits
        
        #print(main_loss)
        #print(main_logits)

        sub_loss, sub_logits = sub_output

        # Transformers 4.0.0+
        #self.log("train_loss", loss, prog_bar=True, logger=True)

        #loss = output.loss
        #logits = output.logits

        #preds = output.argmax(dim=-1)
        y_true = main_labels.detach().cpu().numpy()
        y_pred = main_logits.detach().cpu().numpy()
        #print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}, Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')

        #print("y_true")
        #print(y_true)

        #print("y_pred")
        #print(y_pred)

        return {
            'loss':  main_loss + sub_loss,
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
        # 이유는 모르겠지만 이렇게 하면 된다.
        for i in outputs:
            for true in i['y_true']:
                y_true.append(np.array([ x for x in true]))

            for pred in i['y_pred']:
                 y_pred.append(np.array([1 if x > 0.5 else 0 for x in pred]))
                 #print("for loop")
                 #print([1 if y > 0.5 else 0 for y in x])
        #print(multilabel_confusion_matrix(y_true, y_pred)) # confusion matrixx 구해져서 acc prc rec f1 구하면 된다,

        confusion_mat = multilabel_confusion_matrix(y_true, y_pred)

        for idx ,x in enumerate(confusion_mat):
          acc =  (x[0][0] + x[1][1]) / (x[0][0] + x[0][1] + x[1][0] + x[1][1]) if (x[0][0] + x[0][1] + x[1][0] + x[1][1]) > 0 else 0
          prec =  x[1][1] / (x[0][1] + x[1][1])  if (x[0][1] + x[1][1]) > 0 else 0
          rec = x[1][1] / (x[1][0] + x[1][1]) if (x[1][0] + x[1][1]) > 0 else 0
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
        elif self.hparams.optimizer == 'SWA':
            from torchcontrib.optim import SWA
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
            #optimizer = SWA(swa_lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW , AdamP and SWA is Supported!')
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
            return pd.read_csv(path, sep='\t')
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

    def encode(self, x, **kwargs):
        return self.tokenizer.encode(
                self.clean(str(x)),
                padding='max_length',
                max_length=self.hparams.max_length,
                truncation=True,
                **kwargs,
        )

    def preprocess_dataframe(self, df):
        df['내용'] = df['내용'].map(self.encode)
        # 문장은 input_ids 로 return 해주고 
        #print("리턴 타입 텐서 아니라 list다!")
        
        return df

    def subDataframe(self, main_df):
        write_list = []
        LABEL_COLUMNS = main_df.columns.tolist()[1:]
        
        for index, row in main_df.iterrows():
            #print(row)
            result = 0
            for i in LABEL_COLUMNS:
                if row[i] > 0:
                    result = 1
                    break
            #print(row['내용'], result)
            
            temp = [result]
            write_list.append(temp)


        return pd.DataFrame(write_list, columns=['악플'])

    def dataloader(self, path, shuffle=False):
        main_df = self.read_data(path)
        main_df = self.preprocess_dataframe(main_df)
        LABEL_COLUMNS = main_df.columns.tolist()[1:]

        sub_df=self.subDataframe(main_df) #레이블 결과만 저장했다.
        
        #print(main_df[LABEL_COLUMNS])
        #print(sub_df)

        
        #print(type(main_df[LABEL_COLUMNS]))
        #print(type(sub_df))

        #일단 df에서 다 0 아니면 1로 만들어준다
        for i in LABEL_COLUMNS:
            main_df[i] = main_df[i].map(lambda x : 0 if x == 0 else 1)

        #print(df.columns)

        ## 2차 데이터를 만든다.



        dataset = TensorDataset(
            torch.tensor(main_df['내용'].to_list(), dtype=torch.long),
            torch.tensor(main_df[LABEL_COLUMNS].values.tolist(), dtype=torch.float),
            torch.tensor(sub_df.values.tolist(), dtype=torch.float),
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
    
    def predict_dataloader(self):
        return self.dataloader()
    

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
    'test_name' : '',
    }
    


    '''
    설명 : 모델 변수 설정을 유저로 부터 입력을 받는다. 
    형식 : python3 trainning.py --변수이름 변수 값
    
    예를들어 lr 과 epochs 를 변경하고 싶다면 다음과 같이 입력
    python3 trainning.py --lr 5e-10 --epochs 10 

    '''
    parser = argparse.ArgumentParser(description="usage")

    parser.add_argument('--pretrained_model', type=str, default='beomi/kcbert-large', help='type of model')
    parser.add_argument('--batch_size', type=int, default=4, help='size of batch')
    parser.add_argument('--lr', type=float, default=5e-6, help='number of learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--train_data_path', type=str, default='./data/train.tsv', help='train file path')
    parser.add_argument('--val_data_path', type=str, default='./data/valid.tsv', help='validation file path')
    parser.add_argument('--result_file', type=str, default='result.txt', help='path and name of result file')
    parser.add_argument('--test_mode', type=bool, default=False, help='whether to turn on test')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='type of optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='exp', help='type of learning scheduler')
    parser.add_argument('--test_name', type=str, default='no_name', help='실험 이름 / directory 로 사용한다')


    user_input = parser.parse_args()


    # user_input을 통해 받은 인자를 순회하면서 args에 넣어준다
    for arg in vars(user_input):
        temp = getattr(user_input, arg)
        args[arg] = temp
    
    #check point 와 early_stop_callback을 설정해 준다.
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        dirpath=f"./checkpoint/{args['test_name']}",
        filename="{epoch}-{val_loss:.2f}-{step}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=2,
        verbose=True,
        mode="min",
        check_on_train_epoch_end=True
    )
    
    #argument value check
    #for key, value in args.items():
    #     print(f'key : {key} | value : {value}')
    

    #모델 실행
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
    
    # make file directory
    try:
        os.makedirs(f"./checkpoint/{args['test_name']}")
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(f"./checkpoint/{args['test_name']}"):
            pass
        else: raise

    result = open(f"./checkpoint/{args['test_name']}/{args['result_file']}", 'w')
    
    for arg in vars(user_input):
        temp = getattr(user_input, arg)
        print(f"{arg} : {temp}", end = ' | ', file=result)
    print(file=result)
    
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])
    seed_everything(args['random_seed'])
    print(":: Start Training ::")
    trainer.fit(model)

    result.close()

    TEST = False

    if TEST == True:
        def infer(x):
            return model(**model.tokenizer(x, return_tensors='pt'))

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
