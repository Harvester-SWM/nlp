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

"""# Model 만들기 with Pytorch Lightning"""

TRESHOLD = 0.5

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() # 이 부분에서 self.hparams에 위 kwargs가 저장된다.
        
        self.model = AutoModel.from_pretrained(self.hparams.pretrained_model)
        self.dropout = torch.nn.Dropout(self.model.config.hidden_dropout_prob)
        self.main_classification = torch.nn.Linear(self.model.config.hidden_size, 11) # classification label
        self.sub_classification = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
            if self.hparams.pretrained_tokenizer
            else self.hparams.pretrained_model
        )

    def forward(self, input_ids, main_labels=None, sub_labels=None,**kwargs):
        #forward에 인자 넘기고 싶으면 / self 있는 곳 들에서 인자 넘겨주면 된다.

        '''
        
        모델에 따라서 output layer 수정하기
                             /----- label 11개
        bert ---> dropout ---
                             \----- label 1개
        '''

        outputs = self.model(input_ids)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        #main logtis and loss
        main_logits = self.main_classification(pooled_output)

        main_loss_fct = torch.nn.BCEWithLogitsLoss()
        if main_labels is not None:
            main_loss = main_loss_fct(main_logits, main_labels)
        else:
            main_loss = 0

        main_output = (main_logits,) + outputs[2:]

        #sub logtis and loss
        sub_logits = self.sub_classification(pooled_output)
        
        sub_loss_fct = torch.nn.BCEWithLogitsLoss()
        if sub_labels is not None:
            sub_loss = sub_loss_fct(sub_logits, sub_labels)
        else:
            sub_loss = 0

        sub_output = (sub_logits,) + outputs[2:]


        return ((main_loss,) + main_output) ,((sub_loss, ) + sub_output)

    def step(self, batch, batch_idx):
        data, main_labels, sub_labels = batch

        main_output, sub_output = self(input_ids=data, main_labels=main_labels, sub_labels=sub_labels)

        main_loss, main_logits = main_output

        sub_loss, sub_logits = sub_output

        y_true = main_labels.detach().cpu().numpy()
        y_pred = main_logits.detach().cpu().numpy()
        
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
        
        # loss 더하기
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)
        
        y_true = []
        y_pred = []
        # 이유는 모르겠지만 이렇게 하면 된다.
        for i in outputs:
            for true in i['y_true']:
                y_true.append(np.array([ x for x in true]))

            for pred in i['y_pred']:
                y_pred.append(np.array([1 if x > TRESHOLD else 0 for x in pred]))
        

        confusion_mat = multilabel_confusion_matrix(y_true, y_pred)

        #file open
        result = open(f"./checkpoint/{self.hparams.test_name}/{self.hparams.result_file}", 'a')

        total_acc = 0; total_prec = 0; total_rec = 0; total_f1 = 0

        for idx ,x in enumerate(confusion_mat):
            acc =  (x[0][0] + x[1][1]) / (x[0][0] + x[0][1] + x[1][0] + x[1][1]) if (x[0][0] + x[0][1] + x[1][0] + x[1][1]) > 0 else 0
            prec =  x[1][1] / (x[0][1] + x[1][1])  if (x[0][1] + x[1][1]) > 0 else 0
            rec = x[1][1] / (x[1][0] + x[1][1]) if (x[1][0] + x[1][1]) > 0 else 0
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
            total_acc+=acc; total_prec+=prec; total_rec+=rec; total_f1+=f1
            print(f'class {idx : .5f} |  acc : {acc : .5f} | prec : {prec : .5f} | rec : {rec : .5f} | f1 : {f1 : .5f}')
            
        total_acc /= 11; total_prec /= 11; total_rec /= 11; total_f1 /= 11

        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}')
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}', file=result)
        print(f'acc : {total_acc : .5f} | prec : {total_prec : .5f} | rec : {total_rec : .5f} | f1 : {total_f1 : .5f}', file=result)
        
        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', float(total_acc), on_epoch=True, prog_bar=True)
        self.log(state+'_prec', float(total_prec), on_epoch=True, prog_bar=True)
        self.log(state+'_rec', float(total_rec), on_epoch=True, prog_bar=True)
        self.log(state+'_f1', float(total_f1), on_epoch=True, prog_bar=True)
        
        #file close
        result.close()

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
        emojis = ''.join(emoji.EMOJI_DATA.keys()) 
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

    def subDataframe(self, main_df, LABEL_COLUMNS):
        write_list = []
        
        for index, row in main_df.iterrows():
            result = 0
            for i in LABEL_COLUMNS:
                if row[i] > self.hparams.sensitive:
                    result = 1
                    break
            
            temp = [result]
            write_list.append(temp)

        return pd.DataFrame(write_list, columns=['악플'])

    def dataloader(self, path, shuffle=False):
        main_df = self.read_data(path)
        main_df = self.preprocess_dataframe(main_df)
        LABEL_COLUMNS = main_df.columns.tolist()[1:]

        sub_df=self.subDataframe(main_df, LABEL_COLUMNS) #레이블 결과만 저장했다.
        # 227번째 줄의 코드를 232번째 줄의 반복문 아래로 옮기게 되면 결과가 바뀔 수 있다
        # 위치 이동시 참고 할 것

        #일단 df에서 다 0 아니면 1로 만들어준다
        for i in LABEL_COLUMNS:
            main_df[i] = main_df[i].map(lambda x : 1 if x > self.hparams.sensitive else 0)


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
