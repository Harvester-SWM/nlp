import os
import pandas as pd

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import LightningModule, Trainer, seed_everything

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import re
import emoji
from soynlp.normalizer import repeat_normalize

"""## 기본 학습 Arguments"""
# """# Model 만들기 with Pytorch Lightning"""

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() # 이 부분에서 self.hparams에 위 kwargs가 저장된다.
        
        self.clsfier = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
            if self.hparams.pretrained_tokenizer
            else self.hparams.pretrained_model
        )

    def forward(self, **kwargs):
        return self.clsfier(**kwargs)

    def step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

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
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', acc, on_epoch=True, prog_bar=True)
        self.log(state+'_precision', prec, on_epoch=True, prog_bar=True)
        self.log(state+'_recall', rec, on_epoch=True, prog_bar=True)
        self.log(state+'_f1', f1, on_epoch=True, prog_bar=True)
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}, Acc: {acc}, Prec: {prec}, Rec: {rec}, F1: {f1}')
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
        return df
    
    def subDataframe(self, df, LABEL_COLUMNS):
        write_list = []
        
        for index, row in df.iterrows():
            result = 0
            for i in LABEL_COLUMNS:
                if row[i] > self.hparams.sensitive:
                    result = 1
                    break
            
            temp = [result]
            write_list.append(temp)

        return pd.DataFrame(write_list, columns=['악플'])

    def dataloader(self, path, shuffle=False):
        df = self.read_data(path)
        df = self.preprocess_dataframe(df)

        LABEL_COLUMNS = df.columns.tolist()[1:]

        label_df = self.subDataframe(df, LABEL_COLUMNS)

        dataset = TensorDataset(
            torch.tensor(df['내용'].to_list(), dtype=torch.long),
            torch.tensor(label_df['악플'].to_list(), dtype=torch.long),
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

"""# 학습!

> 💡**NOTE**💡 1epoch별로 GPU P100기준 약50분, GPU V100기준 ~15분이 걸립니다. <br>
> 학습시 약 0.92 이하의 validation acc를 얻을 수 있습니다.

> Update @ 2020.09.01
> 최근 Colab Pro에서 V100이 배정됩니다.

```python
# 1epoch
loss=0.207, v_num=0, val_loss=0.221, val_acc=0.913, val_precision=0.914, val_recall=0.913, val_f1=0.914
# 2epoch
loss=0.152, v_num=0, val_loss=0.213, val_acc=0.918, val_precision=0.912, val_recall=0.926, val_f1=0.919
# 3epoch
loss=0.135, v_num=0, val_loss=0.225, val_acc=0.919, val_precision=0.907, val_recall=0.936, val_f1=0.921
```
"""

# print("Using PyTorch Ver", torch.__version__)
# print("Fix Seed:", args['random_seed'])
# seed_everything(args['random_seed'])
# model = Model(**args)

# print(":: Start Training ::")
# trainer = Trainer(
#     max_epochs=args['epochs'],
#     fast_dev_run=args['test_mode'],
#     num_sanity_val_steps=None if args['test_mode'] else 0,
#     # For GPU Setup
#     deterministic=torch.cuda.is_available(),
#     gpus=[0] if torch.cuda.is_available() else None,  # 0번 idx GPU  사용
#     precision=16 if args['fp16'] and torch.cuda.is_available() else 32,
#     callbacks = [early_stop_callback, checkpoint_callback]
#     # For TPU Setup
#     # tpu_cores=args['tpu_cores'] if args['tpu_cores'] else None,
# )
# trainer.fit(model)

"""# Inference"""

# from glob import glob

# latest_ckpt = sorted(glob('./lightning_logs/version_1/checkpoints/*.ckpt'))[0]

# model = Model.load_from_checkpoint(latest_ckpt)

# def infer(x):
#     return torch.softmax(
#         model(**model.tokenizer(x, return_tensors='pt')
#     ).logits, dim=-1)

# print(infer('노잼 '))

# print(infer('이  영화  꿀잼! 완존  추천요  '))