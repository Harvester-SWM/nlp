import os
import argparse
import errno

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch

"""## 기본 학습 Arguments"""

tpu_cores: int = 0  # Enable TPU with 1 core or 8 cores


''' 이 파일의 전체 변수 '''
args = {
'random_seed': 42, # Random Seed
'pretrained_model': 'beomi/kcbert-large',  # Transformers PLM name
'pretrained_tokenizer': '',  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
'batch_size': 32,
'lr': 5e-6,  # Starting Learning Rate
'epochs': 10,  # Max Epochs
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
'sensitive' : 0,
'test_name' : '',
'result_file': '',
}



'''
설명 : 모델 변수 설정을 유저로 부터 입력을 받는다. 
형식 : python3 trainning.py --변수이름 변수 값

예를들어 lr 과 epochs 를 변경하고 싶다면 다음과 같이 입력
python3 trainning.py --lr 5e-10 --epochs 10 

'''
parser = argparse.ArgumentParser(description="usage")

parser.add_argument('--model_task', type=str, default='multi_task_model', help='single_task_model 선택 혹은 multi_task_model') # 이건 필수
parser.add_argument('--pretrained_model', type=str, default='beomi/kcbert-large', help='type of model')
parser.add_argument('--batch_size', type=int, default=64, help='size of batch')
parser.add_argument('--lr', type=float, default=5e-6, help='number of learning rate')
parser.add_argument('--epochs', type=int, default=15, help='number of epochs')
parser.add_argument('--train_data_path', type=str, default='./data/train_500000.tsv', help='train file path')
parser.add_argument('--val_data_path', type=str, default='./data/valid_500000.tsv', help='validation file path')
parser.add_argument('--test_mode', type=bool, default=False, help='whether to turn on test')
parser.add_argument('--optimizer', type=str, default='AdamW', help='type of optimizer')
parser.add_argument('--lr_scheduler', type=str, default='exp', help='type of learning scheduler')
parser.add_argument('--sensitive', type=int, default=0, help='how sensitive 0이면 sensitive 하기 1 이면 둔감')
parser.add_argument('--test_name', type=str, default='no_name', help='실험 이름 / directory 로 사용한다')
parser.add_argument('--result_file', type=str, default='result.txt', help='path and name of result file')



user_input = parser.parse_args()

# user_input 에서 받은 model_task를 기준으로 import 할 모듈을 정한다.
if user_input.model_task == 'multi_task_model':
    from models.multi_task_model import Model
elif user_input.model_task == 'single_task_model':
    from models.single_task_model import Model
elif user_input.model_task == 'smile_gate':
    from models.inferTest import Model
else:
    print(user_input.model_task)
    raise NotImplementedError('Only single or multitask_model supported!')


# user_input을 통해 받은 인자를 순회하면서 args에 넣어준다
for arg in vars(user_input):
    temp = getattr(user_input, arg)
    args[arg] = temp


# make file directory
try:
    os.makedirs(f"./checkpoint/{args['test_name']}")
except OSError as exc: # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(f"./checkpoint/{args['test_name']}"):
        exit()# 경로 존재하면 프로그램 종료
        pass
    else: raise
#파일 오픈
result = open(f"./checkpoint/{args['test_name']}/{args['result_file']}", 'w')
#args 에 파일 stream 넘겨주기
#args['result_file']=result

for arg in vars(user_input):
    temp = getattr(user_input, arg)
    print(f"{arg} : {temp}", end = ' | ', file=result)
print(file=result)

result.close()


#check point 와 early_stop_callback을 설정해 준다.
checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        dirpath=f"./checkpoint/{args['test_name']}",
        filename="{epoch}-{val_loss:.2f}"
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
    callbacks = [early_stop_callback, checkpoint_callback]
    #callback?
    # For TPU Setup
    # tpu_cores=args.tpu_cores if args.tpu_cores else None,
)



print("Using PyTorch Ver", torch.__version__)
print("Fix Seed:", args['random_seed'])
seed_everything(args['random_seed'])
print(":: Start Training ::")
trainer.fit(model)