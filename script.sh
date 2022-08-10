# 포멧
# python3 (trainning.py) (train) (valid) (result)

#
:<<'END'
parser.add_argument('--model_task', type=str, default='multi_task_model', help='single_task_model 선택 혹은 multi_task_model') # 이건 필수
parser.add_argument('--pretrained_model', type=str, default='beomi/kcbert-large', help='type of model')
parser.add_argument('--batch_size', type=int, default=4, help='size of batch')
parser.add_argument('--lr', type=float, default=5e-6, help='number of learning rate')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--train_data_path', type=str, default='./data/train.tsv', help='train file path')
parser.add_argument('--val_data_path', type=str, default='./data/valid.tsv', help='validation file path')
parser.add_argument('--test_mode', type=bool, default=False, help='whether to turn on test')
parser.add_argument('--optimizer', type=str, default='AdamW', help='type of optimizer')
parser.add_argument('--lr_scheduler', type=str, default='exp', help='type of learning scheduler')
parser.add_argument('--sensitive', type=int, default=0, help='how sensitive 0이면 sensitive 하기 1 이면 둔감')
parser.add_argument('--test_name', type=str, default='no_name', help='실험 이름 / directory 로 사용한다')

변수 이름 지정
END 
# 이번에 조절할꺼 model_task lr pretrarined_model sensitive test_name
python3 trainning.py --model_task $ --pretrained_model $ --batch_size $ --lr $ --epochs $ --train_data_path $ --val_data_path $ --test_mode $ --optimizer --lr_scheduler --sensitive $ --test_name $

python3 trainning.py train_500000.tsv    valid_500000.tsv   result.txt

#python3 trainning.py new_train_1.tsv    new_valid_1.tsv   result_1.txt

#처음이 smile_gate 로만 훈련
#python3 trainning.py train_smile.tsv    valid_smile.tsv   result_smile_6.txt

#두번째가 내 데이터 추가후 훈련

#python3 trainning.py train_my.tsv       valid_smile.tsv   result_my_6.txt