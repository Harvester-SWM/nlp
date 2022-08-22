#!/bin/bash

: << "END"
#parser.add_argument('--model_task', type=str, default='multi_task_model', help='single_task_model 선택 혹은 multi_task_model') # 이건 필수
#parser.add_argument('--pretrained_model', type=str, default='beomi/kcbert-large', help='type of model')
#parser.add_argument('--batch_size', type=int, default=4, help='size of batch')
#parser.add_argument('--lr', type=float, default=5e-6, help='number of learning rate')
#parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
#parser.add_argument('--train_data_path', type=str, default='./data/train.tsv', help='train file path')
#parser.add_argument('--val_data_path', type=str, default='./data/valid.tsv', help='validation file path')
#parser.add_argument('--test_mode', type=bool, default=False, help='whether to turn on test')
#parser.add_argument('--optimizer', type=str, default='AdamW', help='type of optimizer')
#parser.add_argument('--lr_scheduler', type=str, default='exp', help='type of learning scheduler')
#parser.add_argument('--sensitive', type=int, default=0, help='how sensitive 0이면 sensitive 하기 1 이면 둔감')
#parser.add_argument('--test_name', type=str, default='no_name', help='실험 이름 / directory 로 사용한다')
END

#변수 이름 지정
# 이번에 조절할꺼 model_task lr pretrarined_model sensitive test_name
# 모델은 kcbert / kcelectra / kobert

# 모델 이름의 array
# 폴더 이름으로 사용할 문자열
# lr 
# sensitive
# model_task

#python3 trainning.py --model_task multi_task_model --pretrained_model beomi/KcELECTRA-base --lr 0.000005 --sensitive 1 --test_name multi_kcelec_0.000005_1

model_task_list=("single_task_model" "multi_task_model")
model_task_str=("single" "multi")

pretrained_model_list=("beomi/kcbert-large" "beomi/kcbert-base" "HanBert-54kN-torch" "beomi/KcELECTRA-base" "monologg/koelectra-base-v3-discriminator" "monologg/koelectra-small-v3-discriminator")
pretrained_model_str=("kcbert-l" "kcbert-b" "Hanbert" "kcelec" "koelec-b" "koelec-s")
lr_list=(0.000005 0.000001)
sensitive_list=(0 1)

NOW_TEST_NUMBER=1
TOTAL_TEST_NUMBER=`expr ${#model_task_list[@]} \* ${#pretrained_model_list[@]} \* ${#lr_list[@]} \*  ${#sensitive_list[@]}`

# electra에서는 multi-task model 작동하지 않음
# 이번에 들어가는 인수 {model_task} {lr} {pretrarined_model} {sensitive} {test_name} 총 5개

for (( i = 0 ; i < ${#model_task_list[@]} ; i++ ))  ; do
    for (( j = 0 ; j < ${#pretrained_model_list[@]} ; j++ ))  ; do
        for lr in "${lr_list[@]}" ; do
            for sensitive in "${sensitive_list[@]}" ; do
START=$(date +%s)


python3 message.py \
--command \
"python3 trainning.py\
--model_task ${model_task_list[$i]} \
--pretrained_model ${pretrained_model_list[$j]} \
--lr ${lr} \
--sensitive ${sensitive}" \
--now_number ${NOW_TEST_NUMBER} \
--total_number ${TOTAL_TEST_NUMBER} \

#echo "
#python3 trainnig.py \
#--model_task ${model_task_list[$i]} \
#--pretrained_model ${pretrained_model_list[$j]} \
#--lr ${lr} \
#--sensitive ${sensitive} \
#--test_name ${model_task_str[$i]}_${pretrained_model_str[$j]}_${lr}_${sensitive}
#" 
python3 trainning.py \
--model_task ${model_task_list[$i]} \
--pretrained_model ${pretrained_model_list[$j]} \
--lr ${lr} \
--sensitive ${sensitive} \
--test_name ${model_task_str[$i]}_${pretrained_model_str[$j]}_${lr}_${sensitive}
END=$(date +%s)
DIFF=$(( $END - $START ))

python3 message.py \
--command \
"python3 trainning.py\
--model_task ${model_task_list[$i]} \
--pretrained_model ${pretrained_model_list[$j]} \
--lr ${lr} \
--sensitive ${sensitive}" \
--now_number ${NOW_TEST_NUMBER} \
--total_number ${TOTAL_TEST_NUMBER} \
--time_elapsed ${DIFF}

NOW_TEST_NUMBER=$(($NOW_TEST_NUMBER + 1))
            done
        done
    done
done

python3 message.py --command "done!"


#python3 trainning.py --model_task $ --pretrained_model $ --batch_size $ --lr $ --epochs $ --train_data_path $ --val_data_path $ --test_mode $ --optimizer --lr_scheduler --sensitive $ --test_name $

#python3 trainning.py train_500000.tsv    valid_500000.tsv   result.txt

#python3 trainning.py new_train_1.tsv    new_valid_1.tsv   result_1.txt

#처음이 smile_gate 로만 훈련
#python3 trainning.py train_smile.tsv    valid_smile.tsv   result_smile_6.txt

#두번째가 내 데이터 추가후 훈련

#python3 trainning.py train_my.tsv       valid_smile.tsv   result_my_6.txt
