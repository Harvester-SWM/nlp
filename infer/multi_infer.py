import sys
import os

from pickle import FALSE
import torch
import json

from glob import glob
from clean import clean


# inferTest 불러오기
root_path = sys.path[0]
model_path = os.path.join(root_path, '../models')
sys.path.append(model_path)
# inferTest 불러오기

from multi_task_model import Model


#map location 해주면 환경 바뀌어도 가능
def main():
    while True:
        sentence = input("문장을 입력하시오! ")
        judge(sentence=sentence)
    
def infer(x):
    return model(**model.tokenizer(x, return_tensors='pt'))
def judge(sentence):
    sentence=str(sentence)
    if sentence == "":
        print("빈 문장")
    else:
        LABEL_COLUMNS=["욕설","모욕","폭력위협/범죄조장","외설","성혐오","연령","인종/출신지","장애","종교","정치성향","직업혐오"]
        test_prediction, _ = infer(clean(sentence))
        #print(test_prediction)
        output = torch.sigmoid(test_prediction[1])
        output = output.detach().flatten().numpy()
        #for i in zip(LABEL_COLUMNS, output):
            #print(f"{i[0]} {i[1]}에 해당합니다")
                #return 1
            #print(f'probability : {prediction}')
        print(dict(zip(LABEL_COLUMNS, output)))
        return dict(zip(LABEL_COLUMNS, output))
    
if __name__ == "__main__":
    
    # 모델 경로
    MODEL_PATH = "../checkpoint/multi_kcbert-l_0.000005_0/epoch=1-val_loss=0.35.ckpt"
    # 사용할 모델 지정
    latest_ckpt = sorted(glob(MODEL_PATH))[0]
    
    # 모델 로드함 (프로그램 종료되면 다시  load 해야함)
    model = Model.load_from_checkpoint(latest_ckpt)
    # 모델 load 하는 과정에서 불 필요한 함수 제거
    model.eval()
    

# 문장을 평가하는 함수
main()