import torch
import sys
import os
from pytorch_lightning import LightningModule, Trainer, seed_everything

# inferTest 불러오기
root_path = sys.path[0]
model_path = os.path.join(root_path, '../models')
sys.path.append(model_path)
# inferTest 불러오기

from inferTest import Model
from clean import clean


MODEL_PATH = "../checkpoint/smile_kcbert-b_0.000005_0/epoch=0-val_loss=0.31.ckpt"

from glob import glob

latest_ckpt = sorted(glob(MODEL_PATH))[0]
model = Model.load_from_checkpoint(latest_ckpt)
model.eval()
#map location 해주면 환경 바뀌어도 가능
def main():
    #checkpoint = torch.load(MODEL_PATH)
    #print(model["hyper_parameters"])
    while True:
        sentence = input("문장을 입력하시오! ")
        judge(sentence=sentence)


def infer(x):
    return torch.softmax(
        model(**model.tokenizer(x, return_tensors='pt')
    ).logits, dim=-1)

def judge(sentence):
    sentence=str(sentence)
    if sentence == "":
        print("빈 문장")
    else:
        toxic = torch.argmax(infer(clean(sentence)))
        #print(infer(sentence))

        # if toxic == 0:
        #     print("악플이 아닙니다.")
        # elif toxic == 1:
        #     print("악플로 판정되었습니다.")
        return toxic

if __name__ == "__main__":
    main()