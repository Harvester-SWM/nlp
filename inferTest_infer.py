import torch
import sys
from pytorch_lightning import LightningModule, Trainer, seed_everything
import inferTest

# loop = True;

# while loop: 
#   sentence = input("하고싶은 말을 입력해주세요 : ") 
#   if sentence == 0: 
#     break;
#   print(infer(sentence))



MODEL_PATH = "./lightning_logs/version_5/checkpoints/*.ckpt"
HPARAMS_PATH = "./lightning_logs/version_5/hparams.yaml"

from glob import glob

latest_ckpt = sorted(glob(MODEL_PATH))[0]
model = inferTest.Model.load_from_checkpoint(latest_ckpt, hparams_file=HPARAMS_PATH)
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
        toxic = torch.argmax(infer(sentence))
        #print(infer(sentence))

        # if toxic == 0:
        #     print("악플이 아닙니다.")
        # elif toxic == 1:
        #     print("악플로 판정되었습니다.")
        return toxic

if __name__ == "__main__":
    main()