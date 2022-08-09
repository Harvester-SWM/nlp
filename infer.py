import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
import trainning

# loop = True;

# while loop: 
#   sentence = input("하고싶은 말을 입력해주세요 : ") 
#   if sentence == 0: 
#     break;
#   print(infer(sentence))


MODEL_PATH = "./lightning_logs/version_2/checkpoints/*.ckpt"
#HPARAMS_PATH = "./lightning_logs/version_2/hparams.yaml"

from glob import glob

latest_ckpt = sorted(glob(MODEL_PATH))[0]
#model = trainning.Model.load_from_checkpoint(latest_ckpt, hparams_file=HPARAMS_PATH)
model = trainning.Model.load_from_checkpoint(latest_ckpt)

model.eval()
#map location 해주면 환경 바뀌어도 가능
def main():
    while True:
        sentence = input("문장을 입력하시오! ")
        judge(sentence=sentence)
    
def infer(x):
    return model(**model.tokenizer(x, return_tensors='pt'))
def judge(sentence):
    if sentence == "":
        print("빈 문장")
    else:
        LABEL_COLUMNS=["욕설","모욕","폭력위협/범죄조장","외설","성혐오","연령","인종/출신지","장애","종교","정치성향","직업혐오"]
        test_prediction = infer(sentence)
        output = torch.sigmoid(test_prediction.logits)
        output = output.detach().flatten().numpy()
        for i in zip(LABEL_COLUMNS, output):
            #if i[1] > 0.5:
            print(i)
            #print(f'probability : {prediction}')
    
if __name__ == "__main__":
    main()