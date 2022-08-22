from pickle import FALSE
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
import multi_task_model
from glob import glob

#print("how on earth")

# loop = True;

# while loop: 
#   sentence = input("하고싶은 말을 입력해주세요 : ") 
#   if sentence == 0: 
#     break;
#   print(infer(sentence))


MODEL_PATH = "./checkpoint/multi_kcbert-l_0.000005_0/epoch=1-val_loss=0.35.ckpt"
#HPARAMS_PATH = "./lightning_logs/version_2/hparams.yaml"

latest_ckpt = sorted(glob(MODEL_PATH))[0]
#model = trainning.Model.load_from_checkpoint(latest_ckpt, hparams_file=HPARAMS_PATH)

model = multi_task_model.Model.load_from_checkpoint(latest_ckpt)

model.eval()
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
        test_prediction, _ = infer(sentence)
        #print(test_prediction)
        output = torch.sigmoid(test_prediction[1])
        output = output.detach().flatten().numpy()
        for i in zip(LABEL_COLUMNS, output):
            if i[1] > 0.5:
                #print(f"{i[0]} {i[1]}에 해당합니다")
                return 1
            #print(f'probability : {prediction}')
        return 0
    
if __name__ == "__main__":
    #print("workign?")
    main()