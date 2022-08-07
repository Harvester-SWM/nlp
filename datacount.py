import pandas as pd

LABEL_COLUMNS=["욕설","모욕","폭력위협/범죄조장","외설","성혐오","연령","인종/출신지","장애","종교","정치성향","직업혐오"]
#욕설	모욕	폭력위협/범죄조장	외설	성혐오	연령	인종/출신지	장애	종교	정치성향	직업혐오
FILE = "./data/train.tsv"

chart = pd.read_csv(FILE, sep='\t')

print(chart['욕설'].value_counts())
print(chart['모욕'].value_counts())
print(chart['폭력위협/범죄조장'].value_counts())
print(chart['외설'].value_counts())
print(chart['성혐오'].value_counts())
print(chart['연령'].value_counts())
print(chart['인종/출신지'].value_counts())
print(chart['장애'].value_counts())
print(chart['종교'].value_counts())
print(chart['정치성향'].value_counts())
print(chart['직업혐오'].value_counts())

for i in LABEL_COLUMNS:
    chart[i] = chart[i].map(lambda x : 0 if x == 0 else 1)
    print(i)

#print(chart)

