import os

PATH = "result.txt"
MOD = "loss"

f = open(PATH, 'r')

f.readline()

lines = f.readlines()

val_temp = []
train_temp = []

for idx ,line in enumerate(lines):
    data = []
    data = line.split()
    if idx % 2 == 0: # 짝수 번째 acc prec 같은 값 나오는 경우
        #print(data[4])
        if "VAL" in data[2]:
            val_temp.append(data[4])
        else:
            train_temp.append(data[4])
    else:
        if idx % 4 == 1:
            #print(data)
            val_temp.extend([data[2], data[6], data[10], data[14]])
            #print(val_temp)
        else:
            train_temp.extend([data[2], data[6], data[10], data[14]])
            #print(val_temp)
    #print(data)

#print(val_temp)
#print(train_temp)
f.close()

val=[]
train=[]
temp={}

for idx, number in enumerate(val_temp):
    if idx % 5 == 0:
        temp['val'] = number
    elif idx % 5 == 1:
        temp['acc'] = number
    elif idx % 5 == 2:
        temp['prec'] = number
    elif idx % 5 == 3:
        temp['rec'] = number
    else:
        temp['f1'] = number
        val.append(temp)
        temp={}

#print(val)

for idx, number in enumerate(train_temp):
    if idx % 5 == 0:
        temp['val'] = number
    elif idx % 5 == 1:
        temp['acc'] = number
    elif idx % 5 == 2:
        temp['prec'] = number
    elif idx % 5 == 3:
        temp['rec'] = number
    else:
        temp['f1'] = number
        train.append(temp)
        temp={}

#print(val, train)

# 다 string 이라서 float 으로 변환 요구 된다.

temp = [float(i['val']) for i in val]
temp_2 = [float(i['val']) for i in val]