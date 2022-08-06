import infer

f = open('1000.txt', "r")
f1 = open('1000-1.txt', "w")

f1.write(f'번호\t문장\t악플\n')
cnt = 1

while True:
    line = f.readline().strip()
    if not line: break
    #print(line)
    print(f"{cnt}")
    toxic = 0
    toxic = infer.judge(line)
    f1.write(f'{cnt}\t{line}\t{toxic}\n')
    cnt+=1