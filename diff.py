

f = open('proto.txt', 'r')

lines = f.readlines()


for line in lines:
    #print(line)
    line = line.strip().split(',')
    #print(line)
    line = [x.split(':') for x in line]
    line[0] = [x.strip() for x in line[0]]
    line[1] = [x.strip() for x in line[1]]
    line[2] = [x.strip() for x in line[2]]
    line[3] = [x.strip() for x in line[3]]
    line[4] = [x.strip() for x in line[4]]
    #print(line[0])
    #print(line[1])
    #print(line[2])
    #print(line[3])
    #print(line[4])
    #mul sin inf pred sentence
    if line[0][1] == '1' and line[1][1] == '1' and line[2][1] == '0' and line[3][1] == '1':
        print(line[4][1])


    
    """if문 달아서 print"""