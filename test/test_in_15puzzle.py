import commands

ans_hash = {}
for line in open('result_15puzzle.txt', 'r'):
    array = line.split()
    if len(array) == 0:
        break
    ans_hash[array[0]] = array[1]


flag = True
results = commands.getoutput("../src/./a.out")
outputs = results.split('\n');
for res in outputs:
    array = res.split()
    if len(array) != 2:
        break

    if ans_hash[array[0]] != array[1]:
        flag = False
        print("answer is different in " + array[0] + "true ans is " + ans_hash[array[0]] + " : false ans is " + array[1])

if flag == True :
    print("this solver is valid")

