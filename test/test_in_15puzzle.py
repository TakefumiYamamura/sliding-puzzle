import commands
import os

def simple_test(exec_file_name):
    ans_hash = {}
    for line in open('result_15puzzle.txt', 'r'):
        array = line.split()
        if len(array) == 0:
            break
        ans_hash[array[0]] = array[1]


    flag = True
    results = commands.getoutput(exec_file_name)
    outputs = results.split('\n');
    for res in outputs:
        array = res.split()
        if len(array) != 2:
            break

        if ans_hash[array[0]] != array[1]:
            flag = False
            print("answer is different in " + array[0] + " true ans is " + ans_hash[array[0]] + " : false ans is " + array[1])
        else:
            print("answer is same in " + array[0] + " " + ans_hash[array[0]])


    if flag == True :
        print(exec_file_name + " this solver is valid")
        # print(outputs)
    print("")

os.system("g++ -std='c++14' -O3 -o ../src/15puzzle_speed ../src/15puzzle_speed.cc")
simple_test("../src/./15puzzle_speed")
os.system("nvcc -std='c++14' -O3 -o ../src/15puzzle_psimple ../src/15puzzle_psimple.cu")
simple_test("../src/./15puzzle_psimple")
