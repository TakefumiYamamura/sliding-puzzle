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
        if len(array) != 2 and len(array) != 3:
            continue

        if ans_hash[array[0]] != array[1]:
            flag = False
            print("answer is different in " + array[0] + " true ans is " + ans_hash[array[0]] + " : false ans is " + array[1])
        else:
            print("answer is same in " + array[0] + " " + ans_hash[array[0]])


    if flag == True :
        print(exec_file_name + " this solver is valid")
        # print(outputs)
    print("")

# os.system("g++ -std='c++11' -o ../src/15puzzle_speed ../src/15puzzle_speed.cc")
# simple_test("../src/./15puzzle_speed")
# os.system("g++ -std='c++11' -o ../src/15puzzle_expand ../src/15puzzle_expand.cc")
# simple_test("../src/./15puzzle_expand")
# os.system("/usr/local/cuda/bin/nvcc -std='c++11' -O3 -o ../src/15puzzle_psimple ../src/15puzzle_psimple.cu")
# simple_test("../src/./15puzzle_psimple")
# os.system("/usr/local/cuda/bin/nvcc -std='c++11' -O3 -o ../src/15puzzle_psimple_shared ../src/15puzzle_psimple_shared.cu")
# simple_test("../src/./15puzzle_psimple_shared")
# os.system("/usr/local/cuda/bin/nvcc -std='c++11' -o ../src/15puzzle_block_parallel ../src/15puzzle_block_parallel.cu")
# simple_test("../src/./15puzzle_block_parallel")

os.system("g++ -std='c++11' -O3 -o calculate_executed_sum calculate_executed_sum.cpp")

print("cpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_result_speed_100.csv 100")
print("cpu expand in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_result_expand_100.csv 100")
print("psimple gpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_psimple_result_50.csv 50")
print("psimple shared gpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_psimple_result_50_shared.csv 50")
print("block parallel gpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_block_parallel_result_with_staticlb_dfs_100_2048.csv 100")
print("block parallel gpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_block_parallel_result_with_staticlb_100_2048.csv 100")

print("horie block parallel gpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/idas_smem.txt 100")



