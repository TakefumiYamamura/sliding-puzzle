import commands
import os
import time
import sys



def calculate_executed_time(solver_name):
    os.system("/usr/local/cuda/bin/nvcc -o ../src_horie/solver ../src_horie/" + solver_name + ".cu")
    duration_times = []

    for x in xrange(0, 100):
        print(x)
        file_number = ""
        if x < 10:
            file_number =  file_number + "00" + str(x)
        elif x < 100:
            file_number = file_number + "0" + str(x)

        start = time.time()
        os.system("../src_horie/./solver ../benchmarks/korf100/prob" + file_number)
        duration_time = time.time() - start
        print(duration_time)
        duration_times.append(str(duration_time) + "\n")

    f = open("../result/" +  solver_name + "64.txt", 'w')
    f.writelines(duration_times)
    f.close() 


argvs = sys.argv
argc = len(argvs)
i = 1
while i < argc:
    print argvs[i]
    calculate_executed_time(argvs[i])
    i = i + 1

# os.system("g++ -std='c++11' -o ../src/15puzzle_expand ../src/15puzzle_expand.cc")
# simple_test("../src/./15puzzle_expand")

# os.system("g++ -std='c++11' -O3 -o calculate_executed_sum calculate_executed_sum.cpp")
# print("cpu in 15 puzzle")
# os.system("./calculate_executed_sum ../result/korf100_result_speed.csv 50")
# print("psimple gpu in 15 puzzle")
# os.system("./calculate_executed_sum ../result/korf100_psimple_result_50.csv 50")
# print("psimple gpu in 15 puzzle")
# os.system("./calculate_executed_sum ../result/korf100_psimple_result_50_shared.csv 50")
