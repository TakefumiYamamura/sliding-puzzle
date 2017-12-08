import commands
import os

def simple_test(exec_file_name):
	ans_hash = {}
	for line in open('result_24puzzle.txt', 'r'):
		array = line.split()
		if len(array) == 0:
			break
		ans_hash[array[0]] = array[1]


	flag = True
	results = commands.getoutput(exec_file_name)
	print results
	outputs = results.split('\n');
	# print outputs
	for res in outputs:
		# print res
		array = res.split()
		if len(array) != 2 and len(array) != 3:
			continue
		# print res

		if ans_hash[array[0]] != array[1]:
			flag = False
			print("answer is different in " + array[0] + " true ans is " + ans_hash[array[0]] + " : false ans is " + array[1])
		# else:
			# print("answer is same in " + array[0] + " " + ans_hash[array[0]])


	if flag == True:
		print(exec_file_name + " this solver is valid")
	else:
		print(exec_file_name + " invalid !!!")
	print("")

# os.system("g++ -std='c++11' -O3 -o ../src/24puzzle ../src/24puzzle.cc")
# simple_test("../src/./24puzzle")
os.system("g++ -std='c++11' -O3 -o ../src/24puzzle_expand ../src/24puzzle_expand.cc")
simple_test("../src/./24puzzle_expand")
# os.system("g++ -std='c++11' -O3 -o ../src/24puzzle_with_pdb ../src/24puzzle_with_pdb.cc")
# simple_test("../src/./24puzzle_with_pdb")

# os.system("/usr/local/cuda/bin/nvcc -std='c++11' -O3 -o ../src/24puzzle_psimple ../src/24puzzle_psimple.cu")
# simple_test("../src/./24puzzle_psimple")
# os.system("/usr/local/cuda/bin/nvcc -std='c++11' -O3 -o ../src/24puzzle_psimple_with_pdb ../src/24puzzle_psimple_with_pdb.cu")
# simple_test("../src/./24puzzle_psimple_with_pdb")

# os.system("/usr/local/cuda/bin/nvcc -std='c++11' -O3 -o ../src/24puzzle_block_parallel ../src/24puzzle_block_parallel.cu")
# simple_test("../src/./24puzzle_block_parallel")
# os.system("/usr/local/cuda/bin/nvcc -std='c++11' -O3 -o ../src/24puzzle_block_parallel_with_pdb ../src/24puzzle_block_parallel_with_pdb.cu")
# simple_test("../src/./24puzzle_block_parallel_with_pdb")

os.system("g++ -std='c++11' -O3 -o calculate_executed_sum calculate_executed_sum.cpp")
print("cpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_result.csv 50")
print("cpu expand in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_expand_result.csv 50")
print("psimple gpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_psimple_result.csv 50")
print("block parallel gpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_block_parallel_result_with_staticlb_dfs_100_2048.csv 50")


print("cpu with pdb in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_result_pdb_wo_cuda.csv 50")
print("psimple with pdb gpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_psimple_with_pdb_result.csv 50")
print("block parallel with pdb gpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_block_parallel_result_with_pdb_2048.csv 50")


