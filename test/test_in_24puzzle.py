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

os.system("g++ -std='c++11' -O3 -o ../src/24puzzle ../src/24puzzle.cc")
simple_test("../src/./24puzzle")
os.system("g++ -std='c++11' -O3 -o ../src/24puzzle_with_pdb ../src/24puzzle_with_pdb.cc")
simple_test("../src/./24puzzle_with_pdb")

os.system("/usr/local/cuda/bin/nvcc -std='c++11' -O3 -o ../src/24puzzle_psimple ../src/24puzzle_psimple.cu")
simple_test("../src/./24puzzle_psimple")
os.system("/usr/local/cuda/bin/nvcc -std='c++11' -O3 -o ../src/24puzzle_psimple_with_pdb ../src/24puzzle_psimple_with_pdb.cu")
simple_test("../src/./24puzzle_psimple_with_pdb")
