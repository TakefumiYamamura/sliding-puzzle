import commands
import os


os.system("g++ -std='c++11' -O3 -o calculate_executed_sum calculate_executed_sum.cpp")

print("cpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_result_speed.csv 50")
print("psimple gpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_psimple_result_50.csv 50")
print("\n")

print("cpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_result.csv 50")
print("psimple gpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_psimple_result.csv 50")
print("\n")

print("cpu with pdb in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_result_pdb_wo_cuda.csv 50")
print("psimple with pdb gpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_psimple_with_pdb_result.csv 50")
