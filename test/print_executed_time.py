import commands
import os

os.system("g++ -std='c++11' -O3 -o calculate_executed_sum calculate_executed_sum.cpp")

print("cpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_result_speed_100.csv 100")
print("cpu option in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_result8.csv 100")
print("cpu expand in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_result_expand_100.csv 100")

print("cpu horie option in 15 puzzle")
os.system("./calculate_executed_sum ../result/idas_cpu.txt 100")
print("cpu expand horie option in 15 puzzle")
os.system("./calculate_executed_sum ../result/idas_cpu_expand.txt 100")
print("cpu expand option in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_result_expand_with_option100.csv 100")
print("psimple gpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_psimple_result_50.csv 50")

print("psimple gpu 99 in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_psimple_result_99.csv 99")
print("psimple shared gpu in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_psimple_result_50_shared.csv 50")
print("BPIDA* in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_block_parallel_result_with_staticlb_dfs_100_2048.csv 100")
# print("BPIDA* in 15 puzzle")
# os.system("./calculate_executed_sum ../result/korf100_block_parallel_result_with_staticlb_100_2048.csv 100")

print("BPIDA* in 15 puzzle update")
os.system("./calculate_executed_sum ../result/korf100_block_parallel_result_with_staticlb_100_2048.csv 100")

print("BPIDA* all option true in 15 puzzle")
os.system("./calculate_executed_sum ../result/korf100_block_parallel_result_with_staticlb_100_2048_all_true.csv 100")

# print("BPIDA* all option in 15 puzzle")
# os.system("./calculate_executed_sum ../result/korf100_block_parallel_result_with_staticlb_100_2048_all.csv 100")

# print("horie BPIDA* all in 15 puzzle")
# os.system("./calculate_executed_sum ../result/idas_smem.txt 100")

print("horie BPIDA* all (best version) in 15 puzzle")
os.system("./calculate_executed_sum ../result/idas_bestall.txt 100")

print("horie BPIDA* finding one solution(?) in 15 puzzle")
os.system("./calculate_executed_sum ../result/idas_best.txt 100")

print("------------------------------------")


print("cpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_result.csv 50")
print("cpu horie in 24 puzzle")
os.system("./calculate_executed_sum ../result/idas_cpu_25.txt 50")
print("cpu expand in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_expand_result.csv 50")
print("cpu horie expand in 24 puzzle")
os.system("./calculate_executed_sum ../result/idas_cpu_25_expand.txt 50")
print("psimple gpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_psimple_result.csv 50")
print("BPIDA* in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_block_parallel_result_with_staticlb_100_2048.csv 50")
print("BPIDA* dfs in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_block_parallel_result_with_staticlb_dfs_100_2048.csv 50")
print("BPIDA* dfs global in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_block_parallel_result_with_staticlb_dfs_100_2048_global.csv 50")

print("cpu with pdb in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_result_pdb_wo_cuda.csv 50")
print("cpu expand with pdb in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_result_pdb_expand.csv 50")
print("psimple with pdb gpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_psimple_with_pdb_result.csv 50")
print("block parallel with pdb gpu in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_block_parallel_result_with_pdb_2048.csv 50")
print("block parallel with pdb gpu dfs in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_block_parallel_result_with_pdb_2048_dfs.csv 50")
print("block parallel with pdb gpu dfs global in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_med_block_parallel_result_with_pdb_2048_global.csv 50")

print("all hard problems ------------------------------------ ")

print("cpu expand in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_hard_new_expand_option_result.csv 50")
print("BPIDA* in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048.csv 50")
print("BPIDA* in 24 puzzle")
os.system("./calculate_executed_sum ../result/yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global.csv 50")




