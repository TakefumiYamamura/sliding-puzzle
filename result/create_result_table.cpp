#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <queue>
#include <fstream>
#include <time.h>
#include <string>
 
#include <cmath>
#include <algorithm>
#include <string>
#include <set>
#include <climits>
#include <stack>
#include <sstream>
#include <chrono>

// #define RESULT1
// #define RESULT2
// #define RESULT3
#define RESULT4


using namespace std;


int main() {
	ofstream writing_file;
	writing_file.open("results.tex", std::ios::out);

	#ifdef RESULT1
	ifstream ifs_cpu("yama24_hard_new_expand_option_result.csv");
	ifstream ifs_bpida("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048.csv");
	ifstream ifs_bpida_global("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global.csv");
	#endif

	#ifdef RESULT2
	ifstream ifs_4("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_152.csv");
	ifstream ifs_9("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_304.csv");
	ifstream ifs_13("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_456.csv");
	ifstream ifs_18("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_608.csv");
	ifstream ifs_36("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global_1216.csv");
	#endif

	#ifdef RESULT3
	ifstream ifs_cpu_pdb("yama24_hard_new_result_pdb_expand.csv");
	ifstream ifs_bpida_pdb("yama24_hard_new_block_parallel_result_with_pdb_2048_dfs.csv");
	ifstream ifs_bpida_global_pdb("yama24_hard_new_block_parallel_result_with_pdb_2048_global.csv");
	#endif

	#ifdef RESULT4
	ifstream ifs_cpu("yama24_hard_new_expand_option_result.csv");
	ifstream ifs_bpida("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048.csv");
	ifstream ifs_bpida_global("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global.csv");
	ifstream ifs_cpu_pdb("yama24_hard_new_result_pdb_expand.csv");
	ifstream ifs_bpida_pdb("yama24_hard_new_block_parallel_result_with_pdb_2048_dfs.csv");
	ifstream ifs_bpida_global_pdb("yama24_hard_new_block_parallel_result_with_pdb_2048_global.csv");
	#endif


	for (int i = 0; i < 50; ++i) {

		writing_file << i << " & ";

		#ifdef RESULT1
		{
			string tmp;
			ifs_cpu >> tmp;
			writing_file << tmp << " & ";
		}

		{
			string tmp;
			ifs_bpida >> tmp;
			writing_file << tmp << " & ";
		}

		{
			string tmp;
			ifs_bpida_global >> tmp;
			writing_file << tmp << " \\\\" << endl;
		}
		#endif

		#ifdef RESULT2
		{
			string tmp;
			ifs_4 >> tmp;
			writing_file << tmp << " & ";
		}

		{
			string tmp;
			ifs_9 >> tmp;
			writing_file << tmp << " & ";
		}

		{
			string tmp;
			ifs_13 >> tmp;
			writing_file << tmp << " & ";
		}

		{
			string tmp;
			ifs_18 >> tmp;
			writing_file << tmp << " & ";
		}

		{
			string tmp;
			ifs_36 >> tmp;
			writing_file << tmp << " \\\\" << endl;
		}
		#endif

		#ifdef RESULT3
		{
			string tmp;
			ifs_cpu_pdb >> tmp;
			writing_file << tmp << " & ";
		}

		{
			string tmp;
			ifs_bpida_pdb >> tmp;
			writing_file << tmp << " & ";
		}

		{
			string tmp;
			ifs_bpida_global_pdb >> tmp;
			writing_file << tmp << " \\\\" << endl;
		}
		#endif

		#ifdef RESULT4
		{
			double tmp1, tmp2;
			ifs_cpu >> tmp1;
			ifs_cpu_pdb >> tmp2;
			writing_file << tmp1 / tmp2 << " & ";
		}

		{
			double tmp1, tmp2;
			ifs_bpida >> tmp1;
			ifs_bpida_pdb >> tmp2;
			writing_file << tmp1 / tmp2 << " & ";
		}

		{
			double tmp1, tmp2;
			ifs_bpida_global >> tmp1;
			ifs_bpida_global_pdb >> tmp2;
			writing_file << tmp1 / tmp2 << " \\\\" << endl;
		}
		#endif
	}
	#ifdef RESULT1
	ifs_cpu.close();
	ifs_bpida.close();
	ifs_bpida_global.close();
	#endif

	#ifdef RESULT2
	ifs_4.close();
	ifs_9.close();
	ifs_13.close();
	ifs_18.close();
	ifs_36.close();
	#endif

	#ifdef RESULT3
	ifs_cpu_pdb.close();
	ifs_bpida_pdb.close();
	ifs_bpida_global_pdb.close();
	#endif

	#ifdef RESULT4
	ifs_cpu.close();
	ifs_bpida.close();
	ifs_bpida_global.close();
	ifs_cpu_pdb.close();
	ifs_bpida_pdb.close();
	ifs_bpida_global_pdb.close();
	#endif

	writing_file.close();
}

