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


using namespace std;


int main() {
	ofstream writing_file;
	writing_file.open("results.tex", std::ios::out);

	ifstream ifs_cpu("yama24_hard_new_expand_option_result.csv");
	ifstream ifs_bpida("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048.csv");
	ifstream ifs_bpida_global("yama24_hard_new_block_parallel_result_with_staticlb_dfs_100_2048_global.csv");

	for (int i = 0; i < 50; ++i) {

		writing_file << i << " & ";

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
	}
	ifs_cpu.close();
	ifs_bpida.close();
	ifs_bpida_global.close();
	writing_file.close();
}

