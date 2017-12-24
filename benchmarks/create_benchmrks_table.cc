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
	writing_file.open("problems.tex", std::ios::out);

	ifstream ifs_mnh("manhattan_expand_nodes.csv");
	ifstream ifs_pdb("pdb_expand_nodes.csv");
	for (int i = 0; i < 50; ++i) {

		string input_file = "../benchmarks/yama24_50_hard_new/prob";

		if(i < 10) {
			input_file += "00";
		} else if(i < 100) {
			input_file += "0";
		}
		ifstream ifs(input_file + to_string(i));


		writing_file << i << " & ";


		for (int j = 0; j < 25; ++j)
		{
			int tmp;
			ifs >> tmp;
			writing_file << tmp << " ";
		}
		writing_file << " & ";

		{
			string tmp1, tmp2, tmp3;
			ifs_mnh >> tmp1 >> tmp2 >> tmp3;
			writing_file << tmp2 << " & " << tmp3 << " & ";
		}

		{
			string tmp1, tmp2, tmp3;
			ifs_pdb >> tmp1 >> tmp2 >> tmp3;
			writing_file << tmp3 << " \\\\" << endl;
		}
		ifs.close();
	}
	ifs_mnh.close();
	ifs_pdb.close();
	writing_file.close();

}

