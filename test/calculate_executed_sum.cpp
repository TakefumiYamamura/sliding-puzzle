#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <time.h>

#include <cmath>
#include <algorithm>
#include <string>
#include <set>
#include <climits>

using namespace std;

int main(int argc, char* argv[]){
	if(argc != 3) {
		cout<< "usage: " << argv[0] << " <filename>," << argv[1] << " <the number of tests>" << endl;
		return 0;
	}
	string input_file = string(argv[1]);
	// string input_file = "../result/yama24_med_result_pdb_wo_cuda.csv";
	// string input_file = "../result/korf100_result9.csv";
	// string input_file = "../result/korf100_horie_solver.csv";
	// string input_file = "../result/korf100_burns_solver.csv";
	ifstream ifs(input_file);
	double sum = 0;
	int nums = stoi(string(argv[2]) );
	for (int i = 0; i < nums; ++i)
	{
		double tmp;
		ifs >> tmp;
		sum += tmp;
	}
	cout << sum << endl;
}