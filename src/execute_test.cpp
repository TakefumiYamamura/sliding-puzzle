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
#include <stdio.h>
#include <stdlib.h>

using namespace std;

int main (){
    // string output_file = "../result/korf100_horie_solver.csv";
    fstream writing_file;
	// writing_file.open(output_file, std::ios::out);
    for (int i = 0; i < 100; ++i)
	{
		string input_file = "../benchmarks/korf100/prob";
		if(i < 10) {
			input_file += "00";
		} else if(i < 100) {
			input_file += "0";
		}
		input_file += to_string(i);
		// cout << input_file << endl;
		// clock_t start = clock();
		ifstream ifs(input_file);
		string cmdline = "./horie.exe";
		cmdline += " " + input_file;
		// cout << cmdline << endl;
		system(cmdline.c_str());
		// clock_t end = clock();
		// writing_file << (double)(end - start) / CLOCKS_PER_SEC << endl;
	}
    return 0;
}
