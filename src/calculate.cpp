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

int main(){
	string input_file = "../result/korf100_result9.csv";
	// string input_file = "../result/korf100_horie_solver.csv";
	// string input_file = "../result/korf100_burns_solver.csv";
	ifstream ifs(input_file);
	double sum = 0;
	for (int i = 0; i < 100; ++i)
	{
		double tmp;
		ifs >> tmp;
		sum += tmp;
	}
	cout << sum << endl;
}