#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <time.h>
#include <queue>

#include <cmath>
#include <algorithm>
#include <string>
#include <set>
#include <climits>

#define N 5
#define N2 25

using namespace std;


struct Problem
{
    int id;
    double exec;

    bool operator < (const Problem& p) const {
        return exec < p.exec ;
    }
    bool operator > (const Problem& p) const {
        return exec > p.exec ;
    }
};


int main() {
	string input_results = "../result/yama24_hard_expand_result.csv";
	ifstream ifs(input_results);
	priority_queue<Problem, vector<Problem>, greater<Problem> > pq;
	for (int i = 0; i <= 50; ++i)
	{
		double tmp;
		ifs >> tmp;
		Problem tmp_p = {i, tmp};
		pq.push(tmp_p);
	}
	ifs.close();

	vector<Problem> probs; 

	for (int i = 0; i < 50; ++i)
	{
		Problem cur = pq.top();
		pq.pop();
		probs.push_back(cur);
	}
	random_shuffle(probs.begin(), probs.begin() + 5);
	random_shuffle(probs.begin() + 5, probs.begin() + 10);
	random_shuffle(probs.begin() + 10, probs.begin() + 15);
	random_shuffle(probs.begin() + 15, probs.begin() + 20);
	random_shuffle(probs.begin() + 20, probs.begin() + 25);
	random_shuffle(probs.begin() + 35, probs.begin() + 30);
	random_shuffle(probs.begin() + 30, probs.begin() + 35);
	random_shuffle(probs.begin() + 45, probs.begin() + 40);
	random_shuffle(probs.begin() + 40, probs.begin() + 45);
	random_shuffle(probs.begin() + 45, probs.begin() + 48);

	for (int i = 0; i < 50; ++i)
	{
		cout << probs[i].exec << " " << probs[i].id << endl;
	}
	// return 0;



	for (int i = 0; i < 50; ++i)
	{
		Problem cur = probs[i];

		string input_file = "../benchmarks/yama24_50_hard/prob";
		string output_file = "../benchmarks/yama24_50_hard_new/prob";
		ofstream writing_file;
		if(cur.id < 10) {
			input_file += "00";
		} else if(cur.id < 100) {
			input_file += "0";
		}
		ifstream ifs(input_file + to_string(cur.id));
		if(i < 10) {
			output_file += "00";
		} else if(i < 100) {
			output_file += "0";
		}
		writing_file.open(output_file + to_string(i), std::ios::out);

		for (int i = 0; i < N2; ++i)
		{
			int tmp;
			ifs >> tmp;
			writing_file << tmp << " ";
		}
		ifs.close();
		writing_file.close();
	}

}

