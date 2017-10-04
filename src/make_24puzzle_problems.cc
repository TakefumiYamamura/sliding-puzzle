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

#define N 5
#define N2 25

using namespace std;

// const int N = 4;
// const int N2 = 16;

static const int dx[4] = {0, -1, 0, 1};
static const int dy[4] = {1, 0, -1, 0};


struct Node
{
	int puzzle[N][N];
	pair<int, int> space;
};

class Npuzzle
{
private:
	ofstream writing_file;

public:
	Npuzzle(string output_file);
	void rand_proceed();
};


Npuzzle::Npuzzle(string output_file) {
	writing_file.open(output_file, std::ios::out);
}

void Npuzzle::rand_proceed() {
	Node cur;
	for (int i = 0; i < N2; ++i)
	{
		cur.puzzle[i/N][i%N] = i;
	}
	cur.space = pair<int, int>(0, 0);
	for (int i = 0; i < 70; ++i)
	{
		int rnd = rand() % 4;
		pair<int, int> next = pair<int, int>(cur.space.first + dx[rnd], cur.space.second + dy[rnd]);
		if(next.first < 0 || next.first >= N) continue;
		if(next.second < 0 || next.second >= N) continue;
		swap(cur.puzzle[cur.space.first][cur.space.second], cur.puzzle[next.first][next.second]);
		cur.space = next;
	}
	for (int i = 0; i < N2; ++i)
	{
		writing_file << cur.puzzle[i/N][i%N] << " ";
	}
	writing_file << endl;
	writing_file.close();
}


int main() {
	for (int i = 1; i <= 50; ++i)
	{
		string output_file = "../benchmarks/yama24_50/prob";
		if(i < 10) {
			output_file += "00";
		} else if(i < 100) {
			output_file += "0";
		}
		output_file += to_string(i);

		cout << output_file << endl;

		Npuzzle np = Npuzzle(output_file);
		np.rand_proceed();
	}
}

