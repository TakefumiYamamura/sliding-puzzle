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
	vector<vector<int> > md;
	int limit;

public:
	Npuzzle(string output_file, int _limit);
	void rand_proceed();
	int get_md_sum(Node n);
	void set_md();
};


Npuzzle::Npuzzle(string output_file, int _limit) {
	writing_file.open(output_file, std::ios::out);
	limit = _limit;
	md = vector<vector<int> >(N2, vector<int>(N2, 0));
	set_md();
}

void Npuzzle::rand_proceed() {
	Node cur;
	for (int i = 0; i < N2; ++i)
	{
		cur.puzzle[i/N][i%N] = i;
	}
	cur.space = pair<int, int>(0, 0);
	for (int i = 0; i < 150 || get_md_sum(cur) < limit; ++i)
	{
		int rnd = rand() % 4;
		pair<int, int> next = pair<int, int>(cur.space.first + dx[rnd], cur.space.second + dy[rnd]);
		if(next.first < 0 || next.first >= N) continue;
		if(next.second < 0 || next.second >= N) continue;
		swap(cur.puzzle[cur.space.first][cur.space.second], cur.puzzle[next.first][next.second]);
		cur.space = next;
	}
	cout << get_md_sum(cur) << " " << limit << endl;
	for (int i = 0; i < N2; ++i)
	{
		writing_file << cur.puzzle[i/N][i%N] << " ";
	}
	writing_file << endl;
	writing_file.close();
}
void Npuzzle::set_md() {
    for (int i = 0; i < N2; ++i)
    {
        for (int j = 0; j < N2; ++j)
        {
            md[i][j] = abs(i / N - j / N) + abs(i % N - j % N);
        }
    }
}

int Npuzzle::get_md_sum(Node n) {
    int sum = 0;
    for (int i = 0; i < N; ++i)
    {
    	for (int j = 0; j < N; ++j)
    	{
    		if(n.puzzle[i][j] == 0) continue;
    		sum += md[i * N + j][n.puzzle[i][j]];
    	}

    }
    return sum;
}


int main() {
	for (int i = 0; i <= 50; ++i)
	{
		string output_file = "../benchmarks/yama24_50_hard/prob";
		if(i < 10) {
			output_file += "00";
		} else if(i < 100) {
			output_file += "0";
		}
		output_file += to_string(i);

		cout << output_file << endl;

		Npuzzle np = Npuzzle(output_file, 35 + i/5);
		np.rand_proceed();
	}
}

