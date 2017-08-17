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

#define N 4
#define N2 16

using namespace std;

// const int N = 4;
// const int N2 = 16;

static const int dx[4] = {0, -1, 0, 1};
static const int dy[4] = {1, 0, -1, 0};
static const char dir[4] = {'r', 'u', 'l', 'd'};
static const int order[4] = {1, 0, 2, 3};


struct Node
{
	int puzzle[N2];
	int space;
	int md;
};

class Npuzzle
{
private:
	Node s_n;
	Node cur_n;
	int limit;
	int md[N2][N2];
	vector<int> path;
	int ans;
	int node_num;
public:
	Npuzzle();
	Npuzzle(string input_file);
	int get_md_sum(int *puzzle);
	void set_md();
	bool dfs(int depth, int pre);
	void ida_star();
};

Npuzzle::Npuzzle() {
	s_n = Node();
	node_num = 0;
	int in[N2];
	for (int i = 0; i < N2; ++i)
	{
		int tmp;
		cin >> tmp;
		if(tmp == 0) {
			s_n.space = i;
		}
		s_n.puzzle[i] = tmp;
	}
	set_md();
	s_n.md = get_md_sum(s_n.puzzle);
}

Npuzzle::Npuzzle(string input_file) {
	node_num = 0;
	ifstream ifs(input_file);
	int in[N2];
	for (int i = 0; i < N2; ++i)
	{
		int tmp;
		ifs >> tmp;
		if(tmp == 0) {
			s_n.space = i;
		}
		s_n.puzzle[i] = tmp;
	}
	set_md();
	s_n.md = get_md_sum(s_n.puzzle);
}

int Npuzzle::get_md_sum(int *puzzle) {
	int sum = 0;
	for (int i = 0; i < N2; ++i)
	{
		if(puzzle[i] == 0) continue;
		// cout << md[i][puzzle[i]] << " ";
		sum += md[i][puzzle[i]];
	}
	return sum;
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


bool Npuzzle::dfs(int depth, int pre) {
	if(cur_n.md == 0 ) {
		ans = depth;
		return true;
	}
	if(depth + cur_n.md > limit) return false;
	int s_x = cur_n.space / N;
	int s_y = cur_n.space % N;
	for (int operator_order = 0; operator_order < 4; ++operator_order)
	{
		int i = order[operator_order];
		Node tmp_n = cur_n;
		int new_x = s_x + dx[i];
		int new_y = s_y + dy[i];
		if(new_x < 0  || new_y < 0 || new_x >= N || new_y >= N) continue; 
		if(max(pre, i) - min(pre, i) == 2) continue;

		//incremental manhattan distance
		cur_n.md -= md[new_x * N + new_y][cur_n.puzzle[new_x * N + new_y]];
		cur_n.md += md[s_x * N + s_y][cur_n.puzzle[new_x * N + new_y]];

		swap(cur_n.puzzle[new_x * N + new_y], cur_n.puzzle[s_x * N + s_y]);
		cur_n.space = new_x * N + new_y;
		// assert(get_md_sum(cur_n.puzzle) == cur_n.md);
		// return dfs(cur_n, depth+1, i);
		if(dfs(depth + 1, i)){
			// path[depth] = i;
			return true;
		}
		cur_n = tmp_n;
	}
	return false;
}

void Npuzzle::ida_star() {
	for (limit = s_n.md; limit < 1000; ++limit, ++limit)
	{
		// path.resize(limit);
		cur_n = s_n;
		if(dfs(0, -10)) {
			// string str = "";
			// for (int i = 0; i < limit; ++i)
			// {
			// 	str += dir[path[i]];
			// }
			// cout << str << endl;
			return;
		}
	}
}




int main() {
	string output_file = "../result/korf100_result9.csv";
	ofstream writing_file;
	writing_file.open(output_file, std::ios::out);

	for (int i = 0; i < 100; ++i)
	{
		string input_file = "../benchmarks/korf100/prob";
		if(i < 10) {
			input_file += "00";
		} else if(i < 100) {
			input_file += "0";
		}
		input_file += to_string(i);
		cout << input_file << endl;
		clock_t start = clock();
		Npuzzle np = Npuzzle(input_file);
		np.ida_star();
		clock_t end = clock();
		writing_file << (double)(end - start) / CLOCKS_PER_SEC << endl;
	}
}

