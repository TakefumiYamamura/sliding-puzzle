#include <iostream>
#include <vector>
#include <assert.h>
#include <map>
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <set>
#include <climits>

using namespace std;

const int N = 4;
const int N2 = 16;

static const int dx[4] = {0, -1, 0, 1};
static const int dy[4] = {1, 0, -1, 0};
static const char dir[4] = {'r', 'u', 'l', 'd'}; 


struct Node
{
	vector<int> puzzle;
	int space;
	int md;
};

class Npuzzle
{
private:
	Node s_node;
	int limit;
	vector<vector<int> > md;
	vector<int> path;
	int ans;
public:
	Npuzzle() {
		s_node = Node();
		vector<int> in;
		for (int i = 0; i < N2; ++i)
		{
			int tmp;
			cin >> tmp;
			if(tmp == 0) {
				tmp = N2;
				s_node.space = i;
			}
			in.push_back(tmp);
		}
		s_node.puzzle = in;
		set_md();
		s_node.md = get_md_sum(s_node.puzzle);
	}

	int get_md_sum(const vector<int>& puzzle) {
		int sum = 0;
		for (int i = 0; i < N2; ++i)
		{
			if(puzzle[i] == N2) continue;
			sum += md[i][puzzle[i]-1];
		}
		return sum;
	}

	void set_md() {
		md = vector<vector<int> >(N2, vector<int>(N2, 0));
		for (int i = 0; i < N2; ++i)
		{
			for (int j = 0; j < N2; ++j)
			{
				md[i][j] = abs(i / N - j / N) + abs(i % N - j % N);
			}
		}
	}

	bool dfs(Node cur_n, int depth, int pre) {
		if(cur_n.md == 0 ) {
			ans = depth;
			return true;
		}
		if(depth + cur_n.md > limit) return false;
		int s_x = cur_n.space / N;
		int s_y = cur_n.space % N;
		for (int i = 0; i < 4; ++i)
		{
			Node new_n = cur_n;
			int new_x = s_x + dx[i];
			int new_y = s_y + dy[i];
			if(new_x < 0  || new_y < 0 || new_x >= N || new_y >= N) continue; 
			if(max(pre, i) - min(pre, i) == 2) continue;

			new_n.md -= md[new_x * N + new_y][new_n.puzzle[new_x * N + new_y] - 1];
			new_n.md += md[s_x * N + s_y][new_n.puzzle[new_x * N + new_y] - 1];
			swap(new_n.puzzle[new_x * N + new_y], new_n.puzzle[s_x * N + s_y]);
			new_n.space = new_x * N + new_y;
			// assert(get_md_sum(new_n.puzzle) == new_n.md);
			// return dfs(new_n, depth+1, i);
			if(dfs(new_n, depth + 1, i)){
				path[depth] = i;
				return true;
			}
		}
		return false;
	}

	void ida_star() {
		for (limit = s_node.md; limit < 1000; ++limit)
		{
			path.resize(limit);
			if(dfs(s_node, 0, -10)) {
				cout << limit << endl;
				string str = "";
				for (int i = 0; i < limit; ++i)
				{
					str += dir[path[i]];
				}
				cout << str << endl;
				return;
			}
		}
	}

	void exec() {
		ida_star();
	}
};


int main() {
	Npuzzle np = Npuzzle();
	np.exec(); 
}

