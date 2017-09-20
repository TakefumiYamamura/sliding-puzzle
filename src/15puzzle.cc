#include <iostream>
#include <assert.h>
#include <vector>
 
#include <cmath>
#include <algorithm>
#include <string>
#include <set>
#include <climits>
#include <stack>
 
#define N 4
#define N2 16
 
using namespace std;
 
// const int N = 4;
// const int N2 = 16;
 
static const int dx[4] = {0, -1, 0, 1};
static const int dy[4] = {1, 0, -1, 0};
static const char dir[4] = {'r', 'u', 'l', 'd'}; 
 
 
struct Node
{
    int puzzle[N2];
    int space;
    int md;
    int depth;
    int pre;
};
 
class Npuzzle
{
private:
    Node s_node;
    Node cur_n;
    int md[N2][N2];
    int ans;
public:
    Npuzzle() {
        s_node = Node();
        int in[N2];
        for (int i = 0; i < N2; ++i)
        {
            int tmp;
            cin >> tmp;
            if(tmp == 0) {
                tmp = N2;
                s_node.space = i;
            }
            s_node.puzzle[i] = tmp;
        }
        set_md();
        s_node.md = get_md_sum(s_node.puzzle);
        s_node.depth = 0;
        s_node.pre = -10;
    }
 
    int get_md_sum(int *puzzle) {
        int sum = 0;
        for (int i = 0; i < N2; ++i)
        {
            if(puzzle[i] == N2) continue;
            sum += md[i][puzzle[i]-1];
        }
        return sum;
    }
 
    void set_md() {
        for (int i = 0; i < N2; ++i)
        {
            for (int j = 0; j < N2; ++j)
            {
                md[i][j] = abs(i / N - j / N) + abs(i % N - j % N);
            }
        }
    }

    bool dfs(int limit) {
        stack<Node> st;
        st.push(s_node);

        while(!st.empty()) {
            Node cur_n = st.top();
            st.pop();
            if(cur_n.md == 0 ) {
                ans = cur_n.depth;
                return true;
            }
            if(cur_n.depth + cur_n.md > limit) continue;
            int s_x = cur_n.space / N;
            int s_y = cur_n.space % N;
            for (int i = 0; i < 4; ++i)
            {
                Node next_n = cur_n;
                int new_x = s_x + dx[i];
                int new_y = s_y + dy[i];
                if(new_x < 0  || new_y < 0 || new_x >= N || new_y >= N) continue; 
                if(max(cur_n.pre, i) - min(cur_n.pre, i) == 2) continue;
     
                //incremental manhattan distance
                next_n.md -= md[new_x * N + new_y][next_n.puzzle[new_x * N + new_y] - 1];
                next_n.md += md[s_x * N + s_y][next_n.puzzle[new_x * N + new_y] - 1];
     
                swap(next_n.puzzle[new_x * N + new_y], next_n.puzzle[s_x * N + s_y]);
                next_n.space = new_x * N + new_y;
                // assert(get_md_sum(new_n.puzzle) == new_n.md);
                // return dfs(new_n, depth+1, i);
                next_n.depth++;
                if(cur_n.depth + cur_n.md > limit) continue;
                next_n.pre = i;
                st.push(next_n);
                if(next_n.md == 0) {
                    ans = next_n.depth;
                    return true;
                }
            }
        }
        return false;
    }

    void ida_star() {
        for (int limit = s_node.md; limit < 1000; ++limit, ++limit)
        {
            // path.resize(limit);
            cur_n = s_node;
            if(dfs(limit)) {
                cout << limit << endl;
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
