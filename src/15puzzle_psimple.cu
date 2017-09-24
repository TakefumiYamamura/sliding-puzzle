#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <queue>
#include <fstream>
#include <time.h>
 
#include <cmath>
#include <algorithm>
#include <string>
#include <set>
#include <climits>
#include <stack>
#include <sstream>      // std::ostringstream
template <typename T> std::string tostr(const T& t)
{
    std::ostringstream os; os<<t; return os.str();
}
 
#define N 4
#define N2 16
#define STACK_LIMIT 100
// #define CORE_NUM 1
#define CORE_NUM 32
// 1536
using namespace std;
 
// const int N = 4;
// const int N2 = 16;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
 
static const int dx[4] = {0, -1, 0, 1};
static const int dy[4] = {1, 0, -1, 0};
// static const char dir[4] = {'r', 'u', 'l', 'd'}; 
static const int order[4] = {1, 0, 2, 3}; 
 
struct Node
{
    int puzzle[N2];
    int space;
    int md;
    int depth;
    int pre;
    bool operator < (const Node& n) const {
        return depth + md < n.depth + n.md;
    }

    bool operator > (const Node& n) const {
        return depth + md > n.depth + n.md;
    }

};
 

Node s_node;
int md[N2][N2];
int ans;
priority_queue<Node, vector<Node>, greater<Node> > pq;

// struct node stack_nodes[STACK_LIMIT];


int get_md_sum(int *puzzle) {
    int sum = 0;
    for (int i = 0; i < N2; ++i)
    {
        if(puzzle[i] == 0) continue;
        sum += md[i][puzzle[i]];
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

void input_table(char *input_file) {
    s_node = Node();
    fstream ifs(input_file);

    for (int i = 0; i < N2; ++i)
    {
        int tmp;
        // scanf("%d", &tmp);
        ifs >> tmp;
        // cin >> tmp;
        if(tmp == 0) {
            s_node.space = i;
        }
        s_node.puzzle[i] = tmp;
    }
    s_node.md = get_md_sum(s_node.puzzle);
    s_node.depth = 0;
    s_node.pre = -10;
}



bool dfs(int limit, Node s_n) {
    stack<Node> st;
    st.push(s_n);

    while(!st.empty()) {
        Node cur_n = st.top();
        st.pop();
        if(cur_n.md == 0 ) {
            ans = cur_n.depth;
            return true;
        }
        int s_x = cur_n.space / N;
        int s_y = cur_n.space % N;
        for (int operator_order = 0; operator_order < 4; ++operator_order)
        {
            int i = order[operator_order];
            Node next_n = cur_n;
            int new_x = s_x + dx[i];
            int new_y = s_y + dy[i];
            if(new_x < 0  || new_y < 0 || new_x >= N || new_y >= N) continue; 
            if(max(cur_n.pre, i) - min(cur_n.pre, i) == 2) continue;
 
            //incremental manhattan distance
            next_n.md -= md[new_x * N + new_y][next_n.puzzle[new_x * N + new_y]];
            next_n.md += md[s_x * N + s_y][next_n.puzzle[new_x * N + new_y]];
 
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

bool create_root_set() {
    pq.push(s_node);
    while(pq.size() < CORE_NUM) {
        Node cur_n = pq.top();
        pq.pop();
        if(cur_n.md == 0 ) {
            ans = cur_n.depth;
            return true;
        }
        int s_x = cur_n.space / N;
        int s_y = cur_n.space % N;
        for (int operator_order = 0; operator_order < 4; ++operator_order)
        {
            int i = order[operator_order];
            Node next_n = cur_n;
            int new_x = s_x + dx[i];
            int new_y = s_y + dy[i];
            if(new_x < 0  || new_y < 0 || new_x >= N || new_y >= N) continue; 
            if(max(cur_n.pre, i) - min(cur_n.pre, i) == 2) continue;
 
            //incremental manhattan distance
            next_n.md -= md[new_x * N + new_y][next_n.puzzle[new_x * N + new_y]];
            next_n.md += md[s_x * N + s_y][next_n.puzzle[new_x * N + new_y]];
 
            swap(next_n.puzzle[new_x * N + new_y], next_n.puzzle[s_x * N + s_y]);
            next_n.space = new_x * N + new_y;
            // assert(get_md_sum(new_n.puzzle) == new_n.md);
            // return dfs(new_n, depth+1, i);
            next_n.depth++;
            next_n.pre = i;
            if(next_n.md == 0) {
                ans = next_n.depth;
                return true;
            }
            pq.push(next_n);
            if(pq.size() >= CORE_NUM){
                return false;
            }
        }
    }
    return false;
}

__global__ void dfs_kernel(int limit, Node *root_set, bool *dev_flag) {
    int tid = blockIdx.x;
    if(tid % 3 == 0) {
        dev_flag[tid] = false;
    } else {
        dev_flag[tid] = true;
    }

}

void ida_star() {
    if(create_root_set()) {
        printf("%d\n", ans);
        return;
    }
    int pq_size = pq.size();
    Node root_set[CORE_NUM];
    int i = 0;
    while(!pq.empty()) {
        Node n = pq.top();
        pq.pop();
        root_set[i] = n;
        i++;
    }


    //gpu側で使う根集合のポインタ
    Node *dev_root_set;
    //gpu側のメモリ割当て
    HANDLE_ERROR(cudaMalloc((void**)&dev_root_set, pq_size * sizeof(Node) ) );
    //root_setをGPU側のdev_root_setにコピー
    HANDLE_ERROR(cudaMemcpy(dev_root_set, root_set, pq_size * sizeof(Node), cudaMemcpyHostToDevice) );

    for (int limit = s_node.md; limit < 5; ++limit, ++limit)
    {
        // path.resize(limit);
        // priority_queue<Node, vector<Node>, greater<Node> > tmp_pq = pq;

        bool flag[CORE_NUM];
        bool *dev_flag;

        //gpu側にメモリ割当
        HANDLE_ERROR(cudaMalloc( (void**)&dev_flag, pq_size * sizeof(bool) ) );
        dfs_kernel<<<CORE_NUM, 1>>>(limit, dev_root_set, dev_flag);
        HANDLE_ERROR(cudaMemcpy(flag, dev_flag, CORE_NUM * sizeof(bool), cudaMemcpyDeviceToHost));
        for (int i = 0; i < CORE_NUM; ++i)
        {
            cout << dev_flag[i] << " ";
        }
        cout << endl;


        // while(!tmp_pq.empty()) {
        //     Node n = tmp_pq.top();
        //     tmp_pq.pop();
        //     if(dfs(limit, n)) {
        //         printf("%d\n", limit);
        //         return;
        //     }
        // } 
        HANDLE_ERROR(cudaFree(dev_flag) );
    }
    HANDLE_ERROR(cudaFree(dev_root_set));
    cout << "test" << endl;
}

 
int main() {
    string output_file = "../result/korf100_psimple_result.csv";
    // ofstream writing_file;
    // writing_file.open(output_file, std::ios::out);
    for (int i = 0; i < 1; ++i)
    {
        string input_file = "../benchmarks/korf100/prob";
        if(i < 10) {
            input_file += "00";
        } else if(i < 100) {
            input_file += "0";
        }
        input_file += tostr(i);
        cout << input_file << endl;
        set_md();

        clock_t start = clock();

        input_table(const_cast<char*>(input_file.c_str()));
        ida_star();

        clock_t end = clock();

        // writing_file << (double)(end - start) / CLOCKS_PER_SEC << endl;
    }
}
