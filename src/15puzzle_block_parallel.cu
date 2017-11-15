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
#include <sstream>
#include <chrono>

template <typename T> std::string tostr(const T& t)
{
    std::ostringstream os; os<<t; return os.str();
}
 
#define N 4
#define N2 16
#define STACK_LIMIT 64 * 4
// #define CORE_NUM 1536
// #define CORE_NUM 384
#define CORE_NUM 192
// #define WARP_SIZE 8
#define WARP_SIZE 4
#define BLOCK_NUM 48

using namespace std;

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

struct Lock {
    int *mutex;
    Lock( void ) {
        HANDLE_ERROR( cudaMalloc( (void**)&mutex,
                              sizeof(int) ) );
        HANDLE_ERROR( cudaMemset( mutex, 0, sizeof(int) ) );
    }

    ~Lock( void ) {
        cudaFree( mutex );
    }

    __device__ void lock( void ) {
        while( atomicCAS( mutex, 0, 1 ) != 0 );
    __threadfence();
    }

    __device__ void unlock( void ) {
        __threadfence();
        atomicExch( mutex, 0 );
    }
};

Node s_node;
int tmp_md[N2*N2];
__constant__ int md[N2*N2];
int ans;
priority_queue<Node, vector<Node>, greater<Node> > pq;

int get_md_sum(int *puzzle) {
    int sum = 0;
    for (int i = 0; i < N2; ++i)
    {
        if(puzzle[i] == 0) continue;
        sum += tmp_md[i * N2 + puzzle[i]];
    }
    return sum;
}

void set_md() {
    for (int i = 0; i < N2; ++i)
    {
        for (int j = 0; j < N2; ++j)
        {
            tmp_md[i * N2 + j] = abs(i / N - j / N) + abs(i % N - j % N);
        }
    }
    HANDLE_ERROR(cudaMemcpyToSymbol(md, tmp_md, sizeof(int) * N2 * N2));
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
            next_n.md -= tmp_md[(new_x * N + new_y) * N2 + next_n.puzzle[new_x * N + new_y]];
            next_n.md += tmp_md[(s_x * N + s_y) * N2 + next_n.puzzle[new_x * N + new_y]];
 
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
    while(pq.size() < BLOCK_NUM) {
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
            next_n.md -= tmp_md[(new_x * N + new_y) * N2 + next_n.puzzle[new_x * N + new_y]];
            next_n.md += tmp_md[(s_x * N + s_y) * N2 + next_n.puzzle[new_x * N + new_y]];
 
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
            if(pq.size() >= BLOCK_NUM){
                return false;
            }
        }
    }
    return false;
}

__global__ void dfs_kernel(int limit, Node *root_set, int *dev_flag, Lock *lock) {
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // local_stack<Node, STACK_LIMIT> st;
    __shared__ Node st[STACK_LIMIT];
    __shared__ int index;
    index = -1;
    __syncthreads();

    if(threadIdx.x == 0) {
        index++;
        st[index] = root_set[blockIdx.x];
    }
    __syncthreads();

    int order[4] = {1, 0, 2, 3};
    int dx[4] = {0, -1, 0, 1};
    int dy[4] = {1, 0, -1, 0};

    __shared__ Node cur_nodes[WARP_SIZE];
    __shared__ bool flag[WARP_SIZE];

    while(index >= 0) {
        for (int i = 0; i < WARP_SIZE; ++i)
        {
            flag[i] = false;
        }
        __syncthreads();
        for (int i = 0; i < WARP_SIZE; ++i)
        {
            if(i == threadIdx.x && (threadIdx.x % 4) == 0) {
                // lock[blockIdx.x].lock();
                // printf("node_num: %d threadIdx.x: %d \n", index, threadIdx.x);
                if(index >= 0) {
                    flag[threadIdx.x / 4] = true;
                    cur_nodes[threadIdx.x / 4] = st[index];
                    atomicSub(&index, 1);
                }
                // lock[blockIdx.x].unlock();
            }
        }
        if(flag[threadIdx.x / 4] == false) continue;


        Node cur_n = cur_nodes[threadIdx.x / 4];
        if(cur_n.md == 0) {
            *dev_flag = cur_n.depth;
            return;
        }
        if(cur_n.depth + cur_n.md > limit) continue;
        int s_x = cur_n.space / N;
        int s_y = cur_n.space % N; 
        int operator_order = threadIdx.x % 4; 
        int i = order[operator_order];
        Node next_n = cur_n;
        int new_x = s_x + dx[i];
        int new_y = s_y + dy[i];
        if(new_x < 0  || new_y < 0 || new_x >= N || new_y >= N) continue; 
        if(max(cur_n.pre, i) - min(cur_n.pre, i) == 2) continue;

        //incremental manhattan distance
        next_n.md -= md[(new_x * N + new_y) * N2 + next_n.puzzle[new_x * N + new_y]];
        next_n.md += md[(s_x * N + s_y) * N2 + next_n.puzzle[new_x * N + new_y]];

        int a = next_n.puzzle[new_x * N + new_y];
        next_n.puzzle[new_x * N + new_y] = next_n.puzzle[s_x * N + s_y];
        next_n.puzzle[s_x * N + s_y] = a;

        next_n.space = new_x * N + new_y;
        // assert(get_md_sum(new_n.puzzle) == new_n.md);

        next_n.depth++;
        if(next_n.depth + next_n.md > limit) continue;
        next_n.pre = i;
        if(next_n.md == 0) {
            *dev_flag = next_n.depth;
            return;
        }

        for (int j = 0; j < WARP_SIZE; ++j)
        {
            if(j == threadIdx.x) {
                // lock[blockIdx.x].lock();
                atomicAdd(&index, 1);
                // printf("%d:%d:%d\n", index, next_n.depth, next_n.pre);
                st[index] = next_n;
                // lock[blockIdx.x].unlock();
            }
        }
    }
    return;

}

void ida_star() {
    pq = priority_queue<Node, vector<Node>, greater<Node> >();
    if(create_root_set()) {
        printf("%d\n", ans);
        return;
    }
    int pq_size = pq.size();
    Node root_set[BLOCK_NUM];
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

    for (int limit = s_node.md; limit < 100; ++limit, ++limit)
    {
        // path.resize(limit);
        // priority_queue<Node, vector<Node>, greater<Node> > tmp_pq = pq;

        int flag = -1;
        int *dev_flag;


        //gpu側にメモリ割当
        HANDLE_ERROR(cudaMalloc((void**)&dev_flag, sizeof(int)));
        cudaMemcpy(dev_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);

        Lock    lock[BLOCK_NUM];
        Lock    *dev_lock;
        HANDLE_ERROR( cudaMalloc( (void**)&dev_lock,
                              BLOCK_NUM * sizeof( Lock ) ) );
        HANDLE_ERROR( cudaMemcpy( dev_lock, lock,
                              BLOCK_NUM * sizeof( Lock ),
                              cudaMemcpyHostToDevice ) );

        dfs_kernel<<<BLOCK_NUM, WARP_SIZE>>>(limit, dev_root_set, dev_flag, dev_lock);

        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaMemcpy(&flag, dev_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if(flag != -1) {
            cout << flag << endl;
            return;
        }
        HANDLE_ERROR(cudaFree(dev_flag));
    }
    HANDLE_ERROR(cudaFree(dev_root_set));
}

 
int main() {
    // string output_file = "../result/korf100_psimple_result.csv";
    // ofstream writing_file;
    // writing_file.open(output_file, std::ios::out);
    FILE *output_file;
    output_file = fopen("../result/korf100_block_parallel_result_30.csv","w");

    set_md();
    for (int i = 0; i < 4; ++i)
    {
        string input_file = "../benchmarks/korf100/prob";
        if(i < 10) {
            input_file += "00";
        } else if(i < 100) {
            input_file += "0";
        }
        input_file += tostr(i);
        cout << input_file << " ";
        // set_md();

        // clock_t start = clock();
        auto start = std::chrono::system_clock::now();

        input_table(const_cast<char*>(input_file.c_str()));
        ida_star();

        // clock_t end = clock();
        auto end = std::chrono::system_clock::now();
        auto diff = end - start;
        fprintf(output_file,"%f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / (double)1000000000.0);
        printf("%f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / (double)1000000000.0);

        // writing_file << (double)(end - start) / CLOCKS_PER_SEC << endl;
    }
    fclose(output_file);
}
