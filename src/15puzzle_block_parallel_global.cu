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

// #define DEBUG
// #define DFS
// #define MANY_NODE

template <typename T> std::string tostr(const T& t)
{
    std::ostringstream os; os<<t; return os.str();
}
 
#define N 4
#define N2 16
#define STACK_LIMIT 64 * 12
#define MAX_CORE_NUM 100000
#define MAX_BLOCK_SIZE 64535
#define WARP_SIZE 32
#define THREAD_SIZE_PER_BLOCK 64
#define BLOCK_NUM 2048
// #define BLOCK_NUM 512
// #define BLOCK_NUM 4096
// #define BLOCK_NUM 48


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
        return (depth + md) < (n.depth + n.md);
    }

    bool operator > (const Node& n) const {
        return (depth + md) > (n.depth + n.md);
    }
};

Node s_node;
int tmp_md[N2*N2];
__constant__ int md[N2*N2];
int ans;
priority_queue<Node, vector<Node>, greater<Node> > pq;
Node *global_st;

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

bool create_root_set() {
    pq.push(s_node);
    while(pq.size() < BLOCK_NUM ) {
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
            next_n.depth++;
            next_n.pre = i;
            if(next_n.md == 0) {
                ans = next_n.depth;
                return true;
            }
            pq.push(next_n);
        }
    }
    return false;
}

#ifdef MANY_NODE
__global__ void dfs_kernel(int limit, Node *root_set, int *dev_flag, int *loop_set, Node *global_st, int *dev_node_size) {

#else

__global__ void dfs_kernel(int limit, Node *root_set, int *dev_flag, int *loop_set, Node *global_st) {

#endif
    __shared__ int shared_md[N2*N2];
    for (int i = threadIdx.x; i < N2*N2; i += blockDim.x)
    {
        shared_md[i] = md[i];
    }

    __shared__ int mutex;
    mutex = 0;

    __syncthreads();

    __shared__ int index;
    #ifdef MANY_NODE
    index = 0;
    __syncthreads();
    int tmp_id = blockIdx.x * (THREAD_SIZE_PER_BLOCK / 4 ) + threadIdx.x / 4;
    if(threadIdx.x % 4 == 0 && *dev_node_size > tmp_id) {
        atomicAdd(&index, 1);
        //printf("stack index : %d  root node index %d\n", threadIdx.x / 4, blockIdx.x * (THREAD_SIZE_PER_BLOCK / 4 ) + threadIdx.x / 4);
        global_st[threadIdx.x / 4] = root_set[tmp_id];
    }

    #else
    index = 0;
    global_st[blockIdx.x * STACK_LIMIT + 0] = root_set[blockIdx.x];
    #endif

    __syncthreads();

    int order[4] = {1, 0, 2, 3};
    int dx[4] = {0, -1, 0, 1};
    int dy[4] = {1, 0, -1, 0};

    int loop_count = 0;
    while(true) {
        bool stack_is_empty = (index <= -1);
        __syncthreads();
        if(stack_is_empty || *dev_flag != -1) break;
        loop_count++;

        Node cur_n;
        bool find_cur_n = false;
        int cur_n_idx = index - (threadIdx.x / 4);
        if(cur_n_idx >= 0) {
            cur_n = global_st[blockIdx.x * STACK_LIMIT + cur_n_idx];
            assert(cur_n_idx < STACK_LIMIT);
            find_cur_n = true;
        }

        if(threadIdx.x == 0) {
            index = index >= (THREAD_SIZE_PER_BLOCK / 4 - 1) ? (index - THREAD_SIZE_PER_BLOCK / 4) : -1;
        }
        __syncthreads();

        Node next_n;

        if(find_cur_n) {
            if(cur_n.md == 0) {
                *dev_flag = cur_n.depth;
                goto LOOP;
            }
            if(cur_n.depth + cur_n.md > limit) goto LOOP;
            int s_x = cur_n.space / N;
            int s_y = cur_n.space % N; 
            int operator_order = threadIdx.x % 4; 
            int i = order[operator_order];
            next_n = cur_n;
            int new_x = s_x + dx[i];
            int new_y = s_y + dy[i];
            if(new_x < 0  || new_y < 0 || new_x >= N || new_y >= N) goto LOOP; 
            if(max(cur_n.pre, i) - min(cur_n.pre, i) == 2) goto LOOP;

            //incremental manhattan distance
            next_n.md -= shared_md[(new_x * N + new_y) * N2 + next_n.puzzle[new_x * N + new_y]];
            next_n.md += shared_md[(s_x * N + s_y) * N2 + next_n.puzzle[new_x * N + new_y]];

            int a = next_n.puzzle[new_x * N + new_y];
            next_n.puzzle[new_x * N + new_y] = next_n.puzzle[s_x * N + s_y];
            next_n.puzzle[s_x * N + s_y] = a;

            next_n.space = new_x * N + new_y;
            #ifdef DEBUG
            // int sum = 0;
            // for (int i = 0; i < N2; ++i)
            // {
            //     if(next_n.puzzle[i] == 0) continue;
            //     sum += shared_md[i * N2 + next_n.puzzle[i]];
            // }
            // assert(sum == next_n.md);
            #endif

            next_n.depth++;
            if(next_n.depth + next_n.md > limit) goto LOOP;
            next_n.pre = i;
            if(next_n.md == 0) {
                *dev_flag = next_n.depth;
                // for (int k = 0; k < N; ++k)
                // {
                //     for (int t = 0; t < N; ++t)
                //     {
                //         printf("%d ", next_n.puzzle[k*N + t] );
                //     }
                //     printf("\n");
                // }
                //return;
                goto LOOP;
            }
            for (int j = 0; j < WARP_SIZE; ++j)
            {
                if(j == (threadIdx.x % WARP_SIZE) ) {
                    while( atomicCAS(&mutex, 0, 1 ) != 0 );
                    index++;
                    global_st[blockIdx.x * STACK_LIMIT + index] = next_n;
                    atomicExch(&mutex, 0);
                    assert(index < STACK_LIMIT);
                }
            }
        }

        LOOP:
        __syncthreads();
    }
    loop_set[blockIdx.x] = loop_count; 
    return;
}


#ifndef DFS 
void divide_root_set(Node root, Node *new_root_set, int *new_root_set_index, int divide_num){
    priority_queue<Node, vector<Node>, greater<Node> > prq;
    // priority_queue<Node> prq;
    prq.push(root);
    while(!prq.empty() && prq.size() < divide_num ) {
        Node cur_n = prq.top();
        prq.pop();
        if(cur_n.md == 0 ) {
            prq.push(cur_n);
            // break;
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

            #ifdef DEBUG
            assert(get_md_sum(next_n.puzzle) == next_n.md);
            #endif

            next_n.depth++;
            next_n.pre = i;
            // if(next_n.md == 0) {
            //     prq.push(next_n);
            //     break;
            //     // ans = next_n.depth;
            //     // return true;
            // }
            prq.push(next_n);
        }
    }
    while(!prq.empty()) {
        new_root_set[*new_root_set_index] = prq.top();
        prq.pop();
        *new_root_set_index = *new_root_set_index + 1;
    }
    return;
}

#else

void divide_root_set(Node root, Node *new_root_set, int *new_root_set_index, int divide_num){
    stack<Node> st;
    st.push(root);
    while(!st.empty() && st.size() < divide_num ) {
        Node cur_n = st.top();
        st.pop();
        if(cur_n.md == 0 ) {
            st.push(cur_n);
            // break;
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
            #ifdef DEBUG
            assert(get_md_sum(next_n.puzzle) == next_n.md);
            #endif
            next_n.depth++;
            next_n.pre = i;
            st.push(next_n);
        }
    }
    while(!st.empty()) {
        new_root_set[*new_root_set_index] = st.top();
        st.pop();
        *new_root_set_index = *new_root_set_index + 1;
    }
    return;
}

#endif

Node root_set[MAX_BLOCK_SIZE];
Node new_root_set[MAX_BLOCK_SIZE];
int load_set[MAX_BLOCK_SIZE];
//メモリが足りなくなるのでグローバル変数として定義

void ida_star() {
    pq = priority_queue<Node, vector<Node>, greater<Node> >();
    if(create_root_set()) {
        printf("%d\n", ans);
        return;
    }
    int root_node_size = pq.size();
    int i = 0;
    while(!pq.empty()) {
        Node n = pq.top();
        pq.pop();
        root_set[i] = n;
        i++;
    }

    int flag = -1;
    int *dev_flag;
    Node *dev_root_set;
    int *dev_load_set;

    HANDLE_ERROR(cudaMalloc((void**)&dev_root_set, MAX_BLOCK_SIZE * sizeof(Node) ) );
    HANDLE_ERROR(cudaMalloc((void**)&dev_flag, sizeof(int)));
    cudaMemcpy(dev_flag, &flag, sizeof(int), cudaMemcpyHostToDevice);

    for (int limit = s_node.md; limit < 100; ++limit, ++limit)
    {
        #ifdef DEBUG
        auto start = std::chrono::system_clock::now();
        #endif

        HANDLE_ERROR(cudaMemcpy(dev_root_set, root_set, root_node_size * sizeof(Node), cudaMemcpyHostToDevice) );
        HANDLE_ERROR(cudaMalloc((void**)&dev_load_set, root_node_size * sizeof(int)));
        HANDLE_ERROR(cudaMemset(dev_load_set, 0, root_node_size * sizeof(int)));

        #ifdef DEBUG
        cout << "f_limit : " << limit << endl;
        cout << root_node_size << endl;
        #endif

        #ifdef MANY_NODE
        int *dev_node_size;
        HANDLE_ERROR(cudaMalloc((void**)&dev_node_size, sizeof(int)));
        cudaMemcpy(dev_node_size, &root_node_size, sizeof(int), cudaMemcpyHostToDevice);
        dfs_kernel<<<root_node_size / (THREAD_SIZE_PER_BLOCK / 4 ), THREAD_SIZE_PER_BLOCK>>>(limit, dev_root_set, dev_flag, dev_load_set, global_st, dev_node_size);
        #else
        dfs_kernel<<<root_node_size, THREAD_SIZE_PER_BLOCK>>>(limit, dev_root_set, dev_flag, dev_load_set, global_st);
        #endif

        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaMemcpy(&flag, dev_flag, sizeof(int), cudaMemcpyDeviceToHost));
        #ifdef MANY_NODE
        HANDLE_ERROR(cudaMemcpy(&load_set, dev_load_set, root_node_size / (THREAD_SIZE_PER_BLOCK / 4 ) * sizeof(int), cudaMemcpyDeviceToHost));
        #else
        HANDLE_ERROR(cudaMemcpy(&load_set, dev_load_set, root_node_size * sizeof(int), cudaMemcpyDeviceToHost));
        #endif

        #ifdef MANY_NODE
        HANDLE_ERROR(cudaFree(dev_node_size));
        #endif

        if(flag != -1) {
            cout << flag << endl;
            HANDLE_ERROR(cudaFree(dev_flag));
            HANDLE_ERROR(cudaFree(dev_root_set));
            HANDLE_ERROR(cudaFree(dev_load_set));

            return;
        }

        int new_root_node_size = 0;

        //calculate load_balance
        int load_sum = 0;
        int max_load = 0;
        for (int i = 0; i < root_node_size; ++i)
        {
            load_sum += load_set[i];
            max_load = max(load_set[i], max_load);
            // cout << load_set[i] << " ";
        }
        // cout << "load sum " << load_sum << endl;
        int load_av = load_sum / root_node_size;
        #ifdef DEBUG
        cout << "load average " << load_av << endl;
        cout << "max load " << max_load << endl;
        int stat_cnt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        #endif
        for (int i = 0; i < root_node_size; ++i)
        {
            #ifdef DEBUG
            if (load_set[i] < load_av)
                stat_cnt[0]++;
            else if (load_set[i] < 2 * load_av)
                stat_cnt[1]++;
            else if (load_set[i] < 4 * load_av)
                stat_cnt[2]++;
            else if (load_set[i] < 8 * load_av)
                stat_cnt[3]++;
            else if (load_set[i] < 16 * load_av)
                stat_cnt[4]++;
            else if (load_set[i] < 32 * load_av)
                stat_cnt[5]++;
            else if (load_set[i] < 64 * load_av)
                stat_cnt[6]++;
            else if (load_set[i] < 128 * load_av)
                stat_cnt[7]++;
            else
                stat_cnt[8]++;
            #endif
            int divide_num = load_av == 0 ? load_set[i] : (load_set[i]- 1) / load_av + 1;

            if((divide_num > 1 && new_root_node_size + root_node_size - i + divide_num < MAX_BLOCK_SIZE/2) || (divide_num > 2 && new_root_node_size + root_node_size - i + divide_num < MAX_BLOCK_SIZE)) {
                #ifdef DEBUG
                int tmp = new_root_node_size;
                #endif
                divide_root_set(root_set[i], new_root_set, &new_root_node_size, divide_num);
                #ifdef DEBUG
                // cout << tmp << " " << new_root_node_size << endl;
                assert(tmp <= new_root_node_size);
                assert(new_root_node_size < MAX_BLOCK_SIZE);
                #endif
            } else {
                new_root_set[new_root_node_size] = root_set[i];
                new_root_node_size++;
            }

        }
        #ifdef DEBUG
        printf("STAT: distr: av=%d, 2av=%d, 4av=%d, 8av=%d, 16av=%d, 32av=%d, "
             "64av=%d, 128av=%d, more=%d\n",
             stat_cnt[0], stat_cnt[1], stat_cnt[2], stat_cnt[3], stat_cnt[4],
             stat_cnt[5], stat_cnt[6], stat_cnt[7], stat_cnt[8]);
        cout << "root_node_size:" << root_node_size << endl;
        cout << "new_root_node_size:" << new_root_node_size << endl;
        auto end = std::chrono::system_clock::now();
        auto diff = end - start;
        printf("executed time is %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / (double)1000000000.0);
        cout << "------" << endl;
        cout << endl;
        #endif

        assert(new_root_node_size <= MAX_CORE_NUM);


        for (int i = 0; i < new_root_node_size; ++i)
        {
            root_set[i] = new_root_set[i];
        }
        root_node_size = new_root_node_size;
    }
}

 
int main() {
    int problems_num = 100;
    #ifndef DEBUG
    FILE *output_file;
    string output_file_str = "../result/korf100_block_parallel_result_with_staticlb_global" + tostr(problems_num) + "_" + tostr(BLOCK_NUM) + "_" + tostr(THREAD_SIZE_PER_BLOCK) + ".csv";
    output_file = fopen(const_cast<char*>(output_file_str.c_str()),"w");
    #endif

    HANDLE_ERROR(cudaMalloc((void**)&global_st, MAX_BLOCK_SIZE * STACK_LIMIT * sizeof(Node) ) );

    set_md();
    for (int i = 0; i < problems_num; ++i)
    {
        string input_file = "../benchmarks/korf100/prob";
        if(i < 10) {
            input_file += "00";
        } else if(i < 100) {
            input_file += "0";
        }
        input_file += tostr(i);
        cout << input_file << " ";
        auto start = std::chrono::system_clock::now();

        input_table(const_cast<char*>(input_file.c_str()));
        ida_star();

        auto end = std::chrono::system_clock::now();
        auto diff = end - start;
        #ifndef DEBUG
        fprintf(output_file,"%f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / (double)1000000000.0);
        #endif
        #ifdef DEBUG
        printf("%f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / (double)1000000000.0);
        printf("thread per block : %d\n", THREAD_SIZE_PER_BLOCK);
        #endif
    }
    HANDLE_ERROR(cudaFree(global_st));

    #ifndef DEBUG
    fclose(output_file);
    #endif
}
