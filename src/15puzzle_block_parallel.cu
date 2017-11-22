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
#define STACK_LIMIT 64 * 8
#define MAX_CORE_NUM 15360
#define CORE_NUM 1536
// #define CORE_NUM 15360
// #define CORE_NUM 384
// #define CORE_NUM 192
// #define WARP_SIZE 8
// #define WARP_SIZE 4
#define WARP_SIZE 32
#define BLOCK_NUM 48
// #define BLOCK_NUM 480

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
            if(pq.size() >= BLOCK_NUM){
                return false;
            }
        }
    }
    return false;
}

__global__ void dfs_kernel(int limit, Node *root_set, int *dev_flag, Lock *lock, int *loop_set) {
    __shared__ int shared_md[N2*N2];
    for (int i = threadIdx.x; i < N2*N2; i += blockDim.x)
    {
        shared_md[i] = md[i];
    }

    __syncthreads();

    __shared__ Node st[STACK_LIMIT];
    __shared__ int index;
    index = 0;
    st[0] = root_set[blockIdx.x];
    // index = WARP_SIZE / 4 - 1;
    // if(threadIdx.x % 4 == 0) {
    //     index++;
    // printf("stack index : %d  root node index %d\n", threadIdx.x / 4, blockIdx.x * (WARP_SIZE / 4 ) + threadIdx.x / 4);
    // st[threadIdx.x / 4] = root_set[blockIdx.x * (WARP_SIZE / 4 ) + threadIdx.x / 4];
    // }
    __syncthreads();

    int order[4] = {1, 0, 2, 3};
    int dx[4] = {0, -1, 0, 1};
    int dy[4] = {1, 0, -1, 0};

    int loop_count = 0;
    while(true) {
        bool stack_is_empty = (index <= -1);
        __syncthreads();
        if(stack_is_empty || *dev_flag != -1) break;

        Node cur_n;
        bool find_cur_n = false;
        int cur_n_idx = index - (threadIdx.x / 4);
        if(cur_n_idx >= 0) {
            cur_n = st[cur_n_idx];
            find_cur_n = true;
        }

        if(threadIdx.x == 0) {
            index = index >= (WARP_SIZE / 4 - 1) ? (index - WARP_SIZE / 4) : -1;
        }
        __syncthreads();


        if(find_cur_n) {
            if(cur_n.md == 0) {
                *dev_flag = cur_n.depth;
                goto LOOP;
                // return;
            }
            if(cur_n.depth + cur_n.md > limit) goto LOOP;
            int s_x = cur_n.space / N;
            int s_y = cur_n.space % N; 
            int operator_order = threadIdx.x % 4; 
            int i = order[operator_order];
            Node next_n = cur_n;
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
            // assert(get_md_sum(new_n.puzzle) == new_n.md);

            next_n.depth++;
            if(next_n.depth + next_n.md > limit) goto LOOP;
            next_n.pre = i;
            if(next_n.md == 0) {
                *dev_flag = next_n.depth;
                //return;
                goto LOOP;
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
        LOOP:
        loop_count++;
        __syncthreads();
    }
    loop_set[blockIdx.x] = loop_count; 
    return;
}


void divide_root_set(Node root, Node *new_root_set, int *new_root_set_index, int divide_num){
    priority_queue<Node, vector<Node>, greater<Node> > prq;
    prq.push(root);
    while(!prq.empty() && prq.size() < divide_num ) {
        Node cur_n = prq.top();
        prq.pop();
        if(cur_n.md == 0 ) {
            prq.push(cur_n);
            break;
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
            // if(next_n.md == 0) {
            //     prq.push(next_n);
            //     break;
            //     // ans = next_n.depth;
            //     // return true;
            // }
            prq.push(next_n);
        }
        // if(prq.size() >= divide_num){
        //     break
        //     while(prq.empty()) {
        //         new_root_set[*new_root_set_index] = prq.top();
        //         prq.pop();
        //         *new_root_set_index = *new_root_set_index + 1;
        //     }
        //     return;
        // }
    }
    while(!prq.empty()) {
        new_root_set[*new_root_set_index] = prq.top();
        prq.pop();
        *new_root_set_index = *new_root_set_index + 1;
    }
    return;
}

void ida_star() {
    pq = priority_queue<Node, vector<Node>, greater<Node> >();
    if(create_root_set()) {
        printf("%d\n", ans);
        return;
    }
    int root_node_size = pq.size();
    Node root_set[MAX_CORE_NUM];
    int i = 0;
    while(!pq.empty()) {
        Node n = pq.top();
        pq.pop();
        root_set[i] = n;
        i++;
    }

    int load_set[MAX_CORE_NUM];

    for (int limit = s_node.md; limit < 100; ++limit, ++limit)
    {
        int flag = -1;
        int *dev_flag;
        // int load;
        // int *dev_load;

        //gpu側で使う根集合のポインタ
        Node *dev_root_set;
        int *dev_load_set;
        //gpu側のメモリ割当て
        HANDLE_ERROR(cudaMalloc((void**)&dev_root_set, root_node_size * sizeof(Node) ) );
        //root_setをGPU側のdev_root_setにコピー
        HANDLE_ERROR(cudaMemcpy(dev_root_set, root_set, root_node_size * sizeof(Node), cudaMemcpyHostToDevice) );


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
        HANDLE_ERROR(cudaMalloc((void**)&dev_load_set, root_node_size * sizeof(int)));
        HANDLE_ERROR(cudaMemset(dev_load_set, 0, root_node_size * sizeof(int)));

        // cout << root_node_size << endl;
        dfs_kernel<<<root_node_size, WARP_SIZE>>>(limit, dev_root_set, dev_flag, dev_lock, dev_load_set);


        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaMemcpy(&flag, dev_flag, sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(&load_set, dev_load_set, root_node_size * sizeof(int), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaFree(dev_flag));
        HANDLE_ERROR(cudaFree(dev_root_set));

        if(flag != -1) {
            cout << flag << endl;
            return;
        }

        int new_root_node_size = 0;
        Node new_root_set[MAX_CORE_NUM];

        //calculate load_balance
        int load_sum = 0;
        for (int i = 0; i < root_node_size; ++i)
        {
            load_sum += load_set[i];
            // cout << load_set[i] << " ";
        }
        // cout << endl;
        int load_av = load_sum / root_node_size;

        for (int i = 0; i < root_node_size; ++i)
        {
            int divide_num = load_av == 0 ? load_set[i] : (load_set[i]- 1) / load_av + 1;
            if(divide_num > 1) {
                divide_root_set(root_set[i], new_root_set, &new_root_node_size, divide_num);
            } else {
                new_root_set[new_root_node_size] = root_set[i];
                new_root_node_size++;
            }
            // cout << divide_num << " ";
        }
        // cout << "root_node_size:" <<root_node_size << endl;;

        assert(new_root_node_size <= MAX_CORE_NUM);


        for (int i = 0; i < new_root_node_size; ++i)
        {
            root_set[i] = new_root_set[i];
        }
        root_node_size = new_root_node_size;
    }
}

 
int main() {
    FILE *output_file;
    output_file = fopen("../result/korf100_block_parallel_result_with_staticlb_50.csv","w");

    set_md();
    for (int i = 0; i < 50; ++i)
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
        fprintf(output_file,"%f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / (double)1000000000.0);
        // printf("%f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / (double)1000000000.0);
    }
    fclose(output_file);
}
