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
// #define SHARED
// #define USE_LOCK
#define BEST
#define SEARCH_ALL


template <typename T> std::string tostr(const T& t)
{
    std::ostringstream os; os<<t; return os.str();
}
 
#define N 5
#define N2 25
#define PDB_TABLESIZE 244140625
#define STACK_LIMIT 76 * 8
#define MAX_CORE_NUM 65000
#define MAX_BLOCK_SIZE 64535
// #define MAX_CORE_NUM 524288
#define CORE_NUM 1536
// #define CORE_NUM 15360
// #define CORE_NUM 384
// #define CORE_NUM 192
// #define WARP_SIZE 8
// #define WARP_SIZE 4
#define WARP_SIZE 32
#define BLOCK_NUM 2048
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
 
static const char dx[4] = {0, -1, 0, 1};
static const char dy[4] = {1, 0, -1, 0};
// static const char dir[4] = {'r', 'u', 'l', 'd'}; 
static const char order[4] = {1, 0, 2, 3};

static __device__ __constant__ const char dev_rf[] = {0,5,10,15,20,1,6,11,16,21,2,7,12,17,22,3,8,13,18,23,4,9,14,19,24};
static __device__ __constant__ const char dev_rot90[] = {20,15,10,5,0,21,16,11,6,1,22,17,12,7,2,23,18,13,8,3,24,19,14,9,4};
static __device__ __constant__ const char dev_rot90rf[] = {20,21,22,23,24,15,16,17,18,19,10,11,12,13,14,5,6,7,8,9,0,1,2,3,4};
static __device__ __constant__ const char dev_rot180[] = {24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
static __device__ __constant__ const char dev_rot180rf[] = {24,19,14,9,4,23,18,13,8,3,22,17,12,7,2,21,16,11,6,1,20,15,10,5,0};

static  const char rf[] = {0,5,10,15,20,1,6,11,16,21,2,7,12,17,22,3,8,13,18,23,4,9,14,19,24};
static  const char rot90[] = {20,15,10,5,0,21,16,11,6,1,22,17,12,7,2,23,18,13,8,3,24,19,14,9,4};
static  const char rot90rf[] = {20,21,22,23,24,15,16,17,18,19,10,11,12,13,14,5,6,7,8,9,0,1,2,3,4};
static  const char rot180[] = {24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
static  const char rot180rf[] = {24,19,14,9,4,23,18,13,8,3,22,17,12,7,2,21,16,11,6,1,20,15,10,5,0};


__device__ unsigned char dev_h0[PDB_TABLESIZE];
__device__ unsigned char dev_h1[PDB_TABLESIZE];

unsigned char h0[PDB_TABLESIZE];
unsigned char h1[PDB_TABLESIZE];

struct Node
{
    char puzzle[N2];
    char inv_puzzle[N2];
    char space;
    // char md;
    char h;
    char depth;
    char pre;
    bool operator < (const Node& n) const {
        return (depth + h) < (n.depth + n.h);
    }

    bool operator > (const Node& n) const {
        return (depth + h) > (n.depth + n.h);
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
int ans;
priority_queue<Node, vector<Node>, greater<Node> > pq;

class PatternDataBase
{
private:
    // unsigned char h0[PDB_TABLESIZE];
    // unsigned char h1[PDB_TABLESIZE];
    /* the position of each tile in order, reflected about the main diagonal */
public:
    PatternDataBase();
    void init();
    void input_h0(const char *filename);
    void input_h1(const char *filename);
    unsigned int hash0(const char *inv);
    unsigned int hash1(const char *inv);
    unsigned int hash2(const char *inv);
    unsigned int hash3(const char *inv);
    unsigned int hashref0(const char *inv);
    unsigned int hashref1(const char *inv);
    unsigned int hashref2(const char *inv);
    unsigned int hashref3(const char *inv);
    unsigned int get_hash_value(const char *inv);
    // unsigned char get_h0_value(int i);
    // unsigned char get_h1_value(int i);
};

PatternDataBase::PatternDataBase() {}

void PatternDataBase::init() {
    const char *c0 = "../pdb/pat24.1256712.tab";
    const char *c1 = "../pdb/pat24.34891314.tab";
    #ifdef DEBUG
    cout << "pattern 1 2 5 6 7 12 read in" << endl;
    #endif
    input_h0(c0);
    #ifdef DEBUG
    cout << "pattern 3 4 8 9 13 14 read in" << endl;
    #endif
    input_h1(c1);
}

void PatternDataBase::input_h0(const char *filename) {
    FILE *infile;
    infile = fopen(filename, "rb");
    int index;
    int s[6];
    for (s[0] = 0; s[0] < N2; s[0]++) {
        for (s[1] = 0; s[1] < N2; s[1]++) {
            if (s[1] == s[0]) continue;
            for (s[2] = 0; s[2] < N2; s[2]++) {
                if (s[2] == s[0] || s[2] == s[1]) continue;
                for (s[3] = 0; s[3] < N2; s[3]++) {
                    if (s[3] == s[0] || s[3] == s[1] || s[3] == s[2]) continue;
                    for (s[4] = 0; s[4] < N2; s[4]++) {
                        if (s[4] == s[0] || s[4] == s[1] || s[4] == s[2] || s[4] == s[3]) continue;
                        for (s[5] = 0; s[5] < N2; s[5]++)   {
                            if (s[5] == s[0] || s[5] == s[1] || s[5] == s[2] || s[5] == s[3] || s[5] == s[4]) continue;
                            index = ((((s[0]*25+s[1])*25+s[2])*25+s[3])*25+s[4])*25+s[5];
                            h0[index] = getc(infile);
                        }
                    }
                }
            }
        }
    }
    fclose(infile);
}


void PatternDataBase::input_h1(const char *filename) {
    FILE *infile;
    infile = fopen(filename, "rb");
    int index;
    int s[6];
    for (s[0] = 0; s[0] < N2; s[0]++) {
        for (s[1] = 0; s[1] < N2; s[1]++) {
            if (s[1] == s[0]) continue;
            for (s[2] = 0; s[2] < N2; s[2]++) {
                if (s[2] == s[0] || s[2] == s[1]) continue;
                for (s[3] = 0; s[3] < N2; s[3]++) {
                    if (s[3] == s[0] || s[3] == s[1] || s[3] == s[2]) continue;
                    for (s[4] = 0; s[4] < N2; s[4]++) {
                        if (s[4] == s[0] || s[4] == s[1] || s[4] == s[2] || s[4] == s[3]) continue;
                        for (s[5] = 0; s[5] < N2; s[5]++)   {
                            if (s[5] == s[0] || s[5] == s[1] || s[5] == s[2] || s[5] == s[3] || s[5] == s[4]) continue;
                            index = ((((s[0]*25+s[1])*25+s[2])*25+s[3])*25+s[4])*25+s[5];
                            h1[index] = getc(infile);
                        }
                    }
                }
            }
        }
    }
    fclose(infile);
}

unsigned int PatternDataBase::hash0(const char *inv) {
    int hashval;
    hashval = ((((inv[1]*N2+inv[2])*N2+inv[5])*N2+inv[6])*N2+inv[7])*N2+inv[12];
    return h0[hashval];
}

unsigned int PatternDataBase::hash1(const char *inv) {
    int hashval;
    hashval = ((((inv[3]*N2+inv[4])*N2+inv[8])*N2+inv[9])*N2+inv[13])*N2+inv[14];
    return (h1[hashval]);
}

unsigned int PatternDataBase::hash2(const char *inv) {
    int hashval;
    hashval = ((((rot180[inv[21]] * N2
              + rot180[inv[20]]) * N2
             + rot180[inv[16]]) * N2
            + rot180[inv[15]]) * N2
           + rot180[inv[11]]) * N2
          + rot180[inv[10]];
    return (h1[hashval]);
}

unsigned int PatternDataBase::hash3(const char *inv) {
    int hashval;
    hashval = ((((rot90[inv[19]] * N2
              + rot90[inv[24]]) * N2
             + rot90[inv[18]]) * N2
            + rot90[inv[23]]) * N2
           + rot90[inv[17]]) * N2
          + rot90[inv[22]];
    return (h1[hashval]);
}

unsigned int PatternDataBase::hashref0(const char *inv) {
    int hashval;
    hashval = (((((rf[inv[5]] * N2
               + rf[inv[10]]) * N2
              + rf[inv[1]]) * N2
             + rf[inv[6]]) * N2
            + rf[inv[11]]) * N2
           + rf[inv[12]]);
    return (h0[hashval]);
}

unsigned int PatternDataBase::hashref1(const char *inv) {
    int hashval;
    hashval = (((((rf[inv[15]] * N2
               + rf[inv[20]]) * N2
              + rf[inv[16]]) * N2
             + rf[inv[21]]) * N2
            + rf[inv[17]]) * N2
           + rf[inv[22]]);
    return (h1[hashval]);
}
unsigned int PatternDataBase::hashref2(const char *inv) {
    int hashval;
    hashval = (((((rot180rf[inv[9]] * N2
               + rot180rf[inv[4]]) * N2
              + rot180rf[inv[8]]) * N2
             + rot180rf[inv[3]]) * N2
            + rot180rf[inv[7]]) * N2
           + rot180rf[inv[2]]);
    return (h1[hashval]);
}

unsigned int PatternDataBase::hashref3(const char *inv) {
    int hashval;
    hashval = (((((rot90rf[inv[23]] * N2
               + rot90rf[inv[24]]) * N2
              + rot90rf[inv[18]]) * N2
             + rot90rf[inv[19]]) * N2
            + rot90rf[inv[13]]) * N2
           + rot90rf[inv[14]]);
    return (h1[hashval]);
}

unsigned int PatternDataBase::get_hash_value(const char *inv) {
    return max( hash0(inv) + hash1(inv) + hash2(inv) + hash3(inv), 
        hashref0(inv) + hashref1(inv) + hashref2(inv) + hashref3(inv) ); 
}

// unsigned char PatternDataBase::get_h0_value(int i) {
//     return h0[i];
// }
// unsigned char PatternDataBase::get_h1_value(int i) {
//     return h1[i];
// }


PatternDataBase pd;

class local_pdb
{
// private:
//     unsigned char h0[PDB_TABLESIZE];
//     unsigned char h1[PDB_TABLESIZE];
public:
    local_pdb();
    __device__ unsigned int hash0(const char *inv);
    __device__ unsigned int hash1(const char *inv);
    __device__ unsigned int hash2(const char *inv);
    __device__ unsigned int hash3(const char *inv);
    __device__ unsigned int hashref0(const char *inv);
    __device__ unsigned int hashref1(const char *inv);
    __device__ unsigned int hashref2(const char *inv);
    __device__ unsigned int hashref3(const char *inv);
    __device__ unsigned int get_hash_value(const char *inv);

};

local_pdb::local_pdb() {
    // HANDLE_ERROR(cudaMemcpy(dev_h0, h0, PDB_TABLESIZE * sizeof(unsigned char), cudaMemcpyHostToDevice) );
    // HANDLE_ERROR(cudaMemcpy(dev_h1, h1, PDB_TABLESIZE * sizeof(unsigned char), cudaMemcpyHostToDevice) );
}


__device__ unsigned int local_pdb::hash0(const char *inv) {
    int hashval;
    hashval = ((((inv[1]*N2+inv[2])*N2+inv[5])*N2+inv[6])*N2+inv[7])*N2+inv[12];
    return dev_h0[hashval];
}

__device__ unsigned int local_pdb::hash1(const char *inv) {
    int hashval;
    hashval = ((((inv[3]*N2+inv[4])*N2+inv[8])*N2+inv[9])*N2+inv[13])*N2+inv[14];
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::hash2(const char *inv) {
    int hashval;
    hashval = ((((dev_rot180[inv[21]] * N2
              + dev_rot180[inv[20]]) * N2
             + dev_rot180[inv[16]]) * N2
            + dev_rot180[inv[15]]) * N2
           + dev_rot180[inv[11]]) * N2
          + dev_rot180[inv[10]];
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::hash3(const char *inv) {
    int hashval;
    hashval = ((((dev_rot90[inv[19]] * N2
              + dev_rot90[inv[24]]) * N2
             + dev_rot90[inv[18]]) * N2
            + dev_rot90[inv[23]]) * N2
           + dev_rot90[inv[17]]) * N2
          + dev_rot90[inv[22]];
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::hashref0(const char *inv) {
    int hashval;
    hashval = (((((dev_rf[inv[5]] * N2
               + dev_rf[inv[10]]) * N2
              + dev_rf[inv[1]]) * N2
             + dev_rf[inv[6]]) * N2
            + dev_rf[inv[11]]) * N2
           + dev_rf[inv[12]]);
    return (dev_h0[hashval]);
}

__device__ unsigned int local_pdb::hashref1(const char *inv) {
    int hashval;
    hashval = (((((dev_rf[inv[15]] * N2
               + dev_rf[inv[20]]) * N2
              + dev_rf[inv[16]]) * N2
             + dev_rf[inv[21]]) * N2
            + dev_rf[inv[17]]) * N2
           + dev_rf[inv[22]]);
    return (dev_h1[hashval]);
}
__device__ unsigned int local_pdb::hashref2(const char *inv) {
    int hashval;
    hashval = (((((dev_rot180rf[inv[9]] * N2
               + dev_rot180rf[inv[4]]) * N2
              + dev_rot180rf[inv[8]]) * N2
             + dev_rot180rf[inv[3]]) * N2
            + dev_rot180rf[inv[7]]) * N2
           + dev_rot180rf[inv[2]]);
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::hashref3(const char *inv) {
    int hashval;
    hashval = (((((dev_rot90rf[inv[23]] * N2
               + dev_rot90rf[inv[24]]) * N2
              + dev_rot90rf[inv[18]]) * N2
             + dev_rot90rf[inv[19]]) * N2
            + dev_rot90rf[inv[13]]) * N2
           + dev_rot90rf[inv[14]]);
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::get_hash_value(const char *inv) {
    return max( hash0(inv) + hash1(inv) + hash2(inv) + hash3(inv), 
        hashref0(inv) + hashref1(inv) + hashref2(inv) + hashref3(inv) ); 
}

local_pdb  *dev_pd;

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
        s_node.inv_puzzle[tmp] = i;
    }
    s_node.h = pd.get_hash_value(s_node.inv_puzzle);
    s_node.depth = 0;
    s_node.pre = -10;
}

bool create_root_set() {
    pq.push(s_node);
    while(pq.size() < BLOCK_NUM ) {
        Node cur_n = pq.top();
        pq.pop();
        if(cur_n.h == 0 ) {
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

            swap(next_n.inv_puzzle[next_n.puzzle[new_x * N + new_y]], next_n.inv_puzzle[next_n.puzzle[s_x * N + s_y]]); 
            swap(next_n.puzzle[new_x * N + new_y], next_n.puzzle[s_x * N + s_y]);

            next_n.space = new_x * N + new_y;
            next_n.h = pd.get_hash_value(next_n.inv_puzzle);
            // assert(get_md_sum(new_n.puzzle) == new_n.h);
            next_n.depth++;
            next_n.pre = i;
            if(next_n.h == 0) {
                ans = next_n.depth;
                return true;
            }
            pq.push(next_n);
        }
    }
    return false;
}


#ifdef DEBUG
__constant__ int md[N2*N2];
int tmp_md[N2*N2];
__device__ int get_md_sum(char *puzzle) {
    int sum = 0;
    for (int i = 0; i < N2; ++i)
    {
        if(puzzle[i] == 0) continue;
        sum += md[i * N2 + puzzle[i]];
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
#endif

#ifdef USE_LOCK

__global__ void dfs_kernel(int limit, Node *root_set, int *dev_flag, Lock *lock, int *loop_set, local_pdb *dev_pdb) {

#else

__global__ void dfs_kernel(int limit, Node *root_set, int *dev_flag, int *loop_set, local_pdb *dev_pdb) {

#endif
    #ifdef SHARED
    // __shared__ int shared_md[N2*N2];
    // for (int i = threadIdx.x; i < N2*N2; i += blockDim.x)
    // {
    //     shared_md[i] = md[i];
    // }
    #endif

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
        #ifdef SEARCH_ALL
        if(stack_is_empty) break;
        #else
        if(stack_is_empty || *dev_flag != -1) break;
        #endif
        loop_count++;

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

        Node next_n;

        if(find_cur_n) {
            #ifdef DEBUG
            int md_h = get_md_sum(cur_n.puzzle);
            if(md_h == 0) assert(md_h == cur_n.h); 
            #endif
            if(cur_n.h == 0) {
                *dev_flag = cur_n.depth;
                goto LOOP;
                // return;
            }
            if(cur_n.depth + cur_n.h > limit) goto LOOP;
            int s_x = cur_n.space / N;
            int s_y = cur_n.space % N; 
            int operator_order = threadIdx.x % 4; 
            int i = order[operator_order];
            next_n = cur_n;
            int new_x = s_x + dx[i];
            int new_y = s_y + dy[i];
            if(new_x < 0  || new_y < 0 || new_x >= N || new_y >= N) goto LOOP; 
            if(max(cur_n.pre, i) - min(cur_n.pre, i) == 2) goto LOOP;


            int a = next_n.inv_puzzle[next_n.puzzle[new_x * N + new_y]];
            int b = next_n.inv_puzzle[next_n.puzzle[s_x * N + s_y]];
            int c = next_n.puzzle[new_x * N + new_y];
            int d = next_n.puzzle[s_x * N + s_y];

            next_n.inv_puzzle[next_n.puzzle[new_x * N + new_y]] = b;
            next_n.inv_puzzle[next_n.puzzle[s_x * N + s_y]] = a;

            next_n.puzzle[new_x * N + new_y] = d;
            next_n.puzzle[s_x * N + s_y] = c;

            next_n.space = new_x * N + new_y;
            next_n.h = dev_pdb->get_hash_value(next_n.inv_puzzle);

            next_n.depth++;
            if(next_n.depth + next_n.h > limit) goto LOOP;
            next_n.pre = i;
            if(next_n.h == 0) {
                *dev_flag = next_n.depth;
                //return;
                goto LOOP;
            }
            #ifdef BEST
            int tmp = atomicAdd((int *)&index, 1);
            st[tmp + 1] = next_n;
            assert(index < STACK_LIMIT);
            #else
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
            #endif

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
    prq.push(root);
    while(!prq.empty() && prq.size() < divide_num ) {
        Node cur_n = prq.top();
        prq.pop();
        if(cur_n.h == 0 ) {
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

            swap(next_n.inv_puzzle[next_n.puzzle[new_x * N + new_y]], next_n.inv_puzzle[next_n.puzzle[s_x * N + s_y]]); 
            swap(next_n.puzzle[new_x * N + new_y], next_n.puzzle[s_x * N + s_y]);
            next_n.h = pd.get_hash_value(next_n.inv_puzzle);

            next_n.space = new_x * N + new_y;
            // assert(get_md_sum(new_n.puzzle) == new_n.h);
            next_n.depth++;
            next_n.pre = i;
            // if(next_n.h == 0) {                                         
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

#else

void divide_root_set(Node root, Node *new_root_set, int *new_root_set_index, int divide_num){
    stack<Node> st;
    st.push(root);
    while(!st.empty() && st.size() < divide_num ) {
        Node cur_n = st.top();
        st.pop();
        if(cur_n.h == 0 ) {
            st.push(cur_n);
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

            swap(next_n.inv_puzzle[next_n.puzzle[new_x * N + new_y]], next_n.inv_puzzle[next_n.puzzle[s_x * N + s_y]]); 
            swap(next_n.puzzle[new_x * N + new_y], next_n.puzzle[s_x * N + s_y]);
            next_n.space = new_x * N + new_y;
            next_n.h = pd.get_hash_value(next_n.inv_puzzle);
            next_n.depth++;
            next_n.pre = i;
            // if(next_n.h == 0) {
            //     cout << "find ans in divide" << endl;
            // } 
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


Node root_set[MAX_CORE_NUM];
int load_set[MAX_CORE_NUM];
Node new_root_set[MAX_CORE_NUM];

void ida_star() {
    pq = priority_queue<Node, vector<Node>, greater<Node> >();
    if(create_root_set()) {
        printf("%d\n", ans);
        return;
    }
    int root_node_size = pq.size();
    int idx = 0;
    while(!pq.empty()) {
        Node n = pq.top();
        pq.pop();
        root_set[idx] = n;
        idx++;
    }


    for (int limit = s_node.h; limit < 100; ++limit, ++limit)
    {
        #ifdef DEBUG
        auto start = std::chrono::system_clock::now();
        #endif
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

        #ifdef USE_LOCK
        Lock    lock[BLOCK_NUM];
        Lock    *dev_lock;
        HANDLE_ERROR( cudaMalloc( (void**)&dev_lock,
                              BLOCK_NUM * sizeof( Lock ) ) );
        HANDLE_ERROR( cudaMemcpy( dev_lock, lock,
                              BLOCK_NUM * sizeof( Lock ),
                              cudaMemcpyHostToDevice ) );
        #endif

        HANDLE_ERROR(cudaMalloc((void**)&dev_load_set, root_node_size * sizeof(int)));
        HANDLE_ERROR(cudaMemset(dev_load_set, 0, root_node_size * sizeof(int)));

        #ifdef DEBUG
        cout << "f_limit : " << limit << endl;
        #endif

        #ifdef USE_LOCK
        dfs_kernel<<<root_node_size, WARP_SIZE>>>(limit, dev_root_set, dev_flag, dev_lock, dev_load_set, dev_pd);
        #else
        dfs_kernel<<<root_node_size, WARP_SIZE>>>(limit, dev_root_set, dev_flag, dev_load_set, dev_pd);
        #endif


        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaMemcpy(&flag, dev_flag, sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(&load_set, dev_load_set, root_node_size * sizeof(int), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaFree(dev_flag));
        HANDLE_ERROR(cudaFree(dev_root_set));
        #ifdef USE_LOCK
        HANDLE_ERROR(cudaFree(dev_lock));
        #endif
        HANDLE_ERROR(cudaFree(dev_load_set));

        if(flag != -1) {
            cout << flag << endl;
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

        if(flag != -1) {
            cout << flag << endl;
            return;
        }
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

            if((divide_num > 1 && new_root_node_size + root_node_size - i < MAX_BLOCK_SIZE) || (divide_num > 2 && new_root_node_size + root_node_size - i < MAX_BLOCK_SIZE/2)) {
                #ifdef DEBUG
                int tmp = new_root_node_size;
                #endif
                divide_root_set(root_set[i], new_root_set, &new_root_node_size, divide_num);
                #ifdef DEBUG
                // cout << tmp << " " << new_root_node_size << endl;
                assert(tmp <= new_root_node_size);
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
    #ifndef DEBUG
    FILE *output_file;
    output_file = fopen("../result/yama24_hard_new_block_parallel_result_with_pdb_2048_dfs.csv","w");
    #endif

    #ifdef DEBUG
    set_md();
    #endif

    pd = PatternDataBase();
    pd.init();
    //gpu側のメモリ割当て
    HANDLE_ERROR(cudaMalloc((void**)&dev_pd, sizeof(local_pdb) ) );
    local_pdb *lpdb = new local_pdb();
    //root_setをGPU側のdev_pdにコピー
    HANDLE_ERROR(cudaMemcpy(dev_pd, lpdb, sizeof(local_pdb), cudaMemcpyHostToDevice) );

    HANDLE_ERROR(cudaMemcpyToSymbol(dev_h0, &h0, PDB_TABLESIZE * sizeof(unsigned char)));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_h1, &h1, PDB_TABLESIZE * sizeof(unsigned char)));
    for (int i = 0; i < 50; ++i)
    {           
        string input_file = "../benchmarks/yama24_50_hard_new/prob";
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
        #endif
    }
    HANDLE_ERROR(cudaFree(dev_pd));
    cudaDeviceReset();

    #ifndef DEBUG
    fclose(output_file);
    #endif
}
