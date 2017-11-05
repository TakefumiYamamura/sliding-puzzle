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

template <typename T> std::string tostr(const T& t)
{
    std::ostringstream os; os<<t; return os.str();
}
 
#define N 5
#define N2 25
#define PDB_TABLESIZE 244140625
#define STACK_LIMIT 64 * 8
#define CORE_NUM 768
#define WARP_SIZE 32
#define BLOCK_NUM 24

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

static __device__ __constant__ const int dev_rf[] = {0,5,10,15,20,1,6,11,16,21,2,7,12,17,22,3,8,13,18,23,4,9,14,19,24};
static __device__ __constant__ const int dev_rot90[] = {20,15,10,5,0,21,16,11,6,1,22,17,12,7,2,23,18,13,8,3,24,19,14,9,4};
static __device__ __constant__ const int dev_rot90rf[] = {20,21,22,23,24,15,16,17,18,19,10,11,12,13,14,5,6,7,8,9,0,1,2,3,4};
static __device__ __constant__ const int dev_rot180[] = {24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
static __device__ __constant__ const int dev_rot180rf[] = {24,19,14,9,4,23,18,13,8,3,22,17,12,7,2,21,16,11,6,1,20,15,10,5,0};

static  const int rf[] = {0,5,10,15,20,1,6,11,16,21,2,7,12,17,22,3,8,13,18,23,4,9,14,19,24};
static  const int rot90[] = {20,15,10,5,0,21,16,11,6,1,22,17,12,7,2,23,18,13,8,3,24,19,14,9,4};
static  const int rot90rf[] = {20,21,22,23,24,15,16,17,18,19,10,11,12,13,14,5,6,7,8,9,0,1,2,3,4};
static  const int rot180[] = {24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
static  const int rot180rf[] = {24,19,14,9,4,23,18,13,8,3,22,17,12,7,2,21,16,11,6,1,20,15,10,5,0};


__device__ unsigned char dev_h0[PDB_TABLESIZE];
__device__ unsigned char dev_h1[PDB_TABLESIZE];

unsigned char h0[PDB_TABLESIZE];
unsigned char h1[PDB_TABLESIZE];

struct Node
{
    int puzzle[N2];
    int inv_puzzle[N2];
    int space;
    // int md;
    int h;
    int depth;
    int pre;
    bool operator < (const Node& n) const {
        return depth + h < n.depth + n.h;
    }

    bool operator > (const Node& n) const {
        return depth + h > n.depth + n.h;
    }
};

template<class T, int NUM>
class local_stack
{
private:
    T buf[NUM];
    int tos;

public:
    __device__ local_stack() :
    tos(-1)
    {
    }

    __device__ T const & top() const
    {
        return buf[tos];
    }

    __device__ T & top()
    {
        return buf[tos];
    }

    __device__ void push(T const & v)
    {
        buf[++tos] = v;
    }

    __device__ T pop()
    {
        return buf[tos--];
    }

    __device__ bool full()
    {
        return tos == (NUM - 1);
    }

    __device__ bool empty()
    {
        return tos == -1;
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
    unsigned int hash0(const int *inv);
    unsigned int hash1(const int *inv);
    unsigned int hash2(const int *inv);
    unsigned int hash3(const int *inv);
    unsigned int hashref0(const int *inv);
    unsigned int hashref1(const int *inv);
    unsigned int hashref2(const int *inv);
    unsigned int hashref3(const int *inv);
    unsigned int get_hash_value(const int *inv);
    // unsigned char get_h0_value(int i);
    // unsigned char get_h1_value(int i);
};

PatternDataBase::PatternDataBase() {}

void PatternDataBase::init() {
    const char *c0 = "../pdb/pat24.1256712.tab";
    const char *c1 = "../pdb/pat24.34891314.tab";
    cout << "pattern 1 2 5 6 7 12 read in" << endl;
    input_h0(c0);
    cout << "pattern 3 4 8 9 13 14 read in" << endl;
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

unsigned int PatternDataBase::hash0(const int *inv) {
    int hashval;
    hashval = ((((inv[1]*N2+inv[2])*N2+inv[5])*N2+inv[6])*N2+inv[7])*N2+inv[12];
    return h0[hashval];
}

unsigned int PatternDataBase::hash1(const int *inv) {
    int hashval;
    hashval = ((((inv[3]*N2+inv[4])*N2+inv[8])*N2+inv[9])*N2+inv[13])*N2+inv[14];
    return (h1[hashval]);
}

unsigned int PatternDataBase::hash2(const int *inv) {
    int hashval;
    hashval = ((((rot180[inv[21]] * N2
              + rot180[inv[20]]) * N2
             + rot180[inv[16]]) * N2
            + rot180[inv[15]]) * N2
           + rot180[inv[11]]) * N2
          + rot180[inv[10]];
    return (h1[hashval]);
}

unsigned int PatternDataBase::hash3(const int *inv) {
    int hashval;
    hashval = ((((rot90[inv[19]] * N2
              + rot90[inv[24]]) * N2
             + rot90[inv[18]]) * N2
            + rot90[inv[23]]) * N2
           + rot90[inv[17]]) * N2
          + rot90[inv[22]];
    return (h1[hashval]);
}

unsigned int PatternDataBase::hashref0(const int *inv) {
    int hashval;
    hashval = (((((rf[inv[5]] * N2
               + rf[inv[10]]) * N2
              + rf[inv[1]]) * N2
             + rf[inv[6]]) * N2
            + rf[inv[11]]) * N2
           + rf[inv[12]]);
    return (h0[hashval]);
}

unsigned int PatternDataBase::hashref1(const int *inv) {
    int hashval;
    hashval = (((((rf[inv[15]] * N2
               + rf[inv[20]]) * N2
              + rf[inv[16]]) * N2
             + rf[inv[21]]) * N2
            + rf[inv[17]]) * N2
           + rf[inv[22]]);
    return (h1[hashval]);
}
unsigned int PatternDataBase::hashref2(const int *inv) {
    int hashval;
    hashval = (((((rot180rf[inv[9]] * N2
               + rot180rf[inv[4]]) * N2
              + rot180rf[inv[8]]) * N2
             + rot180rf[inv[3]]) * N2
            + rot180rf[inv[7]]) * N2
           + rot180rf[inv[2]]);
    return (h1[hashval]);
}

unsigned int PatternDataBase::hashref3(const int *inv) {
    int hashval;
    hashval = (((((rot90rf[inv[23]] * N2
               + rot90rf[inv[24]]) * N2
              + rot90rf[inv[18]]) * N2
             + rot90rf[inv[19]]) * N2
            + rot90rf[inv[13]]) * N2
           + rot90rf[inv[14]]);
    return (h1[hashval]);
}

unsigned int PatternDataBase::get_hash_value(const int *inv) {
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
    __device__ unsigned int hash0(const int *inv);
    __device__ unsigned int hash1(const int *inv);
    __device__ unsigned int hash2(const int *inv);
    __device__ unsigned int hash3(const int *inv);
    __device__ unsigned int hashref0(const int *inv);
    __device__ unsigned int hashref1(const int *inv);
    __device__ unsigned int hashref2(const int *inv);
    __device__ unsigned int hashref3(const int *inv);
    __device__ unsigned int get_hash_value(const int *inv);

};

local_pdb::local_pdb() {
    // HANDLE_ERROR(cudaMemcpy(dev_h0, h0, PDB_TABLESIZE * sizeof(unsigned char), cudaMemcpyHostToDevice) );
    // HANDLE_ERROR(cudaMemcpy(dev_h1, h1, PDB_TABLESIZE * sizeof(unsigned char), cudaMemcpyHostToDevice) );
}


__device__ unsigned int local_pdb::hash0(const int *inv) {
    int hashval;
    hashval = ((((inv[1]*N2+inv[2])*N2+inv[5])*N2+inv[6])*N2+inv[7])*N2+inv[12];
    return dev_h0[hashval];
}

__device__ unsigned int local_pdb::hash1(const int *inv) {
    int hashval;
    hashval = ((((inv[3]*N2+inv[4])*N2+inv[8])*N2+inv[9])*N2+inv[13])*N2+inv[14];
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::hash2(const int *inv) {
    int hashval;
    hashval = ((((dev_rot180[inv[21]] * N2
              + dev_rot180[inv[20]]) * N2
             + dev_rot180[inv[16]]) * N2
            + dev_rot180[inv[15]]) * N2
           + dev_rot180[inv[11]]) * N2
          + dev_rot180[inv[10]];
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::hash3(const int *inv) {
    int hashval;
    hashval = ((((dev_rot90[inv[19]] * N2
              + dev_rot90[inv[24]]) * N2
             + dev_rot90[inv[18]]) * N2
            + dev_rot90[inv[23]]) * N2
           + dev_rot90[inv[17]]) * N2
          + dev_rot90[inv[22]];
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::hashref0(const int *inv) {
    int hashval;
    hashval = (((((dev_rf[inv[5]] * N2
               + dev_rf[inv[10]]) * N2
              + dev_rf[inv[1]]) * N2
             + dev_rf[inv[6]]) * N2
            + dev_rf[inv[11]]) * N2
           + dev_rf[inv[12]]);
    return (dev_h0[hashval]);
}

__device__ unsigned int local_pdb::hashref1(const int *inv) {
    int hashval;
    hashval = (((((dev_rf[inv[15]] * N2
               + dev_rf[inv[20]]) * N2
              + dev_rf[inv[16]]) * N2
             + dev_rf[inv[21]]) * N2
            + dev_rf[inv[17]]) * N2
           + dev_rf[inv[22]]);
    return (dev_h1[hashval]);
}
__device__ unsigned int local_pdb::hashref2(const int *inv) {
    int hashval;
    hashval = (((((dev_rot180rf[inv[9]] * N2
               + dev_rot180rf[inv[4]]) * N2
              + dev_rot180rf[inv[8]]) * N2
             + dev_rot180rf[inv[3]]) * N2
            + dev_rot180rf[inv[7]]) * N2
           + dev_rot180rf[inv[2]]);
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::hashref3(const int *inv) {
    int hashval;
    hashval = (((((dev_rot90rf[inv[23]] * N2
               + dev_rot90rf[inv[24]]) * N2
              + dev_rot90rf[inv[18]]) * N2
             + dev_rot90rf[inv[19]]) * N2
            + dev_rot90rf[inv[13]]) * N2
           + dev_rot90rf[inv[14]]);
    return (dev_h1[hashval]);
}

__device__ unsigned int local_pdb::get_hash_value(const int *inv) {
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



bool dfs(int limit, Node s_n) {
    stack<Node> st;
    st.push(s_n);

    while(!st.empty()) {
        Node cur_n = st.top();
        st.pop();
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
 
            swap(next_n.puzzle[new_x * N + new_y], next_n.puzzle[s_x * N + s_y]);
            swap(next_n.inv_puzzle[new_x * N + new_y], next_n.inv_puzzle[s_x * N + s_y]);
            next_n.space = new_x * N + new_y;
            next_n.h = pd.get_hash_value(cur_n.inv_puzzle);
            // assert(get_md_sum(new_n.puzzle) == new_n.h);
            // return dfs(new_n, depth+1, i);
            next_n.depth++;
            if(cur_n.depth + cur_n.h > limit) continue;
            next_n.pre = i;
            st.push(next_n);
            if(next_n.h == 0) {
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
 
 
            // swap(next_n.puzzle[new_x * N + new_y], next_n.puzzle[s_x * N + s_y]);
            // swap(next_n.inv_puzzle[new_x * N + new_y], next_n.inv_puzzle[s_x * N + s_y]);
            next_n.space = new_x * N + new_y;
            next_n.h = pd.get_hash_value(next_n.inv_puzzle);

            next_n.depth++;
            next_n.pre = i;
            if(next_n.h == 0) {
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

__global__ void dfs_kernel(int limit, Node *root_set, int *dev_flag, local_pdb *dev_pdb) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    local_stack<Node, STACK_LIMIT> st;
    st.push(root_set[idx]);

    int order[4] = {1, 0, 2, 3};
    int dx[4] = {0, -1, 0, 1};
    int dy[4] = {1, 0, -1, 0};

    while(!st.empty()) {
        Node cur_n = st.top();
        st.pop();
        if(cur_n.h == 0 ) {
            // ans = cur_n.depth;
            *dev_flag = cur_n.depth;
            return;
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
 

 
            int a = next_n.inv_puzzle[next_n.puzzle[new_x * N + new_y]];
            int b = next_n.inv_puzzle[next_n.puzzle[s_x * N + s_y]];
            next_n.inv_puzzle[next_n.puzzle[new_x * N + new_y]] = b;
            next_n.inv_puzzle[next_n.puzzle[s_x * N + s_y]] = a;

            int c = next_n.puzzle[new_x * N + new_y];
            int d = next_n.puzzle[s_x * N + s_y];
            next_n.puzzle[new_x * N + new_y] = d;
            next_n.puzzle[s_x * N + s_y] = c;

            next_n.space = new_x * N + new_y;
            next_n.h = dev_pdb->get_hash_value(next_n.inv_puzzle);
            next_n.depth++;
            if(cur_n.depth + cur_n.h > limit) continue;
            next_n.pre = i;
            st.push(next_n);
            if(next_n.h == 0) {
                // ans = next_n.depth;
                *dev_flag = next_n.depth;
                return;
            }
        }
    }
    return;

}

void ida_star() {
    // cout << "before_create_root" << endl;
    pq = priority_queue<Node, vector<Node>, greater<Node> >();
    if(create_root_set()) {
        printf("%d\n", ans);
        return;
    }
    // cout << "after_create_root" << endl;

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
    // cout << "md" << s_node.h << endl;
    for (int limit = s_node.h; limit < 100; ++limit, ++limit)
    {
        // path.resize(limit);
        // priority_queue<Node, vector<Node>, greater<Node> > tmp_pq = pq;

        int flag = -1;
        int *dev_flag;

        //gpu側にメモリ割当
        HANDLE_ERROR(cudaMalloc( (void**)&dev_flag, sizeof(int) ) );
        HANDLE_ERROR(cudaMemcpy(dev_flag, &flag, sizeof(int), cudaMemcpyHostToDevice));
        dfs_kernel<<<BLOCK_NUM, WARP_SIZE>>>(limit, dev_root_set, dev_flag, dev_pd);
        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaMemcpy(&flag, dev_flag, sizeof(int), cudaMemcpyDeviceToHost));
        if(flag != -1) {
            cout << flag << endl;
            HANDLE_ERROR(cudaFree(dev_flag) );
            HANDLE_ERROR(cudaFree(dev_root_set));
            return;
        }
        HANDLE_ERROR(cudaFree(dev_flag) );
    }
    HANDLE_ERROR(cudaFree(dev_root_set));

}

 
int main() {

    FILE *output_file;
    // output_file = fopen("../result/yama24_psimple_with_pdb_result.csv","w");
    output_file = fopen("../result/yama24_med_psimple_with_pdb_result.csv","w");

    // set_md();
    // pattern database 
    pd = PatternDataBase();
    pd.init();
    //gpu側のメモリ割当て
    HANDLE_ERROR(cudaMalloc((void**)&dev_pd, sizeof(local_pdb) ) );
    local_pdb *lpdb = new local_pdb();
    //root_setをGPU側のdev_pdにコピー
    HANDLE_ERROR(cudaMemcpy(dev_pd, lpdb, sizeof(local_pdb), cudaMemcpyHostToDevice) );

    HANDLE_ERROR(cudaMemcpyToSymbol(dev_h0, &h0, PDB_TABLESIZE * sizeof(unsigned char)));
    HANDLE_ERROR(cudaMemcpyToSymbol(dev_h1, &h1, PDB_TABLESIZE * sizeof(unsigned char)));

    for (int i = 0; i <= 50; ++i)
    {
        // string input_file = "../benchmarks/yama24_50_easy/prob";
        string input_file = "../benchmarks/yama24_50/prob";
        string input_file = "../benchmarks/yama24_50_med/prob";
        // string input_file = "../benchmarks/korf100/prob";
        if(i < 10) {
            input_file += "00";
        } else if(i < 100) {
            input_file += "0";
        }
        input_file += tostr(i);
        cout << input_file << " ";
        // set_md();

        clock_t start = clock();

        input_table(const_cast<char*>(input_file.c_str()));
        ida_star();

        clock_t end = clock();
        fprintf(output_file,"%f\n", (double)(end - start) / CLOCKS_PER_SEC);

        // writing_file << (double)(end - start) / CLOCKS_PER_SEC << endl;
    }
    HANDLE_ERROR(cudaFree(dev_pd));
    // HANDLE_ERROR(cudaFree(dev_h0));
    // HANDLE_ERROR(cudaFree(dev_h1));
    fclose(output_file);
    cudaDeviceReset();
}
