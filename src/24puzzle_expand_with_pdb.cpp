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
#include <chrono>

#define N 5
#define N2 25
#define PDB_TABLESIZE 244140625

using namespace std;

// const int N = 4;
// const int N2 = 16;

static const int dx[4] = {0, -1, 0, 1};
static const int dy[4] = {1, 0, -1, 0};
static const char dir[4] = {'r', 'u', 'l', 'd'};
static const int order[4] = {1, 0, 2, 3};

unsigned char h0[PDB_TABLESIZE];
unsigned char h1[PDB_TABLESIZE];


/* the position of each tile in order, reflected about the main diagonal */
static const int rf[N2] = {0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24};

/* rotates the puzzle 90 degrees */
static const int rot90[N2] = {20, 15, 10, 5, 0, 21, 16, 11, 6, 1, 22, 17, 12, 7, 2, 23, 18, 13, 8, 3, 24, 19, 14, 9, 4};

/* composes the reflection and 90 degree rotation into a single array */
static const int rot90rf[N2] = {20, 21, 22, 23, 24, 15, 16, 17, 18, 19, 10, 11, 12, 13, 14, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4};

/* rotates the puzzle 180 degrees */
static const int rot180[N2] = {24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

/* composes the reflection and 180 degree rotation into a single array */
static const int rot180rf[N2] = {24, 19, 14, 9, 4, 23, 18, 13, 8, 3, 22, 17, 12, 7, 2, 21, 16, 11, 6, 1, 20, 15, 10, 5, 0};


struct Node
{
    int puzzle[N2];
    int inv_puzzle[N2];
    int space;
    int h;
};

class PatternDataBase
{
private:


public:
    PatternDataBase();
    void input(const char *filename, unsigned char *table);
    unsigned int hash0(const int *inv);
    unsigned int hash1(const int *inv);
    unsigned int hash2(const int *inv);
    unsigned int hash3(const int *inv);
    unsigned int hashref0(const int *inv);
    unsigned int hashref1(const int *inv);
    unsigned int hashref2(const int *inv);
    unsigned int hashref3(const int *inv);
    unsigned int get_hash_value(const int *inv);

};

PatternDataBase::PatternDataBase() {
    const char *c0 = "../pdb/pat24.1256712.tab";
    const char *c1 = "../pdb/pat24.34891314.tab";
    cout << "pattern 1 2 5 6 7 12 read in ";
    input(c0, h0);
    cout << "pattern 3 4 8 9 13 14 read in ";
    input(c1, h1);
    cout << "pdb is installed." << endl;
}

void PatternDataBase::input(const char *filename, unsigned char *table) {
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
                            table[index] = getc(infile);
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

PatternDataBase *pd;

class Npuzzle
{
private:
    Node s_n;
    Node cur_n;
    int limit;
    int node_num;
    vector<int> path;
    int ans;
    // PatternDataBase *pd;
public:
    Npuzzle(string input_file);
    bool dfs(int depth, int pre);
    void ida_star();
};

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
        s_n.inv_puzzle[tmp] = i;
    }
    s_n.h = pd->get_hash_value(s_n.inv_puzzle);
    // cout << s_n.h << endl;
}

bool Npuzzle::dfs(int depth, int pre) {
    if(cur_n.h == 0 ) {
        ans = depth;
        // return true;
    }
    if(depth + cur_n.h > limit) return false;
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
        node_num++;
        // //incremental manhattan distance
        // cur_n.md -= md[new_x * N + new_y][cur_n.puzzle[new_x * N + new_y]];
        // cur_n.md += md[s_x * N + s_y][cur_n.puzzle[new_x * N + new_y]];

        swap(cur_n.inv_puzzle[cur_n.puzzle[new_x * N + new_y]], cur_n.inv_puzzle[cur_n.puzzle[s_x * N + s_y]]); 
        swap(cur_n.puzzle[new_x * N + new_y], cur_n.puzzle[s_x * N + s_y]);

        cur_n.h = pd->get_hash_value(cur_n.inv_puzzle);
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
    for (limit = s_n.h; limit < 1000; ++limit, ++limit)
    {
        // path.resize(limit);
        cur_n = s_n;
        ans = -1;
        dfs(0, -10)
        if(ans != -1) {
            // string str = "";
            // for (int i = 0; i < limit; ++i)
            // {
            //  str += dir[path[i]];
            // }
            // cout << str << endl;
            // cout << node_num << " ";
            cout << ans << " ";
            cout << node_num << endl;
            return;
        }
    }
}




int main() {
    // string output_file = "../result/korf50_result.csv";
    pd = new PatternDataBase();
    // string output_file = "../result/yama24_result_pdb_wo_cuda.csv";
    string output_file = "../result/yama24_med_result_pdb_expand.csv";
    ofstream writing_file;
    writing_file.open(output_file, std::ios::out);
    // vector<int> test_array = {25, 32};
//1, 5, 13, 25, 30, 32, 37, 38, 40, 44
    for (int i = 0; i <= 50; ++i)
    // for (auto i : test_array)
    {
// string input_file = "../benchmarks/korf50_24puzzle/";
// string input_file = "../benchmarks/yama24_50/prob";
        string input_file = "../benchmarks/yama24_50_med/prob";
        if(i < 10) {
        input_file += "00";
        } else if(i < 100) {
        input_file += "0";
        }
        input_file += to_string(i);
        cout << input_file << " ";
        // clock_t start = clock();
        auto start = std::chrono::system_clock::now();
        Npuzzle np = Npuzzle(input_file);
        np.ida_star();
        auto end = std::chrono::system_clock::now();
        // clock_t end = clock();
        auto diff = end - start;
        writing_file << std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / 1000000000.0 << endl;
    }
}

