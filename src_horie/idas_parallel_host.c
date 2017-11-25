#include "./idas_parallel_host.h"

#include <cuda_runtime_api.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define exit_failure(...)                                                      \
    do                                                                         \
    {                                                                          \
        printf(__VA_ARGS__);                                                   \
        exit(EXIT_FAILURE);                                                    \
    } while (0)

static int
pop_int_from_str(const char *str, char **end_ptr)
{
    long int rv = strtol(str, end_ptr, 0);
    errno       = 0;

    if (errno != 0)
        exit_failure("%s: %s cannot be converted into long\n", __func__, str);
    else if (end_ptr && str == *end_ptr)
        exit_failure("%s: reach end of string", __func__);

    if (rv > INT_MAX || rv < INT_MIN)
        exit_failure("%s: too big number, %ld\n", __func__, rv);

    return (int) rv;
}

#define MAX_LINE_LEN 100
static void
load_state_from_file(const char *fname, uchar *s)
{
    FILE *fp;
    char  str[MAX_LINE_LEN];
    char *str_ptr = str, *end_ptr;

    fp = fopen(fname, "r");
    if (!fp)
        exit_failure("%s: %s cannot be opened\n", __func__, fname);

    if (!fgets(str, MAX_LINE_LEN, fp))
        exit_failure("%s: fgets failed\n", __func__);

    for (int i = 0; i < STATE_N; ++i)
    {
        s[i]    = pop_int_from_str(str_ptr, &end_ptr);
        str_ptr = end_ptr;
    }

    fclose(fp);
}
#undef MAX_LINE_LEN

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        const cudaError_t e = call;                                            \
        if (e != cudaSuccess)                                                  \
            exit_failure("Error: %s:%d code:%d, reason: %s\n", __FILE__,       \
                         __LINE__, e, cudaGetErrorString(e));                  \
    } while (0)

__host__ static int
calc_hvalue(uchar s_list[])
{
    int from_x[STATE_N], from_y[STATE_N];
    int h_value = 0;

    for (int i = 0; i < STATE_N; ++i)
    {
        from_x[s_list[i]] = POS_X(i);
        from_y[s_list[i]] = POS_Y(i);
    }
    for (int i = 1; i < STATE_N; ++i)
    {
        h_value += abs(from_x[i] - POS_X(i));
        h_value += abs(from_y[i] - POS_Y(i));
    }
    return h_value;
}

#define h_d_t(op, i, dir)                                                      \
    (h_diff_table[(op) *STATE_N * DIR_N + (i) *DIR_N + (dir)])
__host__ static void
init_mdist(signed char h_diff_table[])
{
    for (int opponent = 0; opponent < STATE_N; ++opponent)
    {
        int goal_x = POS_X(opponent), goal_y = POS_Y(opponent);

        for (int i = 0; i < STATE_N; ++i)
        {
            int from_x = POS_X(i), from_y = POS_Y(i);
            for (uchar dir = 0; dir < DIR_N; ++dir)
            {
                if (dir == DIR_LEFT)
                    h_d_t(opponent, i, dir) = goal_x > from_x ? -1 : 1;
                if (dir == DIR_RIGHT)
                    h_d_t(opponent, i, dir) = goal_x < from_x ? -1 : 1;
                if (dir == DIR_UP)
                    h_d_t(opponent, i, dir) = goal_y > from_y ? -1 : 1;
                if (dir == DIR_DOWN)
                    h_d_t(opponent, i, dir) = goal_y < from_y ? -1 : 1;
            }
        }
    }
}
#undef h_d_t

#define m_t(i, d) (movable_table[(i) *DIR_N + (d)])
__host__ static void
init_movable_table(bool movable_table[])
{
    for (int i = 0; i < STATE_N; ++i)
        for (unsigned int d = 0; d < DIR_N; ++d)
        {
            if (d == DIR_RIGHT)
                m_t(i, d) = (POS_X(i) < STATE_WIDTH - 1);
            else if (d == DIR_LEFT)
                m_t(i, d) = (POS_X(i) > 0);
            else if (d == DIR_DOWN)
                m_t(i, d) = (POS_Y(i) < STATE_WIDTH - 1);
            else if (d == DIR_UP)
                m_t(i, d) = (POS_Y(i) > 0);
        }
}
#undef m_t

#include "./distributor.h"
#include "./state.h"
static char dir_char[] = {'U', 'R', 'L', 'D'};

int
main(int argc, char *argv[])
{
    uchar  s_list[STATE_N * N_CORE];
    uchar *d_s_list;
    int    s_list_size = sizeof(uchar) * STATE_N * N_CORE;

    signed char  plan[PLAN_LEN_MAX * N_CORE];
    signed char *d_plan;
    int          plan_size = sizeof(signed char) * PLAN_LEN_MAX * N_CORE;

    int root_h_value = 0;

    bool  movable_table[STATE_N * DIR_N];
    bool *d_movable_table;
    int   movable_table_size = sizeof(bool) * STATE_N * DIR_N;

    signed char  h_diff_table[STATE_N * STATE_N * DIR_N];
    signed char *d_h_diff_table;
    int h_diff_table_size = sizeof(signed char) * STATE_N * STATE_N * DIR_N;

    if (argc < 2)
    {
        printf("usage: bin/cumain <ifname>\n");
        exit(EXIT_FAILURE);
    }

    load_state_from_file(argv[1], s_list);
    root_h_value = calc_hvalue(s_list);

    {
        uchar goal[STATE_N];
        State init_state = state_init(s_list, 0), goal_state;

        for (int i  = 0; i < STATE_N; ++i)
            goal[i] = i;
        goal_state  = state_init(goal, 0);

        if (distributor_bfs(init_state, goal_state, s_list, N_CORE))
        {
            puts("solution is found by distributor");
            return 0;
        }
    }

    init_mdist(h_diff_table);
    init_movable_table(movable_table);

    CUDA_CHECK(cudaMalloc((void **) &d_s_list, s_list_size));
    CUDA_CHECK(cudaMalloc((void **) &d_plan, plan_size));
    CUDA_CHECK(cudaMalloc((void **) &d_movable_table, movable_table_size));
    CUDA_CHECK(cudaMalloc((void **) &d_h_diff_table, h_diff_table_size));
    CUDA_CHECK(cudaMemcpy(d_movable_table, movable_table, movable_table_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h_diff_table, h_diff_table, h_diff_table_size,
                          cudaMemcpyHostToDevice));

    for (uchar f_limit = root_h_value;; ++f_limit)
    {
        printf("f=%d\n", (int) f_limit);
        CUDA_CHECK(
            cudaMemcpy(d_s_list, s_list, s_list_size, cudaMemcpyHostToDevice));

        printf("call idas_kernel(block=%d, thread=%d)\n", N_BLOCK,
               N_CORE / N_BLOCK);
        idas_parallel_main(d_s_list, d_plan, f_limit, d_h_diff_table,
                           d_movable_table);

        CUDA_CHECK(cudaMemcpy(plan, d_plan, plan_size, cudaMemcpyDeviceToHost));

        printf("len=%d: ", 0);
        for (unsigned int j = 0; j < 2 * PLAN_LEN_MAX; ++j)
            printf("%d ", (int) plan[j + 1]);
        putchar('\n');

        for (int i = 0; i < N_CORE; ++i)
            if (plan[i * PLAN_LEN_MAX] != NOT_SOLVED)
            {
                printf("len=%d: ", (int) plan[i * PLAN_LEN_MAX]);
                for (int j = 0; j < plan[i * PLAN_LEN_MAX]; ++j)
                    printf("%c ",
                           dir_char[(int) plan[i * PLAN_LEN_MAX + j + 1]]);
                putchar('\n');
                goto solution_found;
            }
    }
solution_found:

    CUDA_CHECK(cudaFree(d_s_list));
    CUDA_CHECK(cudaFree(d_plan));
    CUDA_CHECK(cudaFree(d_movable_table));
    CUDA_CHECK(cudaFree(d_h_diff_table));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
