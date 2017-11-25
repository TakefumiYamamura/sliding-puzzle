#include <stdbool.h>
#include <stdio.h>

typedef unsigned char uchar;

#define N_THREADS 1
#define N_BLOCKS 96
#define PLAN_LEN_MAX 255

typedef uchar Direction;
#define dir_reverse(dir) ((Direction)(3 - (dir)))
#define DIR_N 4
#define DIR_FIRST 0
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

/* stack implementation */

__device__ __shared__ static struct dir_stack_tag
{
    uchar i;
    uchar buf[PLAN_LEN_MAX];
} stack[N_THREADS];

#define STACK_I (stack[threadIdx.x].i)
#define stack_get(i) (stack[threadIdx.x].buf[i])
#define stack_set(i, val) (stack[threadIdx.x].buf[i] = (val))

__device__ static inline void
stack_init(void)
{
    STACK_I = 0;
}
__device__ static inline void
stack_put(Direction dir)
{
    stack_set(STACK_I, dir);
    ++STACK_I;
}
__device__ static inline bool
stack_is_empty(void)
{
    return STACK_I == 0;
}
__device__ static inline Direction
stack_pop(void)
{
    --STACK_I;
    return stack_get(STACK_I);
}
__device__ static inline Direction
stack_peak(void)
{
    return stack_get(STACK_I - 1);
}

/* state implementation */

#define STATE_WIDTH 4
#define STATE_N (STATE_WIDTH * STATE_WIDTH)

static char assert_state_width_is_four[STATE_WIDTH == 4 ? 1 : -1];
#define POS_X(pos) ((pos) &3)
#define POS_Y(pos) ((pos) >> 2)

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

__device__ __shared__ static struct state_tag
{
    uchar tile[STATE_N];
    uchar empty;
    uchar h_value; /* ub of h_value is 6*16 */
} state[N_THREADS];

#define STATE_TILE(i) (state[threadIdx.x].tile[(i)])
#define STATE_EMPTY (state[threadIdx.x].empty)
#define STATE_HVALUE (state[threadIdx.x].h_value)

__device__ static uchar inline distance(uchar i, uchar j)
{
    return i > j ? i - j : j - i;
}

#define H_DIFF(opponent, empty, empty_dir)                                     \
    h_diff_table[opponent][empty][empty_dir]
__device__ __shared__ static signed char h_diff_table[STATE_N][STATE_N][DIR_N];

__device__ static void
init_mdist(void)
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
                    H_DIFF(opponent, i, dir) = goal_x > from_x ? -1 : 1;
                if (dir == DIR_RIGHT)
                    H_DIFF(opponent, i, dir) = goal_x < from_x ? -1 : 1;
                if (dir == DIR_UP)
                    H_DIFF(opponent, i, dir) = goal_y > from_y ? -1 : 1;
                if (dir == DIR_DOWN)
                    H_DIFF(opponent, i, dir) = goal_y < from_y ? -1 : 1;
            }
        }
    }
}

__device__ static inline void
state_init_hvalue(void)
{
    uchar from_x[STATE_N], from_y[STATE_N];

    for (int i = 0; i < STATE_N; ++i)
    {
        from_x[STATE_TILE(i)] = POS_X(i);
        from_y[STATE_TILE(i)] = POS_Y(i);
    }
    for (int i = 1; i < STATE_N; ++i)
    {
        state[threadIdx.x].h_value += distance(from_x[i], POS_X(i));
        state[threadIdx.x].h_value += distance(from_y[i], POS_Y(i));
    }
}

__device__ static void
state_tile_fill(const uchar v_list[STATE_WIDTH * STATE_WIDTH])
{
    for (int i = 0; i < STATE_N; ++i)
    {
        if (v_list[i] == 0)
            STATE_EMPTY = i;
        STATE_TILE(i)   = v_list[i];
    }
}

__device__ static inline bool
state_is_goal(void)
{
    return state[threadIdx.x].h_value == 0;
}

static char assert_direction2
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ __shared__ static bool movable_table[STATE_N][DIR_N];

__device__ static void
init_movable_table(void)
{
    for (int i = 0; i < STATE_N; ++i)
        for (unsigned int d = 0; d < DIR_N; ++d)
        {
            if (d == DIR_RIGHT)
                movable_table[i][d] = (POS_X(i) < STATE_WIDTH - 1);
            else if (d == DIR_LEFT)
                movable_table[i][d] = (POS_X(i) > 0);
            else if (d == DIR_DOWN)
                movable_table[i][d] = (POS_Y(i) < STATE_WIDTH - 1);
            else if (d == DIR_UP)
                movable_table[i][d] = (POS_Y(i) > 0);
        }
}
__device__ static inline bool
state_movable(Direction dir)
{
    return movable_table[STATE_EMPTY][dir];
}

static char assert_direction
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ __constant__ const static int pos_diff_table[DIR_N] = {
    -STATE_WIDTH, 1, -1, +STATE_WIDTH};

__device__ static inline bool
state_move_with_limit(Direction dir, unsigned int f_limit)
{
    int new_empty   = STATE_EMPTY + pos_diff_table[dir];
    int opponent    = STATE_TILE(new_empty);
    int new_h_value = STATE_HVALUE + H_DIFF(opponent, new_empty, dir);

    if (STACK_I + 1 + new_h_value > f_limit)
        return false;

    STATE_HVALUE            = new_h_value;
    STATE_TILE(STATE_EMPTY) = opponent;
    STATE_EMPTY             = new_empty;

    return true;
}

__device__ static inline void
state_move(Direction dir)
{
    int new_empty = STATE_EMPTY + pos_diff_table[dir];
    int opponent  = STATE_TILE(new_empty);

    STATE_HVALUE += H_DIFF(opponent, new_empty, dir);
    STATE_TILE(STATE_EMPTY) = opponent;
    STATE_EMPTY             = new_empty;
}

/*
 * solver implementation
 */

__device__ static bool
idas_internal(int f_limit, long long *ret_nodes_expanded)
{
    uchar     dir            = 0;
    long long nodes_expanded = 0;

    for (;;)
    {
        if (state_is_goal())
        {
            *ret_nodes_expanded = nodes_expanded;
            return true;
        }

        if ((stack_is_empty() || stack_peak() != dir_reverse(dir)) &&
            state_movable(dir))
        {
            ++nodes_expanded;

            if (state_move_with_limit(dir, f_limit))
            {
                stack_put(dir);
                dir = 0;
                continue;
            }
        }

        while (++dir == DIR_N)
        {
            if (stack_is_empty())
            {
                *ret_nodes_expanded = nodes_expanded;
                return false;
            }

            dir = stack_pop();
            state_move(dir_reverse(dir));
        }
    }
}

__global__ void
idas_kernel(uchar *input, uchar *plan)
{
    long long nodes_expanded = 0;

    init_mdist();
    init_movable_table();

    stack_init();
    state_tile_fill(input); /* the same input for all */
    state_init_hvalue();

    for (int f_limit = STATE_HVALUE;; f_limit += 2)
    {
        nodes_expanded = 0;
        if (idas_internal(f_limit, &nodes_expanded))
            break;
    }

    plan[0] = (int) STACK_I; /* len of plan */
    for (uchar i    = 0; i < STACK_I; ++i)
        plan[i + 1] = stack_get(i);
}

/* host implementation */

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

static void
avoid_unused_static_assertions(void)
{
    (void) assert_direction[0];
    (void) assert_direction2[0];
    (void) assert_state_width_is_four[0];
}

#include <sys/time.h>

int
main(int argc, char *argv[])
{
    uchar  s_list[STATE_N];
    uchar *s_list_device;
    uchar  plan[PLAN_LEN_MAX];
    uchar *plan_device;
    int    insize  = sizeof(uchar) * STATE_N;
    int    outsize = sizeof(uchar) * PLAN_LEN_MAX;
    struct timeval s, e;
    gettimeofday(&s, NULL);


    if (argc < 2)
    {
        printf("usage: bin/cumain <ifname>\n");
        exit(EXIT_FAILURE);
    }

    load_state_from_file(argv[1], s_list);
    CUDA_CHECK(cudaMalloc((void **) &s_list_device, insize));
    CUDA_CHECK(cudaMalloc((void **) &plan_device, outsize));
    CUDA_CHECK(
        cudaMemcpy(s_list_device, s_list, insize, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	gettimeofday(&e, NULL);
	printf("time(init) = %lf\n", (e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec)*1.0E-6);

	gettimeofday(&s, NULL);
	idas_kernel<<<N_BLOCKS, N_THREADS>>>(s_list_device, plan_device);
	cudaDeviceSynchronize();
	gettimeofday(&e, NULL);
	printf("time(kernel) = %lf\n", (e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec)*1.0E-6);

    CUDA_CHECK(cudaMemcpy(plan, plan_device, outsize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(s_list_device));
    CUDA_CHECK(cudaFree(plan_device));

    CUDA_CHECK(cudaDeviceReset());

    printf("len=%d: ", (int) plan[0]);
    for (int i = 0; i < plan[0]; ++i)
        printf("%d ", (int) plan[i + 1]);
    putchar('\n');

    avoid_unused_static_assertions();

    return 0;
}
