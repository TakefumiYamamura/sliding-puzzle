#include <stdbool.h>
#include "idas_parallel_host.h"


/* stack implementation */

__device__ __shared__ static struct dir_stack_tag
{
    uchar i;
    uchar buf[STACK_BUF_BYTES];
} stack[WARP_SIZE];

#define stack_byte(i) (stack[threadIdx.x].buf[(i) >> STACK_DIR_BITS])
#define stack_ofs(i) ((i & STACK_DIR_MASK) << 1)
#define stack_get(i)                                                           \
    ((stack_byte(i) & (STACK_DIR_MASK << stack_ofs(i))) >> stack_ofs(i))

__device__ static inline void
stack_init(void)
{
    stack[threadIdx.x].i = 0;
}

__device__ static inline void
stack_put(DirDev dir)
{
    stack_byte(stack[threadIdx.x].i) &=
        ~(STACK_DIR_MASK << stack_ofs(stack[threadIdx.x].i));
    stack_byte(stack[threadIdx.x].i) |= dir << stack_ofs(stack[threadIdx.x].i);
    ++stack[threadIdx.x].i;
}
__device__ static inline bool
stack_is_empty(void)
{
    return stack[threadIdx.x].i == 0;
    /* how about !stack[threadIdx.x].i */
}
__device__ static inline DirDev
stack_pop(void)
{
    --stack[threadIdx.x].i;
    return stack_get(stack[threadIdx.x].i);
}
__device__ static inline DirDev
stack_peak(void)
{
    return stack_get(stack[threadIdx.x].i - 1);
}

/* state implementation */

#define STATE_EMPTY 0
#define STATE_TILE_BITS 4
#define STATE_TILE_MASK ((1ull << STATE_TILE_BITS) - 1)

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

__device__ __shared__ static struct state_tag
{
    unsigned long long tile; /* packed representation label(4bit)*16pos */
    uchar              empty;
    uchar              h_value; /* ub of h_value is 6*16 */
} state[WARP_SIZE];

#define state_tile_ofs(i) (i << 2)
#define state_tile_get(i)                                                      \
    ((state[threadIdx.x].tile & (STATE_TILE_MASK << state_tile_ofs(i))) >>     \
     state_tile_ofs(i))
#define state_tile_set(i, val)                                                 \
    do                                                                         \
    {                                                                          \
        state[threadIdx.x].tile &= ~((STATE_TILE_MASK) << state_tile_ofs(i));  \
        state[threadIdx.x].tile |= ((unsigned long long) val)                  \
                                   << state_tile_ofs(i);                       \
    } while (0)

__device__ static uchar inline distance(uchar i, uchar j)
{
    return i > j ? i - j : j - i;
}

#define H_DIFF(opponent, empty, empty_dir)                                     \
    h_diff_table_shared[opponent][empty][empty_dir]
__device__ __shared__ static signed char h_diff_table_shared[STATE_N][STATE_N]
                                                            [DIR_N];

__device__ static void
state_init_hvalue(void)
{
    uchar from_x[STATE_N], from_y[STATE_N];
    int   tid = threadIdx.x;

    state[tid].h_value = 0;

    for (int i = 0; i < STATE_N; ++i)
    {
        from_x[state_tile_get(i)] = POS_X(i);
        from_y[state_tile_get(i)] = POS_Y(i);
    }
    for (int i = 1; i < STATE_N; ++i)
    {
        state[tid].h_value += distance(from_x[i], POS_X(i));
        state[tid].h_value += distance(from_y[i], POS_Y(i));
    }
}

__device__ static void
state_tile_fill(const uchar v_list[STATE_WIDTH * STATE_WIDTH])
{
    for (int i = 0; i < STATE_N; ++i)
    {
        if (v_list[i] == STATE_EMPTY)
            state[threadIdx.x].empty = i;
        state_tile_set(i, v_list[i]);
    }
}

__device__ static inline bool
state_is_goal(void)
{
    return state[threadIdx.x].h_value == 0;
}

__device__ static char assert_direction2
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ __shared__ static bool movable_table_shared[STATE_N][DIR_N];

__device__ static inline bool
state_movable(DirDev dir)
{
    return movable_table_shared[state[threadIdx.x].empty][dir];
}

__device__ static char assert_direction
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ __constant__ const static int pos_diff_table[DIR_N] = {
    -STATE_WIDTH, 1, -1, +STATE_WIDTH};

__device__ static inline bool
state_move_with_limit(DirDev dir, unsigned int f_limit)
{
    int new_empty = state[threadIdx.x].empty + pos_diff_table[dir];
    int opponent  = state_tile_get(new_empty);
    int new_h_value =
        state[threadIdx.x].h_value + H_DIFF(opponent, new_empty, dir);

    if (stack[threadIdx.x].i + 1 + new_h_value > f_limit)
        return false;

    state[threadIdx.x].h_value = new_h_value;
    state_tile_set(state[threadIdx.x].empty, opponent);
    state[threadIdx.x].empty = new_empty;

    return true;
}

__device__ static inline void
state_move(DirDev dir)
{
    int new_empty = state[threadIdx.x].empty + pos_diff_table[dir];
    int opponent  = state_tile_get(new_empty);

    state[threadIdx.x].h_value += H_DIFF(opponent, new_empty, dir);
    state_tile_set(state[threadIdx.x].empty, opponent);
    state[threadIdx.x].empty = new_empty;
}

/*
 * solver implementation
 */

__device__ static bool
idas_internal(uchar f_limit)
{
    uchar dir = 0;

    for (;;)
    {
        if (state_is_goal())
            return true;

        if ((stack_is_empty() || stack_peak() != dir_reverse_dev(dir)) &&
            state_movable(dir))
        {
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
                return false;

            dir = stack_pop();
            state_move(dir_reverse_dev(dir));
        }
    }
}

__global__ void
idas_kernel(uchar *input, signed char *plan, int f_limit,
            signed char *h_diff_table, bool *movable_table)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id  = tid + bid * blockDim.x;

    for (int dir = 0; dir < DIR_N; ++dir)
        if (tid < STATE_N)
            movable_table_shared[tid][dir] = movable_table[tid * DIR_N + dir];
    for (int i = 0; i < STATE_N * DIR_N; ++i)
        if (tid < STATE_N)
            h_diff_table_shared[tid][i / DIR_N][i % DIR_N] =
                h_diff_table[tid * STATE_N * DIR_N + i];

    __syncthreads();

    stack_init();
    state_tile_fill(input + id * STATE_N);
    state_init_hvalue();

    if (idas_internal(f_limit))
    {
        plan[id * PLAN_LEN_MAX] = (signed char) stack[tid].i; /* len of plan */
        for (uchar i                        = 0; i < stack[tid].i; ++i)
            plan[i + 1 + id * PLAN_LEN_MAX] = stack_get(i);
    }
    else
        plan[id * PLAN_LEN_MAX] = NOT_SOLVED;
}

void idas_parallel_main(uchar *input, signed char *plan, int f_limit,
signed char *h_diff_table, bool *movable_table)
{
    (void) assert_direction[0];
    (void) assert_direction2[0];
    idas_kernel<<<N_BLOCK, N_CORE / N_BLOCK>>>(input, plan, f_limit, h_diff_table, movable_table);
}
