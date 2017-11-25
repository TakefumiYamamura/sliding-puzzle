#include <stdbool.h>

#define BLOCK_DIM (32)
#define N_BLOCKS (48)
#define N_WORKERS (N_BLOCKS * BLOCK_DIM)
#define N_INIT_DISTRIBUTION (N_WORKERS * 4)
#define N_INPUTS (N_WORKERS * 8)
#define PLAN_LEN_MAX 64

#define STATE_WIDTH 4
#define STATE_N (STATE_WIDTH * STATE_WIDTH)

typedef unsigned char uchar;
typedef signed char   Direction;
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
    uchar i, j;
	uchar parent_dir;
    int   init_depth;
	int   input_i;
    uchar buf[PLAN_LEN_MAX];
} stack[BLOCK_DIM];

#define STACK (stack[threadIdx.x])

typedef struct search_stat_tag
{
    bool      solved;
    int       len;
    unsigned long long int nodes_expanded;
	int thread;
} search_stat;
typedef struct input_tag
{
    uchar     tiles[STATE_N];
    int       init_depth;
    Direction parent_dir;
} Input;

__device__ static inline void
stack_init(Input input, int input_i)
{
    STACK.i          = 0;
	STACK.j          = 0;
	STACK.input_i    = input_i;
	STACK.parent_dir = input.parent_dir;
    STACK.init_depth = input.init_depth;
}

__device__ static inline void
stack_put(Direction dir)
{
    STACK.buf[STACK.i] = dir;
    ++STACK.i;
}
__device__ static inline bool
stack_is_empty(void)
{
    return STACK.i <= STACK.j;
}
__device__ static inline Direction
stack_pop(void)
{
    --STACK.i;
    return STACK.buf[STACK.i];
}
__device__ static inline Direction
stack_peak(void)
{
    return STACK.buf[STACK.i - 1];
}

/* state implementation */

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
} state[BLOCK_DIM];

#define STATE_TILE(i) (state[threadIdx.x].tile[(i)])
#define STATE_EMPTY (state[threadIdx.x].empty)
#define STATE_HVALUE (state[threadIdx.x].h_value)
#define distance(i, j) ((i) > (j) ? (i) - (j) : (j) - (i))

#define H_DIFF(opponent, empty, empty_dir)                                     \
    h_diff_table_shared[opponent][empty][empty_dir]
__device__ __shared__ static signed char h_diff_table_shared[STATE_N][STATE_N]
                                                            [DIR_N];

__device__ static void
state_init_hvalue(void)
{
    uchar from_x[STATE_N], from_y[STATE_N];

    STATE_HVALUE = 0;

    for (int i = 0; i < STATE_N; ++i)
    {
        from_x[STATE_TILE(i)] = POS_X(i);
        from_y[STATE_TILE(i)] = POS_Y(i);
    }
    for (int i = 1; i < STATE_N; ++i)
    {
        STATE_HVALUE += distance(from_x[i], POS_X(i));
        STATE_HVALUE += distance(from_y[i], POS_Y(i));
    }
}

__device__ static void
state_tile_fill(Input input)
{
    for (int i = 0; i < STATE_N; ++i)
    {
        if (input.tiles[i] == 0)
            STATE_EMPTY = i;
        STATE_TILE(i)   = input.tiles[i];
    }
}

__device__ static inline bool
state_is_goal(void)
{
    return STATE_HVALUE == 0;
}

__device__ static char assert_direction2
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ __shared__ static bool movable_table_shared[STATE_N][DIR_N];

__device__ static inline bool
state_movable(Direction dir)
{
    return movable_table_shared[STATE_EMPTY][dir];
}

__device__ static char assert_direction
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
__device__ __constant__ const static int pos_diff_table[DIR_N] = {
    -STATE_WIDTH, 1, -1, +STATE_WIDTH};

__device__ static inline bool
state_move_with_limit(Direction dir, unsigned int f_limit)
{
    int new_empty   = STATE_EMPTY + pos_diff_table[dir];
    int opponent    = STATE_TILE(new_empty);
    int new_h_value = STATE_HVALUE + H_DIFF(opponent, new_empty, dir);

    if (STACK.i + STACK.init_depth + 1 + new_h_value > f_limit)
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

__shared__ unsigned int input_i_shared;
#define thread_running (0)
#define thread_sharing (1)
#define thread_stopping (2)
__shared__ int thread_state[32];

__device__ static bool
get_works(Input *input, uchar *dir)
{
	int tid = threadIdx.x;
	int target = (tid + 1) & 31;
	for (;;)
	{
		uchar old = atomicCAS(&thread_state[target], thread_running, thread_sharing);
		if (old == thread_running)
		{
			int j = stack[target].j;
			if (j == stack[target].i)
			{
				target = (target+1)&31;
				thread_state[target] = thread_running;
				if (target == tid)
					break;
				continue;
			}

			int target_i = stack[target].input_i;
			Input in = input[target_i];

			stack_init(in, target_i);
			state_tile_fill(in);
			state_init_hvalue();

			for (int idx = 0; idx < j; ++idx)
				state_move(stack[target].buf[idx]);

			STACK.parent_dir = j == 0 ? stack[target].parent_dir : stack[target].buf[j - 1];
            *dir = stack[target].buf[j];
			STACK.init_depth += j;
			stack[target].j++;

			thread_state[target] = thread_running;
			thread_state[tid] = thread_running;
			return true;
		}
		else if (old == thread_stopping)
		{
			target = (target+1)&31;
			if (target == tid)
				break;
		}
		return false;
	}
	return false;
}

__device__ static void
idas_internal(int f_limit, Input *input, int *input_ends, search_stat *stat)
{
    int       tid            = threadIdx.x;
    int       bid            = blockIdx.x;
    int       id             = tid + bid * blockDim.x;

	int input_begin = bid == 0 ? 0 : input_ends[bid-1];
	int input_end = input_ends[bid];
	int input_i = input_begin+tid;
	uchar     dir            = 0;
	thread_state[tid] = thread_running;

	STACK.input_i = input_i;
	if (input_begin == input_end)
	{
		thread_state[tid] = thread_stopping;
		if (!get_works(input, &dir))
			return;
	}

	/* input surely includes more warks than #warp by devision condition */
	if (tid == 0)
		input_i_shared = input_begin + 32;
	Input this_input = input[input_i];

	stack_init(this_input, input_i);
	state_tile_fill(this_input);
	state_init_hvalue();

	for (;;)
    {
		unsigned long long nodes_expanded = 0;

		for (;;)
		{
			if (state_is_goal())
				asm("trap;"); /* solution found */
			/*
			{
				stat[input_i].solved = true;
				// copy stack to output
				stat[input_i].len = STACK.i;;
				return;
			}
			*/

			if (((stack_is_empty() && dir_reverse(dir) != STACK.parent_dir) ||
						stack_peak() != dir_reverse(dir)) &&
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
					goto END_THIS_NODE;

				dir = stack_pop();
				state_move(dir_reverse(dir));
			}
		}

END_THIS_NODE:
        atomicAdd(&stat[input_i].nodes_expanded, nodes_expanded);
        stat[input_i].thread = id; /* just a reference, so not atomic for now */

		input_i = atomicInc(&input_i_shared, UINT_MAX);
		//input_i = ++input_i_shared; /* avoiding atomic operation may improve performance */

		if (input_i >= input_end)
		{
			thread_state[tid] = thread_stopping;
			if (get_works(input, &dir))
			{
				continue;
			}
			else
				return;
		}

		this_input = input[input_i];
		dir            = 0;
		stack_init(this_input, input_i);
		state_tile_fill(this_input);
		state_init_hvalue();
    }
}

__global__ void
idas_kernel(Input *input, int *input_ends, signed char *plan, search_stat *stat,
            int f_limit, signed char *h_diff_table, bool *movable_table)
{
    int       tid            = threadIdx.x;

    for (int dir = 0; dir < DIR_N; ++dir)
        for (int i = tid; i < STATE_N; i += blockDim.x)
            if (i < STATE_N)
                movable_table_shared[i][dir] = movable_table[i * DIR_N + dir];
    for (int i = 0; i < STATE_N * DIR_N; ++i)
        for (int j = tid; j < STATE_N; j += blockDim.x)
            if (j < STATE_N)
                h_diff_table_shared[j][i / DIR_N][i % DIR_N] =
                    h_diff_table[j * STATE_N * DIR_N + i];

    __syncthreads();

	idas_internal(f_limit, input, input_ends, stat);
}

/* host library implementation */

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef UNABLE_LOG
#define elog(...) fprintf(stderr, __VA_ARGS__)
#else
#define elog(...) ;
#endif

void *
palloc(size_t size)
{
    void *ptr = malloc(size);
    if (!ptr)
        elog("malloc failed\n");

    return ptr;
}

void *
repalloc(void *old_ptr, size_t new_size)
{
    void *ptr = realloc(old_ptr, new_size);
    if (!ptr)
        elog("realloc failed\n");

    return ptr;
}

void
pfree(void *ptr)
{
    if (!ptr)
        elog("empty ptr\n");
    free(ptr);
}

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char idx_t;
/*
 *  [0,0] [1,0] [2,0] [3,0]
 *  [0,1] [1,1] [2,1] [3,1]
 *  [0,2] [1,2] [2,2] [3,2]
 *  [0,3] [1,3] [2,3] [3,3]
 */

/*
 * goal state is
 * [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

typedef struct state_tag_cpu
{
    int       depth; /* XXX: needed? */
    uchar     pos[STATE_WIDTH][STATE_WIDTH];
    idx_t     i, j; /* pos of empty */
    Direction parent_dir;
    int       h_value;
} * State;

#define v(state, i, j) ((state)->pos[i][j])
#define ev(state) (v(state, state->i, state->j))
#define lv(state) (v(state, state->i - 1, state->j))
#define dv(state) (v(state, state->i, state->j + 1))
#define rv(state) (v(state, state->i + 1, state->j))
#define uv(state) (v(state, state->i, state->j - 1))

static uchar from_x[STATE_WIDTH * STATE_WIDTH],
    from_y[STATE_WIDTH * STATE_WIDTH];

static inline void
fill_from_xy(State from)
{
    for (idx_t x = 0; x < STATE_WIDTH; ++x)
        for (idx_t y = 0; y < STATE_WIDTH; ++y)
        {
            from_x[v(from, x, y)] = x;
            from_y[v(from, x, y)] = y;
        }
}

static char assert_state_width_is_four2[STATE_WIDTH == 4 ? 1 : -1];
static inline int
heuristic_manhattan_distance(State from)
{
    int h_value = 0;

    fill_from_xy(from);

    for (idx_t i = 1; i < STATE_N; ++i)
    {
        h_value += distance(from_x[i], i & 3);
        h_value += distance(from_y[i], i >> 2);
    }

    return h_value;
}

bool
state_is_goal(State state)
{
    return state->h_value == 0;
}

static inline State
state_alloc(void)
{
    return (State) palloc(sizeof(struct state_tag_cpu));
}

static inline void
state_free(State state)
{
    pfree(state);
}

State
state_init(uchar v_list[STATE_WIDTH * STATE_WIDTH], int init_depth)
{
    State state = state_alloc();
    int   cnt   = 0;

    state->depth      = init_depth;
    state->parent_dir = (Direction) -1;

    for (idx_t j = 0; j < STATE_WIDTH; ++j)
        for (idx_t i = 0; i < STATE_WIDTH; ++i)
        {
            if (v_list[cnt] == 0)
            {
                state->i = i;
                state->j = j;
            }
            v(state, i, j) = v_list[cnt++];
        }

    state->h_value = heuristic_manhattan_distance(state);

    return state;
}

void
state_fini(State state)
{
    state_free(state);
}

State
state_copy(State src)
{
    State dst = state_alloc();

    memcpy(dst, src, sizeof(*src));

    return dst;
}

static inline bool
state_left_movable(State state)
{
    return state->i != 0;
}
static inline bool
state_down_movable(State state)
{
    return state->j != STATE_WIDTH - 1;
}
static inline bool
state_right_movable(State state)
{
    return state->i != STATE_WIDTH - 1;
}
static inline bool
state_up_movable(State state)
{
    return state->j != 0;
}

bool
state_movable(State state, Direction dir)
{
    return (dir != DIR_LEFT || state_left_movable(state)) &&
           (dir != DIR_DOWN || state_down_movable(state)) &&
           (dir != DIR_RIGHT || state_right_movable(state)) &&
           (dir != DIR_UP || state_up_movable(state));
}

#define h_diff(who, from_i, from_j, dir)                                       \
    (h_diff_table[((who) << 6) + ((from_j) << 4) + ((from_i) << 2) + (dir)])
static int h_diff_table[STATE_N * STATE_N * DIR_N] = {
    1,  1,  1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,
    1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,
    -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1,
    1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,
    1,  1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1,
    1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, -1,
    1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,  1,  1,
    -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  -1, -1, 1,  1,  -1, -1, 1,  1,
    -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,
    1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,
    -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1,
    -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, -1, 1,  1,
    -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, -1, 1,  1,  -1, -1, 1,
    1,  -1, -1, 1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,
    -1, -1, 1,  1,  -1, -1, 1,  1,  1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,
    1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,
    -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,
    -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,
    1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1,
    1,  -1, 1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,
    -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1,
    -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  -1, -1,
    1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1,
    -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1,
    1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,
    1,  1,  1,  1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,
    1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  1,
    1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  1,  -1,
    1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  1,  1,  1,  1,  -1,
    1,  1,  1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,
    -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,
    1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1,
    1,  -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  -1, -1, 1,
    1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  -1,
    1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,
    1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,
    1,  1,  -1, 1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, 1,  -1,
    1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1,
    1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,
    -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,
    -1, -1, 1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1,
    -1, 1,  1,  -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,
    -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,
    1,  1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  -1, 1,  -1,
    1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  1,
    -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,
    -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,
    1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1,
    1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,
    -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  -1,
    1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,
    -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1,
    1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  1,
    -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  1,  1,  1};

void
state_move(State state, Direction dir)
{
    idx_t who;
    assert(state_movable(state, dir));

    switch (dir)
    {
    case DIR_LEFT:
        who = ev(state) = lv(state);
        state->i--;
        break;
    case DIR_DOWN:
        who = ev(state) = dv(state);
        state->j++;
        break;
    case DIR_RIGHT:
        who = ev(state) = rv(state);
        state->i++;
        break;
    case DIR_UP:
        who = ev(state) = uv(state);
        state->j--;
        break;
    default:
        elog("unexpected direction");
        assert(false);
    }

    state->h_value =
        state->h_value + h_diff(who, state->i, state->j, dir_reverse(dir));
    state->parent_dir = dir;
}

bool
state_pos_equal(State s1, State s2)
{
    for (idx_t i = 0; i < STATE_WIDTH; ++i)
        for (idx_t j = 0; j < STATE_WIDTH; ++j)
            if (v(s1, i, j) != v(s2, i, j))
                return false;

    return true;
}

size_t
state_hash(State state)
{
    /* FIXME: for A* */
    size_t hash_value = 0;
    for (idx_t i = 0; i < STATE_WIDTH; ++i)
        for (idx_t j = 0; j < STATE_WIDTH; ++j)
            hash_value ^= (v(state, i, j) << ((i * 3 + j) << 2));
    return hash_value;
}
int
state_get_hvalue(State state)
{
    return state->h_value;
}

int
state_get_depth(State state)
{
    return state->depth;
}

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#ifndef SIZE_MAX
#define SIZE_MAX ((size_t) -1)
#endif

typedef enum {
    HT_SUCCESS = 0,
    HT_FAILED_FOUND,
    HT_FAILED_NOT_FOUND,
} HTStatus;

/* XXX: hash function for State should be surveyed */
inline static size_t
hashfunc(State key)
{
    return state_hash(key);
}

typedef struct ht_entry_tag *HTEntry;
struct ht_entry_tag
{
    HTEntry next;
    State   key;
    int     value;
};

static HTEntry
ht_entry_init(State key)
{
    HTEntry entry = (HTEntry) palloc(sizeof(*entry));

    entry->key  = state_copy(key);
    entry->next = NULL;

    return entry;
}

static void
ht_entry_fini(HTEntry entry)
{
    pfree(entry);
}

typedef struct ht_tag
{
    size_t   n_bins;
    size_t   n_elems;
    HTEntry *bin;
} * HT;

static bool
ht_rehash_required(HT ht)
{
    return ht->n_bins <= ht->n_elems; /* TODO: local policy is also needed */
}

static size_t
calc_n_bins(size_t required)
{
    /* NOTE: n_bins is used for mask and hence it should be pow of 2, fon now */
    size_t size = 1;
    assert(required > 0);

    while (required > size)
        size <<= 1;

    return size;
}

HT
ht_init(size_t init_size_hint)
{
    size_t n_bins = calc_n_bins(init_size_hint);
    HT     ht     = (HT) palloc(sizeof(*ht));

    ht->n_bins  = n_bins;
    ht->n_elems = 0;

    assert(sizeof(*ht->bin) <= SIZE_MAX / n_bins);
    ht->bin = (HTEntry *) palloc(sizeof(*ht->bin) * n_bins);
    memset(ht->bin, 0, sizeof(*ht->bin) * n_bins);

    return ht;
}

static void
ht_rehash(HT ht)
{
    HTEntry *new_bin;
    size_t   new_size = ht->n_bins << 1;

    assert(ht->n_bins<SIZE_MAX>> 1);

    new_bin = (HTEntry *) palloc(sizeof(*new_bin) * new_size);
    memset(new_bin, 0, sizeof(*new_bin) * new_size);

    for (size_t i = 0; i < ht->n_bins; ++i)
    {
        HTEntry entry = ht->bin[i];

        while (entry)
        {
            HTEntry next = entry->next;

            size_t idx   = hashfunc(entry->key) & (new_size - 1);
            entry->next  = new_bin[idx];
            new_bin[idx] = entry;

            entry = next;
        }
    }

    pfree(ht->bin);
    ht->n_bins = new_size;
    ht->bin    = new_bin;
}

void
ht_fini(HT ht)
{
    for (size_t i = 0; i < ht->n_bins; ++i)
    {
        HTEntry entry = ht->bin[i];
        while (entry)
        {
            HTEntry next = entry->next;
            state_fini(entry->key);
            ht_entry_fini(entry);
            entry = next;
        }
    }

    pfree(ht->bin);
    pfree(ht);
}

HTStatus
ht_insert(HT ht, State key, int **value)
{
    size_t  i;
    HTEntry entry, new_entry;

    if (ht_rehash_required(ht))
        ht_rehash(ht);

    i     = hashfunc(key) & (ht->n_bins - 1);
    entry = ht->bin[i];

    while (entry)
    {
        if (state_pos_equal(key, entry->key))
        {
            *value = &entry->value;
            return HT_FAILED_FOUND;
        }

        entry = entry->next;
    }

    new_entry = ht_entry_init(key);

    new_entry->next = ht->bin[i];
    ht->bin[i]      = new_entry;
    *value          = &new_entry->value;

    assert(ht->n_elems < SIZE_MAX);
    ht->n_elems++;

    return HT_SUCCESS;
}

/*
 * Priority Queue implementation
 */

#include <assert.h>
#include <stdint.h>

typedef struct pq_entry_tag
{
    State state;
    int   f, g;
} PQEntryData;
typedef PQEntryData *PQEntry;

/* tiebreaking is done comparing g value */
static inline bool
pq_entry_higher_priority(PQEntry e1, PQEntry e2)
{
    return e1->f < e2->f || (e1->f == e2->f && e1->g >= e2->g);
}

/*
 * NOTE:
 * This priority queue is implemented doubly reallocated array.
 * It will only extend and will not shrink, for now.
 * It may be improved by using array of layers of iteratively widened array
 */
typedef struct pq_tag
{
    size_t       n_elems;
    size_t       capa;
    PQEntryData *array;
} * PQ;

static inline size_t
calc_init_capa(size_t capa_hint)
{
    size_t capa = 1;
    assert(capa_hint > 0);

    while (capa < capa_hint)
        capa <<= 1;
    return capa - 1;
}

PQ
pq_init(size_t init_capa_hint)
{
    PQ pq = (PQ) palloc(sizeof(*pq));

    pq->n_elems = 0;
    pq->capa    = calc_init_capa(init_capa_hint);

    assert(pq->capa <= SIZE_MAX / sizeof(PQEntryData));
    pq->array = (PQEntryData *) palloc(sizeof(PQEntryData) * pq->capa);

    return pq;
}

void
pq_fini(PQ pq)
{
    for (size_t i = 0; i < pq->n_elems; ++i)
        state_fini(pq->array[i].state);

    pfree(pq->array);
    pfree(pq);
}

static inline bool
pq_is_full(PQ pq)
{
    assert(pq->n_elems <= pq->capa);
    return pq->n_elems == pq->capa;
}

static inline void
pq_extend(PQ pq)
{
    pq->capa = (pq->capa << 1) + 1;
    assert(pq->capa <= SIZE_MAX / sizeof(PQEntryData));

    pq->array =
        (PQEntryData *) repalloc(pq->array, sizeof(PQEntryData) * pq->capa);
}

static inline void
pq_swap_entry(PQ pq, size_t i, size_t j)
{
    PQEntryData tmp = pq->array[i];
    pq->array[i]    = pq->array[j];
    pq->array[j]    = tmp;
}

static inline size_t
pq_up(size_t i)
{
    /* NOTE: By using 1-origin, it may be written more simply, i >> 1 */
    return (i - 1) >> 1;
}

static inline size_t
pq_left(size_t i)
{
    return (i << 1) + 1;
}

static void
heapify_up(PQ pq)
{
    for (size_t i = pq->n_elems; i > 0;)
    {
        size_t ui = pq_up(i);
        assert(i > 0);
        if (!pq_entry_higher_priority(&pq->array[i], &pq->array[ui]))
            break;

        pq_swap_entry(pq, i, ui);
        i = ui;
    }
}

void
pq_put(PQ pq, State state, int f, int g)
{
    if (pq_is_full(pq))
        pq_extend(pq);

    pq->array[pq->n_elems].state = state_copy(state);
    pq->array[pq->n_elems].f     = f; /* this may be abundant */
    pq->array[pq->n_elems].g     = g;
    heapify_up(pq);
    ++pq->n_elems;
}

static void
heapify_down(PQ pq)
{
    size_t sentinel = pq->n_elems;

    for (size_t i = 0;;)
    {
        size_t ri, li = pq_left(i);
        if (li >= sentinel)
            break;

        ri = li + 1;
        if (ri >= sentinel)
        {
            if (pq_entry_higher_priority(&pq->array[li], &pq->array[i]))
                pq_swap_entry(pq, i, li);
            /* Reached the bottom */
            break;
        }

        /* NOTE: If p(ri) == p(li), it may be good to go right
         * since the filling order is left-first */
        if (pq_entry_higher_priority(&pq->array[li], &pq->array[ri]))
        {
            if (!pq_entry_higher_priority(&pq->array[li], &pq->array[i]))
                break;

            pq_swap_entry(pq, i, li);
            i = li;
        }
        else
        {
            if (!pq_entry_higher_priority(&pq->array[ri], &pq->array[i]))
                break;

            pq_swap_entry(pq, i, ri);
            i = ri;
        }
    }
}

State
pq_pop(PQ pq)
{
    State ret_state;

    if (pq->n_elems == 0)
        return NULL;

    ret_state = pq->array[0].state;

    --pq->n_elems;
    pq->array[0] = pq->array[pq->n_elems];
    heapify_down(pq);

    return ret_state;
}

void
pq_dump(PQ pq)
{
    elog("%s: n_elems=%zu, capa=%zu\n", __func__, pq->n_elems, pq->capa);
    for (size_t i = 0, cr_required = 1; i < pq->n_elems; i++)
    {
        if (i == cr_required)
        {
            elog("\n");
            cr_required = (cr_required << 1) + 1;
        }
        elog("%d,", pq->array[i].f);
        elog("%d ", pq->array[i].g);
    }
    elog("\n");
}

#include <stdlib.h>
#include <string.h>

int rrand(int m)
{
	return (int)((double)m * ( rand() / (RAND_MAX+1.0) ));
}

void shuffle_input(Input input[], search_stat stat[], int n_inputs)
{
	Input tmp;
	search_stat tmp_stat;
	size_t n = n_inputs;
	while ( n > 1 ) {
		size_t k = rrand(n--);

		memcpy(&tmp, &input[n], sizeof(Input));
		memcpy(&input[n], &input[k], sizeof(Input));
		memcpy(&input[k], &tmp, sizeof(Input));

		if (stat)
		{
			memcpy(&tmp_stat, &stat[n], sizeof(Input));
			memcpy(&stat[n], &stat[k], sizeof(Input));
			memcpy(&stat[k], &tmp_stat, sizeof(Input));
		}
	}
}

static HT closed;

bool
distribute_astar(State init_state, Input input[], int input_ends[], int distr_n,
                 int *cnt_inputs, int *min_fvalue)
{
    int      cnt = 0;
    State    state;
    PQ       q = pq_init(distr_n + 10);
    HTStatus ht_status;
    int *    ht_value;
    bool     solved = false;
    closed          = ht_init(10000);

    ht_status = ht_insert(closed, init_state, &ht_value);
    *ht_value = 0;
    pq_put(q, state_copy(init_state), state_get_hvalue(init_state), 0);
    ++cnt;

    while ((state = pq_pop(q)))
    {
        --cnt;
        if (state_is_goal(state))
        {
            solved = true;
            break;
        }

        ht_status = ht_insert(closed, state, &ht_value);
        if (ht_status == HT_FAILED_FOUND && *ht_value < state_get_depth(state))
        {
            state_fini(state);
            continue;
        }
        else
            *ht_value = state_get_depth(state);

        for (int dir = 0; dir < DIR_N; ++dir)
        {
            if (state->parent_dir != dir_reverse(dir) &&
                state_movable(state, (Direction) dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, (Direction) dir);
                next_state->depth++;

                ht_status = ht_insert(closed, next_state, &ht_value);
                if (ht_status == HT_FAILED_FOUND &&
                    *ht_value <= state_get_depth(next_state))
                    state_fini(next_state);
                else
                {
                    ++cnt;
                    *ht_value = state_get_depth(next_state);
                    pq_put(q, next_state,
                           *ht_value + state_get_hvalue(next_state), *ht_value);
                }
            }
        }

        state_fini(state);

        if (cnt >= distr_n)
            break;
    }

    *cnt_inputs = cnt;
    if (!solved)
    {
        int minf = INT_MAX;
        for (int id = 0; id < cnt; ++id)
        {
            State state = pq_pop(q);
            assert(state);

            for (int i = 0; i < STATE_N; ++i)
                input[id].tiles[i] =
                    state->pos[i % STATE_WIDTH][i / STATE_WIDTH];
            input[id].tiles[state->i + (state->j * STATE_WIDTH)] = 0;

            input[id].init_depth = state_get_depth(state);
            input[id].parent_dir = state->parent_dir;
            if (minf > state_get_depth(state) + state_get_hvalue(state))
                minf = state_get_depth(state) + state_get_hvalue(state);
        }
		shuffle_input(input, NULL, cnt);
        *min_fvalue = minf;

        printf("distr_n=%d, n_worers=%d, cnt=%d\n", distr_n, N_WORKERS, cnt);
        for (int id               = 0; id < N_BLOCKS; ++id)
            input_ends[id]        = (distr_n / N_BLOCKS) * (id + 1) - 1;
        input_ends[N_BLOCKS - 1] = cnt;
    }

    pq_fini(q);

    return solved;
}

static int
input_devide(Input input[], search_stat stat[], int i, int devide_n, int tail)
{
    int   cnt = 0;
    int * ht_value;
    State state       = state_init(input[i].tiles, input[i].init_depth);
    state->parent_dir = input[i].parent_dir;
    PQ       pq       = pq_init(32);
    HTStatus ht_status;
    pq_put(pq, state, state_get_hvalue(state), 0);
    ++cnt;

    while ((state = pq_pop(pq)))
    {
        --cnt;
        if (state_is_goal(state))
        {
            /* It may not be optimal goal */
            pq_put(pq, state, state_get_depth(state) + state_get_hvalue(state),
                   state_get_depth(state));
            ++cnt;
            break;
        }

        ht_status = ht_insert(closed, state, &ht_value);
        if (ht_status == HT_FAILED_FOUND && *ht_value < state_get_depth(state))
        {
            state_fini(state);
            continue;
        }
        else
            *ht_value = state_get_depth(state);

        for (int dir = 0; dir < DIR_N; ++dir)
        {
            if (state->parent_dir != dir_reverse(dir) &&
                state_movable(state, (Direction) dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, (Direction) dir);
                next_state->depth++;

                ht_status = ht_insert(closed, next_state, &ht_value);
                if (ht_status == HT_FAILED_FOUND &&
                    *ht_value < state_get_depth(next_state))
                    state_fini(next_state);
                else
                {
                    ++cnt;
                    *ht_value = state_get_depth(next_state);
                    pq_put(pq, next_state,
                           *ht_value + state_get_hvalue(next_state), *ht_value);
                }
            }
        }

        state_fini(state);

        if (cnt >= devide_n)
            break;
    }

    for (int id = 0; id < cnt; ++id)
    {
        int   estimation_after_devision = stat[i].nodes_expanded / cnt; /* XXX: fix to consider f-value */
        int   ofs                       = id == 0 ? i : tail - 1 + id;
        State state                     = pq_pop(pq);
        assert(state);

        for (int j              = 0; j < STATE_N; ++j)
            input[ofs].tiles[j] = state->pos[j % STATE_WIDTH][j / STATE_WIDTH];
        input[ofs].tiles[state->i + (state->j * STATE_WIDTH)] = 0;

        input[ofs].init_depth = state_get_depth(state);
        input[ofs].parent_dir = state->parent_dir;

        stat[ofs].nodes_expanded = estimation_after_devision;
    }

    pq_fini(pq);

    return cnt == 0 ? 0 : cnt - 1;
}

/* main */

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

static void
avoid_unused_static_assertions(void)
{
    (void) assert_direction[0];
    (void) assert_direction2[0];
    (void) assert_state_width_is_four[0];
    (void) assert_state_width_is_four2[0];
}

static char dir_char[] = {'U', 'R', 'L', 'D'};

int
main(int argc, char *argv[])
{
    int cnt_inputs;

    int    input_size = sizeof(Input) * N_INPUTS;
    Input  input[N_INPUTS];
    Input *d_input;

    int  input_ends_size = sizeof(int) * N_BLOCKS;
    int  input_ends[N_BLOCKS];
    int *d_input_ends;

    int          plan_size = sizeof(signed char) * PLAN_LEN_MAX * N_INPUTS;
    signed char  plan[PLAN_LEN_MAX * N_INPUTS];
    signed char *d_plan;

    int          stat_size = sizeof(search_stat) * N_INPUTS;
    search_stat  stat[N_INPUTS];
    search_stat *d_stat;

    bool         movable_table[STATE_N * DIR_N];
    bool *       d_movable_table;
    int          movable_table_size = sizeof(bool) * STATE_N * DIR_N;
    signed char  h_diff_table[STATE_N * STATE_N * DIR_N];
    signed char *d_h_diff_table;
    int h_diff_table_size = sizeof(signed char) * STATE_N * STATE_N * DIR_N;

    int min_fvalue = 0;

    if (argc < 2)
    {
        printf("usage: bin/cumain <ifname>\n");
        exit(EXIT_FAILURE);
    }

    load_state_from_file(argv[1], input[0].tiles);

    {
        State init_state = state_init(input[0].tiles, 0);

        if (distribute_astar(init_state, input, input_ends, N_INIT_DISTRIBUTION,
                             &cnt_inputs, &min_fvalue))
        {
            puts("solution is found by distributor");
            return 0;
        }
    }

    init_mdist(h_diff_table);
    init_movable_table(movable_table);

    CUDA_CHECK(cudaMalloc((void **) &d_input, input_size));
    CUDA_CHECK(cudaMalloc((void **) &d_input_ends, input_ends_size));
    CUDA_CHECK(cudaMalloc((void **) &d_plan, plan_size));
    CUDA_CHECK(cudaMalloc((void **) &d_stat, stat_size));
    CUDA_CHECK(cudaMalloc((void **) &d_movable_table, movable_table_size));
    CUDA_CHECK(cudaMalloc((void **) &d_h_diff_table, h_diff_table_size));
    CUDA_CHECK(cudaMemcpy(d_movable_table, movable_table, movable_table_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h_diff_table, h_diff_table, h_diff_table_size,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_input, 0, input_size));
    CUDA_CHECK(cudaMemset(d_plan, 0, plan_size));

    for (uchar f_limit = min_fvalue;; f_limit += 2)
    {
		CUDA_CHECK(cudaMemset(d_stat, 0, stat_size));
        elog("f=%d\n", (int) f_limit);

        CUDA_CHECK(
            cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input_ends, input_ends, input_ends_size,
                              cudaMemcpyHostToDevice));

        elog("call idas_kernel(block=%d, thread=%d)\n", N_BLOCKS, BLOCK_DIM);
        idas_kernel<<<N_BLOCKS, BLOCK_DIM>>>(d_input, d_input_ends, d_plan,
                                             d_stat, f_limit, d_h_diff_table,
                                             d_movable_table);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(plan, d_plan, plan_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(stat, d_stat, stat_size, cudaMemcpyDeviceToHost));

        for (int i = 0; i < cnt_inputs; ++i)
            if (stat[i].solved)
            {
                elog("core id = %d\n", i);
                printf("cpu len=%d: \n", input[i].init_depth);

                /* CPU side output */
                // FIXME: Not implemented, for now. It is easy to search path
                // from init state to this root.

                /* GPU side output */
                printf("gpu len=%d: ", stat[i].len);
                for (int j = 0; j < stat[i].len; ++j)
                    printf("%c ", dir_char[(int) plan[i * PLAN_LEN_MAX + j]]);
                putchar('\n');

                goto solution_found;
            }

		unsigned long long int nodes_expanded_by_threads[N_WORKERS];
		memset(nodes_expanded_by_threads, 0, sizeof(nodes_expanded_by_threads[0]) * N_WORKERS);
        unsigned long long int sum_of_expansion = 0;
        for (int i = 0; i < cnt_inputs; ++i)
		{
            sum_of_expansion += stat[i].nodes_expanded;
			nodes_expanded_by_threads[stat[i].thread] += stat[i].nodes_expanded;
		}

        printf("STAT: nodes_expanded\n");
        for (int i = 0; i < cnt_inputs; ++i)
            printf("%lld, ", stat[i].nodes_expanded);
        putchar('\n');
        printf("STAT: threads_loads\n");
        for (int i = 0; i < N_WORKERS; ++i)
            printf("%lld, ", nodes_expanded_by_threads[i]);
        putchar('\n');

        int increased             = 0;
		unsigned long long int avarage_expected_load = sum_of_expansion / N_WORKERS;

        int stat_cnt[10] = {0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < cnt_inputs; ++i)
        {
            if (stat[i].nodes_expanded < avarage_expected_load)
                stat_cnt[0]++;
            else if (stat[i].nodes_expanded < 2 * avarage_expected_load)
                stat_cnt[1]++;
            else if (stat[i].nodes_expanded < 4 * avarage_expected_load)
                stat_cnt[2]++;
            else if (stat[i].nodes_expanded < 8 * avarage_expected_load)
                stat_cnt[3]++;
            else if (stat[i].nodes_expanded < 16 * avarage_expected_load)
                stat_cnt[4]++;
            else if (stat[i].nodes_expanded < 32 * avarage_expected_load)
                stat_cnt[5]++;
            else
                stat_cnt[6]++;

            int policy =
                (stat[i].nodes_expanded - 1)/ avarage_expected_load + 1;

            if (policy > 1 && stat[i].nodes_expanded > 100)
	    {
                increased += input_devide(input, stat, i, policy,
                                          cnt_inputs + increased);
	    }
        }
        elog("STAT: sum of expanded nodes: %lld\n", sum_of_expansion);
        elog("STAT: avarage expanded nodes: %lld\n", avarage_expected_load);
        elog("STAT: av=%d, 2av=%d, 4av=%d, 8av=%d, 16av=%d, 32av=%d, more=%d\n",
             stat_cnt[0], stat_cnt[1], stat_cnt[2], stat_cnt[3], stat_cnt[4],
             stat_cnt[5], stat_cnt[6]);

	elog("DEBUG: cnt_inputs=%d, increased=%d\n", cnt_inputs, increased);

        if (cnt_inputs + increased > N_INPUTS)
        {
            elog("cnt_inputs too large, cnt_inputs=%d\n", cnt_inputs + increased);
            abort();
        }

        cnt_inputs += increased;
        elog("input count: %d\n", cnt_inputs);

        int stat_thread[10] = {0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < N_WORKERS; ++i)
        {
            if (nodes_expanded_by_threads[i]< avarage_expected_load)
                stat_thread[0]++;
            else if (nodes_expanded_by_threads[i]< 2 * avarage_expected_load)
                stat_thread[1]++;
            else if (nodes_expanded_by_threads[i]< 4 * avarage_expected_load)
                stat_thread[2]++;
            else if (nodes_expanded_by_threads[i]< 8 * avarage_expected_load)
                stat_thread[3]++;
            else if (nodes_expanded_by_threads[i]< 16 * avarage_expected_load)
                stat_thread[4]++;
            else if (nodes_expanded_by_threads[i]< 32 * avarage_expected_load)
                stat_thread[5]++;
            else
                stat_thread[6]++;
        }
        elog("STAT: avarage thread_wors: %lld\n", avarage_expected_load);
        elog("STAT: av=%d, 2av=%d, 4av=%d, 8av=%d, 16av=%d, 32av=%d, more=%d\n",
             stat_thread[0], stat_thread[1], stat_thread[2], stat_thread[3],
			 stat_thread[4], stat_thread[5], stat_thread[6]);

		shuffle_input(input, stat, cnt_inputs);

        /* NOTE: optionally sort here by expected cost or g/h-value */

        int id = 0;
        for (int i = 0, load = 0; i < cnt_inputs; ++i)
        {
            load += stat[i].nodes_expanded;
            if ((unsigned int) load >= avarage_expected_load*BLOCK_DIM)
            {
                load             = 0;
                input_ends[id++] = i;
            }
        }

        while (id < N_BLOCKS)
            input_ends[id++] = cnt_inputs;
	input_ends[N_BLOCKS-1] = cnt_inputs;
    }
solution_found:

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_input_ends));
    CUDA_CHECK(cudaFree(d_plan));
    CUDA_CHECK(cudaFree(d_stat));
    CUDA_CHECK(cudaFree(d_movable_table));
    CUDA_CHECK(cudaFree(d_h_diff_table));
    CUDA_CHECK(cudaDeviceReset());

    avoid_unused_static_assertions();

    return 0;
}
