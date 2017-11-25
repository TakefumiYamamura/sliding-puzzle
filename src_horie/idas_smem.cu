#include <stdbool.h>

#define PACKED
#define SEARCH_ALL_THE_BEST
#define COLLECT_LOG
#undef USE_PRECOMPUTED_HDIFF

#define BLOCK_DIM (32)
#define N_INIT_DISTRIBUTION (BLOCK_DIM * 64)
#define MAX_GPU_PLAN (24)
#define MAX_BUF_RATIO (256)

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
#define POS_X(pos) ((pos) &3)
#define POS_Y(pos) ((pos) >> 2)

typedef struct search_stat_tag
{
    bool                   solved;
    int                    len;
    unsigned long long int loads;
#ifdef COLLECT_LOG
	unsigned long long int nodes_expanded;
#endif
	//bool assert_failed;
} search_stat;
typedef struct input_tag
{
    uchar     tiles[STATE_N];
    int       init_depth;
    Direction parent_dir;
} Input;

/* state implementation */

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

#ifdef USE_PRECOMPUTED_HDIFF
__device__ __shared__ static signed char h_diff_table_shared[STATE_N][STATE_N] [DIR_N];
#endif

typedef struct state_tag
{
#ifndef PACKED
    uchar tile[STATE_N];
#else
    unsigned long long tile;
#endif
    uchar     empty;
    uchar     depth;
    Direction parent_dir;
    uchar     h_value; /* ub of h_value is STATE_WIDTH*2*(STATE_N-1), e.g. 90 */
} d_State;

#ifndef PACKED
#define state_tile_get(i) (state->tile[i])
#define state_tile_set(i, v) (state->tile[i] = (v))

#else
#define STATE_TILE_BITS 4
#define STATE_TILE_MASK ((1ull << STATE_TILE_BITS) - 1)
#define state_tile_ofs(i) (i << 2)
#define state_tile_get(i)                                                      \
    ((state->tile & (STATE_TILE_MASK << state_tile_ofs(i))) >>                 \
     state_tile_ofs(i))
#define state_tile_set(i, val)                                                 \
    do                                                                         \
    {                                                                          \
        state->tile &= ~((STATE_TILE_MASK) << state_tile_ofs(i));              \
        state->tile |= ((unsigned long long) val) << state_tile_ofs(i);        \
    } while (0)
#endif

#define distance(i, j) ((i) > (j) ? (i) - (j) : (j) - (i))
__device__ static void
state_init(d_State *state, Input *input)
{
    state->depth      = input->init_depth;
    state->parent_dir = input->parent_dir;
    for (int i = 0; i < STATE_N; ++i)
    {
        if (input->tiles[i] == 0)
            state->empty = i;
        state_tile_set(i, input->tiles[i]);
    }

    state->h_value = 0;
    for (int i = 0; i < STATE_N; ++i)
    {
	    uchar tile = state_tile_get(i);
	    if (tile == 0)
		    continue;
	    state->h_value += distance(POS_X(i), POS_X(tile));
	    state->h_value += distance(POS_Y(i), POS_Y(tile));
    }
}

__device__ static inline bool
state_is_goal(d_State state)
{
    return state.h_value == 0;
}

__device__ static inline int
state_get_f(d_State state)
{
    return state.depth + state.h_value;
}

__device__ __shared__ static bool movable_table_shared[STATE_N][DIR_N];

__device__ static inline bool
state_movable(d_State state, Direction dir)
{
    return movable_table_shared[state.empty][dir];
}

__device__ __constant__ const static int pos_diff_table[DIR_N] = {
    -STATE_WIDTH, 1, -1, +STATE_WIDTH};

__device__ static inline int
calc_h_diff(int opponent, int from, int rev_dir)
{
	int goal_x = POS_X(opponent), goal_y = POS_Y(opponent);
	int from_x = POS_X(from), from_y = POS_Y(from);
	if (rev_dir == DIR_LEFT)
		return goal_x > from_x ? -1 : 1;
	else if (rev_dir == DIR_RIGHT)
		return goal_x < from_x ? -1 : 1;
	else if (rev_dir == DIR_UP)
		return goal_y > from_y ? -1 : 1;
	else
		return goal_y < from_y ? -1 : 1;
}

__device__ static inline void
state_move(d_State *state, Direction dir)
{
    int new_empty = state->empty + pos_diff_table[dir];
    int opponent  = state_tile_get(new_empty);

#ifdef USE_PRECOMPUTED_HDIFF
    state->h_value += h_diff_table_shared[opponent][new_empty][dir];
#else
    state->h_value += calc_h_diff(opponent, new_empty, dir);
#endif
    state_tile_set(state->empty, opponent);
    state->empty      = new_empty;
    state->parent_dir = dir;
    ++state->depth;
}

/* stack implementation */
#define STACK_BUF_LEN (MAX_GPU_PLAN * (BLOCK_DIM/DIR_N))

typedef struct div_stack_tag
{
    unsigned int n;
    d_State      buf[STACK_BUF_LEN];
} d_Stack;

__device__ static inline bool
stack_is_empty(d_Stack *stack)
{
	bool ret = (stack->n == 0);
	__syncthreads();
	return ret;
}

__device__ static inline void
stack_put(d_Stack *stack, d_State *state, bool put)
{
	if (put)
	{
		unsigned int i = atomicInc( &stack->n, UINT_MAX); /* slow? especially in old CC environment */
		stack->buf[i] = *state;
	}
	__syncthreads();
}
__device__ static inline bool
stack_pop(d_Stack *stack, d_State *state)
{
    int tid = threadIdx.x;
    int i   = (int) stack->n - 1 - (int) (tid >> 2);
    if (i >= 0)
        *state = stack->buf[i];
    __syncthreads();
    if (tid == 0)
        stack->n = stack->n >= BLOCK_DIM / DIR_N ?
			stack->n - BLOCK_DIM / DIR_N : 0;
	__syncthreads();
    return i >= 0;
}

//__device__ __shared__ Direction candidate_dir_table[4][3] = {}

/*
 * solver implementation
 */
__device__ static void
idas_internal(d_Stack *stack, int f_limit, search_stat *stat)
{
	d_State state;
    unsigned long long int loop_cnt = 0;
#ifdef COLLECT_LOG
    unsigned long long int nodes_expanded = 0;
#endif
	if (threadIdx.x == 0)
		stat->solved = false;

    for (;;)
    {
        if (stack_is_empty(stack))
		{
			stat->loads = loop_cnt;
#ifdef COLLECT_LOG
			atomicAdd(&stat->nodes_expanded, nodes_expanded);
#endif
			break;
		}

        ++loop_cnt;
        bool found = stack_pop(stack, &state),
			 put = false;

        if (found)
        {
            Direction dir = threadIdx.x & 3;
#ifdef COLLECT_LOG
			nodes_expanded++;
#endif

			/* NOTE: candidate_dir_table may be effective to avoid divergence */
            if (state.parent_dir == dir_reverse(dir))
                continue;

            if (state_movable(state, dir))
            {
                state_move(&state, dir);

                if (state_get_f(state) <= f_limit)
                {
                    if (state_is_goal(state))
					{
#ifndef SEARCH_ALL_THE_BEST
						asm("trap;");
#else
						stat->loads = loop_cnt;
						stat->len = state.depth;
						stat->solved = true;
#endif

#ifdef COLLECT_LOG
						atomicAdd(&stat->nodes_expanded, nodes_expanded);
#endif
					}
                    else
                        put = true;
                }
            }
        }

		stack_put(stack, &state, put);
    }
}

/* XXX: movable table is effective in this case? */
__global__ void
idas_kernel(Input *input, search_stat *stat, int f_limit,
            signed char *h_diff_table, bool *movable_table)
{
    __shared__ d_Stack     stack;
    int tid = threadIdx.x;
	int bid = blockIdx.x;
	if (tid == 0)
		stat[bid].loads = 0;

	d_State state;
	state_init(&state, &input[bid]);
	if (state_get_f(state) > f_limit)
		return;

	if (tid == 0)
	{
		stack.buf[0] = state;
		stack.n      = 1;
	}

    for (int i = tid; i < STATE_N * DIR_N; i += blockDim.x)
        if (i < STATE_N * DIR_N)
            movable_table_shared[i / DIR_N][i % DIR_N] = movable_table[i];

#ifdef USE_PRECOMPUTED_HDIFF
    for (int dir = 0; dir < DIR_N; ++dir)
        for (int j = tid; j < STATE_N * STATE_N; j += blockDim.x)
            if (j < STATE_N * STATE_N)
                h_diff_table_shared[j / STATE_N][j % STATE_N][dir] =
                    h_diff_table[j * DIR_N + dir];
#endif

	__syncthreads();
    idas_internal(&stack, f_limit, &stat[bid]);
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

static void
state_dump(State state)
{
    elog("LOG(state): depth=%d, h=%d, f=%d, ", state->depth, state->h_value,
         state->depth + state->h_value);
    for (int i = 0; i < STATE_N; ++i)
        elog("%d%c", i == state->i + STATE_WIDTH * state->j
                         ? 0
                         : state->pos[i % STATE_WIDTH][i / STATE_WIDTH],
             i == STATE_N - 1 ? '\n' : ',');
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
    return ht->n_bins <= ht->n_elems;
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

int
rrand(int m)
{
    return (int) ((double) m * (rand() / (RAND_MAX + 1.0)));
}

void
shuffle_input(Input input[], int n_inputs)
{
    Input  tmp;
    size_t n = n_inputs;
    while (n > 1)
    {
        size_t k = rrand(n--);
        memcpy(&tmp, &input[n], sizeof(Input));
        memcpy(&input[n], &input[k], sizeof(Input));
        memcpy(&input[k], &tmp, sizeof(Input));
    }
}

static HT closed;

bool
distribute_astar(State init_state, Input input[], int distr_n, int *cnt_inputs,
                 int *min_fvalue)
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
    elog("LOG: init_distr, cnt=%d\n", cnt);
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
        assert(pq_pop(q) == NULL);
        // shuffle_input(input, cnt);
        *min_fvalue = minf;
    }

    pq_fini(q);

    return solved;
}

static int
input_devide(Input input[], search_stat stat[], int i, int devide_n, int tail,
             int *buf_len)
{
    int   cnt = 0;
    int * ht_value;
    State state       = state_init(input[i].tiles, input[i].init_depth);
    state->parent_dir = input[i].parent_dir;
    PQ       pq       = pq_init(devide_n);
    HTStatus ht_status;
    pq_put(pq, state, state_get_hvalue(state), 0);
    ++cnt;
    assert(devide_n > 0);

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

    int new_buf_len = *buf_len;
    while (tail + cnt >= new_buf_len)
        new_buf_len <<= 1;
    if (new_buf_len != *buf_len)
    {
        *buf_len = new_buf_len;
        repalloc(input, sizeof(*input) * new_buf_len);
        elog("LOG: host buf resize\n");
    }

    input[i] = input[tail - 1];

    for (int id = 0; id < cnt; ++id)
    {
        int   ofs   = tail - 1 + id;
        State state = pq_pop(pq);
        assert(state);

        for (int j              = 0; j < STATE_N; ++j)
            input[ofs].tiles[j] = state->pos[j % STATE_WIDTH][j / STATE_WIDTH];
        input[ofs].tiles[state->i + (state->j * STATE_WIDTH)] = 0;

        input[ofs].init_depth = state_get_depth(state);
        input[ofs].parent_dir = state->parent_dir;
    }

    pq_fini(pq);

    return cnt - 1;
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

__host__ static void *
cudaPalloc(size_t size)
{
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

__host__ static void
cudaPfree(void *ptr)
{
    CUDA_CHECK(cudaFree(ptr));
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

// static char dir_char[] = {'U', 'R', 'L', 'D'};

#define INPUT_SIZE (sizeof(Input) * buf_len)
#define STAT_SIZE (sizeof(search_stat) * buf_len)
#define MOVABLE_TABLE_SIZE (sizeof(bool) * STATE_N * DIR_N)
#define H_DIFF_TABLE_SIZE (STATE_N * STATE_N * DIR_N)
int
main(int argc, char *argv[])
{
    int n_roots;

    int buf_len = N_INIT_DISTRIBUTION * MAX_BUF_RATIO;

    Input *input                = (Input *) palloc(INPUT_SIZE),
          *d_input              = (Input *) cudaPalloc(INPUT_SIZE);
    search_stat *stat           = (search_stat *) palloc(STAT_SIZE),
                *d_stat         = (search_stat *) cudaPalloc(STAT_SIZE);
    bool *movable_table         = (bool *) palloc(MOVABLE_TABLE_SIZE),
         *d_movable_table       = (bool *) cudaPalloc(MOVABLE_TABLE_SIZE);
    signed char *h_diff_table   = (signed char *) palloc(H_DIFF_TABLE_SIZE),
                *d_h_diff_table = (signed char *) cudaPalloc(H_DIFF_TABLE_SIZE);

    int min_fvalue = 0;

    if (argc != 2)
        exit_failure("usage: bin/cumain <ifname>\n");

    load_state_from_file(argv[1], input[0].tiles);

    {
        State init_state = state_init(input[0].tiles, 0);
        state_dump(init_state);
        if (distribute_astar(init_state, input, N_INIT_DISTRIBUTION, &n_roots,
                             &min_fvalue))
        {
            elog("solution is found by distributor\n");
            goto solution_found;
        }
        state_fini(init_state);
    }

    init_mdist(h_diff_table);
    init_movable_table(movable_table);

    CUDA_CHECK(cudaMemcpy(d_movable_table, movable_table, MOVABLE_TABLE_SIZE,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h_diff_table, h_diff_table, H_DIFF_TABLE_SIZE,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_input, 0, INPUT_SIZE));

    for (uchar f_limit = min_fvalue;; f_limit += 2)
    {
        CUDA_CHECK(cudaMemset(d_stat, 0, STAT_SIZE));
        CUDA_CHECK(
            cudaMemcpy(d_input, input, INPUT_SIZE, cudaMemcpyHostToDevice));

        elog("f_limit=%d\n", (int) f_limit);
        idas_kernel<<<n_roots, BLOCK_DIM>>>(d_input, d_stat, f_limit,
                                            d_h_diff_table, d_movable_table);
        CUDA_CHECK(
            cudaGetLastError()); /* asm trap is called when find solution */

        CUDA_CHECK(cudaMemcpy(stat, d_stat, STAT_SIZE, cudaMemcpyDeviceToHost));

        unsigned long long int loads_sum = 0;
        for (int i = 0; i < n_roots; ++i)
            loads_sum += stat[i].loads;

#ifdef COLLECT_LOG
        elog("STAT: loop\n");
        for (int i = 0; i < n_roots; ++i)
            elog("%lld, ", stat[i].loads);
        putchar('\n');
        elog("STAT: nodes_expanded\n");
        for (int i = 0; i < n_roots; ++i)
            elog("%lld, ", stat[i].nodes_expanded);
        putchar('\n');
        elog("STAT: efficiency\n");
        for (int i = 0; i < n_roots; ++i)
		if (stat[i].loads != 0)
            elog("%lld, ", stat[i].nodes_expanded / stat[i].loads);
        putchar('\n');
#endif

        int                    increased = 0;
        unsigned long long int loads_av  = loads_sum / n_roots;

        int stat_cnt[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < n_roots; ++i)
        {
            if (stat[i].loads < loads_av)
                stat_cnt[0]++;
            else if (stat[i].loads < 2 * loads_av)
                stat_cnt[1]++;
            else if (stat[i].loads < 4 * loads_av)
                stat_cnt[2]++;
            else if (stat[i].loads < 8 * loads_av)
                stat_cnt[3]++;
            else if (stat[i].loads < 16 * loads_av)
                stat_cnt[4]++;
            else if (stat[i].loads < 32 * loads_av)
                stat_cnt[5]++;
            else if (stat[i].loads < 64 * loads_av)
                stat_cnt[6]++;
            else if (stat[i].loads < 128 * loads_av)
                stat_cnt[7]++;
            else
                stat_cnt[8]++;

            int policy = loads_av == 0 ? stat[i].loads
                                       : (stat[i].loads - 1) / loads_av + 1;

            int buf_len_old = buf_len;
            if (policy > 1 && stat[i].loads > 10)
                increased += input_devide(input, stat, i, policy,
                                          n_roots + increased, &buf_len);

            if (buf_len != buf_len_old)
            {
                elog("XXX: fix MAX_BUF_RATIO\n");
                stat = (search_stat *) repalloc(stat, STAT_SIZE);

                cudaPfree(d_input);
                cudaPfree(d_stat);
                d_input = (Input *) cudaPalloc(INPUT_SIZE);
                d_stat  = (search_stat *) cudaPalloc(STAT_SIZE);
            }
        }

        elog("STAT: loads: sum=%lld, av=%lld\n", loads_sum, loads_av);
        elog("STAT: distr: av=%d, 2av=%d, 4av=%d, 8av=%d, 16av=%d, 32av=%d, "
             "64av=%d, 128av=%d, more=%d\n",
             stat_cnt[0], stat_cnt[1], stat_cnt[2], stat_cnt[3], stat_cnt[4],
             stat_cnt[5], stat_cnt[6], stat_cnt[7], stat_cnt[8]);

        n_roots += increased;
        elog("STAT: n_roots=%d(+%d)\n", n_roots, increased);

#ifdef SEARCH_ALL_THE_BEST
        for (int i = 0; i < n_roots; ++i)
            if (stat[i].solved)
            {
                elog("find all the optimal solution(s), at depth=%d\n", stat[i].len);
                goto solution_found;
            }
#endif

        // shuffle_input(input, n_roots); /* it may not be needed in case of idas_global */
    }

solution_found:
    cudaPfree(d_input);
    cudaPfree(d_stat);
    cudaPfree(d_movable_table);
    cudaPfree(d_h_diff_table);

    CUDA_CHECK(cudaDeviceReset());

    pfree(input);
    pfree(stat);
    pfree(movable_table);
    pfree(h_diff_table);

    return 0;
}
