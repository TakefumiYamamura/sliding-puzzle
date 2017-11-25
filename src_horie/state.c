#include "./state.h"
#include "./utils.h"

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

struct state_tag
{
    int         depth; /* XXX: needed? */
    state_panel pos[STATE_WIDTH][STATE_WIDTH];
    idx_t       i, j; /* pos of empty */
    Direction   parent;
    int         h_value;
};

#define v(state, i, j) ((state)->pos[i][j])
#define ev(state) (v(state, state->i, state->j))
#define lv(state) (v(state, state->i - 1, state->j))
#define dv(state) (v(state, state->i, state->j + 1))
#define rv(state) (v(state, state->i + 1, state->j))
#define uv(state) (v(state, state->i, state->j - 1))

static state_panel from_x[STATE_WIDTH * STATE_WIDTH],
    from_y[STATE_WIDTH * STATE_WIDTH];

static inline int
distance(int i, int j)
{
    return i > j ? i - j : j - i;
}

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

    for (idx_t i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
    {
        h_value += distance(from_x[i], i % STATE_WIDTH);
        h_value += distance(from_y[i], i / STATE_WIDTH);
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
    return (State) palloc(sizeof(struct state_tag));
}

static inline void
state_free(State state)
{
    pfree(state);
}

State
state_init(state_panel v_list[STATE_WIDTH * STATE_WIDTH], int depth)
{
    State state = state_alloc();
    int   cnt   = 0;

    assert(depth >= 0);

    state->depth  = depth;
    state->parent = (Direction) -1;

    for (idx_t j = 0; j < STATE_WIDTH; ++j)
        for (idx_t i = 0; i < STATE_WIDTH; ++i)
        {
            if (v_list[cnt] == STATE_EMPTY)
            {
                state->i = i;
                state->j = j;
            }
            v(state, i, j) = v_list[cnt++];
        }

    state->h_value = heuristic_manhattan_distance(state);
    state_dump(state);

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
    return (dir != LEFT || state_left_movable(state)) &&
           (dir != DOWN || state_down_movable(state)) &&
           (dir != RIGHT || state_right_movable(state)) &&
           (dir != UP || state_up_movable(state));
}

/*
static inline int
calc_h_diff(idx_t who, idx_t from_x, idx_t from_y, Direction rdir)
{
    idx_t right_x = who % STATE_WIDTH;
    idx_t right_y = who / STATE_WIDTH;

    switch (rdir)
    {
    case LEFT:
        return right_x > from_x ? -1 : 1;
    case RIGHT:
        return right_x < from_x ? -1 : 1;
    case UP:
        return right_y > from_y ? -1 : 1;
    case DOWN:
        return right_y < from_y ? -1 : 1;
    }
}
*/
#define h_diff(who, from_i, from_j, dir)                                       \
    (h_diff_table[((who) << 6) + ((from_j) << 4) + ((from_i) << 2) + (dir)])
static int h_diff_table[STATE_N * STATE_N * N_DIR] = {
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
    case LEFT:
        who = ev(state) = lv(state);
        state->i--;
        break;
    case DOWN:
        who = ev(state) = dv(state);
        state->j++;
        break;
    case RIGHT:
        who = ev(state) = rv(state);
        state->i++;
        break;
    case UP:
        who = ev(state) = uv(state);
        state->j--;
        break;
    default:
        elog("unexpected direction");
        assert(false);
    }

    state->h_value =
        state->h_value + h_diff(who, state->i, state->j, dir_reverse(dir));
    // state->h_value = state->h_value + calc_h_diff(who, state->i, state->j,
    // dir);
    state->parent = dir;
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

void
state_dump(State state)
{
    elog("%s: h_value=%d, (i,j)=(%u,%u)\n", __func__, state->h_value, state->i,
         state->j);

    for (idx_t j = 0; j < STATE_WIDTH; ++j)
    {
        for (idx_t i = 0; i < STATE_WIDTH; ++i)
            elog("%u ", i == state->i && j == state->j ? 0 : v(state, i, j));
        elog("\n");
    }
    elog("-----------\n");
}

void
state_fill_slist(State state, unsigned char slist[])
{
    for (int i   = 0; i < STATE_N; ++i)
        slist[i] = state->pos[i % STATE_WIDTH][i / STATE_WIDTH];
    slist[state->i + (state->j * STATE_WIDTH)] = 0;
}

/*
 * Heuristic functions
 */

/*
static state_panel from_x[STATE_WIDTH * STATE_WIDTH],
    from_y[STATE_WIDTH * STATE_WIDTH], to_x[STATE_WIDTH * STATE_WIDTH],
    to_y[STATE_WIDTH * STATE_WIDTH];

static int inline distance(int i, int j)
{
    return i > j ? i - j : j - i;
}

static void inline fill_from_to_xy(State from, State to)
{
    for (idx_t x = 0; x < STATE_WIDTH; ++x)
        for (idx_t y = 0; y < STATE_WIDTH; ++y)
        {
            from_x[v(from, x, y)] = x;
            from_y[v(from, x, y)] = y;
            to_x[v(to, x, y)]     = x;
            to_y[v(to, x, y)]     = y;
        }
}

static inline int
heuristic_manhattan_distance(State from, State to)
{
    int h_value = 0;

    fill_from_to_xy(from, to);

    for (idx_t i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
    {
        h_value += distance(from_x[i], to_x[i]);
        h_value += distance(from_y[i], to_y[i]);
    }

    return h_value;
}

static inline int
heuristic_misplaced_tiles(State from, State to)
{
    int h_value = 0;

    fill_from_to_xy(from, to);

    for (idx_t i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
        if (from_x[i] != to_x[i] || from_y[i] != to_y[i])
            h_value += 1;

    return h_value;
}

static inline int
heuristic_tiles_out_of_row_col(State from, State to)
{
    int h_value = 0;

    fill_from_to_xy(from, to);

    for (idx_t i = 1; i < STATE_WIDTH * STATE_WIDTH; ++i)
    {
        if (from_x[i] != to_x[i])
            h_value += 1;
        if (from_y[i] != to_y[i])
            h_value += 1;
    }

    return h_value;
}

int
calc_h_value(Heuristic heuristic, State from, State to)
{
    switch (heuristic)
    {
    case HeuristicManhattanDistance:
        return heuristic_manhattan_distance(from, to);
    case HeuristicTilesOutOfRowCol:
        return heuristic_tiles_out_of_row_col(from, to);
    case HeuristicMisplacedTiles:
        return heuristic_misplaced_tiles(from, to);
    default:
        elog("%s: unexpected label\n", __func__);
        exit(EXIT_FAILURE);
    }
}
*/
