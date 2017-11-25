#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define elog(...) fprintf(stderr, __VA_ARGS__)

typedef unsigned char idx_t;

#define N_DIR 4
#define dir_reverse(dir) (3 - (dir))
typedef enum direction_tag {
    DIR_NIL = -30,

    UP    = 0,
    RIGHT = 1,
    LEFT  = 2,
    DOWN  = 3,
} Direction;

/* stack for dir (expand recursion) */
#include "utils.h"
#include <assert.h>
#include <stdint.h>

static struct dir_stack_tag
{
    size_t     capa, i;
    Direction *buf;
} stack;

static inline size_t
calc_init_capa(size_t hint)
{
    size_t capa;
    assert(hint > 0);
    for (capa = 1;; capa <<= 1)
        if (capa >= hint)
            break;
    return capa;
}

static inline size_t
calc_larger_capa(size_t old_size)
{
    return old_size << 1;
}

static inline void
stack_init(size_t init_capa_hint)
{
    size_t capa = calc_init_capa(init_capa_hint);

    assert(capa <= SIZE_MAX / sizeof(Direction));

    stack.buf  = palloc(sizeof(Direction) * capa);
    stack.capa = capa;
    stack.i    = 0;
}

static inline void
stack_fini(void)
{
    pfree(stack.buf);
}

static inline void
stack_put(Direction dir)
{
    assert(stack.i < SIZE_MAX);
    stack.buf[stack.i++] = dir;

    if (stack.i >= stack.capa)
    {
        size_t new_capa = calc_larger_capa(stack.capa);
        assert(new_capa <= SIZE_MAX / sizeof(Direction));
        stack.buf  = repalloc(stack.buf, sizeof(Direction) * new_capa);
        stack.capa = new_capa;
    }
}

static inline Direction
stack_pop(void)
{
    return stack.i == 0 ? DIR_NIL : stack.buf[--stack.i];
}

static inline Direction
stack_top(void)
{
    return stack.i == 0 ? DIR_NIL : stack.buf[stack.i - 1];
}

static inline void
stack_dump(void)
{
    elog("%s: capa=%zu, i=%zu\n", __func__, stack.capa, stack.i);
}

/*
 * state implementation
 */

#define STATE_EMPTY 0
#define STATE_WIDTH 4
#define STATE_N STATE_WIDTH *STATE_WIDTH

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

typedef struct state_tag
{
    unsigned char pos[STATE_WIDTH][STATE_WIDTH];
    idx_t         i, j; /* pos of empty */
    int           h_value;
} * State;

/* TODO: state globalization is effective? */

#define v(state, i, j) ((state)->pos[i][j])
#define ev(state) (v(state, state->i, state->j))
#define lv(state) (v(state, state->i - 1, state->j))
#define dv(state) (v(state, state->i, state->j + 1))
#define rv(state) (v(state, state->i + 1, state->j))
#define uv(state) (v(state, state->i, state->j - 1))

static unsigned char from_x[STATE_WIDTH * STATE_WIDTH],
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

static inline bool
state_is_goal(State state)
{
    return state->h_value == 0;
}

static void
state_init(State state, const unsigned char v_list[STATE_WIDTH * STATE_WIDTH])
{
    int                 cnt           = 0;
    const unsigned char uninitialized = 111;
    state->i = state->j = uninitialized;

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

    assert(state->i != uninitialized && state->j != uninitialized);

    state->h_value = heuristic_manhattan_distance(state);
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

static inline bool
state_movable(State state, Direction dir)
{
    return (dir != LEFT || state_left_movable(state)) &&
           (dir != DOWN || state_down_movable(state)) &&
           (dir != RIGHT || state_right_movable(state)) &&
           (dir != UP || state_up_movable(state));
}

#define h_diff(who, from_i, from_j, dir)                                       \
    (h_diff_table[((who) << 6) + ((from_j) << 4) + ((from_i) << 2) + (dir)])
static const int h_diff_table[STATE_N * STATE_N * N_DIR] = {
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

static void
state_move(State state, Direction dir)
{
    idx_t who;

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
        exit(EXIT_FAILURE);
    }

    state->h_value =
        state->h_value + h_diff(who, state->i, state->j, dir_reverse(dir));
}

static inline int
state_get_hvalue(State state)
{
    return state->h_value;
}

static void
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

/*
 * solver implementation
 */

static bool
idas_internal(State state, int f_limit)
{
    Direction dir = 0;

    for (;;)
    {
        if (state_is_goal(state))
        {
            elog("\n");
            state_dump(state);
            stack_dump(); /* plan */
            return true;
        }

        if (stack_top() != dir_reverse(dir) && state_movable(state, dir))
        {
            state_move(state, dir);

            if (stack.i + state_get_hvalue(state) > (size_t) f_limit)
                state_move(state, dir_reverse(dir));
            else
            {
                stack_put(dir);
                dir = 0;
                continue;
            }
        }

        while (++dir == N_DIR)
        {
            dir = stack_pop();
            if (dir == DIR_NIL)
                return false;
            state_move(state, dir_reverse(dir));
        }
    }
}

void
idas_main(unsigned char input[])
{
    struct state_tag init_state;
    state_init(&init_state, input);

    stack_init(1000);

    elog("%s: f_limit -> ", __func__);
    for (int f_limit = 1;; ++f_limit)
    {
        elog(".");

        if (idas_internal(&init_state, f_limit))
        {
            elog("\n%s: solved\n", __func__);
            break;
        }
    }

    stack_fini();
}
