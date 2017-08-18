#include <stdbool.h>
#include <stdio.h>
#include <time.h>

typedef unsigned char uchar;

typedef uchar Direction;
#define dir_reverse(dir) ((Direction)(3 - (dir)))
#define DIR_N 4
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

#define PLAN_LEN_MAX 255

/* stack implementation */

#define stack_get(i) (stack.buf[(i)])

static struct dir_stack_tag
{
    uchar i;
    uchar buf[PLAN_LEN_MAX];
} stack;

static inline void
stack_put(Direction dir)
{
    stack.buf[stack.i] = dir;
    ++stack.i;
}
static inline bool
stack_is_empty(void)
{
    return stack.i == 0;
}

static inline Direction
stack_pop(void)
{
    --stack.i;
    return stack_get(stack.i);
}
static inline Direction
stack_peak(void)
{
    return stack_get(stack.i - 1);
}

static void
stack_dump(void)
{
    printf("len=%d: ", stack.i);
    for (int i = 0; i < stack.i; ++i)
        printf("%d ", (int) stack_get(i));
    putchar('\n');
}

/* state implementation */

#define STATE_EMPTY 0
#define STATE_WIDTH 4
#define STATE_N (STATE_WIDTH * STATE_WIDTH)

static char assert_state_width_is_four[STATE_WIDTH == 4 ? 1 : -1];
#define POS_X(pos) ((pos) &3)
#define POS_Y(pos) ((pos) >> 2)

/*
 * goal: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
 */

static struct state_tag
{
    uchar tile[STATE_N];
    uchar empty;
    uchar h_value; /* ub of h_value is 6*16 for manhattan dist */
} state;

#define state_tile_get(i) (state.tile[(i)])
#define state_tile_set(i, val) (state.tile[(i)] = (val))

static inline unsigned int
distance(uchar i, uchar j)
{
    return i > j ? i - j : j - i;
}

#define H_DIFF(opponent, empty, empty_dir)                                     \
    h_diff_table[opponent][empty][empty_dir]
static int h_diff_table[STATE_N][STATE_N][DIR_N];

static void
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

static inline void
state_init_hvalue(void)
{
    uchar from_x[STATE_N], from_y[STATE_N];

    for (int i = 0; i < STATE_N; ++i)
    {
        from_x[state_tile_get(i)] = POS_X(i);
        from_y[state_tile_get(i)] = POS_Y(i);
    }
    for (int i = 1; i < STATE_N; ++i)
    {
        state.h_value += distance(from_x[i], POS_X(i));
        state.h_value += distance(from_y[i], POS_Y(i));
    }
}

static void
state_tile_fill(const uchar v_list[STATE_N])
{
    for (int i = 0; i < STATE_N; ++i)
    {
        if (v_list[i] == STATE_EMPTY)
            state.empty = i;
        state_tile_set(i, v_list[i]);
    }
}

static inline bool
state_is_goal(void)
{
    return state.h_value == 0;
}

static char assert_direction2
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
static bool movable_table[STATE_N][DIR_N];

static void
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
static inline bool
state_movable(Direction dir)
{
    return movable_table[state.empty][dir];
}

static void
state_dump(void)
{
    printf("%s: h_value=%d, (x,y)=(%u,%u)\n", __func__, state.h_value,
           POS_X(state.empty), POS_Y(state.empty));

    for (int i = 0; i < STATE_N; ++i)
        printf("%d%c", i == state.empty ? 0 : (int) state_tile_get(i),
               POS_X(i) == STATE_WIDTH - 1 ? '\n' : ' ');
    printf("-----------\n");
}

static char assert_direction
    [DIR_UP == 0 && DIR_RIGHT == 1 && DIR_LEFT == 2 && DIR_DOWN == 3 ? 1 : -1];
static int pos_diff_table[DIR_N] = {-STATE_WIDTH, 1, -1, +STATE_WIDTH};

static inline bool
state_move_with_limit(Direction dir, int f_limit)
{
    int new_empty   = state.empty + pos_diff_table[dir];
    int opponent    = state_tile_get(new_empty);
    int new_h_value = state.h_value + H_DIFF(opponent, new_empty, dir);

    if (stack.i + 1 + new_h_value > f_limit)
        return false;

    state.h_value = new_h_value;
    state_tile_set(state.empty, opponent);
    state.empty = new_empty;

    return true;
}

static inline void
state_move(Direction dir)
{
    int new_empty = state.empty + pos_diff_table[dir];
    int opponent  = state_tile_get(new_empty);

    state.h_value += H_DIFF(opponent, new_empty, dir);
    state_tile_set(state.empty, opponent);
    state.empty = new_empty;
}

/*
 * solver implementation
 */

static bool
idas_internal(int f_limit, long long *ret_nodes_expanded)
{
    uchar     dir            = 0;
    long long nodes_expanded = 0;

    for (;;)
    {
        if (state_is_goal())
        {
            state_dump();
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

void
idas_kernel(uchar *input)
{
    long long nodes_expanded = 0, nodes_expanded_first = 0;
    int       f_limit;
    bool      found;
    init_mdist();
    init_movable_table();
    state_tile_fill(input);
    state_init_hvalue();

    state_dump();

    {
        f_limit              = state.h_value;
        nodes_expanded_first = 0;
        found                = idas_internal(f_limit, &nodes_expanded_first);
        printf("f_limit=%3d, expanded nodes = %lld\n", f_limit, nodes_expanded);
    }
    if (!found)
    {
        ++f_limit;
        nodes_expanded = 0;
        found          = idas_internal(f_limit, &nodes_expanded);
        printf("f_limit=%3d, expanded nodes = %lld\n", f_limit, nodes_expanded);

        f_limit += nodes_expanded == nodes_expanded_first ? 1 : 2;

        for (;; f_limit += 2)
        {
            nodes_expanded = 0;
            found          = idas_internal(f_limit, &nodes_expanded);
            printf("f_limit=%3d, expanded nodes = %lld\n", f_limit,
                   nodes_expanded);

            if (found)
                break;
        }
    }
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

    return (int) rv;
}

#define MAX_LINE_LEN 100
static void
load_state_from_file(const char *fname, uchar *s)
{
    FILE *fp;
    char  str[MAX_LINE_LEN];
    char *str_ptr = str, *end_ptr;
    int   i;

    fp = fopen(fname, "r");
    if (!fp)
        exit_failure("%s: %s cannot be opened\n", __func__, fname);

    if (!fgets(str, MAX_LINE_LEN, fp))
        exit_failure("%s: fgets failed\n", __func__);

    for (i = 0; i < STATE_N; ++i)
    {
        s[i]    = pop_int_from_str(str_ptr, &end_ptr);
        str_ptr = end_ptr;
    }

    fclose(fp);
}
#undef MAX_LINE_LEN

static void
avoid_unused_static_assertions(void)
{
    (void) assert_direction[0];
    (void) assert_direction2[0];
    (void) assert_state_width_is_four[0];
}

int
main(int argc, char *argv[])
{
    uchar s_list[STATE_N];

    if (argc < 2)
    {
        printf("usage: ./c <ifname>\n");
        exit(EXIT_FAILURE);
    }

    clock_t start = clock();

    load_state_from_file(argv[1], s_list);
    idas_kernel(s_list);

    stack_dump();
    clock_t end = clock();
    printf("minutes : %f \n", (double)(end - start) / CLOCKS_PER_SEC );


    avoid_unused_static_assertions();

    return 0;
}