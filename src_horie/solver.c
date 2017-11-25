#include "./solver.h"
#include "./ht.h"
#include "./pq.h"
#include "./queue.h"
#include "./stack.h"
#include "./state.h"
#include "./utils.h"

/*
 * IDA* / f-value limited A* Search
 */
static bool
solver_flastar(State state, int f_limit, int depth, Direction parent)
{
    if (state_is_goal(state))
    {
        elog("\n");
        state_dump(state);
        return true;
    }

    for (int dir = 0; dir < N_DIR; ++dir)
    {
        if (parent != (Direction) dir_reverse(dir) && state_movable(state, dir))
        {
            state_move(state, dir);

            if (depth + state_get_hvalue(state) <= f_limit &&
                solver_flastar(state, f_limit, depth + 1, dir))
            {
                elog("%d ", dir);
                return true;
            }

            state_move(state, dir_reverse(dir));
        }
    }

    return false;
}

void
solver_idastar(State init_state)
{
    elog("%s: f_limit -> ", __func__);
    for (int f_limit = 1;; ++f_limit)
    {
        elog(".");

        if (solver_flastar(init_state, f_limit, 0, -1))
        {
            elog("\n%s: solved\n", __func__);
            break;
        }
    }
}

/*
 * A* Search
 */
void
solver_astar(State init_state)
{
    /*
    State    state;
    PQ       pq = pq_init(123);
    HTStatus ht_status;
    int *    ht_value;
    HT       closed = ht_init(123);
    bool     solved = false;

    pq_put(pq, state_copy(init_state), state_get_hvalue(init_state), 0);

    while ((state = pq_pop(pq)))
    {
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

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, dir);

                ht_status = ht_insert(closed, next_state, &ht_value);
                if (ht_status == HT_FAILED_FOUND &&
                    *ht_value < state_get_depth(next_state))
                    state_fini(next_state);
                else
                {
                    *ht_value = state_get_depth(next_state);
                    pq_put(pq, next_state,
                           *ht_value +
                               calc_h_value(heuristic, next_state, goal_state));
                }
            }
        }

        state_fini(state);
    }

    if (solved)
    {
        state_dump(state);
        elog("%s: solved\n", __func__);
    }
    else
        elog("%s: not solved\n", __func__);

    ht_fini(closed);
    pq_fini(pq);
    */
}

/*
 * Iterative Deepring Depth First Search(IDDFS) & Depth Limited Search
 */

/*
bool
solver_dls(State init_state, State goal_state, int depth_limit)
{
    State    state;
    Stack    stack = stack_init(123);
    HTStatus ht_status;
    int *    ht_min_depth;
    HT       closed = ht_init(123);
    bool     solved = false;

    ht_status     = ht_insert(closed, init_state, &ht_min_depth);
    *ht_min_depth = 0;
    stack_put(stack, state_copy(init_state));

    while ((state = stack_pop(stack)))
    {
        if (state_get_depth(state) > depth_limit)
        {
            state_fini(state);
            continue;
        }

        if (state_pos_equal(state, goal_state))
        {
            solved = true;
            break;
        }

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, dir);

                ht_status = ht_insert(closed, next_state, &ht_min_depth);
                if (ht_status == HT_SUCCESS)
                    stack_put(stack, next_state);
                else
                {
                    int next_state_depth = state_get_depth(next_state);
                    assert(ht_status == HT_FAILED_FOUND);

                    if (next_state_depth)
                    {
                        *ht_min_depth = next_state_depth;
                        stack_put(stack, next_state);
                    }
                    else
                        state_fini(next_state);
                }
            }
        }

        state_fini(state);
    }

    if (solved)
    {
        state_dump(state);
        elog("%s: solved\n", __func__);
    }
    else
        elog("%s: not solved\n", __func__);

    ht_fini(closed);
    stack_fini(stack);

    return solved;
}

void
solver_iddfs(State init_state, State goal_state)
{
    for (int depth = 1;; ++depth)
    {
        if (solver_dls(init_state, goal_state, depth))
        {
            elog("%s: solved\n", __func__);
            break;
        }
        else
            elog("%s: not solved at the depth=%d\n", __func__, depth);
    }
}
*/

/*
 * BFS/DFS
 */

/*
void
solver_dfs(State init_state, State goal_state)
{
    State    state;
    Stack    stack = stack_init(123);
    HTStatus ht_status;
    int *    ht_place_holder;
    HT       closed = ht_init(123);
    bool     solved = false;

    ht_status = ht_insert(closed, init_state, &ht_place_holder);
    stack_put(stack, state_copy(init_state));

    while ((state = stack_pop(stack)))
    {
        if (state_pos_equal(state, goal_state))
        {
            solved = true;
            break;
        }

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, dir);

                ht_status = ht_insert(closed, next_state, &ht_place_holder);
                if (ht_status == HT_SUCCESS)
                    stack_put(stack, next_state);
                else
                    state_fini(next_state);
            }
        }

        state_fini(state);
    }

    if (solved)
    {
        state_dump(state);
        elog("%s: solved\n", __func__);
    }
    else
        elog("%s: not solved\n", __func__);

    ht_fini(closed);
    stack_fini(stack);
}

void
solver_bfs(State init_state, State goal_state)
{
    State    state;
    Queue    q = queue_init();
    HTStatus ht_status;
    int *    ht_place_holder;
    HT       closed = ht_init(123);
    bool     solved = false;

    ht_status = ht_insert(closed, init_state, &ht_place_holder);
    queue_put(q, state_copy(init_state));

    while ((state = queue_pop(q)))
    {
        if (state_pos_equal(state, goal_state))
        {
            solved = true;
            break;
        }

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, dir);

                ht_status = ht_insert(closed, next_state, &ht_place_holder);
                if (ht_status == HT_SUCCESS)
                    queue_put(q, next_state);
                else
                    state_fini(next_state);
            }
        }

        state_fini(state);
    }

    if (solved)
    {
        state_dump(state);
        elog("%s: solved\n", __func__);
    }
    else
        elog("%s: not solved\n", __func__);

    ht_fini(closed);
    queue_fini(q);
}
*/

/*
 * BFS/DFS without closed list
 */

/*
static bool
solver_recursive_dfs_without_closed_internal(State s)
{
    State next_state;

    if (state_pos_equal(s, goal))
    {
        state_dump(s);
        state_fini(s);
        return true;
    }

    for (int dir = 0; dir < N_DIR; ++dir)
    {
        if (state_movable(s, dir))
        {
            next_state = state_copy(s);
            state_move(next_state, dir);
            if (solver_recursive_dfs_without_closed_internal(next_state))
            {
                state_fini(s);
                return true;
            }
        }
    }

    state_fini(s);
    return false;
}

void
solver_recursive_dfs_without_closed(State init_state, State goal_state)
{
    State s = state_copy(init_state);
    goal    = goal_state;

    if (solver_recursive_dfs_without_closed_internal(s))
        elog("%s: solved\n", __func__);
    else
        elog("%s: not solved\n", __func__);
}

void
solver_dfs_without_closed(State init_state, State goal_state)
{
    State state;
    Stack stack  = stack_init(123);
    bool  solved = false;
    stack_put(stack, state_copy(init_state));

    while ((state = stack_pop(stack)))
    {
        if (state_pos_equal(state, goal_state))
        {
            solved = true;
            break;
        }

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, dir);
                stack_put(stack, next_state);
            }
        }

        state_fini(state);
    }

    if (solved)
    {
        state_dump(state);
        elog("%s: solved\n", __func__);
    }
    else
        elog("%s: not solved\n", __func__);

    stack_fini(stack);
}

void
solver_bfs_without_closed(State init_state, State goal_state)
{
    State state;
    Queue q      = queue_init();
    bool  solved = false;
    queue_put(q, state_copy(init_state));

    while ((state = queue_pop(q)))
    {
        if (state_pos_equal(state, goal_state))
        {
            solved = true;
            break;
        }

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, dir);
                queue_put(q, next_state);
            }
        }

        state_fini(state);
    }

    if (solved)
    {
        state_dump(state);
        elog("%s: solved\n", __func__);
    }
    else
        elog("%s: not solved\n", __func__);

    queue_fini(q);
}
*/
