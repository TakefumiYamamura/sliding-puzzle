#include "./ht.h"
#include "./queue.h"
#include "./state.h"

bool
node_generator(State init_state, int gen_n, unsigned char s_list[])
{
    /* A* way generation */
    State    state;
    PQ       pq = pq_init(1550);
    HTStatus ht_status;
    int *    ht_value;
    HT       closed = ht_init(123);
    bool     solved = false;
    int      cnt    = 1;

    pq_put(pq, state_copy(init_state), state_get_hvalue(init_state), 0);

    while ((state = pq_pop(pq)))
    {
        if (state_is_goal(state))
        {
            solved = true;
            break;
        }
        --cnt;

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
                    ++cnt;
                    if (cnt == gen_n)
                        break;
                }
            }
        }

        state_fini(state);
    }

    if (solved)
    {
        state_dump(state);
        return true;
    }

    while ((state = pq_pop(pq)))
        state_to_slist;

    ht_fini(closed);
    pq_fini(pq);

    return false;
}
