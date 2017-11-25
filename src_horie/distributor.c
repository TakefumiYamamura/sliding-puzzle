#include "./ht.h"
#include "./queue.h"
#include "./state.h"

#define DIR_N 4
#define DIR_FIRST 0
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

bool
distributor_bfs(State init_state, State goal_state, unsigned char *s_list_ret,
                int distr_n)
{
    int      cnt = 0;
    State    state;
    Queue    q = queue_init();
    HTStatus ht_status;
    int *    ht_place_holder;
    HT       closed = ht_init(123);
    bool     solved = false;

    ht_status = ht_insert(closed, init_state, &ht_place_holder);
    queue_put(q, state_copy(init_state));
    ++cnt;

    while ((state = queue_pop(q)))
    {
        --cnt;
        if (state_pos_equal(state, goal_state))
        {
            solved = true;
            break;
        }

        for (int dir = 0; dir < N_DIR; ++dir)
        {
            if (state_movable(state, (Direction) dir))
            {
                State next_state = state_copy(state);
                state_move(next_state, (Direction) dir);

                ht_status = ht_insert(closed, next_state, &ht_place_holder);
                if (ht_status == HT_SUCCESS)
                {
                    if (++cnt == distr_n)
                    {
                        /* NOTE: put parent.
                         * FIXME: There are duplicated younger siblings */
                        queue_put(q, state);

                        state_fini(next_state);
                        break;
                    }
                    else
                        queue_put(q, next_state);
                }
                else
                    state_fini(next_state);
            }
        }

        state_fini(state);
    }

    if (!solved)
        for (int i = 0; i < distr_n; ++i)
            state_fill_slist(queue_pop(q), s_list_ret + STATE_N * i);

    ht_fini(closed);
    queue_fini(q);

    return solved;
}
