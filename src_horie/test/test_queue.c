#include "../queue.h"
#include "../state.h"
#include "./test.h"

TEST_GROUP(queue);

static Queue q;

TEST_SETUP(queue)
{
    q = queue_init();
}

TEST_TEAR_DOWN(queue)
{
    queue_fini(q);
}

TEST(queue, initialization)
{
}

TEST(queue, empty_queue_should_return_null)
{
    TEST_ASSERT_NULL(queue_pop(q));
}

TEST(queue, create_large_queue)
{
    /* TODO: implement */
}

TEST(queue, poped_is_what_i_put)
{
    value v_list[WIDTH * WIDTH] = {1, 2, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};
    State state                 = state_init(v_list, 0);
    State ret_state;

    queue_put(q, state);
    ret_state = queue_pop(q);
    TEST_ASSERT(state_pos_equal(state, ret_state));

    TEST_ASSERT_NULL(queue_pop(q));
}

TEST(queue, poped_is_what_i_put_in_lifo_order)
{
    value v_list1[WIDTH * WIDTH] = {1, 2, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};
    value v_list2[WIDTH * WIDTH] = {2, 1, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};
    State state1                 = state_init(v_list1, 0);
    State state2                 = state_init(v_list2, 0);
    State ret_state1, ret_state2;

    queue_put(q, state1);
    queue_put(q, state2);
    ret_state1 = queue_pop(q);
    ret_state2 = queue_pop(q);
    TEST_ASSERT_NOT_NULL(ret_state1);
    TEST_ASSERT_NOT_NULL(ret_state2);

    TEST_ASSERT(state_pos_equal(state1, ret_state1));
    TEST_ASSERT(state_pos_equal(state2, ret_state2));

    TEST_ASSERT_FALSE(state_pos_equal(state2, ret_state1));
    TEST_ASSERT_FALSE(state_pos_equal(state1, ret_state2));

    TEST_ASSERT_NULL(queue_pop(q));
}

TEST(queue, put_many_items)
{
    value v_list[WIDTH * WIDTH] = {1, 2, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};
    for (int i = 0; i < 3000; ++i)
    {
        State state = state_init(v_list, 123);
        queue_put(q, state);
    }
}

TEST_GROUP_RUNNER(queue)
{
    RUN_TEST_CASE(queue, initialization);
    RUN_TEST_CASE(queue, empty_queue_should_return_null);
    RUN_TEST_CASE(queue, poped_is_what_i_put);
    RUN_TEST_CASE(queue, poped_is_what_i_put_in_lifo_order);
    RUN_TEST_CASE(queue, put_many_items);
    RUN_TEST_CASE(queue, create_large_queue);
}
