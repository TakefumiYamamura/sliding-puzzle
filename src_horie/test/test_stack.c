#include "../stack.h"
#include "../state.h"
#include "./test.h"

TEST_GROUP(stack);

static Stack st;

TEST_SETUP(stack)
{
    st = stack_init(3);
}

TEST_TEAR_DOWN(stack)
{
    stack_fini(st);
}

TEST(stack, initialization)
{
}

TEST(stack, empty_stack_should_return_null)
{
    TEST_ASSERT_NULL(stack_pop(st));
}

TEST(stack, create_large_stack)
{
    Stack st2 = stack_init(12345678);
    stack_fini(st2);
}

TEST(stack, poped_is_what_i_put)
{
    value v_list[WIDTH * WIDTH] = {1, 2, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};
    State state                 = state_init(v_list, 0);
    State ret_state;

    stack_put(st, state);
    ret_state = stack_pop(st);
    TEST_ASSERT(state_pos_equal(state, ret_state));

    TEST_ASSERT_NULL(stack_pop(st));
}

TEST(stack, poped_is_what_i_put_in_lifo_order)
{
    value v_list1[WIDTH * WIDTH] = {1, 2, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};
    value v_list2[WIDTH * WIDTH] = {2, 1, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};
    State state1                 = state_init(v_list1, 0);
    State state2                 = state_init(v_list2, 0);
    State ret_state1, ret_state2;

    stack_put(st, state1);
    stack_put(st, state2);
    ret_state2 = stack_pop(st);
    ret_state1 = stack_pop(st);
    TEST_ASSERT_NOT_NULL(ret_state1);
    TEST_ASSERT_NOT_NULL(ret_state2);

    TEST_ASSERT(state_pos_equal(state1, ret_state1));
    TEST_ASSERT(state_pos_equal(state2, ret_state2));

    TEST_ASSERT_FALSE(state_pos_equal(state2, ret_state1));
    TEST_ASSERT_FALSE(state_pos_equal(state1, ret_state2));

    TEST_ASSERT_NULL(stack_pop(st));
}

TEST(stack, put_many_items)
{
    value v_list[WIDTH * WIDTH] = {1, 2, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};
    for (int i = 0; i < 1000; ++i)
    {
        State state = state_init(v_list, 123);
        stack_put(st, state);
    }
}

TEST_GROUP_RUNNER(stack)
{
    RUN_TEST_CASE(stack, initialization);
    RUN_TEST_CASE(stack, empty_stack_should_return_null);
    RUN_TEST_CASE(stack, poped_is_what_i_put);
    RUN_TEST_CASE(stack, poped_is_what_i_put_in_lifo_order);
    RUN_TEST_CASE(stack, put_many_items);
    RUN_TEST_CASE(stack, create_large_stack);
}
