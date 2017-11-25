#include "../state.h"
#include "./test.h"

TEST_GROUP(state);

static State s;

TEST_SETUP(state)
{
    /*
     * 1 2 3
     * 4 _ 8
     * 7 6 5
     */
    value v_list[WIDTH * WIDTH] = {1, 2, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};

    s = state_init(v_list, 0);
}

TEST_TEAR_DOWN(state)
{
    state_fini(s);
}

TEST(state, initalization)
{
}

TEST(state, copied_state_should_be_the_same)
{
    State t = state_copy(s);
    TEST_ASSERT(state_pos_equal(s, t));
    TEST_ASSERT(state_pos_equal(t, s));
    state_fini(t);
}
TEST(state, init_state_should_be_left_movalble)
{
    TEST_ASSERT(state_movable(s, LEFT));
}
TEST(state, init_state_should_be_down_movalble)
{
    TEST_ASSERT(state_movable(s, DOWN));
}
TEST(state, init_state_should_be_right_movalble)
{
    TEST_ASSERT(state_movable(s, RIGHT));
}
TEST(state, init_state_should_be_up_movalble)
{
    TEST_ASSERT(state_movable(s, UP));
}
TEST(state, init_state_should_not_be_left_movalble_twice)
{
    state_move(s, LEFT);
    TEST_ASSERT_FALSE(state_movable(s, LEFT));
}
TEST(state, init_state_should_not_be_down_movalble_twice)
{
    state_move(s, DOWN);
    TEST_ASSERT_FALSE(state_movable(s, DOWN));
}
TEST(state, init_state_should_not_be_right_movalble_twice)
{
    state_move(s, RIGHT);
    TEST_ASSERT_FALSE(state_movable(s, RIGHT));
}
TEST(state, init_state_should_not_be_up_movalble_twice)
{
    state_move(s, UP);
    TEST_ASSERT_FALSE(state_movable(s, UP));
}
TEST(state, go_to_the_same_state_in_different_ways)
{
    State s1 = state_copy(s), s2 = state_copy(s);

    state_move(s1, LEFT);
    state_move(s1, UP);

    state_move(s2, UP);
    state_move(s2, DOWN);
    state_move(s2, UP);
    state_move(s2, DOWN);
    state_move(s2, LEFT);
    state_move(s2, UP);

    TEST_ASSERT(state_pos_equal(s1, s2));

    state_fini(s1);
    state_fini(s2);
}

TEST(state, stroll)
{
    state_move(s, LEFT);
    state_move(s, UP);
    state_move(s, RIGHT);
    state_move(s, RIGHT);
    state_move(s, DOWN);
    state_move(s, DOWN);
    state_move(s, LEFT);
    state_move(s, LEFT);
    state_move(s, UP);
    state_move(s, UP);
    state_move(s, RIGHT);
    state_move(s, RIGHT);
    state_move(s, DOWN);
    state_move(s, DOWN);
    state_move(s, LEFT);
    state_move(s, LEFT);
    state_move(s, UP);
    state_move(s, UP);
}

TEST(state, state_pos_equal_regard_different_depth_as_the_same_state)
{
    value v_list[WIDTH * WIDTH] = {1, 2, 3, 4, VALUE_EMPTY, 8, 7, 6, 5};
    State s1                    = state_init(v_list, 0);
    State s2                    = state_init(v_list, 1323);

    TEST_ASSERT(state_pos_equal(s1, s2));
}

TEST_GROUP_RUNNER(state)
{
    RUN_TEST_CASE(state, initalization);
    RUN_TEST_CASE(state, copied_state_should_be_the_same);
    RUN_TEST_CASE(state, init_state_should_be_left_movalble);
    RUN_TEST_CASE(state, init_state_should_be_down_movalble);
    RUN_TEST_CASE(state, init_state_should_be_right_movalble);
    RUN_TEST_CASE(state, init_state_should_be_up_movalble);
    RUN_TEST_CASE(state, init_state_should_not_be_left_movalble_twice);
    RUN_TEST_CASE(state, init_state_should_not_be_down_movalble_twice);
    RUN_TEST_CASE(state, init_state_should_not_be_right_movalble_twice);
    RUN_TEST_CASE(state, init_state_should_not_be_up_movalble_twice);
    RUN_TEST_CASE(state, go_to_the_same_state_in_different_ways);
    RUN_TEST_CASE(state, stroll)
    RUN_TEST_CASE(state,
                  state_pos_equal_regard_different_depth_as_the_same_state)
}
