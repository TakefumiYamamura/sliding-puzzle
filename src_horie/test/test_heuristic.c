#include "../state.h"
#include "./test.h"

TEST_GROUP(heuristic);

static State s, g;

TEST_SETUP(heuristic)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, 3, 4,          5,
                                                     6, 7, 8, STATE_EMPTY};
    s = state_init(v_list, 0);
}

TEST_TEAR_DOWN(heuristic)
{
    state_fini(s);
}

TEST(heuristic, zero)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, 3, 4,          5,
                                                     6, 7, 8, STATE_EMPTY};
    g = state_init(v_list, 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicManhattanDistance, s, g), 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicMisplacedTiles, s, g), 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicTilesOutOfRowCol, s, g), 0);
    state_fini(g);
}
TEST(heuristic, yoko1)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, 3,           4, 5,
                                                     6, 7, STATE_EMPTY, 8};
    g = state_init(v_list, 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicManhattanDistance, s, g), 1);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicMisplacedTiles, s, g), 1);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicTilesOutOfRowCol, s, g), 1);
    state_fini(g);
}
TEST(heuristic, yoko2)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1, 2,           3, 4, 5,
                                                     6, STATE_EMPTY, 7, 8};
    g = state_init(v_list, 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicManhattanDistance, s, g), 2);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicMisplacedTiles, s, g), 2);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicTilesOutOfRowCol, s, g), 2);
    state_fini(g);
}
TEST(heuristic, yoko3)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1, 2,           3, 4, 5,
                                                     6, STATE_EMPTY, 8, 7};
    g = state_init(v_list, 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicManhattanDistance, s, g), 2);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicMisplacedTiles, s, g), 1);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicTilesOutOfRowCol, s, g), 1);
    state_fini(g);
}
TEST(heuristic, tate1)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1,           2, 3, 4, 5,
                                                     STATE_EMPTY, 7, 8, 6};
    g = state_init(v_list, 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicManhattanDistance, s, g), 1);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicMisplacedTiles, s, g), 1);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicTilesOutOfRowCol, s, g), 1);
    state_fini(g);
}
TEST(heuristic, tate2)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, STATE_EMPTY, 4, 5,
                                                     3, 7, 8,           6};
    g = state_init(v_list, 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicManhattanDistance, s, g), 2);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicMisplacedTiles, s, g), 2);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicTilesOutOfRowCol, s, g), 2);
    state_fini(g);
}
TEST(heuristic, tate3)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, STATE_EMPTY, 4, 5,
                                                     6, 7, 8,           3};
    g = state_init(v_list, 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicManhattanDistance, s, g), 2);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicMisplacedTiles, s, g), 1);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicTilesOutOfRowCol, s, g), 1);
    state_fini(g);
}
TEST(heuristic, yoko1tate1)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, 3, 4, STATE_EMPTY,
                                                     6, 7, 8, 5};
    g = state_init(v_list, 0);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicManhattanDistance, s, g), 2);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicMisplacedTiles, s, g), 1);
    TEST_ASSERT_EQUAL(calc_h_value(HeuristicTilesOutOfRowCol, s, g), 2);
    state_fini(g);
}

TEST_GROUP_RUNNER(heuristic)
{
    RUN_TEST_CASE(heuristic, zero);
    RUN_TEST_CASE(heuristic, yoko1);
    RUN_TEST_CASE(heuristic, yoko2);
    RUN_TEST_CASE(heuristic, yoko3);
    RUN_TEST_CASE(heuristic, tate1);
    RUN_TEST_CASE(heuristic, tate2);
    RUN_TEST_CASE(heuristic, tate3);
    RUN_TEST_CASE(heuristic, yoko1tate1);
}
