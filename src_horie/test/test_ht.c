#include "../ht.h"
#include "../state.h"
#include "./test.h"

#include <assert.h>

TEST_GROUP(hash);

static HT    ht;
static State key;

TEST_SETUP(hash)
{
    state_panel v_list[STATE_WIDTH * STATE_WIDTH] = {1, 2, 3, 4, STATE_EMPTY,
                                                     8, 7, 6, 5};

    ht  = ht_init(3);
    key = state_init(v_list, 0);
}

TEST_TEAR_DOWN(hash)
{
    state_fini(key);
    ht_fini(ht);
}

TEST(hash, initialization)
{
}

TEST(hash, search_empty_hash)
{
    ht_value value;
    HTStatus status = ht_search(ht, key, &value);

    TEST_ASSERT_EQUAL(HT_FAILED_NOT_FOUND, status);
}

TEST(hash, delete_empty_hash)
{
    HTStatus status = ht_delete(ht, key);

    TEST_ASSERT_EQUAL(HT_FAILED_NOT_FOUND, status);
}

TEST(hash, inserted_key_should_be_found)
{
    ht_value *value, ret_value;
    HTStatus  status = ht_insert(ht, key, &value);
    *value           = 1234;

    TEST_ASSERT_EQUAL(HT_SUCCESS, status);

    status = ht_search(ht, key, &ret_value);
    TEST_ASSERT_EQUAL(HT_SUCCESS, status);
    TEST_ASSERT_EQUAL(1234, ret_value);
}

TEST(hash, inserted_key_should_be_deletable)
{
    ht_value *value, ret_value;
    HTStatus  status = ht_insert(ht, key, &value);
    *value           = 1234;

    TEST_ASSERT_EQUAL(HT_SUCCESS, status);

    status = ht_delete(ht, key);
    TEST_ASSERT_EQUAL(HT_SUCCESS, status);

    status = ht_search(ht, key, &ret_value);
    TEST_ASSERT_EQUAL(HT_FAILED_NOT_FOUND, status);

    status = ht_delete(ht, key);
    TEST_ASSERT_EQUAL(HT_FAILED_NOT_FOUND, status);
}

TEST(hash, insert_the_same_keys)
{
    ht_value *value;
    HTStatus  status = ht_insert(ht, key, &value);
    *value           = 1234;

    TEST_ASSERT_EQUAL(HT_SUCCESS, status);

    for (int i = 0; i < 1000; ++i)
    {
        status = ht_insert(ht, key, &value);
        TEST_ASSERT_EQUAL(HT_FAILED_FOUND, status);
    }
}

TEST(hash, insert_the_different_keys)
{
    ht_value *value;
    HTStatus  status = ht_insert(ht, key, &value);
    *value           = 1234;

    TEST_ASSERT_EQUAL(HT_SUCCESS, status);

    for (int i = 0; i < 1000; ++i)
    {
        State       fake_key;
        state_panel fake_list[STATE_WIDTH * STATE_WIDTH] = {
            1, 2, 3, 4, STATE_EMPTY, 8, 7, 6, 5};

        /* generate different state less than 10000 times */
        assert(i < 10000);
        fake_list[0] = 30 + i % 100;
        fake_list[1] = 130 + i / 100;

        fake_key = state_init(fake_list, 9);
        status   = ht_insert(ht, fake_key, &value);

        TEST_ASSERT_EQUAL(HT_SUCCESS, status);

        state_fini(fake_key);
    }
}

TEST_GROUP_RUNNER(hash)
{
    RUN_TEST_CASE(hash, initialization);
    RUN_TEST_CASE(hash, search_empty_hash);
    RUN_TEST_CASE(hash, delete_empty_hash);
    RUN_TEST_CASE(hash, inserted_key_should_be_found);
    RUN_TEST_CASE(hash, inserted_key_should_be_deletable);
    RUN_TEST_CASE(hash, insert_the_same_keys);
    RUN_TEST_CASE(hash, insert_the_different_keys);
}
