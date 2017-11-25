#include "./test.h"
#include "stdio.h"

int
main(void)
{
    RUN_TEST_GROUP(state);
    RUN_TEST_GROUP(stack);
    RUN_TEST_GROUP(queue);
    RUN_TEST_GROUP(hash);
    RUN_TEST_GROUP(heuristic);
    RUN_TEST_GROUP(pq);

    puts("\nTest finished.");
    return 0;
}
