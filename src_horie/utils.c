#include "./utils.h"
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdlib.h>

int
pop_int_from_str(const char *str, char **end_ptr)
{
    long int rv;
    errno = 0;
    rv    = strtol(str, end_ptr, 0);

    if (errno != 0)
    {
        elog("%s: %s cannot be converted into long\n", __func__, str);
        exit(EXIT_FAILURE);
    }
    else if (end_ptr && str == *end_ptr)
    {
        elog("%s: reach end of string", __func__);
        exit(EXIT_FAILURE);
    }

    if (rv > INT_MAX || rv < INT_MIN)
    {
        elog("%s: too big number, %ld\n", __func__, rv);
        exit(EXIT_FAILURE);
    }

    return (int) rv;
}

void *
palloc(size_t size)
{
    void *ptr = malloc(size);
    if (!ptr)
        elog("malloc failed\n");

    return ptr;
}

void *
repalloc(void *old_ptr, size_t new_size)
{
    void *ptr = realloc(old_ptr, new_size);
    if (!ptr)
        elog("realloc failed\n");

    return ptr;
}

void
pfree(void *ptr)
{
    if (!ptr)
        elog("empty ptr\n");
    free(ptr);
}
