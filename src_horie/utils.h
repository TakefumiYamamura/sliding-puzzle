#pragma once

#include <assert.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#define elog(...) fprintf(stderr, __VA_ARGS__)
#define exit_with_log(...)                                                     \
    do                                                                         \
    {                                                                          \
        elog(__VA_ARGS__);                                                     \
        exit(EXIT_FAILURE);                                                    \
    } while (0)

int pop_int_from_str(const char *str, char **end_ptr);

void *palloc(size_t size);
void *repalloc(void *ord_ptr, size_t size);
void pfree(void *ptr);
