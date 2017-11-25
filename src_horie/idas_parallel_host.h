#pragma once

#define N_BLOCK 48
#define N_CORE N_BLOCK * 32

#define STATE_WIDTH 4
#define STATE_N (STATE_WIDTH * STATE_WIDTH)
#define POS_X(pos) ((pos) % STATE_WIDTH)
#define POS_Y(pos) ((pos) / STATE_WIDTH)

#define WARP_SIZE 32

#define STACK_SIZE_BYTES 64
#define STACK_BUF_BYTES (STACK_SIZE_BYTES - sizeof(uchar))
#define STACK_DIR_BITS 2
#define STACK_DIR_MASK ((1 << STACK_DIR_BITS) - 1)
#define PLAN_LEN_MAX ((1 << STACK_DIR_BITS) * STACK_BUF_BYTES)

typedef unsigned char uchar;
typedef uchar         DirDev;
#define dir_reverse_dev(dir) ((DirDev)(3 - (dir)))
#define DIR_N 4
#define DIR_FIRST 0
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

#define NOT_SOLVED -1

#include <stdbool.h>

void idas_parallel_main(uchar *input, signed char *plan, int f_limit,
                        signed char *h_diff_table, bool *movable_table);
