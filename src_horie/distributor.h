#pragma once

#include "./state.h"
#include <stdbool.h>

bool distributor_bfs(State init_s, State geal_s, unsigned char s_list_ret[],
                     int distr_n);
