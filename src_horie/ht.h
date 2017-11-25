#pragma once

#include "./state.h"

#include <stddef.h>

typedef struct ht_tag *HT;

typedef int ht_value;

HT ht_init(size_t init_size_hint);
void ht_fini(HT ht);

typedef enum {
    HT_SUCCESS = 0,
    HT_FAILED_FOUND,
    HT_FAILED_NOT_FOUND,
} HTStatus;

HTStatus ht_insert(HT ht, State key, ht_value **value);
HTStatus ht_search(HT ht, State key, ht_value *ret_value);
HTStatus ht_delete(HT ht, State key);
void ht_dump(HT ht);
