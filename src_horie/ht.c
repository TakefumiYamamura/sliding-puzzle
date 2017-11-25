#include "./ht.h"
#include "./state.h"
#include "./utils.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* XXX: hash function for State should be surveyed */
inline static size_t
hashfunc(State key)
{
    return state_hash(key);
}

typedef struct ht_entry_tag *HTEntry;
struct ht_entry_tag
{
    HTEntry  next;
    State    key;
    ht_value value;
};

static HTEntry
ht_entry_init(State key)
{
    HTEntry entry = palloc(sizeof(*entry));

    entry->key  = state_copy(key);
    entry->next = NULL;

    return entry;
}

static void
ht_entry_fini(HTEntry entry)
{
    pfree(entry);
}

struct ht_tag
{
    size_t   n_bins;
    size_t   n_elems;
    HTEntry *bin;
};

static bool
ht_rehash_required(HT ht)
{
    return ht->n_bins <= ht->n_elems; /* TODO: local policy is also needed */
}

static size_t
calc_n_bins(size_t required)
{
    /* NOTE: n_bins is used for mask and hence it should be pow of 2, fon now */
    size_t size = 1;
    assert(required > 0);

    while (required > size)
        size <<= 1;

    return size;
}

HT
ht_init(size_t init_size_hint)
{
    size_t n_bins = calc_n_bins(init_size_hint);
    HT     ht     = palloc(sizeof(*ht));

    ht->n_bins  = n_bins;
    ht->n_elems = 0;

    assert(sizeof(*ht->bin) <= SIZE_MAX / n_bins);
    ht->bin = palloc(sizeof(*ht->bin) * n_bins);
    memset(ht->bin, 0, sizeof(*ht->bin) * n_bins);

    return ht;
}

static void
ht_rehash(HT ht)
{
    HTEntry *new_bin;
    size_t   new_size = ht->n_bins << 1;

    assert(ht->n_bins<SIZE_MAX>> 1);

    new_bin = palloc(sizeof(*new_bin) * new_size);
    memset(new_bin, 0, sizeof(*new_bin) * new_size);

    for (size_t i = 0; i < ht->n_bins; ++i)
    {
        HTEntry entry = ht->bin[i];

        while (entry)
        {
            HTEntry next = entry->next;

            size_t idx   = hashfunc(entry->key) & (new_size - 1);
            entry->next  = new_bin[idx];
            new_bin[idx] = entry;

            entry = next;
        }
    }

    pfree(ht->bin);
    ht->n_bins = new_size;
    ht->bin    = new_bin;
}

void
ht_fini(HT ht)
{
    for (size_t i = 0; i < ht->n_bins; ++i)
    {
        HTEntry entry = ht->bin[i];
        while (entry)
        {
            HTEntry next = entry->next;
            state_fini(entry->key);
            ht_entry_fini(entry);
            entry = next;
        }
    }

    pfree(ht->bin);
    pfree(ht);
}

HTStatus
ht_search(HT ht, State key, ht_value *ret_value)
{
    size_t  i     = hashfunc(key) & (ht->n_bins - 1);
    HTEntry entry = ht->bin[i];

    while (entry)
    {
        if (state_pos_equal(key, entry->key))
        {
            *ret_value = entry->value;
            return HT_SUCCESS;
        }

        entry = entry->next;
    }

    return HT_FAILED_NOT_FOUND;
}

HTStatus
ht_insert(HT ht, State key, ht_value **value)
{
    size_t  i;
    HTEntry entry, new_entry;

    if (ht_rehash_required(ht))
        ht_rehash(ht);

    i     = hashfunc(key) & (ht->n_bins - 1);
    entry = ht->bin[i];

    while (entry)
    {
        if (state_pos_equal(key, entry->key))
        {
            *value = &entry->value;
            return HT_FAILED_FOUND;
        }

        entry = entry->next;
    }

    new_entry = ht_entry_init(key);

    new_entry->next = ht->bin[i];
    ht->bin[i]      = new_entry;
    *value          = &new_entry->value;

    assert(ht->n_elems < SIZE_MAX);
    ht->n_elems++;

    return HT_SUCCESS;
}

HTStatus
ht_delete(HT ht, State key)
{
    size_t  i     = hashfunc(key) & (ht->n_bins - 1);
    HTEntry entry = ht->bin[i], prev;

    if (!entry)
        return HT_FAILED_NOT_FOUND;

    if (state_pos_equal(key, entry->key))
    {
        ht->bin[i] = entry->next;
        ht_entry_fini(entry);
        return HT_SUCCESS;
    }

    prev  = entry;
    entry = entry->next;

    while (entry)
    {
        if (state_pos_equal(key, entry->key))
        {
            prev->next = entry->next;
            ht_entry_fini(entry);

            assert(ht->n_elems > 0);
            ht->n_elems--;

            return HT_SUCCESS;
        }

        prev  = entry;
        entry = entry->next;
    }

    return HT_FAILED_NOT_FOUND;
}

void
ht_dump(HT ht)
{
    elog("%s: n_elems=%zu, n_bins=%zu\n", __func__, ht->n_elems, ht->n_bins);
}
