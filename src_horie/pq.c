#include "./pq.h"
#include "./utils.h"

#include <assert.h>
#include <stdint.h>

typedef struct pq_entry_tag
{
    State state;
    int   f, g;
} PQEntryData;
typedef PQEntryData *PQEntry;

/* tiebreaking is done comparing g value */
static inline bool
pq_entry_higher_priority(PQEntry e1, PQEntry e2)
{
    return e1->f < e2->f || (e1->f == e2->f && e1->g >= e2->g);
}

/*
 * NOTE:
 * This priority queue is implemented doubly reallocated array.
 * It will only extend and will not shrink, for now.
 * It may be improved by using array of layers of iteratively widened array
 */
struct pq_tag
{
    size_t       n_elems;
    size_t       capa;
    PQEntryData *array;
};

static inline size_t
calc_init_capa(size_t capa_hint)
{
    size_t capa = 1;
    assert(capa_hint > 0);

    while (capa < capa_hint)
        capa <<= 1;
    return capa - 1;
}

PQ
pq_init(size_t init_capa_hint)
{
    PQ pq = palloc(sizeof(*pq));

    pq->n_elems = 0;
    pq->capa    = calc_init_capa(init_capa_hint);

    assert(pq->capa <= SIZE_MAX / sizeof(PQEntryData));
    pq->array = palloc(sizeof(PQEntryData) * pq->capa);

    return pq;
}

void
pq_fini(PQ pq)
{
    for (size_t i = 0; i < pq->n_elems; ++i)
        state_fini(pq->array[i].state);

    pfree(pq->array);
    pfree(pq);
}

static inline bool
pq_is_full(PQ pq)
{
    assert(pq->n_elems <= pq->capa);
    return pq->n_elems == pq->capa;
}

static inline void
pq_extend(PQ pq)
{
    pq->capa = (pq->capa << 1) + 1;
    assert(pq->capa <= SIZE_MAX / sizeof(PQEntryData));

    pq->array = repalloc(pq->array, sizeof(PQEntryData) * pq->capa);
}

static inline void
pq_swap_entry(PQ pq, size_t i, size_t j)
{
    PQEntryData tmp = pq->array[i];
    pq->array[i]    = pq->array[j];
    pq->array[j]    = tmp;
}

static inline size_t
pq_up(size_t i)
{
    /* NOTE: By using 1-origin, it may be written more simply, i >> 1 */
    return (i - 1) >> 1;
}

static inline size_t
pq_left(size_t i)
{
    return (i << 1) + 1;
}

static void
heapify_up(PQ pq)
{
    for (size_t i = pq->n_elems; i > 0;)
    {
        size_t ui = pq_up(i);
        assert(i > 0);
        if (!pq_entry_higher_priority(&pq->array[i], &pq->array[ui]))
            break;

        pq_swap_entry(pq, i, ui);
        i = ui;
    }
}

void
pq_put(PQ pq, State state, int f, int g)
{
    if (pq_is_full(pq))
        pq_extend(pq);

    pq->array[pq->n_elems].state = state_copy(state);
    pq->array[pq->n_elems].f     = f; /* this may be abundant */
    pq->array[pq->n_elems].g     = g;
    heapify_up(pq);
    ++pq->n_elems;
}

static void
heapify_down(PQ pq)
{
    size_t sentinel = pq->n_elems;

    for (size_t i = 0;;)
    {
        size_t ri, li = pq_left(i);
        if (li >= sentinel)
            break;

        ri = li + 1;
        if (ri >= sentinel)
        {
            if (pq_entry_higher_priority(&pq->array[li], &pq->array[i]))
                pq_swap_entry(pq, i, li);
            /* Reached the bottom */
            break;
        }

        /* NOTE: If p(ri) == p(li), it may be good to go right
         * since the filling order is left-first */
        if (pq_entry_higher_priority(&pq->array[li], &pq->array[ri]))
        {
            if (!pq_entry_higher_priority(&pq->array[i], &pq->array[li]))
                break;

            pq_swap_entry(pq, i, li);
            i = li;
        }
        else
        {
            if (!pq_entry_higher_priority(&pq->array[i], &pq->array[ri]))
                break;

            pq_swap_entry(pq, i, ri);
            i = ri;
        }
    }
}

State
pq_pop(PQ pq)
{
    State ret_state;

    if (pq->n_elems == 0)
        return NULL;

    ret_state = pq->array[0].state;

    --pq->n_elems;
    pq->array[0] = pq->array[pq->n_elems];
    heapify_down(pq);

    return ret_state;
}

void
pq_dump(PQ pq)
{
    elog("%s: n_elems=%zu, capa=%zu\n", __func__, pq->n_elems, pq->capa);
    for (size_t i = 0, cr_required = 1; i < pq->n_elems; i++)
    {
        if (i == cr_required)
        {
            elog("\n");
            cr_required = (cr_required << 1) + 1;
        }
        elog("%d,", pq->array[i].f);
        elog("%d ", pq->array[i].g);
    }
    elog("\n");
}
