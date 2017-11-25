#include "./queue.h"
#include "./utils.h"

#include <stdlib.h>
#include <unistd.h>
static long pagesize = 0;
static long n_states_par_page;

/*
 * Queue Page implementation
 */

typedef struct queue_page *QPage;
typedef struct queue_page
{
    size_t out, in;
    QPage  next;
    State  buf[];
} QPageData;

static void
set_pagesize(void)
{
    pagesize = sysconf(_SC_PAGESIZE);
    if (pagesize < 0)
    {
        elog("%s: sysconf(_SC_PAGESIZE) failed\n", __func__);
        exit(EXIT_FAILURE);
    }

    n_states_par_page = (pagesize - sizeof(QPageData)) / sizeof(State);

    elog("%s: pagesize=%ld, n_states/page=%ld\n", __func__, pagesize,
         n_states_par_page);
}

static QPage
qpage_init(void)
{
    QPage qp = (QPage) palloc(sizeof(*qp) + sizeof(State) * n_states_par_page);
    qp->in = qp->out = 0;
    qp->next         = NULL;
    return qp;
}

static void
qpage_fini(QPage qp)
{
    while (qp->out < qp->in)
        state_fini(qp->buf[qp->out++]);
    pfree(qp);
}

static inline bool
qpage_have_space(QPage qp)
{
    return (long) (qp->in + 1) < n_states_par_page;
}

static inline void
qpage_put(QPage qp, State state)
{
    assert(qpage_have_space(qp));
    qp->buf[qp->in++] = state;
}

static inline State
qpage_pop(QPage qp)
{
    return qp->out == qp->in ? NULL : qp->buf[qp->out++];
}

/*
 * Queue implementation
 */

struct queue_tag
{
    QPage head, tail;
};

Queue
queue_init(void)
{
    Queue q = (Queue) palloc(sizeof(*q));

    if (!pagesize)
        set_pagesize();

    q->tail = q->head = qpage_init();
    q->head->in = q->head->out = 0;
    q->head->next              = NULL;

    return q;
}

void
queue_fini(Queue q)
{
    QPage page = q->head;

    while (page)
    {
        QPage next = page->next;
        qpage_fini(page);
        page = next;
    }

    pfree(q);
}

void
queue_put(Queue q, State state)
{
    if (!qpage_have_space(q->tail))
    {
        q->tail->next = qpage_init();
        q->tail       = q->tail->next;
    }

    qpage_put(q->tail, state);
}

State
queue_pop(Queue q)
{
    State state = qpage_pop(q->head);

    if (!state)
    {
        QPage next = q->head->next;
        if (!next)
            return NULL;

        state = qpage_pop(next);
        assert(state);

        qpage_fini(q->head);
        q->head = next;
    }

    return state;
}

void
queue_dump(Queue q)
{
    QPage page = q->head;
    int   cnt  = 0;

    while (page)
    {
        elog("%s: page#%d in=%zu, out=%zu", __func__, cnt++, page->in,
             page->out);
        page = page->next;
    }
}
