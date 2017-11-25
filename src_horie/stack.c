#include "./stack.h"

#include "./utils.h"

#include <stdint.h>

struct stack_tag
{
    size_t capa, i;
    State *buf;
};

static size_t
calc_init_capa(size_t hint)
{
    size_t capa;
    assert(hint > 0);
    for (capa = 1;; capa <<= 1)
        if (capa >= hint)
            break;
    return capa;
}

static inline size_t
calc_larger_capa(size_t old_size)
{
    return old_size << 1;
}

Stack
stack_init(size_t init_capa_hint)
{
    size_t capa  = calc_init_capa(init_capa_hint);
    Stack  stack = palloc(sizeof(*stack));

    assert(capa <= SIZE_MAX / sizeof(State));

    stack->buf  = palloc(sizeof(State) * capa);
    stack->capa = capa;
    stack->i    = 0;

    return stack;
}

void
stack_fini(Stack stack)
{
    assert(stack);
    while (stack->i != 0)
        state_fini(stack->buf[--stack->i]);
    pfree(stack);
}

void
stack_put(Stack stack, State state)
{
    assert(stack->i < SIZE_MAX);
    stack->buf[stack->i++] = state;

    if (stack->i >= stack->capa)
    {
        size_t new_capa = calc_larger_capa(stack->capa);
        assert(new_capa <= SIZE_MAX / sizeof(State));
        stack->buf  = repalloc(stack->buf, sizeof(State) * new_capa);
        stack->capa = new_capa;
    }
}

State
stack_pop(Stack stack)
{
    /* shrinking is needed ? */
    return stack->i == 0 ? NULL : stack->buf[--stack->i];
}

void
stack_dump(Stack stack)
{
    elog("%s: capa=%zu, i=%zu\n", __func__, stack->capa, stack->i);
}
