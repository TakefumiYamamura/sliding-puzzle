#pragma once

#include "./state.h"

#include <stddef.h>

typedef struct queue_tag *Queue;

/* This queue is implemented not by ring-buffer but by page chain
 * since open-list seems to have tendency to extend at first,
 * and will be consumed slowly after the large extension
 */

/* NOTE: Queue just holds references.
 * One should not put two references of the same state into this Queue.
 */

Queue queue_init(void);
void queue_fini(Queue q);
void queue_put(Queue q, State state);
State queue_pop(Queue q); /* return NULL if empty */
void queue_dump(Queue q);
