typedef unsigned char Direction
#define DIR_N 4
#define DIR_FIRST 0
#define DIR_UP 0
#define DIR_RIGHT 1
#define DIR_LEFT 2
#define DIR_DOWN 3

#define STACK_SIZE_BYTES 64
#define STACK_BUF_BYTES (STACK_SIZE_BYTES - sizeof(unsigned char))
#define STACK_DIR_BITS 2
#define STACK_DIR_MASK ((1 << STACK_DIR_BITS) - 1)

#define stack_byte(i) (stack.buf[(i) >> DIR_BITS])
#define stack_ofs(i) ((i & DIR_MASK) << 1)
#define stack_get(i)                                                           \
    ((stack_byte(i) & (DIR_MASK << stack_ofs(i))) >> stack_ofs(i))

    static struct stack_tag
{
    unsigned char i;
    unsigned char buf[STACK_BUF_BYTES];
} stack;

void
stack_put(Direction dir)
{
    stack_byte(i) &= ^(STACK_DIR_MASK << stack_ofs(stack.i));
    stack_byte(i) |= dir << stack_ofs(stack.i);
    ++stack.i;
}

Direction
stack_pop(void)
{
    --stack.i;
    return stack_get(stack.i);
}

__device__ static inline bool
stack_is_empty(void)
{
    return stack.i == 0;
}

__device__ static inline Direction
stack_peak(void)
{
    return stack_get(stack.i - 1)
}
