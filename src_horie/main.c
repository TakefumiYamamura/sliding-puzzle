#include "./emb_idas.h"
#include "./solver.h"
#include "./state.h"
#include "./utils.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define BUFLEN 100

typedef struct option_tag
{
    Solver solver;
    int    depth_limit;
    int    f_limit;
    char   ifname[BUFLEN];
} * MainOption;

static void
OptionInit(MainOption opt)
{
    opt->solver      = SolverNotSet;
    opt->depth_limit = 0;
    opt->f_limit     = 0;
    opt->ifname[0]   = '\0';
}

static void
OptionValidate(MainOption opt)
{
    assert(opt->solver != SolverNotSet);
    assert(opt->solver == SolverIDAStar || opt->solver == SolverAStar ||
           opt->solver == SolverIDAMini);

    assert(opt->depth_limit >= 0);
    if (opt->depth_limit > 0)
        assert(opt->solver == SolverDlS);

    assert(opt->f_limit >= 0);
    if (opt->f_limit > 0)
        assert(opt->solver == SolverFLAStar);

    assert(opt->ifname[0] != '\0');
}

static void
solver_main(MainOption opt, State init_state)
{
    switch (opt->solver)
    {
    case SolverAStar:
        solver_astar(init_state);
        break;
    case SolverIDAStar:
        solver_idastar(init_state);
        break;
    default:
        exit_with_log("%s: unrecognized solver\n", __func__);
    }
}

static void
show_help(void)
{
    elog("-h              : show help\n");
    elog("-s <int>        : astar(%d), idastar(%d), miniida*(%d)\n",
         SolverAStar, SolverIDAStar, SolverIDAMini);
    elog("-f <int>        : fvalue limit (LFA*)\n");
    elog("-d <int>        : depth\n");
    elog("-i <filename>   : input file\n");
}

#define MAX_LINE_LEN 100
static void
load_state_from_file(const char *fname, state_panel *s)
{
    FILE *fp;
    char  str[MAX_LINE_LEN];
    char *str_ptr = str, *end_ptr;

    fp = fopen(fname, "r");
    if (!fp)
        exit_with_log("%s: %s cannot be opened\n", __func__, fname);

    if (!fgets(str, MAX_LINE_LEN, fp))
        exit_with_log("%s: fgets failed\n", __func__);

    for (int i = 0; i < STATE_N; ++i)
    {
        s[i]    = pop_int_from_str(str_ptr, &end_ptr);
        str_ptr = end_ptr;
    }

    fclose(fp);
}
#undef MAX_LINE_LEN

int
main(int argc, char *argv[])
{
    state_panel       s_list[STATE_N];
    State             s;
    struct option_tag opt;
    int               ch;

    OptionInit(&opt);

    while ((ch = getopt(argc, argv, "hm:s:f:d:i:")) != -1)
    {
        switch (ch)
        {
        case 's':
            opt.solver = pop_int_from_str(optarg, NULL);
            break;
        case 'f':
            opt.f_limit = pop_int_from_str(optarg, NULL);
            break;
        case 'd':
            opt.depth_limit = pop_int_from_str(optarg, NULL);
            break;
        case 'i':
            strncpy(opt.ifname, optarg, BUFLEN);
            break;

        case 'h':
            show_help();
            exit_with_log(" ");
        default:
            show_help();
            exit_with_log("%s: unknown option %c specified.\n", __func__, ch);
        }
    }

    OptionValidate(&opt);

    load_state_from_file(opt.ifname, s_list);

    if (opt.solver == SolverIDAMini)
        idas_main(s_list);
    else
    {
        s = state_init(s_list, 0);
        solver_main(&opt, s);
        state_fini(s);
    }

    return 0;
}
