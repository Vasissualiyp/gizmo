/* Driver for test_rt_source_luminosity.py: calls rt_get_source_luminosity on a gas cell with the output array
   pre-filled with garbage, as the callers' stack arrays are, and prints PASS|FAIL per band. */
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "declarations/allvars.h"
#include "core/proto.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
    MPI_Comm_size(MPI_COMM_WORLD, &NTask);
    memset(&All, 0, sizeof(All));
    strncpy(ParameterFile, argv[1], sizeof(ParameterFile)-1);
    read_parameter_file(ParameterFile);
    set_units();

    struct particle_data *pp; struct gas_cell_data *cell; /* aligned like mymalloc */
    if(posix_memalign((void **)&pp, 64, sizeof(*pp)) || posix_memalign((void **)&cell, 64, sizeof(*cell))) {return 1;}
    memset((void *)pp, 0, sizeof(*pp)); memset((void *)cell, 0, sizeof(*cell));
    pp[0].Type = 0; pp[0].Mass = cell[0].Mass = 1.0; cell[0].Density = 1.0;
    for(int k = 0; k < N_RT_FREQ_BINS; k++) {cell[0].Rad_Je[k] = 0.1 * (k + 1);}

    int n_fail = 0;
    const double garbage[2] = {NAN, 1e30};
    for(int g = 0; g < 2; g++)
    {
        double lum[N_RT_FREQ_BINS];
        for(int k = 0; k < N_RT_FREQ_BINS; k++) {lum[k] = garbage[g];}
        int active = rt_get_source_luminosity(0, 0, lum, pp, cell);
        printf("%s gas_is_a_source[%d] got=%d want=1\n", active ? "PASS" : "FAIL", g, active); n_fail += !active;
        for(int k = 0; k < N_RT_FREQ_BINS; k++)
        {
            double want = cell[0].Rad_Je[k] * (C_LIGHT_CODE_REDUCED / C_LIGHT_CODE);
            int ok = std::isfinite(lum[k]) && fabs(lum[k] - want) <= 1e-12 * want;
            printf("%s lum[%d]_with_garbage[%d] got=%g want=%g\n", ok ? "PASS" : "FAIL", k, g, lum[k], want); n_fail += !ok;
        }
    }
    printf("DONE failed=%d\n", n_fail);
    MPI_Finalize();
    return n_fail;
}
