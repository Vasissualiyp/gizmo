#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <climits>
#include <float.h>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include "../declarations/allvars.h"
#include "../core/proto.h"
#include "../mesh/kernel.h"
#include "../mesh/neighbor_list.h"
#include "../mesh/sfc_tiles.h"
#include "../mesh/ghost_writeback.h"
#include "compute_finitevol_faces_functions.h"
#include "hydro_corridor.h"

/*! \file mg_gradient_correction.cc
 *  \brief Modified-gradient (MG) method for exact div(B)=0 correction.
 *
 *  Implements Tu, Wang, Gao & Tang (2026), arXiv:2603.04077. The discrete divergence
 *  of cell i is the face sum b_i = sum_j (1/2)(B'_i + B'_j).A_ij, with B' the linear
 *  extrapolation of the (once-limited) cell gradients to the face midpoint. Shifting
 *  both face states by (c_j - c_i)(r^2/8) A_ij (along the face area vector only)
 *  changes that face's term by (c_j - c_i)(r^2/8)|A_ij|^2, so b = 0 on every cell is
 *  the graph-Laplacian system  R c = b  over all gas cells.
 *
 *  This module:
 *    1) assembles R and b over every local gas cell (the graph must be complete: on
 *       an MG step the hydro corridor imports every gas cell's neighbourhood);
 *    2) checks R is symmetric, labels its connected pieces (each has one constant null
 *       mode), checks b sums to roundoff on every piece, and skips pieces whose b is
 *       already at roundoff relative to their face-flux scale;
 *    3) fixes one unknown per piece at zero and solves the now definite system. The
 *       fixed (anchor) row is the one with the largest diagonal: the coefficients can span
 *       tens of orders of magnitude, and an anchor among small-diagonal rows gives the
 *       large-diagonal rows a common offset in c far above their face differences, which
 *       then lose precision. c is used only through differences across faces, so the
 *       choice of anchor never enters the physics;
 *    4) checks the true residual of R c = b: a solve that misses the tolerance is
 *       still applied if it lowers the residual (less accuracy, not a broken state);
 *       otherwise the coefficients and pieces of the last accepted solve are kept and
 *       the step proceeds as one without the solve. Either way a warning is printed.
 *  c_i is stored in CellP[i].MG_cgcoeff (with its piece in CellP[i].MG_piece_token) for the
 *  face reconstruction in hydro and for the constrained-gradient correction on steps
 *  without the solve.
 */
/*
 * This file was written by Phil Hopkins (phopkins@caltech.edu) for GIZMO,
 * based on the method described in Tu, Wang, Gao & Tang (2026), arXiv:2603.04077.
 */

#ifdef MHD_MODIFIED_GRADIENT

extern void gpu_build_symmetric_neighbor_list(struct particle_data *P, int num_total,
    int *active_indices, int num_active, neighbor_list_t *out,
    double search_radius_factor);

#if !defined(MHD_MODIFIED_GRADIENT_CG_ONLY) && !defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"
#endif

#ifdef MHD_MODIFIED_GRADIENT_USE_PARDISO
#include <mkl.h>
#include <mkl_pardiso.h>
#endif

#define MG_CG_MAX_ITER 500
#define MG_CG_TOL 1.0e-10  /* relative-residual tolerance of the solve: tighter values sit below the roundoff
                              floor of the divB-flux right-hand side, so converged solves would never meet them */
#define MG_NOISE_FLOOR 1.0e-10                /* a piece whose ||b|| is below this fraction of its face-flux scale ||s|| is already divergence-free to roundoff: c = 0 there, no solve */
#define MG_RHS_IMBALANCE_STRUCTURAL 1.0e-6    /* |sum b| / sum s on a piece beyond this is not roundoff: the graph or the face fluxes are not antisymmetric */
#define MG_SYMMETRY_TOL 1.0e-12               /* both rows of a face compute its coefficient from the same expression */
#define MG_REPLICATED_PIECE_GRAPH_MAX_BYTES (64LL << 20) /* above this the cross-rank piece graph is resolved by label propagation instead of being gathered everywhere */

/* ==================================================================================== */
/* Matrix storage                                                                         */
/* ==================================================================================== */

struct mg_local_entry { int j; double Qnorm2; };                                          /* neighbour on this task */
struct mg_remote_entry { int remote_task; int remote_index; int exchange_pos; double Qnorm2; }; /* neighbour on another task; exchange_pos = slot in MG_ExchangeValues */

struct mg_row_data {
    double Rdiag;        /* sum_j Qnorm2_ij over all faces */
    double rhs;          /* b_i: face sum of (1/2)(B'_i+B'_j).A_ij */
    double flux_scale;   /* s_i: face sum of (1/2)(|B'_i|+|B'_j|)|A_ij|, the scale b_i is roundoff against */
    long long piece;     /* connected piece label (identity, and the task that reduces the piece): its smallest global row id */
    long long anchor;    /* the piece's anchor row (global id), whose unknown is fixed at zero */
    int n_local, n_remote, alloc_local, alloc_remote;
    struct mg_local_entry *local_entries;
    struct mg_remote_entry *remote_entries;
};

static struct mg_row_data *MG_Rows = NULL;
static std::vector<long long> MG_TaskRowOffsets;  /* global row id of task t's row 0 */
static std::vector<char> MG_RowSolved;            /* 1 = unknown solved for; 0 = c fixed at 0 (a piece's anchor, an isolated cell, or a piece below the noise floor) */
static std::vector<double> MG_RhsSolved;          /* b on solved rows, 0 elsewhere */
static std::vector<char> MG_RowInSolvedPiece;     /* 1 = the row's piece is above the noise floor (its residual is judged) */

static inline long long mg_global_row(int task, int index) {return MG_TaskRowOffsets[task] + (long long)index;}
static inline int mg_row_owner(long long global_row)
{
    return (int)(std::upper_bound(MG_TaskRowOffsets.begin(), MG_TaskRowOffsets.end(), global_row) - MG_TaskRowOffsets.begin()) - 1;
}
/* the anchor unknown of a piece is fixed at zero, so its column drops out of the other rows */
static inline bool mg_column_kept(int i, long long global_col) {return global_col != MG_Rows[i].anchor;}

static int MG_EntryAllocFailed = 0;   /* set when a row's entry storage could not grow; checked collectively after the build */

static inline void mg_add_local_entry(int i, int j, double Qnorm2)
{
    if(MG_Rows[i].n_local >= MG_Rows[i].alloc_local) {
        int n_alloc = DMAX(2 * MG_Rows[i].alloc_local, 32);
        void *grown = realloc(MG_Rows[i].local_entries, n_alloc * sizeof(struct mg_local_entry));
        if(!grown) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
            MG_EntryAllocFailed = 1;
            return;
        }
        MG_Rows[i].local_entries = (struct mg_local_entry *) grown; MG_Rows[i].alloc_local = n_alloc;
    }
    MG_Rows[i].local_entries[MG_Rows[i].n_local].j = j;
    MG_Rows[i].local_entries[MG_Rows[i].n_local].Qnorm2 = Qnorm2;
    MG_Rows[i].n_local++;
}

static inline void mg_add_remote_entry(int i, int remote_task, int remote_index, double Qnorm2)
{
    if(MG_Rows[i].n_remote >= MG_Rows[i].alloc_remote) {
        int n_alloc = DMAX(2 * MG_Rows[i].alloc_remote, 16);
        void *grown = realloc(MG_Rows[i].remote_entries, n_alloc * sizeof(struct mg_remote_entry));
        if(!grown) {
#ifdef _OPENMP
#pragma omp atomic write
#endif
            MG_EntryAllocFailed = 1;
            return;
        }
        MG_Rows[i].remote_entries = (struct mg_remote_entry *) grown; MG_Rows[i].alloc_remote = n_alloc;
    }
    struct mg_remote_entry *e = &MG_Rows[i].remote_entries[MG_Rows[i].n_remote];
    e->remote_task = remote_task; e->remote_index = remote_index; e->exchange_pos = -1; e->Qnorm2 = Qnorm2;
    MG_Rows[i].n_remote++;
}

/* ==================================================================================== */
/* Values of remote neighbours: each task fetches, for every remote entry, a value its   */
/* owner holds for that row (piece labels, solution components).                         */
/* ==================================================================================== */

static int *mg_xch_send_counts = NULL, *mg_xch_recv_counts = NULL, *mg_xch_send_offsets = NULL, *mg_xch_recv_offsets = NULL;
static int *mg_xch_send_indices = NULL;   /* local rows other tasks asked for, grouped by task */
static int mg_xch_n_recv = 0, mg_xch_n_send = 0;
static std::vector<double> MG_ExchangeValues;   /* received values, slot = remote entry's exchange_pos */

static void mg_setup_exchange(void)
{
    mg_xch_send_counts = (int *) calloc(NTask, sizeof(int)); mg_xch_recv_counts = (int *) calloc(NTask, sizeof(int));
    mg_xch_send_offsets = (int *) calloc(NTask, sizeof(int)); mg_xch_recv_offsets = (int *) calloc(NTask, sizeof(int));
    for(int i = 0; i < N_gas; i++) for(int n = 0; n < MG_Rows[i].n_remote; n++) {mg_xch_recv_counts[MG_Rows[i].remote_entries[n].remote_task]++;}
    MPI_Alltoall(mg_xch_recv_counts, 1, MPI_INT, mg_xch_send_counts, 1, MPI_INT, MPI_COMM_WORLD);
    mg_xch_n_recv = mg_xch_n_send = 0;
    for(int t = 0; t < NTask; t++) {
        mg_xch_recv_offsets[t] = mg_xch_n_recv; mg_xch_send_offsets[t] = mg_xch_n_send;
        mg_xch_n_recv += mg_xch_recv_counts[t]; mg_xch_n_send += mg_xch_send_counts[t];
    }
    /* each remote entry gets its slot in the task-grouped receive buffer */
    std::vector<int> fill(NTask, 0), requested((size_t)DMAX(mg_xch_n_recv, 1));
    for(int i = 0; i < N_gas; i++) for(int n = 0; n < MG_Rows[i].n_remote; n++) {
        struct mg_remote_entry *e = &MG_Rows[i].remote_entries[n];
        e->exchange_pos = mg_xch_recv_offsets[e->remote_task] + fill[e->remote_task]++;
        requested[e->exchange_pos] = e->remote_index;
    }
    mg_xch_send_indices = (int *) malloc((size_t)DMAX(mg_xch_n_send, 1) * sizeof(int));
    MPI_Alltoallv(requested.data(), mg_xch_recv_counts, mg_xch_recv_offsets, MPI_INT,
                  mg_xch_send_indices, mg_xch_send_counts, mg_xch_send_offsets, MPI_INT, MPI_COMM_WORLD);
    MG_ExchangeValues.assign((size_t)DMAX(mg_xch_n_recv, 1), 0.0);
}

static void mg_exchange_values(const double *local_values)
{
    std::vector<double> send((size_t)DMAX(mg_xch_n_send, 1));
    for(int k = 0; k < mg_xch_n_send; k++) {send[k] = local_values[mg_xch_send_indices[k]];}
    MPI_Alltoallv(send.data(), mg_xch_send_counts, mg_xch_send_offsets, MPI_DOUBLE,
                  MG_ExchangeValues.data(), mg_xch_recv_counts, mg_xch_recv_offsets, MPI_DOUBLE, MPI_COMM_WORLD);
}

static void mg_cleanup_exchange(void)
{
    free(mg_xch_send_indices); mg_xch_send_indices = NULL;
    free(mg_xch_recv_offsets); free(mg_xch_send_offsets); free(mg_xch_recv_counts); free(mg_xch_send_counts);
    mg_xch_recv_offsets = mg_xch_send_offsets = mg_xch_recv_counts = mg_xch_send_counts = NULL;
    MG_ExchangeValues.clear(); MG_ExchangeValues.shrink_to_fit();
}


/* ==================================================================================== */
/* Assembly over a symmetric CSR whose rows cover every local gas cell                   */
/* ==================================================================================== */

static void mg_build_matrix(const int *rows, int n_rows, const int64_t *offsets, const int *neighbors)
{
    const int num_local = ghost_get_num_local();
    int *ghost_home_rank  = ghost_get_home_rank();
    int *ghost_home_index = ghost_get_home_index();

    /* Each pair appears in both rows' CSR; a row accumulates only itself, so rows are independent. */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int aa = 0; aa < n_rows; aa++) {
        int i = rows[aa];
        if(P[i].Type != 0 || P[i].Mass <= 0 || CellP[i].Density <= 0 || P[i].KernelRadius <= 0) continue;
        double h_i = P[i].KernelRadius, h2_i = h_i * h_i, hinv_i, hinv3_i, hinv4_i;
        kernel_hinv(h_i, &hinv_i, &hinv3_i, &hinv4_i);
        double V_i = P[i].Mass / CellP[i].Density;
        double Particle_Size_i = pow(V_i, 1./NUMDIMS);
        double cnumcrit2 = ((double)CONDITION_NUMBER_DANGER)*((double)CONDITION_NUMBER_DANGER) - CellP[i].ConditionNumber*CellP[i].ConditionNumber;
        Vec3<double> Bi = CellP[i].Bfield();

        for(int64_t nn = offsets[aa]; nn < offsets[aa+1]; nn++) {
            int j = neighbors[nn];
            if(P[j].Type != 0 || P[j].Mass <= 0 || CellP[j].Density <= 0 || P[j].KernelRadius <= 0) continue;
            struct { Vec3<double> dp; double r, wk_i, wk_j, dwk_i, dwk_j, h_i; } kernel;
            kernel.dp = P[i].Pos - P[j].Pos;
            nearest_xyz(kernel.dp);
            double r2 = kernel.dp.norm_sq(), h_j = P[j].KernelRadius;
            if(r2 <= 0 || (r2 >= h2_i && r2 >= h_j * h_j)) continue;
            kernel.r = sqrt(r2); kernel.h_i = h_i;
            double u;
            if(kernel.r < h_i) {u = kernel.r * hinv_i; kernel_main(u, hinv3_i, hinv4_i, &kernel.wk_i, &kernel.dwk_i, 0);} else {kernel.wk_i = kernel.dwk_i = 0;}
            if(kernel.r < h_j) {
                double hinv_j, hinv3_j, hinv4_j; kernel_hinv(h_j, &hinv_j, &hinv3_j, &hinv4_j);
                u = kernel.r * hinv_j; kernel_main(u, hinv3_j, hinv4_j, &kernel.wk_j, &kernel.dwk_j, 0);
            } else {kernel.wk_j = kernel.dwk_j = 0;}

            double V_j = P[j].Mass / CellP[j].Density;
            Vec3<double> Face_Area_Vec; double Face_Area_Norm, Vi_inv_corr_unused, Vj_inv_corr_unused;
            compute_finitevol_faces(CellP[i], CellP[j], kernel, 1.0 / (MIN_REAL_NUMBER + kernel.r), r2, V_i, V_j,
                                    Particle_Size_i, P[j].Get_Particle_Size(), cnumcrit2,
                                    Face_Area_Vec, Face_Area_Norm, Vi_inv_corr_unused, Vj_inv_corr_unused);
            if(Face_Area_Norm <= 0) continue;

            double Qnorm2 = 0.125 * r2 * Face_Area_Vec.norm_sq() + 1.0e-60;
            MG_Rows[i].Rdiag += Qnorm2;

            /* face states as hydro reconstructs them at the face midpoint (x_ij - x_i = -dp/2, x_ij - x_j = +dp/2) */
            Vec3<double> Bj = CellP[j].Bfield();
            NGB_SHEARBOX_BOUNDARY_BCORR_(P[i].Pos, P[j].Pos, Bj, -1);
            Vec3<double> Bface_i = Bi, Bface_j = Bj;
            for(int k = 0; k < 3; k++) for(int k2 = 0; k2 < 3; k2++) {
                Bface_i[k] -= 0.5 * CellP[i].Gradients.B[k][k2] * kernel.dp[k2];
                Bface_j[k] += 0.5 * CellP[j].Gradients.B[k][k2] * kernel.dp[k2];
            }
            MG_Rows[i].rhs += 0.5 * dot(Bface_i + Bface_j, Face_Area_Vec);
            MG_Rows[i].flux_scale += 0.5 * (sqrt(Bface_i.norm_sq()) + sqrt(Bface_j.norm_sq())) * Face_Area_Norm;

            if(j < num_local) {mg_add_local_entry(i, j, Qnorm2);}
            else {int g = j - num_local; mg_add_remote_entry(i, ghost_home_rank[g], ghost_home_index[g], Qnorm2);}
        }
    }
}


/* ==================================================================================== */
/* Symmetry: every entry (i -> j, q) has its reverse (j -> i, q).                         */
/* ==================================================================================== */

static long long mg_count_asymmetric_entries(void)
{
    long long bad = 0;
    /* sorted rows make the reverse lookups binary searches */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int i = 0; i < N_gas; i++) {
        std::sort(MG_Rows[i].local_entries, MG_Rows[i].local_entries + MG_Rows[i].n_local,
                  [](const mg_local_entry &a, const mg_local_entry &b) {return a.j < b.j;});
        std::sort(MG_Rows[i].remote_entries, MG_Rows[i].remote_entries + MG_Rows[i].n_remote,
                  [](const mg_remote_entry &a, const mg_remote_entry &b) {return (a.remote_task != b.remote_task) ? (a.remote_task < b.remote_task) : (a.remote_index < b.remote_index);});
    }
    auto close = [](double a, double b) {return fabs(a - b) <= MG_SYMMETRY_TOL * DMAX(fabs(a), fabs(b));};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) reduction(+:bad)
#endif
    for(int i = 0; i < N_gas; i++) for(int n = 0; n < MG_Rows[i].n_local; n++) {
        int j = MG_Rows[i].local_entries[n].j;
        mg_local_entry key = {i, 0};
        mg_local_entry *e = std::lower_bound(MG_Rows[j].local_entries, MG_Rows[j].local_entries + MG_Rows[j].n_local, key,
                                             [](const mg_local_entry &a, const mg_local_entry &b) {return a.j < b.j;});
        if(e == MG_Rows[j].local_entries + MG_Rows[j].n_local || e->j != i || !close(e->Qnorm2, MG_Rows[i].local_entries[n].Qnorm2)) {bad++;}
    }
    /* cross-task entries: send (my row, its row, q) to the owner, which looks for the reverse */
    std::vector<int> scount(NTask, 0), rcount(NTask, 0), soff(NTask, 0), roff(NTask, 0), fill(NTask, 0);
    for(int i = 0; i < N_gas; i++) for(int n = 0; n < MG_Rows[i].n_remote; n++) {scount[MG_Rows[i].remote_entries[n].remote_task] += 3;}
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int ns = 0, nr = 0; for(int t = 0; t < NTask; t++) {soff[t] = ns; roff[t] = nr; ns += scount[t]; nr += rcount[t];}
    std::vector<double> sbuf((size_t)DMAX(ns, 1)), rbuf((size_t)DMAX(nr, 1));
    for(int i = 0; i < N_gas; i++) for(int n = 0; n < MG_Rows[i].n_remote; n++) {
        const mg_remote_entry &e = MG_Rows[i].remote_entries[n];
        size_t k = (size_t)(soff[e.remote_task] + fill[e.remote_task]); fill[e.remote_task] += 3;
        sbuf[k] = i; sbuf[k+1] = e.remote_index; sbuf[k+2] = e.Qnorm2;
    }
    MPI_Alltoallv(sbuf.data(), scount.data(), soff.data(), MPI_DOUBLE, rbuf.data(), rcount.data(), roff.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for(int t = 0; t < NTask; t++) for(int k = roff[t]; k < roff[t] + rcount[t]; k += 3) {
        int sender_row = (int) rbuf[k], j = (int) rbuf[k+1];
        if(j < 0 || j >= N_gas) {bad++; continue;}
        mg_remote_entry key = {t, sender_row, 0, 0};
        mg_remote_entry *e = std::lower_bound(MG_Rows[j].remote_entries, MG_Rows[j].remote_entries + MG_Rows[j].n_remote, key,
            [](const mg_remote_entry &a, const mg_remote_entry &b) {return (a.remote_task != b.remote_task) ? (a.remote_task < b.remote_task) : (a.remote_index < b.remote_index);});
        if(e == MG_Rows[j].remote_entries + MG_Rows[j].n_remote || e->remote_task != t || e->remote_index != sender_row || !close(e->Qnorm2, rbuf[k+2])) {bad++;}
    }
    long long bad_global = 0;
    MPI_Allreduce(&bad, &bad_global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return bad_global;
}


/* ==================================================================================== */
/* Connected pieces of the graph                                                          */
/* ==================================================================================== */

static int mg_find(std::vector<int> &parent, int a) {while(parent[a] != a) {parent[a] = parent[parent[a]]; a = parent[a];} return a;}

struct mg_piece_graph_report {long long pieces, cross_task_links; int rounds; int replicated;};

/* Label each row with its piece: the smallest global row id in the piece. Pieces are
 * found on each task first; the links between task-local pieces across tasks are then
 * resolved either by gathering the (deduplicated) link list everywhere, when it is small
 * enough, or by propagating the smallest label along the links until nothing changes. */
static struct mg_piece_graph_report mg_label_pieces(void)
{
    struct mg_piece_graph_report rep = {0, 0, 0, 1};
    std::vector<int> parent((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {parent[i] = i;}
    for(int i = 0; i < N_gas; i++) for(int n = 0; n < MG_Rows[i].n_local; n++) {
        int a = mg_find(parent, i), b = mg_find(parent, MG_Rows[i].local_entries[n].j);
        if(a != b) {if(a < b) {parent[b] = a;} else {parent[a] = b;}}   /* the root stays the smallest row */
    }
    std::vector<double> local_label((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {local_label[i] = (double) mg_global_row(ThisTask, mg_find(parent, i));} /* exact in a double below 2^53 */
    mg_exchange_values(local_label.data());

    struct link {long long mine, theirs; int task;};
    std::vector<link> links;
    for(int i = 0; i < N_gas; i++) for(int n = 0; n < MG_Rows[i].n_remote; n++) {
        const mg_remote_entry &e = MG_Rows[i].remote_entries[n];
        links.push_back({(long long) local_label[i], (long long) MG_ExchangeValues[e.exchange_pos], e.remote_task});
    }
    std::sort(links.begin(), links.end(), [](const link &a, const link &b) {return (a.mine != b.mine) ? (a.mine < b.mine) : ((a.theirs != b.theirs) ? (a.theirs < b.theirs) : (a.task < b.task));});
    links.erase(std::unique(links.begin(), links.end(), [](const link &a, const link &b) {return a.mine == b.mine && a.theirs == b.theirs;}), links.end());
    long long n_links = (long long) links.size();
    MPI_Allreduce(&n_links, &rep.cross_task_links, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    std::unordered_map<long long, long long> resolved;   /* task-local piece label -> global piece label, for pieces with cross-task links */
    for(const link &l : links) {resolved[l.mine] = l.mine;}
    if(rep.cross_task_links > 0) {
        const long long replicated_bytes = 2LL * rep.cross_task_links * (long long)sizeof(long long);
        rep.replicated = (replicated_bytes <= MG_REPLICATED_PIECE_GRAPH_MAX_BYTES) && (2LL * rep.cross_task_links < (long long)INT_MAX);
        if(rep.replicated) {
            int my_count = 2 * (int) links.size();
            std::vector<int> counts(NTask), displs(NTask);
            MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            int total = 0; for(int t = 0; t < NTask; t++) {displs[t] = total; total += counts[t];}
            std::vector<long long> mine_flat((size_t)DMAX(my_count, 1)), all_flat((size_t)DMAX(total, 1));
            for(size_t k = 0; k < links.size(); k++) {mine_flat[2*k] = links[k].mine; mine_flat[2*k+1] = links[k].theirs;}
            MPI_Allgatherv(mine_flat.data(), my_count, MPI_LONG_LONG, all_flat.data(), counts.data(), displs.data(), MPI_LONG_LONG, MPI_COMM_WORLD);
            std::vector<long long> labels(all_flat.begin(), all_flat.begin() + total);
            std::sort(labels.begin(), labels.end()); labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
            std::vector<int> lparent(labels.size());
            for(size_t k = 0; k < labels.size(); k++) {lparent[k] = (int)k;}
            auto index_of = [&labels](long long v) {return (int)(std::lower_bound(labels.begin(), labels.end(), v) - labels.begin());};
            for(int k = 0; k < total; k += 2) {
                int a = mg_find(lparent, index_of(all_flat[k])), b = mg_find(lparent, index_of(all_flat[k+1]));
                if(a != b) {if(a < b) {lparent[b] = a;} else {lparent[a] = b;}}   /* sorted labels: the root is the smallest */
            }
            for(auto &kv : resolved) {kv.second = labels[mg_find(lparent, index_of(kv.first))];}
        } else {
            /* propagate the smallest label along the links (both directions are present in a symmetric graph) */
            std::vector<int> scount(NTask), rcount(NTask), soff(NTask), roff(NTask), fill(NTask);
            for(int changed_global = 1; changed_global; ) {
                rep.rounds++;
                std::fill(scount.begin(), scount.end(), 0); std::fill(fill.begin(), fill.end(), 0);
                for(const link &l : links) {scount[l.task] += 2;}
                MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
                int ns = 0, nr = 0; for(int t = 0; t < NTask; t++) {soff[t] = ns; roff[t] = nr; ns += scount[t]; nr += rcount[t];}
                std::vector<long long> sbuf((size_t)DMAX(ns, 1)), rbuf((size_t)DMAX(nr, 1));
                for(const link &l : links) {size_t k = (size_t)(soff[l.task] + fill[l.task]); fill[l.task] += 2; sbuf[k] = l.theirs; sbuf[k+1] = resolved[l.mine];}
                MPI_Alltoallv(sbuf.data(), scount.data(), soff.data(), MPI_LONG_LONG, rbuf.data(), rcount.data(), roff.data(), MPI_LONG_LONG, MPI_COMM_WORLD);
                int changed = 0;
                for(int k = 0; k < nr; k += 2) {
                    auto it = resolved.find(rbuf[k]);
                    if(it == resolved.end()) {resolved[rbuf[k]] = rbuf[k+1]; changed = 1;}
                    else if(rbuf[k+1] < it->second) {it->second = rbuf[k+1]; changed = 1;}
                }
                MPI_Allreduce(&changed, &changed_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            }
        }
    }
    long long pieces = 0;
    for(int i = 0; i < N_gas; i++) {
        long long l = (long long) local_label[i];
        auto it = resolved.find(l);
        MG_Rows[i].piece = (it == resolved.end()) ? l : it->second;
        if(MG_Rows[i].piece == mg_global_row(ThisTask, i)) {pieces++;}
    }
    MPI_Allreduce(&pieces, &rep.pieces, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return rep;
}

enum mg_piece_op {MG_PIECE_SUM, MG_PIECE_MAX, MG_PIECE_MIN};
static inline double mg_piece_identity(int op) {return (op == MG_PIECE_MAX) ? -HUGE_VAL : ((op == MG_PIECE_MIN) ? HUGE_VAL : 0.0);}
static inline void mg_piece_combine(int op, double *acc, double v) {if(op == MG_PIECE_MAX) {*acc = DMAX(*acc, v);} else if(op == MG_PIECE_MIN) {*acc = DMIN(*acc, v);} else {*acc += v;}}

/* Reduce nv per-row values over each piece, across tasks (sum, max or min): row_values and
 * row_totals are N_gas x nv. Each piece is reduced at the task owning its label, which
 * returns the result to every task holding part of the piece. Local partial sums are
 * compensated (the piece sum of b is compared against roundoff). */
static void mg_piece_reduce(int nv, const double *row_values, double *row_totals, int op)
{
    std::unordered_map<long long, int> slot;
    std::vector<long long> slot_label;
    std::vector<double> sum, comp;
    for(int i = 0; i < N_gas; i++) {
        auto it = slot.find(MG_Rows[i].piece);
        int s;
        if(it == slot.end()) {s = (int) slot_label.size(); slot[MG_Rows[i].piece] = s; slot_label.push_back(MG_Rows[i].piece); sum.resize(sum.size() + nv, mg_piece_identity(op)); comp.resize(comp.size() + nv, 0.0);}
        else {s = it->second;}
        for(int v = 0; v < nv; v++) {
            double x = row_values[(size_t)i*nv + v];
            if(op == MG_PIECE_SUM) {   /* Kahan */
                double y = x - comp[(size_t)s*nv + v], t = sum[(size_t)s*nv + v] + y;
                comp[(size_t)s*nv + v] = (t - sum[(size_t)s*nv + v]) - y; sum[(size_t)s*nv + v] = t;
            } else {mg_piece_combine(op, &sum[(size_t)s*nv + v], x);}
        }
    }
    /* partial results of pieces owned elsewhere go to the owner */
    const int rec = nv + 1;
    std::vector<int> scount(NTask, 0), rcount(NTask), soff(NTask), roff(NTask), fill(NTask, 0);
    for(size_t s = 0; s < slot_label.size(); s++) {int o = mg_row_owner(slot_label[s]); if(o != ThisTask) {scount[o] += rec;}}
    MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int ns = 0, nr = 0; for(int t = 0; t < NTask; t++) {soff[t] = ns; roff[t] = nr; ns += scount[t]; nr += rcount[t];}
    std::vector<double> sbuf((size_t)DMAX(ns, 1)), rbuf((size_t)DMAX(nr, 1));
    for(size_t s = 0; s < slot_label.size(); s++) {
        int o = mg_row_owner(slot_label[s]); if(o == ThisTask) continue;
        size_t k = (size_t)(soff[o] + fill[o]); fill[o] += rec;
        sbuf[k] = (double) slot_label[s]; for(int v = 0; v < nv; v++) {sbuf[k+1+v] = sum[s*nv + v];}
    }
    MPI_Alltoallv(sbuf.data(), scount.data(), soff.data(), MPI_DOUBLE, rbuf.data(), rcount.data(), roff.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for(int k = 0; k < nr; k += rec) {
        long long l = (long long) rbuf[k];
        auto it = slot.find(l);
        int s;
        if(it == slot.end()) {s = (int) slot_label.size(); slot[l] = s; slot_label.push_back(l); sum.resize(sum.size() + nv, mg_piece_identity(op));}
        else {s = it->second;}
        for(int v = 0; v < nv; v++) {mg_piece_combine(op, &sum[(size_t)s*nv + v], rbuf[k+1+v]);}
    }
    /* the owner returns the result to every task that contributed (the reverse of the send pattern) */
    for(int k = 0; k < nr; k += rec) {int s = slot[(long long) rbuf[k]]; for(int v = 0; v < nv; v++) {rbuf[k+1+v] = sum[(size_t)s*nv + v];}}
    MPI_Alltoallv(rbuf.data(), rcount.data(), roff.data(), MPI_DOUBLE, sbuf.data(), scount.data(), soff.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for(int k = 0; k < ns; k += rec) {int s = slot[(long long) sbuf[k]]; for(int v = 0; v < nv; v++) {sum[(size_t)s*nv + v] = sbuf[k+1+v];}}
    for(int i = 0; i < N_gas; i++) {int s = slot[MG_Rows[i].piece]; for(int v = 0; v < nv; v++) {row_totals[(size_t)i*nv + v] = sum[(size_t)s*nv + v];}}
}

/* Each piece's anchor: its row of largest Rdiag, the smallest global row id among equal ones
 * (row ids are exact in a double below 2^53). A row without faces never anchors a piece that has them. */
static void mg_select_anchors(void)
{
    std::vector<double> v((size_t)DMAX(N_gas, 1)), best((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {v[i] = (MG_Rows[i].Rdiag > 1.0e-60 && isfinite(MG_Rows[i].Rdiag)) ? MG_Rows[i].Rdiag : 0.0;}
    mg_piece_reduce(1, v.data(), best.data(), MG_PIECE_MAX);
    for(int i = 0; i < N_gas; i++) {v[i] = (v[i] == best[i]) ? (double) mg_global_row(ThisTask, i) : HUGE_VAL;}
    mg_piece_reduce(1, v.data(), best.data(), MG_PIECE_MIN);
    for(int i = 0; i < N_gas; i++) {MG_Rows[i].anchor = (long long) best[i];}
}


/* ==================================================================================== */
/* Operator application on the true (unpinned) matrix: y = R x, remote x exchanged       */
/* ==================================================================================== */

static void mg_apply_matrix(const double *x, double *y)
{
    mg_exchange_values(x);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i = 0; i < N_gas; i++) {
        /* edge form sum_j q_ij (x_i - x_j): exact for Rdiag = sum_j q_ij, and free of the cancellation of
           Rdiag x_i - sum_j q_ij x_j when the coefficients span tens of orders of magnitude */
        double v = 0;
        for(int n = 0; n < MG_Rows[i].n_local; n++) {v += MG_Rows[i].local_entries[n].Qnorm2 * (x[i] - x[MG_Rows[i].local_entries[n].j]);}
        for(int n = 0; n < MG_Rows[i].n_remote; n++) {v += MG_Rows[i].remote_entries[n].Qnorm2 * (x[i] - MG_ExchangeValues[MG_Rows[i].remote_entries[n].exchange_pos]);}
        y[i] = v;
    }
}




/* The metrics a solve is judged by, from the true edge-form residual r = b - R x over every row of
 * a solved piece (anchor rows included: a pinned solve that has not converged leaves on the anchor
 * row minus the sum of the other rows' residuals, which a global norm can hide):
 *   G = |r|/|b|,   P = sqrt(sum r^2 / sum s^2),   M = max_i (|r_i| - e_i)+ / d_i,
 * with d_i = s_i + sum_j q_ij |x_j - x_i| the row's face-flux scale plus the flux the correction
 * itself puts through its faces (a componentwise backward-error scale: rows whose field is zero or
 * vanishing are judged against the field the correction gives them), and e_i = 16 eps sum_j q_ij
 * (|x_i| + |x_j|) the rounding bound of evaluating r_i. The uncorrected state c = 0 has r = b and
 * d = s: G0 = 1, P0 = sqrt(sum b^2 / sum s^2), M0 = max |b_i| / s_i. The correction flux relative
 * to the field's, C = sqrt(sum (d - s)^2 / sum s^2), is reported so a large correction is visible.
 * A row with d_i = 0 has b_i = 0 and no correction, so a residual there beyond e_i is a structural
 * failure (counted in n_unexplained). MG_ExchangeValues must hold x (mg_apply_matrix just ran). */
struct mg_solve_metrics {double G, P, M, P0, M0, C; int any_rows; long long n_unexplained;};
static struct mg_solve_metrics mg_judge_solution(const double *x, const double *resid)
{
    double sum_loc[4] = {0, 0, 0, 0}, sum[4], mx_loc[2] = {0, 0}, mx[2];   /* sum r^2, b^2, s^2, (d-s)^2; max M, M0 */
    long long unexplained_loc = 0, unexplained = 0;
    for(int i = 0; i < N_gas; i++) {
        if(!MG_RowInSolvedPiece[i]) continue;
        double b = MG_Rows[i].rhs, s = MG_Rows[i].flux_scale;
        double e = 0, corr = 0;
        for(int n = 0; n < MG_Rows[i].n_local + MG_Rows[i].n_remote; n++) {
            double q, xj;
            if(n < MG_Rows[i].n_local) {q = MG_Rows[i].local_entries[n].Qnorm2; xj = x[MG_Rows[i].local_entries[n].j];}
            else {const struct mg_remote_entry *er = &MG_Rows[i].remote_entries[n - MG_Rows[i].n_local]; q = er->Qnorm2; xj = MG_ExchangeValues[er->exchange_pos];}
            e += q * (fabs(x[i]) + fabs(xj)); corr += q * fabs(xj - x[i]);
        }
        sum_loc[0] += resid[i] * resid[i]; sum_loc[1] += b * b; sum_loc[2] += s * s; sum_loc[3] += corr * corr;
        double excess = DMAX(fabs(resid[i]) - 16.0 * DBL_EPSILON * e, 0.0), d = s + corr;
        if(d > 0) {mx_loc[0] = DMAX(mx_loc[0], excess / d);} else if(excess > 0) {unexplained_loc++;}
        if(s > 0) {mx_loc[1] = DMAX(mx_loc[1], fabs(b) / s);}
    }
    MPI_Allreduce(sum_loc, sum, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(mx_loc, mx, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&unexplained_loc, &unexplained, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    struct mg_solve_metrics m;
    m.any_rows = (sum[1] > 0);
    m.G = m.any_rows ? sqrt(sum[0] / sum[1]) : 0;
    m.P = (sum[2] > 0) ? sqrt(sum[0] / sum[2]) : 0; m.P0 = (sum[2] > 0) ? sqrt(sum[1] / sum[2]) : 0;
    m.M = mx[0]; m.M0 = mx[1]; m.n_unexplained = unexplained;
    m.C = (sum[2] > 0) ? sqrt(sum[3] / sum[2]) : 0;
    return m;
}


/* ==================================================================================== */
/* Built-in solver (MHD_MODIFIED_GRADIENT_CG_ONLY): SSOR-preconditioned CG on the pinned */
/* system. SSOR acts on the local block only (block-Jacobi between tasks).               */
/* ==================================================================================== */

#ifdef MHD_MODIFIED_GRADIENT_CG_ONLY
static void mg_ssor_precondition(const double *r, double *z)
{
    for(int i = 0; i < N_gas; i++) {z[i] = 0;}
    for(int sweep = 0; sweep < 2; sweep++) {   /* forward, then backward */
        for(int ii = 0; ii < N_gas; ii++) {
            int i = (sweep == 0) ? ii : N_gas - 1 - ii;
            if(!MG_RowSolved[i]) continue;
            double sum = r[i];
            for(int n = 0; n < MG_Rows[i].n_local; n++) {
                int j = MG_Rows[i].local_entries[n].j;
                if(mg_column_kept(i, mg_global_row(ThisTask, j))) {sum += MG_Rows[i].local_entries[n].Qnorm2 * z[j];}
            }
            z[i] = sum / MG_Rows[i].Rdiag;
        }
    }
}

/* y = R_pinned x on solved rows (x is 0 on every unsolved row, so their columns contribute nothing) */
static void mg_apply_pinned(const double *x, double *y)
{
    mg_exchange_values(x);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i = 0; i < N_gas; i++) {
        if(!MG_RowSolved[i]) {y[i] = 0; continue;}
        double v = MG_Rows[i].Rdiag * x[i];
        for(int n = 0; n < MG_Rows[i].n_local; n++) {v -= MG_Rows[i].local_entries[n].Qnorm2 * x[MG_Rows[i].local_entries[n].j];}
        for(int n = 0; n < MG_Rows[i].n_remote; n++) {v -= MG_Rows[i].remote_entries[n].Qnorm2 * MG_ExchangeValues[MG_Rows[i].remote_entries[n].exchange_pos];}
        y[i] = v;
    }
}

static double mg_global_dot(const double *a, const double *b)
{
    double loc = 0, glob;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:loc)
#endif
    for(int i = 0; i < N_gas; i++) {loc += a[i] * b[i];}
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return glob;
}

/* returns the iteration count; x holds the last iterate (the caller judges it by its true residual) */
static int mg_solve_backend(double *x)
{
    std::vector<double> r(MG_RhsSolved), z((size_t)DMAX(N_gas, 1)), p((size_t)DMAX(N_gas, 1)), Ap((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {x[i] = 0;}
    double bnorm2 = mg_global_dot(r.data(), r.data());
    if(bnorm2 <= 0) {return 0;}
    mg_ssor_precondition(r.data(), z.data());
    p = z;
    double rz = mg_global_dot(r.data(), z.data()), rnorm2 = bnorm2;
    int iter;
    for(iter = 0; iter < MG_CG_MAX_ITER; iter++) {
        mg_apply_pinned(p.data(), Ap.data());
        double pAp = mg_global_dot(p.data(), Ap.data());
        if(!(pAp > 0) || !(rz > 0)) {   /* both are positive for an SPD system and preconditioner; anything else is a breakdown */
            if(ThisTask == 0) {PRINT_WARNING("MG built-in CG broke down at iteration %d (p.Ap = %g, r.z = %g); the iterate is judged by its true residual", iter, pAp, rz);}
            break;
        }
        double alpha = rz / pAp;
        for(int i = 0; i < N_gas; i++) {x[i] += alpha * p[i]; r[i] -= alpha * Ap[i];}
        rnorm2 = mg_global_dot(r.data(), r.data());
        if(rnorm2 <= MG_CG_TOL * MG_CG_TOL * bnorm2) {iter++; break;}
        mg_ssor_precondition(r.data(), z.data());
        double rz_new = mg_global_dot(r.data(), z.data()), beta = rz_new / rz;
        rz = rz_new;
        for(int i = 0; i < N_gas; i++) {p[i] = z[i] + beta * p[i];}
    }
    return iter;
}
#endif /* MHD_MODIFIED_GRADIENT_CG_ONLY */


/* ==================================================================================== */
/* HYPRE: BoomerAMG-preconditioned PCG on the pinned system (default)                    */
/* ==================================================================================== */

#if !defined(MHD_MODIFIED_GRADIENT_CG_ONLY) && !defined(MHD_MODIFIED_GRADIENT_USE_PARDISO)
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#define MG_HYPRE_ON_DEVICE   /* a GPU-built HYPRE takes every array it is handed in device memory */
#endif
static int mg_solve_backend(double *x)
{
    static int hypre_initialized = 0;   /* HYPRE's handle (on a GPU build: device, streams, memory pools) is set up once per run */
    if(!hypre_initialized) {HYPRE_Init(); hypre_initialized = 1;}
    HYPRE_BigInt ilower = (HYPRE_BigInt) MG_TaskRowOffsets[ThisTask], iupper = (HYPRE_BigInt)(MG_TaskRowOffsets[ThisTask] + N_gas - 1);
    /* all rows in one insertion: an unsolved row is the equation c_i = 0; a solved row drops
       the column of its piece's anchor unknown */
    std::vector<HYPRE_Int> ncols((size_t)DMAX(N_gas, 1));
    std::vector<HYPRE_BigInt> row_ids((size_t)DMAX(N_gas, 1));
    size_t nnz = 0;
    for(int i = 0; i < N_gas; i++) {nnz += MG_RowSolved[i] ? (size_t)(1 + MG_Rows[i].n_local + MG_Rows[i].n_remote) : 1;}
    std::vector<HYPRE_BigInt> cols(DMAX(nnz, (size_t)1));
    std::vector<HYPRE_Real> vals(DMAX(nnz, (size_t)1));
    size_t pos = 0;
    for(int i = 0; i < N_gas; i++) {
        row_ids[i] = (HYPRE_BigInt) mg_global_row(ThisTask, i);
        size_t start = pos;
        cols[pos] = row_ids[i]; vals[pos] = MG_RowSolved[i] ? MG_Rows[i].Rdiag : 1.0; pos++;
        if(MG_RowSolved[i]) {
            for(int n = 0; n < MG_Rows[i].n_local; n++) {
                long long g = mg_global_row(ThisTask, MG_Rows[i].local_entries[n].j);
                if(mg_column_kept(i, g)) {cols[pos] = (HYPRE_BigInt) g; vals[pos] = -MG_Rows[i].local_entries[n].Qnorm2; pos++;}
            }
            for(int n = 0; n < MG_Rows[i].n_remote; n++) {
                long long g = mg_global_row(MG_Rows[i].remote_entries[n].remote_task, MG_Rows[i].remote_entries[n].remote_index);
                if(mg_column_kept(i, g)) {cols[pos] = (HYPRE_BigInt) g; vals[pos] = -MG_Rows[i].remote_entries[n].Qnorm2; pos++;}
            }
        }
        ncols[i] = (HYPRE_Int)(pos - start);
    }
    std::vector<HYPRE_Real> x_vals((size_t)DMAX(N_gas, 1), 0.0);
    HYPRE_Int *ncols_in = ncols.data(); HYPRE_BigInt *rows_in = row_ids.data(), *cols_in = cols.data();
    HYPRE_Real *vals_in = vals.data(), *rhs_in = MG_RhsSolved.data(), *x_in = x_vals.data();
#ifdef MG_HYPRE_ON_DEVICE
    /* the system is built on the host and copied to the device once; HYPRE assembles, sets up and solves
       there, and the solution comes back once. HYPRE's global memory location and execution policy are
       switched for this solve only and restored to whatever they were. */
    const size_t n_rows_alloc = (size_t)DMAX(N_gas, 1), n_entries_alloc = DMAX(nnz, (size_t)1);
    ncols_in = (HYPRE_Int *)    gizmo_gpu_alloc_device(n_rows_alloc * sizeof(HYPRE_Int), "mg_hypre_ncols");
    rows_in  = (HYPRE_BigInt *) gizmo_gpu_alloc_device(n_rows_alloc * sizeof(HYPRE_BigInt), "mg_hypre_rows");
    cols_in  = (HYPRE_BigInt *) gizmo_gpu_alloc_device(n_entries_alloc * sizeof(HYPRE_BigInt), "mg_hypre_cols");
    vals_in  = (HYPRE_Real *)   gizmo_gpu_alloc_device(n_entries_alloc * sizeof(HYPRE_Real), "mg_hypre_values");
    rhs_in   = (HYPRE_Real *)   gizmo_gpu_alloc_device(n_rows_alloc * sizeof(HYPRE_Real), "mg_hypre_rhs");
    x_in     = (HYPRE_Real *)   gizmo_gpu_alloc_device(n_rows_alloc * sizeof(HYPRE_Real), "mg_hypre_solution");
    int alloc_failed = (!ncols_in || !rows_in || !cols_in || !vals_in || !rhs_in || !x_in);
    MPI_Allreduce(MPI_IN_PLACE, &alloc_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(alloc_failed) {   /* every task skips the solve together, so no task waits in a HYPRE collective */
        gizmo_gpu_free_device(ncols_in); gizmo_gpu_free_device(rows_in); gizmo_gpu_free_device(cols_in);
        gizmo_gpu_free_device(vals_in); gizmo_gpu_free_device(rhs_in); gizmo_gpu_free_device(x_in);
        if(ThisTask == 0) {PRINT_WARNING("MG: device memory for the HYPRE system could not be allocated on some task (task 0 needs %g MB)",
                                         (double)(n_rows_alloc * (sizeof(HYPRE_Int) + sizeof(HYPRE_BigInt) + 2 * sizeof(HYPRE_Real))
                                                  + n_entries_alloc * (sizeof(HYPRE_BigInt) + sizeof(HYPRE_Real))) / (1024. * 1024.));}
        return -1;
    }
    gizmo_gpu_copy_to_device(ncols_in, ncols.data(), (size_t)N_gas * sizeof(HYPRE_Int));
    gizmo_gpu_copy_to_device(rows_in, row_ids.data(), (size_t)N_gas * sizeof(HYPRE_BigInt));
    gizmo_gpu_copy_to_device(cols_in, cols.data(), nnz * sizeof(HYPRE_BigInt));
    gizmo_gpu_copy_to_device(vals_in, vals.data(), nnz * sizeof(HYPRE_Real));
    gizmo_gpu_copy_to_device(rhs_in, MG_RhsSolved.data(), (size_t)N_gas * sizeof(HYPRE_Real));
    gizmo_gpu_copy_to_device(x_in, x_vals.data(), (size_t)N_gas * sizeof(HYPRE_Real));
    HYPRE_MemoryLocation prior_location; HYPRE_ExecutionPolicy prior_policy;
    HYPRE_GetMemoryLocation(&prior_location); HYPRE_GetExecutionPolicy(&prior_policy);
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
#endif
    HYPRE_IJMatrix A;
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);
    if(N_gas > 0) {HYPRE_IJMatrixSetValues(A, (HYPRE_Int) N_gas, ncols_in, rows_in, cols_in, vals_in);}
    HYPRE_IJMatrixAssemble(A);
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);

    HYPRE_IJVector b_vec, x_vec;
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b_vec); HYPRE_IJVectorSetObjectType(b_vec, HYPRE_PARCSR); HYPRE_IJVectorInitialize(b_vec);
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x_vec); HYPRE_IJVectorSetObjectType(x_vec, HYPRE_PARCSR); HYPRE_IJVectorInitialize(x_vec);
    if(N_gas > 0) {
        HYPRE_IJVectorSetValues(b_vec, (HYPRE_Int) N_gas, rows_in, rhs_in);
        HYPRE_IJVectorSetValues(x_vec, (HYPRE_Int) N_gas, rows_in, x_in);
    }
    HYPRE_IJVectorAssemble(b_vec); HYPRE_IJVectorAssemble(x_vec);
    HYPRE_ParVector par_b, par_x;
    HYPRE_IJVectorGetObject(b_vec, (void **)&par_b); HYPRE_IJVectorGetObject(x_vec, (void **)&par_x);

    HYPRE_Solver solver, precond;
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_PCGSetMaxIter(solver, (HYPRE_Int) MG_CG_MAX_ITER);
    HYPRE_PCGSetTol(solver, (HYPRE_Real) MG_CG_TOL);
    HYPRE_PCGSetTwoNorm(solver, 1);
    HYPRE_PCGSetPrintLevel(solver, 0);
    HYPRE_BoomerAMGCreate(&precond);
    HYPRE_BoomerAMGSetPrintLevel(precond, 0);
#ifdef MG_HYPRE_ON_DEVICE
    HYPRE_BoomerAMGSetCoarsenType(precond, 8);  /* PMIS coarsening (Falgout is not available in a device setup) */
    HYPRE_BoomerAMGSetRelaxType(precond, 18);   /* l1-scaled Jacobi (GPU-parallel) */
#else
    HYPRE_BoomerAMGSetCoarsenType(precond, 6);  /* Falgout coarsening */
    HYPRE_BoomerAMGSetRelaxType(precond, 6);    /* symmetric Gauss-Seidel (CPU) */
#endif
    HYPRE_BoomerAMGSetNumSweeps(precond, 1);
    HYPRE_BoomerAMGSetMaxLevels(precond, 25);
    HYPRE_BoomerAMGSetMaxIter(precond, 1);      /* 1 V-cycle per PCG iteration */
    HYPRE_BoomerAMGSetTol(precond, 0.0);
    HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
    HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
    HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
    HYPRE_Int num_iterations;
    HYPRE_PCGGetNumIterations(solver, &num_iterations);
    if(N_gas > 0) {HYPRE_IJVectorGetValues(x_vec, (HYPRE_Int) N_gas, rows_in, x_in);}
    HYPRE_ClearAllErrors();   /* the outcome is judged by the true residual, not HYPRE's flags */

    HYPRE_ParCSRPCGDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b_vec);
    HYPRE_IJVectorDestroy(x_vec);
#ifdef MG_HYPRE_ON_DEVICE
    HYPRE_SetMemoryLocation(prior_location);
    HYPRE_SetExecutionPolicy(prior_policy);
    gizmo_gpu_copy_to_host(x_vals.data(), x_in, (size_t)N_gas * sizeof(HYPRE_Real));
    gizmo_gpu_free_device(ncols_in); gizmo_gpu_free_device(rows_in); gizmo_gpu_free_device(cols_in);
    gizmo_gpu_free_device(vals_in); gizmo_gpu_free_device(rhs_in); gizmo_gpu_free_device(x_in);
#endif
    for(int i = 0; i < N_gas; i++) {x[i] = MG_RowSolved[i] ? x_vals[i] : 0;}
    return (int) num_iterations;
}
#endif


/* ==================================================================================== */
/* MKL PARDISO direct solver on rank 0 (MHD_MODIFIED_GRADIENT_USE_PARDISO)               */
/* ==================================================================================== */

#ifdef MHD_MODIFIED_GRADIENT_USE_PARDISO
static int mg_solve_backend(double *x)
{
    /* the pinned system as upper-triangle CSR rows with sorted columns (the symmetric PARDISO input format) */
    std::vector<int> row_ptr(1, 0), col_idx;
    std::vector<double> vals;
    for(int i = 0; i < N_gas; i++) {
        long long gi = mg_global_row(ThisTask, i);
        std::vector<std::pair<long long, double>> row;
        row.push_back({gi, MG_RowSolved[i] ? MG_Rows[i].Rdiag : 1.0});
        if(MG_RowSolved[i]) {
            for(int n = 0; n < MG_Rows[i].n_local; n++) {
                long long g = mg_global_row(ThisTask, MG_Rows[i].local_entries[n].j);
                if(g > gi && mg_column_kept(i, g)) {row.push_back({g, -MG_Rows[i].local_entries[n].Qnorm2});}
            }
            for(int n = 0; n < MG_Rows[i].n_remote; n++) {
                long long g = mg_global_row(MG_Rows[i].remote_entries[n].remote_task, MG_Rows[i].remote_entries[n].remote_index);
                if(g > gi && mg_column_kept(i, g)) {row.push_back({g, -MG_Rows[i].remote_entries[n].Qnorm2});}
            }
        }
        std::sort(row.begin(), row.end());
        for(auto &e : row) {col_idx.push_back((int) e.first); vals.push_back(e.second);}
        row_ptr.push_back((int) col_idx.size());
    }
    int my_rows = N_gas, my_nnz = (int) col_idx.size();
    std::vector<int> all_rows(NTask), all_nnz(NTask), disp_rows(NTask), disp_nnz(NTask);
    MPI_Gather(&my_rows, 1, MPI_INT, all_rows.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&my_nnz, 1, MPI_INT, all_nnz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int n_global = 0, nnz_global = 0;
    if(ThisTask == 0) {for(int t = 0; t < NTask; t++) {disp_rows[t] = n_global; disp_nnz[t] = nnz_global; n_global += all_rows[t]; nnz_global += all_nnz[t];}}
    std::vector<int> g_row_counts((size_t)DMAX(n_global, 1)), g_col((size_t)DMAX(nnz_global, 1));
    std::vector<double> g_val((size_t)DMAX(nnz_global, 1)), g_rhs((size_t)DMAX(n_global, 1)), g_x((size_t)DMAX(n_global, 1), 0.0);
    std::vector<int> row_counts((size_t)DMAX(N_gas, 1));
    for(int i = 0; i < N_gas; i++) {row_counts[i] = row_ptr[i+1] - row_ptr[i];}
    MPI_Gatherv(row_counts.data(), my_rows, MPI_INT, g_row_counts.data(), all_rows.data(), disp_rows.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(col_idx.data(), my_nnz, MPI_INT, g_col.data(), all_nnz.data(), disp_nnz.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(vals.data(), my_nnz, MPI_DOUBLE, g_val.data(), all_nnz.data(), disp_nnz.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(MG_RhsSolved.data(), my_rows, MPI_DOUBLE, g_rhs.data(), all_rows.data(), disp_rows.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(ThisTask == 0 && n_global > 0) {
        std::vector<MKL_INT> ia((size_t)n_global + 1), ja((size_t)nnz_global);
        ia[0] = 1; for(int r = 0; r < n_global; r++) {ia[r+1] = ia[r] + g_row_counts[r];}
        for(int k = 0; k < nnz_global; k++) {ja[k] = g_col[k] + 1;}   /* 1-based */
        void *pt[64]; for(int k = 0; k < 64; k++) pt[k] = NULL;
        MKL_INT iparm[64]; for(int k = 0; k < 64; k++) iparm[k] = 0;
        MKL_INT maxfct = 1, mnum = 1, mtype = 2 /* real symmetric positive definite (pinned) */, phase, nrhs = 1, msglvl = 0, error = 0, perm_dummy, N = n_global;
        iparm[0] = 1; iparm[1] = 2; iparm[7] = 0; iparm[9] = 13;
        phase = 13; /* analysis, factorization, solve */
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, g_val.data(), ia.data(), ja.data(), &perm_dummy, &nrhs, iparm, &msglvl, g_rhs.data(), g_x.data(), &error);
        if(error != 0) {PRINT_WARNING("MG PARDISO error %d; the solve is judged by its residual below", (int)error);}
        phase = -1;
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, g_val.data(), ia.data(), ja.data(), &perm_dummy, &nrhs, iparm, &msglvl, g_rhs.data(), g_x.data(), &error);
    }
    MPI_Scatterv(g_x.data(), all_rows.data(), disp_rows.data(), MPI_DOUBLE, x, my_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(int i = 0; i < N_gas; i++) {if(!MG_RowSolved[i]) {x[i] = 0;}}
    return 1;
}
#endif /* MHD_MODIFIED_GRADIENT_USE_PARDISO */


/* ==================================================================================== */
/* Public entry point                                                                     */
/* ==================================================================================== */

/* An accepted solve replaces the coefficients and the piece tokens together; anything else leaves the last accepted pair. */
static void mg_commit_solution(const double *x)
{
    for(int i = 0; i < N_gas; i++) {
        bool in_graph = (P[i].Type == 0) && (P[i].Mass > 0) && (CellP[i].Density > 0) && (P[i].KernelRadius > 0);
        CellP[i].MG_cgcoeff = in_graph ? x[i] : 0;
        CellP[i].MG_piece_token = in_graph ? (MG_Rows[i].piece + 1) : 0;
    }
}

void mg_gradient_correction_calc(void)
{
    if(ThisTask == 0) PRINT_STATUS("Computing MG gradient correction (exact div(B)=0) ...");

    /* the right-hand side reads neighbour gradients, which ghosts carry only after this refresh */
    gizmo_hydro_corridor_refresh_ghost_values("pre_mg");
    if(NTask > 1 && !ghost_pool_is_live()) {
        PRINT_WARNING("MG solve reached without a ghost pool: the matrix would be missing every cross-task face");
        endrun(8804);
        return;
    }

    MG_Rows = (struct mg_row_data *) calloc((size_t)DMAX(N_gas, 1), sizeof(struct mg_row_data));
    MG_TaskRowOffsets.assign(NTask + 1, 0);
    {long long my_rows = N_gas, offset = 0; MPI_Exscan(&my_rows, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); if(ThisTask == 0) {offset = 0;}
     MPI_Allgather(&offset, 1, MPI_LONG_LONG, MG_TaskRowOffsets.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);
     long long total = 0; MPI_Allreduce(&my_rows, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD); MG_TaskRowOffsets[NTask] = total;}

    /* The corridor's CSR is the MG graph when its rows are every local gas cell (an all-active
       step); otherwise build one over every local gas cell on the corridor's all-gas pool. */
    int n_eligible = 0;
    for(int i = 0; i < N_gas; i++) {if(P[i].Type == 0 && P[i].Mass > 0) {n_eligible++;}}
    const int *rows = NULL; const int64_t *offsets = NULL; const int *neighbors = NULL; int n_rows = 0;
    int reuse = (gizmo_hydro_corridor_csr_arrays(&rows, &n_rows, &offsets, &neighbors) && n_rows == n_eligible) ? 1 : 0, reuse_all = 0;
    MPI_Allreduce(&reuse, &reuse_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    neighbor_list_t private_list = {NULL, NULL, 0, 0};
    int *private_rows = NULL;
    if(!reuse_all) {
        private_rows = (int *) mymalloc("mg_allgas_rows", (size_t)DMAX(n_eligible, 1) * sizeof(int));
        n_rows = 0;
        for(int i = 0; i < N_gas; i++) {if(P[i].Type == 0 && P[i].Mass > 0) {private_rows[n_rows++] = i;}}
        gpu_build_symmetric_neighbor_list(P, NumPart, private_rows, n_rows, &private_list, 1.0);
        rows = private_rows; offsets = private_list.offsets; neighbors = private_list.neighbors;
    }
    mg_build_matrix(rows, n_rows, offsets, neighbors);
    if(!reuse_all) {free_neighbor_list(&private_list); myfree(private_rows);}

    mg_setup_exchange();
    std::vector<double> x((size_t)DMAX(N_gas, 1), 0.0), resid((size_t)DMAX(N_gas, 1), 0.0);
    int stop_code = 0;
    {
        int alloc_failed = MG_EntryAllocFailed;
        MPI_Allreduce(MPI_IN_PLACE, &alloc_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MG_EntryAllocFailed = 0;
        if(alloc_failed) {
            if(ThisTask == 0) {PRINT_WARNING("MG matrix: a row's entry storage could not be enlarged (out of memory)");}
            stop_code = 8805;
        }
    }
    if(!stop_code) {
        long long asymmetric = mg_count_asymmetric_entries();
        if(asymmetric > 0) {
            if(ThisTask == 0) {PRINT_WARNING("MG matrix is not symmetric: %lld entries have no matching reverse entry -- the face graph is incomplete or one-sided", asymmetric);}
            stop_code = 8801;
        }
    }
    struct mg_piece_graph_report pieces = {0, 0, 0, 1};
    if(!stop_code) {   /* labelling and anchors assume every link has its reverse */
        pieces = mg_label_pieces();
        mg_select_anchors();
    }

    /* per piece: sum b against sum s (b telescopes to roundoff over a closed piece), and ||b|| against ||s|| (the noise floor) */
    long long n_floor_pieces = 0;
    double worst_imbalance = 0;
    MG_RowSolved.assign((size_t)DMAX(N_gas, 1), 0);
    MG_RhsSolved.assign((size_t)DMAX(N_gas, 1), 0.0);
    MG_RowInSolvedPiece.assign((size_t)DMAX(N_gas, 1), 0);
    if(!stop_code) {
        std::vector<double> v((size_t)5 * DMAX(N_gas, 1)), tot((size_t)5 * DMAX(N_gas, 1));
        for(int i = 0; i < N_gas; i++) {double b = MG_Rows[i].rhs, s = MG_Rows[i].flux_scale; v[5*i] = b; v[5*i+1] = s; v[5*i+2] = b*b; v[5*i+3] = s*s; v[5*i+4] = 1.0;}
        mg_piece_reduce(5, v.data(), tot.data(), MG_PIECE_SUM);
        long long floor_loc = 0, nonfinite_loc = 0, nonfinite_coeff = 0;
        double imbalance_loc = 0;
        for(int i = 0; i < N_gas; i++) {
            bool labels_piece = (MG_Rows[i].piece == mg_global_row(ThisTask, i));   /* one row per piece, for per-piece counts */
            bool anchor = (MG_Rows[i].anchor == mg_global_row(ThisTask, i));
            bool below_floor = (tot[5*i+2] <= MG_NOISE_FLOOR * MG_NOISE_FLOOR * tot[5*i+3]);
            if(!isfinite(MG_Rows[i].Rdiag) || !isfinite(MG_Rows[i].rhs) || !isfinite(MG_Rows[i].flux_scale)) {nonfinite_loc++;}
            if(labels_piece && tot[5*i+1] > 0) {imbalance_loc = DMAX(imbalance_loc, fabs(tot[5*i]) / tot[5*i+1]);}
            if(labels_piece && below_floor && MG_Rows[i].Rdiag > 1.0e-60) {floor_loc++;}
            MG_RowInSolvedPiece[i] = !below_floor;
            MG_RowSolved[i] = (MG_Rows[i].Rdiag > 1.0e-60) && !anchor && !below_floor;
            MG_RhsSolved[i] = MG_RowSolved[i] ? MG_Rows[i].rhs : 0.0;
        }
        MPI_Allreduce(&floor_loc, &n_floor_pieces, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&imbalance_loc, &worst_imbalance, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&nonfinite_loc, &nonfinite_coeff, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(nonfinite_coeff > 0) {
            if(ThisTask == 0) {PRINT_WARNING("MG matrix or right-hand side has %lld non-finite rows", nonfinite_coeff);}
            stop_code = 8803;
        } else if(worst_imbalance > MG_RHS_IMBALANCE_STRUCTURAL) {
            if(ThisTask == 0) {PRINT_WARNING("MG right-hand side does not sum to roundoff on a connected piece (|sum b|/sum s = %g): the face fluxes are not antisymmetric", worst_imbalance);}
            stop_code = 8802;
        } else if(worst_imbalance > MG_CG_TOL && ThisTask == 0) {
            PRINT_WARNING("MG right-hand side sums to |sum b|/sum s = %g on a connected piece, above the solve tolerance %g: that piece cannot converge below it", worst_imbalance, MG_CG_TOL);
        }
    }

    int iterations = 0;
    const char *outcome = "converged";
    double rel_resid = 0;
    struct mg_solve_metrics metrics = {0, 0, 0, 0, 0, 0, 0, 0};
    if(!stop_code) {
        iterations = mg_solve_backend(x.data());
        if(iterations < 0) {stop_code = 8806;}   /* the backend could not allocate its system (same on every task) */
    }
    if(!stop_code) {
        int nonfinite_loc = 0, nonfinite = 0;
        for(int i = 0; i < N_gas; i++) {if(!isfinite(x[i])) {nonfinite_loc = 1;}}
        MPI_Allreduce(&nonfinite_loc, &nonfinite, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if(nonfinite) {
            if(ThisTask == 0) {PRINT_WARNING("MG solve returned a non-finite solution");}
            stop_code = 8803;
        }
    }
    if(!stop_code) {
        /* The anchored solution (c = 0 at each piece's anchor row) is the gauge kept: c spans as many decades
           as the coefficients, so subtracting a piece mean would round away c on the large-coefficient rows
           (measured on FIF: every such row collapsed to the same value). c enters the physics only through
           differences across faces, so a per-piece constant changes nothing. */
        mg_apply_matrix(x.data(), resid.data());
        for(int i = 0; i < N_gas; i++) {resid[i] = MG_Rows[i].rhs - resid[i];}
        metrics = mg_judge_solution(x.data(), resid.data());
        rel_resid = metrics.G;
        if(metrics.n_unexplained > 0) {
            if(ThisTask == 0) {PRINT_WARNING("MG solve left a residual on %lld rows that carry neither field nor correction flux", metrics.n_unexplained);}
            endrun(8803);
            outcome = "stopped";
        } else {
            bool converged = (metrics.G <= MG_CG_TOL);
            /* applied only if it lowers the divergence globally, in the field-scaled norm, and in its worst row */
            bool improves = !metrics.any_rows || ((metrics.G < 1) && (metrics.P < metrics.P0) && (metrics.M <= metrics.M0));
            if(!improves) {
                /* keep the last accepted coefficients and pieces; this step is then a step without the solve, except that
                   the gradients pass has already run without its CG correction */
                outcome = converged ? "converged globally but raises the worst row; previous MG correction kept, no CG correction this step"
                                    : "not converged; previous MG correction kept, no CG correction this step";
                All.Flag_SkipMGSolve = 1;
            } else if(!converged) {outcome = "not converged; applied (lowers the divergence)";}
            if((!converged || !improves) && ThisTask == 0) {
                PRINT_WARNING("MG solve %s after %d iterations: |r|/|b| %g (tolerance %g), sqrt(sum r^2/sum s^2) %g -> %g, max |r|/d %g -> %g",
                              outcome, iterations, metrics.G, MG_CG_TOL, metrics.P0, metrics.P, metrics.M0, metrics.M);
            }
            if(improves) {mg_commit_solution(x.data());}
#ifdef DIVBCLEANING_DEDNER
            if(converged && improves) {   /* the face divergence is now zero, so the cleaning potential it sourced is spurious */
                for(int i = 0; i < N_gas; i++) {if(P[i].Type == 0) {CellP[i].Phi = CellP[i].PhiPred = CellP[i].DtPhi = 0;}}
            }
#endif
        }
    } else {
        endrun(stop_code);
        outcome = "stopped";
    }
    if(ThisTask == 0) {
        PRINT_STATUS(" ..MG solve: %lld connected pieces (%lld below the noise floor), %lld cross-task piece links (%s), %d iterations, |r|/|b| = %g, %s",
                     pieces.pieces, n_floor_pieces, pieces.cross_task_links, pieces.replicated ? "gathered" : "labels propagated",
                     iterations, rel_resid, outcome);
        if(!All.Flag_SkipMGSolve && !stop_code && (metrics.n_unexplained == 0)) {
            PRINT_STATUS(" ..MG residuals (uncorrected -> applied): sqrt(sum r^2/sum s^2) %g -> %g, max |r|/d %g -> %g; correction flux / field flux %g",
                         metrics.P0, metrics.P, metrics.M0, metrics.M, metrics.C);
        } else if(!stop_code) {
            PRINT_STATUS(" ..MG residuals (uncorrected): sqrt(sum r^2/sum s^2) %g, max |r|/d %g; the kept MG correction is not re-evaluated on this graph",
                         metrics.P0, metrics.M0);
        }
        if(!pieces.replicated) {PRINT_STATUS(" ..MG piece labels propagated in %d rounds", pieces.rounds);}
    }

    mg_cleanup_exchange();
    for(int i = 0; i < N_gas; i++) {free(MG_Rows[i].local_entries); free(MG_Rows[i].remote_entries);}
    free(MG_Rows); MG_Rows = NULL;
    MG_RowSolved.clear(); MG_RhsSolved.clear(); MG_RowInSolvedPiece.clear();
    if(ThisTask == 0) PRINT_STATUS(" ..MG correction complete.");
}

#endif /* MHD_MODIFIED_GRADIENT */
