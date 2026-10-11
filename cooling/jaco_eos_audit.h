/* JACO_EOS_PREDICT_AUDIT: each anchored prediction set_eos_pressure takes (jaco_cell_eos) is repeated with the full EOS,
   discarded, and the relative differences in T, P and the sound speed are histogrammed by |u - u0|/u0 and by the
   anchor's regime; full evaluations are counted by why the anchor did not apply. Each rank writes
   jaco_eos_audit_<rank>.txt at exit (cumulative over the run). Included by jaco.cc only. */
#include <stdlib.h>

enum { JA_MOLECULAR, JA_ATOMIC, JA_PARTLY_IONIZED, JA_HOT, JA_NREG };
static const char *jaco_audit_regime_name[JA_NREG] = {"molecular_fH2>0.1", "atomic_T<5e3", "partly_ionized_5e3<=T<3e4",
                                                       "hot_T>=3e4"};
enum { JA_T, JA_P, JA_CS, JA_NQ };
static const char *jaco_audit_q_name[JA_NQ] = {"T", "P", "c_s"};
#define JA_NDU 34  /* bin 0: u == u0; bin k >= 1: log10|du/u0| in [-8 + (k-2)/4, -8 + (k-1)/4), bin 1 below -8 */
#define JA_NERR 70 /* bin 0: exact; bin k >= 1: log10 err in [-17 + (k-1)/4, -17 + k/4), last bin open */
static long ja_hits[JA_NREG][JA_NDU], ja_err_hist[JA_NQ][JA_NREG][JA_NERR], ja_err_above[JA_NREG][JA_NDU][3];
static double ja_err_max[JA_NQ][JA_NREG][JA_NDU];
static long ja_miss[5]; /* no anchor (incl. every solve), composition changed, heated directly, beyond the bound, other */
static int ja_registered = 0;

static int jaco_audit_bin(double x, double lo, int n) {
    if (!(x > 0)) return 0;
    int k = 1 + (int)floor((log10(x) - lo) * 4);
    return k < 1 ? 1 : (k > n - 1 ? n - 1 : k);
}

static void jaco_audit_write(void) {
    char fname[64];
    snprintf(fname, sizeof(fname), "jaco_eos_audit_%d.txt", ThisTask);
    FILE *fp = fopen(fname, "w");
    if (!fp) return;
    fprintf(fp, "# JACO_EOS_PREDICT_AUDIT max_du %g\nmiss no_anchor %ld composition %ld reheated %ld bound %ld other %ld\n",
            (double)JACO_EOS_PREDICT_MAX_DU, ja_miss[0], ja_miss[1], ja_miss[2], ja_miss[3], ja_miss[4]);
    fprintf(fp, "# hits: regime du_bin(0: du=0; k: log10|du/u0| < -8+(k-1)/4) n maxerr_T maxerr_P maxerr_cs nT>1e-6 nT>1e-4 nT>1e-3\n");
    for (int r = 0; r < JA_NREG; r++)
        for (int k = 0; k < JA_NDU; k++)
            if (ja_hits[r][k])
                fprintf(fp, "hits %s %d %ld %.3e %.3e %.3e %ld %ld %ld\n", jaco_audit_regime_name[r], k, ja_hits[r][k],
                        ja_err_max[JA_T][r][k], ja_err_max[JA_P][r][k], ja_err_max[JA_CS][r][k], ja_err_above[r][k][0],
                        ja_err_above[r][k][1], ja_err_above[r][k][2]);
    fprintf(fp, "# err_hist: quantity regime counts per bin (0: exact; k: log10 err < -17+k/4)\n");
    for (int q = 0; q < JA_NQ; q++)
        for (int r = 0; r < JA_NREG; r++) {
            fprintf(fp, "err_hist %s %s", jaco_audit_q_name[q], jaco_audit_regime_name[r]);
            for (int k = 0; k < JA_NERR; k++) fprintf(fp, " %ld", ja_err_hist[q][r][k]);
            fprintf(fp, "\n");
        }
    fclose(fp);
}

static void jaco_eos_audit_add(int i, struct particle_data *pp, struct gas_cell_data *cell, double u, double rho,
                               const struct jaco_eos_state *eos) {
    const struct jaco_eos_anchor *a = &cell[i].JacoEos;
    struct jaco_eos_state full;
    double cv, err[JA_NQ];
    jaco_cell_eos_full(i, pp, cell, u, rho, &full, &cv);
    err[JA_T] = fabs(eos->T / full.T - 1);
    err[JA_P] = fabs(eos->P_over_rho / full.P_over_rho - 1);
    err[JA_CS] = fabs(sqrt(eos->gamma * eos->P_over_rho / (full.gamma * full.P_over_rho)) - 1);
    int r = (cell[i].MolecularMassFraction > 0.1) ? JA_MOLECULAR : (a->T < 5e3 ? JA_ATOMIC : (a->T < 3e4 ? JA_PARTLY_IONIZED : JA_HOT));
    int k = jaco_audit_bin(fabs(u * UNIT_SPECEGY_IN_CGS / a->u - 1), -8, JA_NDU);
#ifdef _OPENMP
#pragma omp critical(jaco_eos_audit)
#endif
    {
        if (!ja_registered) { ja_registered = 1; atexit(jaco_audit_write); }
        ja_hits[r][k]++;
        for (int q = 0; q < JA_NQ; q++) {
            ja_err_hist[q][r][jaco_audit_bin(err[q], -17, JA_NERR)]++;
            if (err[q] > ja_err_max[q][r][k]) ja_err_max[q][r][k] = err[q];
        }
        ja_err_above[r][k][0] += err[JA_T] > 1e-6;
        ja_err_above[r][k][1] += err[JA_T] > 1e-4;
        ja_err_above[r][k][2] += err[JA_T] > 1e-3;
    }
}

/* after a full evaluation for set_eos_pressure: why the anchor did not apply */
static void jaco_eos_audit_miss(int i, struct particle_data *pp, struct gas_cell_data *cell, double u, uint64_t key,
                                const struct jaco_eos_anchor *old) {
    int why = !(old->u > 0) ? 0 : (old->key != key ? 1 : (cell[i].JacoReheated ? 2 :
              (!(fabs(u * UNIT_SPECEGY_IN_CGS - old->u) <= JACO_EOS_PREDICT_MAX_DU * old->u) ? 3 : 4)));
#ifdef _OPENMP
#pragma omp critical(jaco_eos_audit)
#endif
    {
        if (!ja_registered) { ja_registered = 1; atexit(jaco_audit_write); }
        ja_miss[why]++;
    }
}
