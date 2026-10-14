// cvode_driver.cc -- port of evolve_abundances.F (driver loop, TNOCHEM bypass,
// check_abundances, init_tolerances) on SUNDIALS CVODE (BDF + dense internal
// difference-quotient Jacobian == DVODE mf=22).
//
// Port-semantics decisions (audit §7 register; full rationale in DEVIATIONS.md):
//  R1/R4 KEEP: the OUTER substep loop is kept literally (safety factors 0.1
//    species / 0.3 energy, ctrl_floor pool gating, dt = min(dt_est, remaining)).
//    One CVode instance per cell call; successful substeps CONTINUE the
//    integrator (no reinit) == DVODE istate=2, so check_abundances clips do
//    NOT feed back into the integrator trajectory (clip-no-feedback KEPT).
//    Failures restore y/tdust, halve dt, CVodeReInit == istate=1.
//  R3 REDESIGN: RHS input violations act on a LOCAL COPY of y and signal via a
//    positive (recoverable) return instead of the ipar(1)+zero-ydot convention;
//    CVODE's internal recoverable retry substitutes for some driver halvings.
//    If CVODE cannot recover it returns a negative flag and the driver's
//    restore/halve path runs.
//  A10 KEEP: atol(energy) = 0 (pure relative control).  Legal in CVODE;
//    safe because the energy component is strictly positive (floored at
//    rho*MinEgySpec before entry and physically bounded below by CMB/floor).
//  R6 KEEP: TNOCHEM hot bypass ported wholesale (molecule zero + hot-CIE ion
//    set + explicit 10%-per-step energy Euler subcycle + mid-step handback +
//    1e-10 relative completion tolerance + ipar-clear == ignore RHS retval).
//  R7 KEEP: failure taxonomy: dt < 1e-8 s -> return 100 ("ABORT(100)");
//    substep budget 9e6 -> return 1 ("ABORT(0)"); negative dt -> return 2.
//    The one-zone harness maps nonzero to the chemcool_fatal exit path.

#include "chem_types.h"
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_context.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define DTCOOL_SCALE_FACTOR 0.3
#define CHEM_RTOL 1e-4
#define CHEM_NSTEP 9000000
#ifndef CHEMCOOL_TNOCHEM
#define CHEMCOOL_TNOCHEM 5e5
#endif

// solver statistics, exposed to the harness for the DVODE/CVODE comparison
// thread_local for the same reason as the workspace: the counters would
// otherwise be a data race.  chemcxx_stats() therefore reports the CALLING
// thread's totals; single-threaded runs (the Gate-2/3 harnesses) are unchanged.
thread_local long cxx_stat_calls = 0, cxx_stat_substeps = 0, cxx_stat_retries = 0;
thread_local long cxx_stat_rhs_evals = 0, cxx_stat_cvode_steps = 0;

// audit R10 switch: 1 = keep the Fortran "last RHS evaluation wins" semantics
static int chem_legacy_lambda()
{
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("CHEMCXX_LEGACY_LAMBDA");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

static inline int rhs_dispatch(double t, const double* y, double* ydot, ChemCell& c)
{
    return (c.net == 5) ? chem_rhs_net5(t, y, ydot, c)
                        : chem_rhs_net17(t, y, ydot, c);
}

// CVODE RHS trampoline
static int cvode_rhs(sunrealtype t, N_Vector yy, N_Vector yp, void* user_data)
{
    ChemCell* c = (ChemCell*)user_data;
    const double* y = N_VGetArrayPointer(yy);
    double* ydot = N_VGetArrayPointer(yp);
    cxx_stat_rhs_evals++;
    return rhs_dispatch(t, y, ydot, *c);   // 0 ok, +1 recoverable
}

// DVODE is silent about recoverable trouble; keep CVODE equally quiet so the
// benchmark logs stay readable.  Set CHEMCXX_VERBOSE=1 to restore the chatter.
static void cvode_err_handler(int error_code, const char* module,
                              const char* function, char* msg, void* /*eh_data*/)
{
    static int verbose = -1;
    if (verbose < 0) { const char* e = std::getenv("CHEMCXX_VERBOSE"); verbose = (e && *e != '0'); }
    if (verbose && error_code < 0)
        std::fprintf(stderr, "[CVODE] %s::%s: %s\n", module, function, msg);
}

// init_tolerances port (evolve_abundances.F:647-756, nets 5/17 branches)
static void chem_atol(int net, double* atol)
{
    if (net == 5) {
        atol[CIH2] = 1e-7;  atol[CIHP] = 1e-12; atol[CICO] = 1e-14;
        atol[3] = 0.0;                       // ATOL_TMP = 0d0
    } else {
        atol[CIH2] = 1e-7;  atol[CIHP] = 1e-12; atol[CICO] = 1e-14;
        atol[CIHEP] = 1e-14; atol[CIHEPP] = 1e-14;
        atol[CIDP] = 1e-14;  atol[CIHD] = 1e-12;
        atol[7] = 0.0;
    }
}

// check_abundances port (evolve_abundances.F:430-631, nets 5/17 branches).
// Returns 0 ok (y possibly clipped), 1 reject.
static int chem_check_abundances(double* y, ChemCell& c, const double* atol)
{
    const int nspec = c.nspec;
    const int itmp = nspec - 1;
    double max_abundance[8];
    max_abundance[CIH2] = 0.5;
    max_abundance[CIHP] = 1.0;
    max_abundance[CICO] = (c.abundc < c.abundo) ? c.abundc : c.abundo;
    if (c.net == 17) {
        max_abundance[CIHEP] = c.abhe;
        max_abundance[CIHEPP] = c.abhe;
        max_abundance[CIDP] = c.abundD;
        max_abundance[CIHD] = c.abundD;
    }
    max_abundance[itmp] = 0.0;   // unused (J != itmp below)

    for (int j = 0; j < nspec; j++)
        if (y[j] != y[j]) return 1;                         // NaN

    for (int j = 0; j < nspec; j++) {
        if (y[j] < 0.0) {
            if (y[j] < -atol[j]) return 1;                  // large negative
            y[j] = 0.0;                                     // small: clip
        }
    }
    for (int j = 0; j < nspec; j++) {
        if (j != itmp && y[j] > max_abundance[j]) {
            double diff = y[j] - max_abundance[j];
            if (diff < CHEM_RTOL * y[j]) y[j] = max_abundance[j];
            else return 1;                                  // large excess
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Persistent CVODE workspace.
// DVODE reuses a static rwork/iwork across calls and re-initializes with
// istate=1; allocating a fresh SUNContext + CVode memory per cell call would
// charge the port a malloc/free cost DVODE never pays and would make the
// wall-time comparison meaningless.  CVodeReInit on a cached instance is the
// exact istate=1 equivalent.  NOT thread-safe (neither is the Fortran side --
// audit R2); one workspace per nspec.
// ---------------------------------------------------------------------------
struct CvWork {
    SUNContext ctx = nullptr;
    void* mem = nullptr;
    N_Vector yv = nullptr, atolv = nullptr;
    SUNMatrix A = nullptr;
    SUNLinearSolver LS = nullptr;
    int nspec = 0;
};
// Gate 3 / audit R2: ONE workspace cache PER THREAD.  DVODE reuses a static
// rwork/iwork and re-initialises with istate=1; caching the CVODE equivalent
// keeps the wall-time comparison fair (DEVIATIONS D-7), and making the cache
// thread_local is what makes the C++ path OpenMP-safe -- a shared CVode memory
// object cannot be entered concurrently.  Cost: one workspace per (thread,net).
static thread_local CvWork g_work[2];   // [0] -> nspec 4 (net 5), [1] -> nspec 8 (net 17)

static CvWork* cv_get(int nspec, const double* atol, double t0, const double* y)
{
    CvWork* w = &g_work[(nspec == 4) ? 0 : 1];
    if (w->mem == nullptr) {
        if (SUNContext_Create(NULL, &w->ctx)) return nullptr;
        w->nspec = nspec;
        w->yv = N_VNew_Serial(nspec, w->ctx);
        w->atolv = N_VNew_Serial(nspec, w->ctx);
        for (int i = 0; i < nspec; i++) {
            NV_Ith_S(w->yv, i) = y[i];
            NV_Ith_S(w->atolv, i) = atol[i];
        }
        w->mem = CVodeCreate(CV_BDF, w->ctx);
        if (!w->mem) return nullptr;
        CVodeSetErrHandlerFn(w->mem, cvode_err_handler, NULL);
        if (CVodeInit(w->mem, cvode_rhs, t0, w->yv) < 0) return nullptr;
        w->A = SUNDenseMatrix(nspec, nspec, w->ctx);
        w->LS = SUNLinSol_Dense(w->yv, w->A, w->ctx);
        CVodeSetLinearSolver(w->mem, w->LS, w->A);   // internal DQ dense Jac (mf=22)
    } else {
        for (int i = 0; i < nspec; i++) {
            NV_Ith_S(w->yv, i) = y[i];
            NV_Ith_S(w->atolv, i) = atol[i];
        }
        if (CVodeReInit(w->mem, t0, w->yv) < 0) return nullptr;
    }
    CVodeSVtolerances(w->mem, CHEM_RTOL, w->atolv);
    CVodeSetMaxNumSteps(w->mem, 200000);             // iwork(6) = 200000
    return w;
}

// evolve_abundances port.
// non_eq_abundances has c.trac_num entries; energy in/out [erg cm^-3].
// Returns 0 ok; 100 = dt collapse (ABORT(100)); 1 = nstep exceeded (ABORT(0));
// 2 = negative/zero dt (ABORT(1)/(2)); 3 = CVODE setup failure.
int cxx_evolve_abundances(double time, double dl, double yn, double divv,
                          double* energy, double* non_eq_abundances, ChemCell& c)
{
    const int nspec = c.nspec;
    const int itmp = nspec - 1;
    const int ih2 = CIH2, ihp = CIHP, ico = CICO;

    if (time < 0.0) return 0;                                // .F:72-74

    cxx_stat_calls++;
    c.yn = yn; c.dl = dl; c.divv = divv;

    double y[8], y_old[8], ydot[8];
    for (int i = 0; i < c.trac_num; i++) y[i] = non_eq_abundances[i];
    y[itmp] = *energy;
    for (int i = 0; i < nspec; i++) { y_old[i] = y[i]; ydot[i] = 0.0; }

    double atol[8];
    chem_atol(c.net, atol);

    // safety factors + ctrl_floor (.F:35-43, 166-181)
    double safety[8];
    for (int i = 0; i < nspec; i++) safety[i] = 1e-1;
    safety[itmp] = DTCOOL_SCALE_FACTOR;
    const double eps_ctrl = 1e-6;
    double ctrl_floor[8];
    for (int j = 0; j < nspec; j++) ctrl_floor[j] = atol[j];
    {
        double co_pool = (c.abundc < c.abundo) ? c.abundc : c.abundo;
        double f = eps_ctrl * co_pool;
        if (f > ctrl_floor[ico]) ctrl_floor[ico] = f;
        if (c.net == 17) {
            f = eps_ctrl * c.abundD;
            if (f > ctrl_floor[CIDP]) ctrl_floor[CIDP] = f;
            if (f > ctrl_floor[CIHD]) ctrl_floor[CIHD] = f;
            f = eps_ctrl * c.abhe;
            if (f > ctrl_floor[CIHEP])  ctrl_floor[CIHEP]  = f;
            if (f > ctrl_floor[CIHEPP]) ctrl_floor[CIHEPP] = f;
        }
    }

    double t_start = 0.0, t_end = time;

    // ---- TNOCHEM hot bypass (.F:200-260) ----
    {
        double yntot_byp = (1.0 + c.abhe + y[ihp] - y[ih2]) * yn;
        double ekn_byp = y[itmp] / (CHEM_KBOLTZ * yntot_byp);
        double temp_byp, gam_byp;
        cxx_lookup_temp_gamma(y[ih2], ekn_byp, &temp_byp, &gam_byp, *c.tab);
        if (temp_byp > CHEMCOOL_TNOCHEM) {
            y[ih2] = 0.0;
            y[ico] = 0.0;
            if (c.net == 17) y[CIHD] = 0.0;
            // HOTIONFIX hot-CIE ion set (.F:223-228)
            y[ihp] = 1.0;
            if (c.net == 17) {
                y[CIHEP] = 0.0;
                y[CIHEPP] = c.abhe;
                y[CIDP] = c.abundD;
            }
            int done = 0;
            for (long nbyp = 1; nbyp <= 100000; nbyp++) {
                // ipar-clear semantics: RHS retval ignored; a flagged eval
                // left ydot all-zero, exactly like the Fortran zero-ydot path.
                if (c.net == 5) chem_rhs_net5(t_start, y, ydot, c);
                else            chem_rhs_net17(t_start, y, ydot, c);
                double dt_byp;
                if (std::fabs(ydot[itmp]) > 0.0)
                    dt_byp = std::fmin(t_end - t_start,
                                       1e-1 * std::fabs(y[itmp] / ydot[itmp]));
                else
                    dt_byp = t_end - t_start;
                y[itmp] += dt_byp * ydot[itmp];
                t_start += dt_byp;
                if (t_start >= t_end - 1e-10 * t_end) { done = 1; break; }  // WEDGEFIX 3
                yntot_byp = (1.0 + c.abhe + y[ihp] - y[ih2]) * yn;
                ekn_byp = y[itmp] / (CHEM_KBOLTZ * yntot_byp);
                cxx_lookup_temp_gamma(y[ih2], ekn_byp, &temp_byp, &gam_byp, *c.tab);
                if (temp_byp <= CHEMCOOL_TNOCHEM) break;    // hand off to stiff path
            }
            // budget exhausted while hot: accept state (.F:257-259)
            if (done || t_start >= t_end - 1e-10 * t_end) {
                for (int i = 0; i < c.trac_num; i++) non_eq_abundances[i] = y[i];
                *energy = y[itmp];
                return 0;
            }
            // else: fall through to the stiff path for the remaining interval
        }
    }

    // ---- CVODE setup (cached workspace; ReInit == istate=1) ----
    CvWork* w = cv_get(nspec, atol, t_start, y);
    if (!w) return 3;
    CVodeSetUserData(w->mem, &c);

    int new_step = 1;
    int need_reinit = 0;         // istate=1 equivalent pending
    double dt = 0.0, dt_est = 0.0;
    int retcode = 1;             // if the loop exhausts: "nstep exceeded"
    double tdust_old = c.tdust;

    for (long I = 1; I <= CHEM_NSTEP; I++) {
        if (new_step == 1) {
            for (int j = 0; j < nspec; j++) y_old[j] = y[j];
            tdust_old = c.tdust;

            int rerr = (c.net == 5) ? chem_rhs_net5(t_start, y, ydot, c)
                                    : chem_rhs_net17(t_start, y, ydot, c);
            dt_est = 1e18;
            for (int j = 0; j < nspec; j++) {
                if (ydot[j] != 0.0 && safety[j] != 0.0 && y[j] > ctrl_floor[j]) {
                    double e = std::fabs(safety[j] * y[j] / ydot[j]);
                    if (e < dt_est) dt_est = e;
                }
            }
            dt = std::fmin(dt_est, t_end - t_start);
            if (rerr) {
                // Driver-level RHS eval flagged.  In the Fortran, ipar(1)
                // persists through the dvode call and the substep is rejected
                // afterwards; here we take the failure path directly.
                c.tdust = tdust_old;
                dt = dt / 2.0;
                if (dt < 1e-8) { retcode = 100; goto out; }
                cxx_stat_retries++;
                new_step = 0; need_reinit = 1;
                continue;
            }
        }

        if (dt < 0.0) { retcode = 2; goto out; }
        if (dt == 0.0) {
            if (t_end == t_start) { retcode = 0; goto out_nocopy; }
            retcode = 2; goto out;
        }

        {
            if (need_reinit) {
                for (int i = 0; i < nspec; i++) NV_Ith_S(w->yv, i) = y[i];
                CVodeReInit(w->mem, t_start, w->yv);         // istate = 1
                need_reinit = 0;
            }
            double tret = t_start;
            int flag = CVode(w->mem, t_start + dt, w->yv, &tret, CV_NORMAL);
            cxx_stat_substeps++;

            int fail = (flag < 0);
            if (!fail) {
                for (int i = 0; i < nspec; i++) y[i] = NV_Ith_S(w->yv, i);
                fail = chem_check_abundances(y, c, atol);
            }

            if (fail) {
                for (int j = 0; j < nspec; j++) y[j] = y_old[j];
                c.tdust = tdust_old;
                dt = dt / 2.0;
                if (dt < 1e-8) {
                    std::fprintf(stderr,
                        "cxx_evolve: excessively small timestep! dt=%g dt_est=%g "
                        "t=[%g,%g] ID=%ld yn=%g\n",
                        dt, dt_est, t_start, t_end, c.id_current, yn);
                    retcode = 100; goto out;
                }
                cxx_stat_retries++;
                new_step = 0;
                need_reinit = 1;                          // istate = 1
            } else {
                t_start = t_start + dt;
                new_step = 1;                             // istate stays 2:
                // NB check_abundances clips modified OUR y (final answer /
                // future y_old) but were NOT pushed into CVODE -- continuation
                // resumes from CVODE's internal (unclipped) trajectory,
                // exactly the DVODE clip-no-feedback semantics.
            }
        }

        if (t_start >= t_end - 1e-10 * t_end) { retcode = 0; goto out; }
    }
    std::fprintf(stderr, "cxx_evolve: nstep exceeded (ID %ld)\n", c.id_current);
    retcode = 1;

out:
    if (retcode == 0 || retcode == 1) {
        for (int i = 0; i < c.trac_num; i++) non_eq_abundances[i] = y[i];
        *energy = y[itmp];
        // ---- audit R10: state-consistent lambda diagnostics ----
        // The Fortran leaves lambda/lambda_chem at whatever the LAST RHS
        // evaluation wrote.  With a multistep integrator that evaluation is at
        // an internal stage or, worse, at a REJECTED trial point, so the
        // published cooling-rate breakdown does not correspond to the state
        // that was actually returned -- and it is the breakdown that lands in
        // snapshots and in every "why did this cell cool" post-mortem.
        // One extra RHS evaluation at the ACCEPTED (t_end, y) fixes that.
        // Cost: one RHS per cell call, ~0.4% of the ~240 evaluations a typical
        // call already makes.  Set CHEMCXX_LEGACY_LAMBDA=1 to restore the
        // Fortran semantics for a bit-comparable A/B (DEVIATIONS D-19).
        if (!chem_legacy_lambda()) {
            double ydot_final[8];
            (void)rhs_dispatch(t_start, y, ydot_final, c);
        }
    }
out_nocopy:
    {
        long ns = 0;
        if (w && w->mem && CVodeGetNumSteps(w->mem, &ns) == CV_SUCCESS)
            cxx_stat_cvode_steps += ns;
    }
    return retcode;
}
