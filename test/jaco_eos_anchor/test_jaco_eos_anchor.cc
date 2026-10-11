/* Standalone check of cooling/jaco_eos_anchor.h, the EOS a gas cell carries between jaco solves, against jaco's full
   EOS: the u -> T inversion at the composition held fixed (Newton to round-off, as jaco_T_from_u), then P and the
   first adiabatic index there, the metals' free electrons re-evaluated at that T as jaco_cell_eos_full does.

   1. At the anchor (u = u0) the prediction is the anchor's own state, bit for bit.
   2. Over molecular, atomic, partly ionized and ionized compositions from 10 K to 1e7 K, every |u - u0| up to the
      bound JACO_EOS_PREDICT_MAX_DU: T and P within TOL of the full EOS, the sound speed within TOL_CS. Two controls show the check
      sees the failures it guards against: the chord through the origin (T = u/c_v0) and the ideal-gas pressure
      (gamma0 - 1) rho u, each off by more than TOL in molecular gas inside the bound.
   3. Beyond the bound, with no anchor, or after any change of a composition input, the anchor is refused: a stale
      composition would be off by more than TOL.
       make -C test/jaco_eos_anchor [MODEL=...] */
#include <cmath>
#include <cstdio>
#include "microphysics_func_jac.h" /* generated into build/MODEL/ by the Makefile */
#include "jaco_eos_anchor.h"

double jaco_T_to_u(double T, const SolveVars *sv, const Params *pr, double *cv_out);
double jaco_eos_pressure(const SolveVars *sv, const Params *pr);
double jaco_electron_abundance(const SolveVars *sv, const Params *pr);

#if !defined(JACO_FAMILY_STARFORGE)
#error "this test fills the STARFORGE family's Params"
#endif

/* the accuracy the anchor must hold inside its bound, relative: T and P; the sound speed, whose gamma is held at the
   anchor's and so is off at first order in T - T0 where H2's heat capacity varies */
static const double TOL = 2e-4, TOL_CS = 2.5e-3;
static int failures = 0, checks = 0;

struct comp {
    const char *label;
    double xHp, xHep, xHepp, xH2, Z; /* Z: metallicity in solar units */
    double T[12];                    /* anchor temperatures, 0-terminated */
};
static const comp comps[] = {
    {"molecular", 1e-8, 1e-12, 1e-12, 0.4999, 1, {10, 20, 40, 70, 100, 150, 200, 300, 500, 1000, 3000, 0}},
    {"half molecular", 1e-6, 1e-12, 1e-12, 0.25, 1, {10, 30, 70, 100, 200, 300, 500, 1000, 3000, 0}},
    {"atomic", 1e-4, 1e-12, 1e-12, 1e-8, 1, {10, 30, 100, 300, 1000, 3000, 6000, 0}},
    {"atomic, Z = 0.01", 1e-4, 1e-12, 1e-12, 1e-8, 0.01, {10, 100, 1000, 6000, 0}},
    {"partly ionized", 0.5, 1e-3, 1e-12, 1e-10, 1, {5e3, 8e3, 1e4, 1.5e4, 2e4, 3e4, 0}},
    {"ionized", 0.9999, 0.05, 0.0494, 1e-12, 1, {1e4, 3e4, 1e5, 1e6, 1e7, 0}},
};

static void fill(const comp &c, SolveVars *sv, Params *pr) {
    for (int k = 0; k < N_VARS; k++) sv->data[k] = 0;
    for (int k = 0; k < N_PARAMS; k++) pr->data[k] = 0;
    sv->x_Hplus = c.xHp;
    sv->x_Heplus = c.xHep;
    sv->x_Heplusplus = c.xHepp;
    sv->x_H_2 = c.xH2;
    pr->n_Htot = 100;
    pr->y = 0.0994;
    pr->X = 0.7155;
    pr->G_0 = 1;
    pr->ISRF = 1;
    pr->N_H = 1e21;
    pr->f_d = 1;
    pr->Z_d = c.Z;
    const double x_solar[9] = {2.1e-4, 6.8e-5, 4.9e-4, 8.5e-5, 3.2e-5, 3.2e-5, 1.3e-5, 2.2e-6, 2.5e-5};
    pr->x_C_tot = x_solar[0] * c.Z;
    pr->x_N = x_solar[1] * c.Z;
    pr->x_O_tot = x_solar[2] * c.Z;
    pr->x_Ne = x_solar[3] * c.Z;
    pr->x_Mg = x_solar[4] * c.Z;
    pr->x_Si = x_solar[5] * c.Z;
    pr->x_S = x_solar[6] * c.Z;
    pr->x_Ca = x_solar[7] * c.Z;
    pr->x_Fe = x_solar[8] * c.Z;
}

/* mass density at these Params (n_Htot included), as jaco_mass_per_H takes it: with no H2 every species carries (3/2) kT */
static double mass_density(const Params *pr) {
    SolveVars s = {};
    s.T = 1e4;
    s.x_Hplus = 1e-4;
    s.x_Heplus = s.x_Heplusplus = 1e-20;
    return 1.5 * jaco_eos_pressure(&s, pr) / jaco_T_to_u(s.T, &s, pr, NULL);
}

struct state { double T, P_over_rho, gamma; };

/* the full EOS at specific energy u for the composition in sv, as jaco_cell_eos_full evaluates it */
static state full_eos(double u, double T_seed, SolveVars sv, const Params *pr, double rho, double *cv) {
    double T = T_seed, lo = 1e-3, hi = 1e12;
    for (int it = 0; it < 200; it++) {
        double du = jaco_T_to_u(T, &sv, pr, cv) - u;
        if (std::fabs(du) <= 1e-13 * std::fabs(u)) break;
        if (du > 0) hi = std::fmin(hi, T); else lo = std::fmax(lo, T);
        double Tn = T - du / *cv;
        T = (Tn > lo && Tn < hi) ? Tn : std::sqrt(lo * hi);
    }
    jaco_T_to_u(T, &sv, pr, cv);
    sv.T = T;
    state s;
    s.T = T;
    s.P_over_rho = jaco_eos_pressure(&sv, pr) / rho;
    s.gamma = 1 + s.P_over_rho / (*cv * T);
    return s;
}

/* the composition inputs a GIZMO cell keys its anchor on: Ne, MolecularMassFraction, metallicity */
static uint64_t comp_key(const SolveVars *sv, const Params *pr) {
    SolveVars s = *sv;
    s.T = 100;
    uint64_t key = jaco_eos_key_add(0, jaco_electron_abundance(&s, pr));
    key = jaco_eos_key_add(key, 2 * sv->x_H_2);
    return jaco_eos_key_add(key, 0.0142 * pr->Z_d);
}

static jaco_eos_anchor make_anchor(double T0, const SolveVars *sv, const Params *pr, double rho) {
    double cv;
    jaco_eos_anchor a = {};
    a.u = jaco_T_to_u(T0, sv, pr, &cv);
    state s = full_eos(a.u, T0, *sv, pr, rho, &cv);
    a.T = s.T;
    a.dTdu = 1 / cv;
    a.P_over_rho = s.P_over_rho;
    a.gamma = s.gamma;
    a.key = comp_key(sv, pr);
    return a;
}

static double rel(double a, double b) { return std::fabs(a / b - 1); }

static void expect(int ok, const char *what) {
    checks++;
    if (!ok) {
        failures++;
        std::printf("  FAIL: %s\n", what);
    }
}

int main() {
    const double max_du = JACO_EOS_PREDICT_MAX_DU;
    /* |u - u0|/u0 inside the bound (the bound itself last), then beyond it */
    const double grid[] = {1e-6, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3};
    double fracs[8], beyond[9];
    int nin = 0, nout = 0;
    for (double f : grid)
        if (f < max_du) fracs[nin++] = f; else if (f > max_du) beyond[nout++] = f;
    fracs[nin++] = max_du;
    beyond[nout++] = max_du * (1 + 1e-9);
    std::printf("anchor bound |u - u0|/u0 <= %g, required accuracy %g (T, P), %g (c_s)\n", max_du, TOL, TOL_CS);
    std::printf("%-18s %10s | max rel. error over the anchor temperatures at |u - u0|/u0 =\n", "composition", "quantity");
    double worst[3] = {0, 0, 0}, chord_mol = 0, ideal_mol = 0;
    for (const comp &c : comps) {
        SolveVars sv;
        Params pr;
        fill(c, &sv, &pr);
        const double rho = mass_density(&pr);
        double err[3][8] = {};
        for (int t = 0; c.T[t] > 0; t++) {
            jaco_eos_anchor a = make_anchor(c.T[t], &sv, &pr, rho);
            /* 1. at the anchor: its own state */
            double T, Pr;
            int ok = jaco_eos_anchor_predict(&a, a.key, a.u, max_du, &T, &Pr);
            expect(ok && T == a.T && Pr == a.P_over_rho, "prediction at the anchor is not the anchor's state");
            for (int f = 0; f < nout; f++) /* 3. beyond the bound: refused */
                for (int sgn = -1; sgn <= 1; sgn += 2)
                    expect(!jaco_eos_anchor_predict(&a, a.key, a.u * (1 + sgn * beyond[f]), max_du, &T, &Pr),
                           "anchor used beyond its bound");
            for (int f = 0; f < nin; f++)
                for (int sgn = -1; sgn <= 1; sgn += 2) {
                    /* 2. inside the bound: the full EOS's state */
                    double u = a.u + sgn * fracs[f] * a.u * (1 - 1e-12), cv;
                    if (!jaco_eos_anchor_predict(&a, a.key, u, max_du, &T, &Pr)) {
                        expect(0, "anchor refused inside its bound");
                        continue;
                    }
                    state s = full_eos(u, a.T, sv, &pr, rho, &cv);
                    double e[3] = {rel(T, s.T), rel(Pr, s.P_over_rho), std::fabs(std::sqrt(a.gamma * Pr / (s.gamma * s.P_over_rho)) - 1)};
                    for (int q = 0; q < 3; q++) {
                        err[q][f] = std::fmax(err[q][f], e[q]);
                        worst[q] = std::fmax(worst[q], e[q]);
                    }
                    if (c.xH2 > 0.2) {
                        chord_mol = std::fmax(chord_mol, rel(u * a.dTdu, s.T));
                        ideal_mol = std::fmax(ideal_mol, rel((a.gamma - 1) * u, s.P_over_rho));
                    }
                }
        }
        const char *qn[3] = {"T", "P", "c_s"};
        for (int q = 0; q < 3; q++) {
            std::printf("%-18s %10s |", q ? "" : c.label, qn[q]);
            for (int f = 0; f < nin; f++) std::printf(" %7.1e:%8.2e", fracs[f], err[q][f]);
            std::printf("\n");
        }
    }
    std::printf("worst inside the bound: T %.3e  P %.3e  c_s %.3e\n", worst[0], worst[1], worst[2]);
    expect(worst[0] <= TOL && worst[1] <= TOL && worst[2] <= TOL_CS, "anchor off by more than TOL inside its bound");
    std::printf("controls, molecular gas inside the bound: chord T = u/c_v0 off by %.3e, ideal-gas P = (gamma0-1) rho u by %.3e\n",
                chord_mol, ideal_mol);
    expect(chord_mol > TOL && ideal_mol > TOL, "a control is within TOL: the check could not see that failure");

    /* 3. composition changes: each one changes the key; the stale anchor would be off by more than TOL */
    {
        const comp &m = comps[1];
        SolveVars sv;
        Params pr;
        fill(m, &sv, &pr);
        const double rho = mass_density(&pr);
        jaco_eos_anchor a = make_anchor(150, &sv, &pr, rho);
        jaco_eos_anchor none = a;
        none.u = 0;
        double T, Pr, cv;
        expect(!jaco_eos_anchor_predict(&none, a.key, a.u, max_du, &T, &Pr), "anchor used with none set");
        struct { const char *what; double dxHp, xH2_scale, Z_scale; } changes[] = {
            {"H2 dissociated by half", 0, 0.5, 1}, {"ionized 10%", 0.1, 1, 1}, {"metallicity doubled", 0, 1, 2}};
        for (auto &ch : changes) {
            comp c2 = m;
            c2.xHp += ch.dxHp;
            c2.xH2 *= ch.xH2_scale;
            c2.Z *= ch.Z_scale;
            SolveVars sv2;
            Params pr2;
            fill(c2, &sv2, &pr2);
            const double rho2 = mass_density(&pr2);
            uint64_t key2 = comp_key(&sv2, &pr2);
            state s = full_eos(a.u, a.T, sv2, &pr2, rho2, &cv);
            double Ts = a.T, Ps = a.P_over_rho; /* what the stale anchor gives at u0 */
            double stale = std::fmax(rel(Ts, s.T), rel(Ps, s.P_over_rho));
            int used = jaco_eos_anchor_predict(&a, key2, a.u, max_du, &T, &Pr);
            std::printf("composition change '%s': key %s, stale anchor off by %.3e\n", ch.what, used ? "UNCHANGED" : "changed", stale);
            expect(!used || std::fmax(rel(T, s.T), rel(Pr, s.P_over_rho)) <= TOL, "stale anchor used after a composition change");
        }
    }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
