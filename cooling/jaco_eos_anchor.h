#pragma once
/* The EOS a gas cell carries between jaco solves. The state of its cooling solve, or of a later full evaluation of
   jaco's EOS, is stored as an anchor; at any other specific energy u the anchor's composition is held fixed and the
   state continued along its tangent: T = T0 + (u - u0)/c_v0, with c_v0 = du/dT at T0, and P/rho = (P/rho)_0 T/T0,
   exact at fixed composition (P = n k T) and the same at every density, the abundances being per H nucleus.
   Predictions never re-anchor. The anchor applies only to the composition it was evaluated at (key) and within
   |u - u0| <= max_du u0. No GIZMO dependencies, so test/jaco_eos_anchor can check it against the full inversion. */
#include <math.h>
#include <stdint.h>
#include <string.h>

/* Largest |u - u0|/u0 over which an anchor is continued: T, P and c_s then stay within 1.4e-4 of the full EOS where
   H2's rotational levels make u(T) most curved, and within 1e-6 in H2-free gas (test/jaco_eos_anchor) */
#ifndef JACO_EOS_PREDICT_MAX_DU
#define JACO_EOS_PREDICT_MAX_DU 0.03
#endif

struct jaco_eos_anchor {
    double u;          /* erg/g; 0: no anchor */
    double T;          /* K */
    double dTdu;       /* 1/c_v at T, composition fixed */
    double P_over_rho; /* erg/g */
    double gamma;      /* 1 + P/(rho c_v T) */
    double x_e, x_Hplus, x_Heplus, x_Heplusplus, y; /* as in jaco_eos_state */
    uint64_t key;      /* jaco_eos_key_add over the composition inputs the anchor was evaluated at */
};

/* Fold x into a composition key: a change in any bit of any input changes the key, up to a 2^-64 collision */
static inline uint64_t jaco_eos_key_add(uint64_t key, double x) {
    uint64_t z;
    memcpy(&z, &x, sizeof(z));
    z ^= key + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* T and P/rho at specific energy u (erg/g) from anchor a, for a cell whose composition key is key: returns 1, or 0 where
   the anchor does not apply */
static inline int jaco_eos_anchor_predict(const struct jaco_eos_anchor *a, uint64_t key, double u, double max_du,
                                          double *T, double *P_over_rho) {
    if (!(a->u > 0) || a->key != key || !(fabs(u - a->u) <= max_du * a->u))
        return 0;
    double t = a->T + (u - a->u) * a->dTdu;
    if (!(t > 0))
        return 0;
    *T = t;
    *P_over_rho = a->P_over_rho * (t / a->T);
    return 1;
}
