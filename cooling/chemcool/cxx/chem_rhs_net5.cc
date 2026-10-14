// chem_rhs_net5.cc -- verbatim-in-structure port of rate_eq_simple.F
// (CHEMISTRYNETWORK=5 branch only; nets 4/6 branches not ported).
// Source: /orion/ptmp/uli/orion/test/gizmo/cooling/chemcool/rate_eq_simple.F
// (canonical, read 2026-07-29, includes no net-5-relevant fixes pending).
//
// ODE vector (0-based): y[0]=H2, y[1]=H+, y[2]=CO, y[3]=energy density.
// Deviations from the Fortran (all per the audit port-semantics register):
//  - validate_input works on a local copy (R3); flagged input returns
//    CHEM_RHS_RECOVERABLE instead of zero-ydot+ipar.
//  - temp<=0 poisoned-input path returns CHEM_RHS_RECOVERABLE immediately
//    (Fortran computes a tmin-evaluated ydot AND rejects the substep after
//    the dvode call returns -- same net effect, documented in DEVIATIONS.md).
//  - validate_output NaN-ydot ABORT(1) is replaced by a loud recoverable
//    return (unified on the net-17 behavior; DEVIATIONS.md).
//  - the unreachable temp<0 bare `stop` (rate_eq_simple.F:505-511) is not
//    ported (audit R7 cleanup item).
//  - debug/compile-out switches BS19_XH2 / CHEM_EQ_H2 / NO_CO_FORM /
//    ADIABATIC_DENSITY_THRESHOLD / CONST_NCRIT_UVPUMP are OFF in production
//    and not ported.
//  - RT_INFRARED is OFF in the one-zone/production-subgrid config: the
//    calc_dust_temp call uses nocalc=0.

#include "chem_types.h"
#include "chem_rhs_common.h"
#include <cstdio>

int chem_rhs_net5(double /*t*/, const double* y_in, double* ydot, ChemCell& c)
{
    const ChemTables& T = *c.tab;
    const int nspec = 4;
    const int ih2 = CIH2, ihp = CIHP, ico = CICO, itmp = 3;

    const double yn = c.yn, dl = c.dl, divv = c.divv;
    const double NH = yn * dl;                       // rate_eq_simple.F:73

    // --- validate_input on a local copy (R3) ---
    double y[4];
    for (int i = 0; i < nspec; i++) { y[i] = y_in[i]; ydot[i] = 0.0; }
    if (chem_validate_input(y, nspec, itmp, ih2, c.id_current))
        return CHEM_RHS_RECOVERABLE;

    // --- tracked species (floors as in .F:92-97) ---
    double abhp = (y[ihp] > 0.0) ? y[ihp] : 0.0;
    double abh2 = (y[ih2] > 0.0) ? y[ih2] : 0.0;
    double abco = (y[ico] > 0.0) ? y[ico] : 0.0;

    // --- conservation-law species (.F:106-113) ---
    double abHI = 1.0 - 2.0 * abh2 - abhp; if (abHI < 0.0) abHI = 0.0;
    double abcp = c.abundc - abco;         if (abcp < 0.0) abcp = 0.0;
    double abe  = abhp + abcp + c.abundsi;

    // --- energy, temperature (.F:117-126) ---
    double energy = y[itmp];
    double yntot = (1.0 + c.abhe - abh2 + abe) * yn;
    double ekn = energy / (yntot * CHEM_KBOLTZ);
    double temp, gamma;
    cxx_lookup_temp_gamma(abh2, ekn, &temp, &gamma, T);

    // --- temperature bounds / table index (.F:150-168) ---
    ChemTIdx ti = chem_temp_index(temp, T);
    if (ti.flag_bad) return CHEM_RHS_RECOVERABLE;    // temp<=0 poisoned input
    temp = ti.temp;

    // --- rate coefficients (.F:172-194) ---
    const double ch1  = chem_CH(T, 1,  ti);
    const double ch2  = chem_CH(T, 2,  ti);
    const double ch3  = chem_CH(T, 3,  ti);
    const double ch4  = chem_CH(T, 4,  ti);
    const double ch5  = chem_CH(T, 5,  ti);
    const double ch6  = chem_CH(T, 6,  ti);
    const double ch11 = chem_CH(T, 11, ti);
    const double ch13 = chem_CH(T, 13, ti);
    const double ch25 = chem_CH(T, 25, ti);
    const double ch34 = chem_CH(T, 34, ti);
    const double ch35 = chem_CH(T, 35, ti);

    // --- H2 collisional dissociation, density-dependent (.F:199-204) ---
    double ncrinv = 2.0 * abh2 * (ch6 - ch5) + ch5;
    double h2var0 = 1.0 / (1.0 + yn * ncrinv);
    double h2var1 = std::pow(ch3, h2var0);
    double h2var2 = std::pow(ch4, h2var0);
    double ykdh  = ch1 * h2var1;
    double ykdh2 = ch2 * h2var2;

    // --- photodissociation (.F:208-225) ---
    double fshield = (c.iphoto == 0) ? 1.0
                     : cxx_calc_H2_shielding_factor(temp, abh2, NH, c);
    double ykdis = c.phrates[0] * fshield;           // phrates(1)
    double gamma_co  = c.phrates[9];                 // phrates(10)
    double gamma_chx = 2.94e-10 * c.G_dust;
    double ph38 = c.phrates[37];
    double ph42 = c.phrates[41];

    // --- grain recombination (.F:232-253) ---
    double h_gr = chem_h_gr(temp, yn, abe, c.G_dust, c.dust_to_gas_ratio, ch34, ch35);

    // --- dust temperature + H2 formation on grains (.F:255-264) ---
    double RH2, Rdust;
    cxx_calc_dust_temp(yn, c.chi_isrf, temp, abHI, &RH2, &Rdust, 0, c);

    // --- cosmic rays (.F:269-272) ---
    double cr1  = T.crtab[0];
    double cr2  = T.crtab[1];
    double cr10 = T.crtab[9];
    double cr11 = T.crtab[10];

    // --- protons (.F:276-281) ---
    ydot[ihp] = (ch11 * abe  * abHI
              -  ch13 * abhp * abe
              -  h_gr * abhp * 1.0) * yn
              +  cr1  * abHI
              +  cr10 * abh2
              +  ph38 * abHI;

    // --- molecular hydrogen (.F:285-292) ---
    ydot[ih2] = (-ch25  * abe  * abh2
              -   ykdh  * abHI * abh2
              -   ykdh2 * abh2 * abh2
              +   Rdust * abHI * 1.0) * yn
              -   ykdis * abh2
              -   0.5 * cr2 * abh2   // "fudge factor of two gives correct HI production"
              -   cr10 * abh2
              -   cr11 * abh2;

    // --- CO, NL97 (.F:302-324) ---
    const double k0 = 5e-16, k1 = 5e-10;
    double abo = c.abundo - abco; if (abo < 0.0) abo = 0.0;  // .F:309 max(0.0,...)
    double beta;
    if (abo == 0.0)               beta = 0.0;
    else if (gamma_chx < 1e-30)   beta = 1.0;
    else                          beta = k1 * abo / (k1 * abo + gamma_chx / yn);
    ydot[ico] = k0 * abcp * abh2 * beta * yn - gamma_co * abco;

    // --- chemical heating/cooling, rates_chem(1..6) (.F:354-418) ---
    double rates_chem[6];
    rates_chem[0] = 4.48 * CHEM_EV * (ykdh  * abHI * abh2 +
                                      ykdh2 * abh2 * abh2 +
                                      ch25  * abe  * abh2) * yn * yn;
    rates_chem[1] = -4e-1 * CHEM_EV * abh2 * yn * ykdis;

    double cdex = (1.4 * std::exp(-1.81e4 / (temp + 1200)) * abh2
                 + 1.0 * std::exp(-1e3 / temp) * abHI) * 1e-12 * std::sqrt(temp) * yn;
    double rdex = 2e-7;
    double fpheat = cdex / (cdex + rdex);
    rates_chem[2] = -2.0 * CHEM_EV * abh2 * yn * ykdis * 8.5 * fpheat;

    double f_ex = 0.0;                                // .F:391-397 (call commented out)
    double f_tot = c.h2_form_kin + f_ex;
    rates_chem[3] = -4.48 * CHEM_EV * f_tot * RH2;

    rates_chem[4] = 13.6 * CHEM_EV * ch11 * abe * abHI * yn * yn;

    rates_chem[5] = CHEM_KBOLTZ * temp * abhp * (ch13 * abe + h_gr) * yn * yn;
    if (temp < 100.0) rates_chem[5] = 0.0;            // .F:412-414

    double ylam_chem = rates_chem[0] + rates_chem[1] + rates_chem[2]
                     + rates_chem[3] + rates_chem[4] + rates_chem[5]
                     - (ph42 * abHI) * yn;            // M1 photoheating add-on

    for (int i = 0; i < 6; i++) c.lambda_chem[i] = rates_chem[i];  // .F:422-427
    for (int i = 6; i < CHEM_NRATES_CHEM_MAX; i++) c.lambda_chem[i] = 0.0;

    // --- cool_func abundance vector (.F:446-462) ---
    double abundances[16];
    for (int i = 0; i < 16; i++) abundances[i] = 0.0;
    abundances[0]  = abh2;
    { double a2 = c.abundo - abco; if (a2 < 0.0) a2 = 0.0; abundances[1] = a2; }
    abundances[4]  = abco;
    abundances[6]  = abcp;
    abundances[8]  = c.abundsi;
    abundances[9]  = abe;
    abundances[10] = abhp;
    abundances[11] = abHI;

    // --- Jeans length (.F:473-476; NOTE: no cs2<0 guard in net 5) ---
    double rho = (1.0 + 4.0 * c.abhe) * CHEM_PROTONMASS * yn;
    double cs2 = gamma * (gamma - 1.0) * energy / rho;
    double L_jeans = std::sqrt(CHEM_PI * cs2 / (CHEM_GRAVITY * rho));

    double ylam, rates[CHEM_NRATES];
    int cferr = cxx_cool_func(temp, yn, L_jeans, divv, c.G_dust, abundances,
                              &ylam, rates, 0, c);
    if (cferr) return CHEM_RHS_RECOVERABLE;           // Fortran ABORT(0)/(2) sites

    // pdv_term is hard-zeroed in production (chemcool.cc:721) -- not carried.
    ydot[itmp] = -(ylam + ylam_chem);

    // --- validate_output: NaN ydot (.F:502; unified recoverable, DEVIATIONS) ---
    for (int i = 0; i < nspec; i++) {
        if (ydot[i] != ydot[i]) {
            std::fprintf(stderr,
                "chem_rhs_net5: NaN ydot[%d] (ID %ld, T=%g yn=%g)\n",
                i, c.id_current, temp, yn);
            for (int j = 0; j < nspec; j++) ydot[j] = 0.0;
            return CHEM_RHS_RECOVERABLE;
        }
    }
    return CHEM_RHS_OK;
}
