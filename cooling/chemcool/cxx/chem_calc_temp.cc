// chem_calc_temp.cc -- Gate-3 port of calc_temp.F and compute_gamma.F.
//
//   lookup_temp_gamma  (calc_temp.F:159-211)  -- the T/gamma table lookup used
//                                                by rate_eq_* and the driver
//   calc_spec          (calc_temp.F:49-67)
//   calc_gamma_temp    (calc_temp.F:84-99)
//   calc_temp/calc_gamma are one-line wrappers around lookup_temp_gamma
//   compute_gamma      (compute_gamma.F:9-64)  -- the high-n gamma table
//
// The tables themselves (temp_table / gamma_table / eh2_table, from
// energy-table.txt and eh2-table.txt) are loaded in chem_tables_build.cc,
// which is the port of init_temperature_lookup.

#include "chem_types.h"
#include <cmath>

// ===========================================================================
// lookup_temp_gamma (calc_temp.F:159-211)
// ===========================================================================
void cxx_lookup_temp_gamma(double abh2, double ekn, double* temp, double* gamma,
                           const ChemTables& T)
{
    if (ekn < 1.0 || ekn > 5e4 || abh2 == 0.0) {
        // For ekn > 50000 the H2 content of the gas is negligible.
        *temp  = (2.0 / 3.0) * ekn;
        *gamma = 5.0 / 3.0;
        return;
    }

    int idx_en     = (int)ekn;
    double diff_en = ekn - (int)ekn;

    if (abh2 == 0.5) {
        double t1 = T.temp_table[200000 + idx_en - 1];
        double t2 = T.temp_table[200000 + idx_en];
        *temp = t1 + (t2 - t1) * diff_en;
        double g1 = T.gamma_table[200000 + idx_en - 1];
        double g2 = T.gamma_table[200000 + idx_en];
        *gamma = g1 + (g2 - g1) * diff_en;
        return;
    }

    // calc_temp.F:185-186 spells `0.1` without a D exponent, so it is a
    // DEFAULT-REAL literal: float32(0.1) = 0.100000001490116119.  Reading it as
    // the exact decimal shifts diff_h2 by ~1.5e-8 (measured: 2.6e-8 on temp).
    // ... AND `0.1 * idx_h2` is REAL(4)*INTEGER, i.e. the product is rounded
    // back to float32 before the REAL(8) subtraction.  That extra rounding is
    // visible only when 0.1f*idx_h2 is not exactly representable -- idx_h2 = 3
    // is the sole such case on this grid, which is exactly the one abh2 value
    // (0.4) the Gate-3 golden reported failing at 1.45e-8 once the division
    // literal had been fixed.
    const float  p1f = 0.1f;
    const double p1  = (double)p1f;
    int idx_h2      = (int)(abh2 / p1);
    double diff_h2  = (abh2 - (double)(p1f * (float)idx_h2)) / p1;
    double temp_1, gamma_1;
    if (idx_h2 == 0) {
        temp_1  = (2.0 / 3.0) * ekn;
        gamma_1 = (5.0 / 3.0);
    } else {
        double t1 = T.temp_table [50000 * (idx_h2 - 1) + idx_en - 1];
        double g1 = T.gamma_table[50000 * (idx_h2 - 1) + idx_en - 1];
        double t2 = T.temp_table [50000 * (idx_h2 - 1) + idx_en];
        double g2 = T.gamma_table[50000 * (idx_h2 - 1) + idx_en];
        temp_1  = t1 + (t2 - t1) * diff_en;
        gamma_1 = g1 + (g2 - g1) * diff_en;
    }
    double t1 = T.temp_table [50000 * idx_h2 + idx_en - 1];
    double g1 = T.gamma_table[50000 * idx_h2 + idx_en - 1];
    double t2 = T.temp_table [50000 * idx_h2 + idx_en];
    double g2 = T.gamma_table[50000 * idx_h2 + idx_en];
    double temp_2  = t1 + (t2 - t1) * diff_en;
    double gamma_2 = g1 + (g2 - g1) * diff_en;

    *temp  = temp_1  + (temp_2  - temp_1)  * diff_h2;
    *gamma = gamma_1 + (gamma_2 - gamma_1) * diff_h2;
}

// ===========================================================================
// calc_spec (calc_temp.F:49-67) -- H2 contribution to the specific energy
// ===========================================================================
double cxx_calc_spec(const ChemTables& T, double temp)
{
    if (temp < 1.0 || temp > 5e4) return 1.5 * temp;
    int idx = (int)temp;
    double dtemp = temp - idx;
    return T.eh2_table[idx - 1] + dtemp * (T.eh2_table[idx] - T.eh2_table[idx - 1]);
}

// ===========================================================================
// calc_gamma_temp (calc_temp.F:84-99)
// ===========================================================================
double cxx_calc_gamma_temp(const ChemTables& T, double abh2, double temp,
                           double abhe)
{
    double en = cxx_calc_spec(T, temp);
    double ekn = (abh2 / (1.0 + abhe - abh2)) * en
               + ((1.0 + abhe - 2.0 * abh2) / (1.0 + abhe - abh2)) * 1.5 * temp;
    double t, g;
    cxx_lookup_temp_gamma(abh2, ekn, &t, &g, T);
    return g;
}

// ===========================================================================
// compute_gamma (compute_gamma.F:9-64) -- gamma(e/kn, fh2) from highn_gamma.h
// ===========================================================================
double cxx_compute_gamma(const ChemTables& T, double abh2, double ekn)
{
    double lekn = std::log10(ekn);
    int index; double scale;
    if (lekn <= 1.5) {
        index = 1; scale = 0.0;
    } else if (lekn >= 4.5) {
        index = 25; scale = 0.0;
    } else {
        index = (int)((lekn - 1.5) / 0.125) + 1;
        scale = (lekn - 1.5 - 0.125 * (index - 1)) / 0.125;
    }

    const double* h = T.hng;
    if (abh2 == 0.0) return 5.0 / 3.0;

    if (abh2 < 0.1) {
        double gl = 5.0 / 3.0;
        double gh = (scale > 0.0) ? h[index-1] + (h[index] - h[index-1]) * scale
                                  : h[index-1];
        return gl + (abh2 / 0.1) * (gh - gl);
    }
    if (abh2 == 0.5) {
        index = 100 + index;
        return (scale > 0.0) ? h[index-1] + (h[index] - h[index-1]) * scale
                             : h[index-1];
    }
    double scale_h2 = (abh2 / 0.1) - (int)(abh2 / 0.1);
    int index1 = index + 25 * ((int)(abh2 / 0.1) - 1);
    int index2 = index + 25 * (int)(abh2 / 0.1);
    double gl, gh;
    if (scale > 0.0) {
        gl = h[index1-1] + (h[index1] - h[index1-1]) * scale;
        gh = h[index2-1] + (h[index2] - h[index2-1]) * scale;
    } else {
        gl = h[index1-1];
        gh = h[index2-1];
    }
    return gl + (gh - gl) * scale_h2;
}
