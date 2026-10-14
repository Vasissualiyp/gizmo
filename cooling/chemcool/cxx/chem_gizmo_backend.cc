// chem_gizmo_backend.cc -- the CHEMCOOL_CXX entry point that GIZMO calls.
//
// This is the ONLY file that couples the C++/CVODE chemistry to GIZMO proper.
// It sits behind the Gate-0 seam (cooling/chemcool/chem_interface.h): by the
// time chem_evolve_cxx() runs, do_chemcool_step() has already executed
//
//     chem_fill_inputs(target, &chem_in);      // gather, no COMMON writes
//     chem_commit_to_fortran(&chem_in);        // the ONLY per-cell COMMON write
//     CALC_PHOTO_WRAPPER(...);                 // fills /current_photo/
//
// so every per-cell input and the frozen photorate cache are already in the
// COMMON blocks.  chemcxx_init() snapshots them into the explicit ChemCell
// context (audit R2) and cxx_evolve_abundances() integrates.  The signature is
// deliberately identical to EVOLVE_ABUNDANCES so the call site in chemcool.cc
// is a one-line #ifdef swap and nothing else in GIZMO changes.
//
// Scope note, UPDATED AFTER GATE 3.  The leaf kernels and the table builders
// are now C++ too (chem_cool_func.cc, chem_cool_util.cc, chem_calc_photo.cc,
// chem_calc_temp.cc, chem_tables_build.cc), and golden/build_purecxx.sh proves
// that set links with g++ alone -- no Fortran object, no libgfortran.
//
// A GIZMO build with CHEMCOOL_CXX ON nonetheless STILL LINKS the Fortran
// chemistry objects, for two deliberate reasons:
//   1. the Fortran DVODE path is retained as the fallback (Uli's standing
//      requirement) and chemcool.cc keeps both call sites under #ifdef;
//   2. do_chemcool_step() still uses COOLINMO/CHEMINMO/CALC_PHOTO_WRAPPER as
//      the INPUT transport -- chem_commit_to_fortran() writes the per-cell
//      inputs into /coolr/ and CALC_PHOTO_WRAPPER fills /current_photo/, and
//      chemcxx_init() snapshots both into ChemCell.
// Closing (2) is the remaining Gate-4 step and is now a short one: replace the
// chem_commit_to_fortran + CALC_PHOTO_WRAPPER pair with a ChemCell filled
// directly from `struct ChemInputs` plus cxx_calc_photo(), i.e. call
// chem_backend_cxx.cc instead of chem_backend_c_api.cc.  Everything it needs
// already exists.
//
// THREAD SAFETY: the C++ side is now safe (audit R2) -- ChemTables is
// immutable after init, ChemCell is per-cell, and the CVODE workspace and the
// statistics counters are thread_local (DEVIATIONS D-15); 64 concurrent solves
// on 8 threads reproduce the serial answers bit-for-bit.  The COMMON-block
// INPUT transport described above is still shared, so OPENMP + CHEMCOOL_CXX
// must wait for the Gate-4 step, not for any further porting.

#include "GIZMO_config.h"

#if defined(CHEMCOOL) && defined(CHEMCOOL_CXX)

#include "chem_types.h"

extern "C" {
int  chemcxx_init(int net);
int  chemcxx_evolve(double timestep, double dl, double yn, double divv,
                    double* energy, double* abundances);
double chemcxx_tdust(void);
void chemcxx_set_tdust(double t);
void chemcxx_get_lambda(double* lam, int nlam, double* lamchem, int nchem);
}

// the Fortran /coolr/ common -- we only need tdust here (in/out algebraic state)
extern "C" {
#include "cooling/chemcool/f2c.h"
#include "cooling/chemcool/chemcool_consts.h"
extern struct {
    double temptab[NMD], cltab[NCLTAB][NMD], chtab[NCHTAB][NMD],
           dtcltab[NCLTAB][NMD], dtchtab[NCHTAB][NMD],
           crtab[NCRTAB], crphot[NCRPHOT],
           phtab[NPHTAB], cst[NCONST], dtlog, tdust, tmax, tmin,
           deff, abundc, abundo, abundsi, abundD,
           abundM, abundN, G0, G0_LW, G0_dust,
           f_rsc, phi_pah,
           dust_to_gas_ratio, AV_conversion_factor,
           cosmic_ray_ion_rate, redshift, AV_ext,
           pdv_term, h2_form_ex, h2_form_kin,
           lambda[28], lambda_chem[NRATES_CHEM],
#ifdef OUTPUT_SHIELD_FAC
           fac_shield_h2, fac_shield_dust,
#endif
#ifdef WSS_CIE_COOL
           Zmass[12],
           C_tbl[352], N_tbl[352], O_tbl[352],
           Ne_tbl[352], Mg_tbl[352], Si_tbl[352],
           S_tbl[352], Ca_tbl[352], Fe_tbl[352],
           HeI_tbl[51], HeII_tbl[51],
#endif
           dm_density,
           rt_phot_HI, rt_phot_HeI, rt_phot_HeII,
           rt_heat_HI, rt_heat_HeI, rt_heat_HeII,
           cr_energy_density;
} COOLR;

void chemcool_fatal_(int* code);
}

#if CHEMISTRYNETWORK != 5 && CHEMISTRYNETWORK != 17
#error "CHEMCOOL_CXX currently supports CHEMISTRYNETWORK 5 and 17 only. \
Build without CHEMCOOL_CXX (the Fortran path) for the other networks."
#endif

extern "C"
void chem_evolve_cxx(double* timestep, double* dl, double* yn, double* divv,
                     double* energy, double* abundances, double* /*column_est*/)
{
    // column_est is hard-zeroed by the caller in every supported config
    // (chemcool.cc; audit §5.1) and the iphoto=4/5 path that consumed it is
    // not ported (DEVIATIONS D-5) -- ignored deliberately, not accidentally.

    static int inited = 0;
    if (!inited) { if (chemcxx_init(CHEMISTRYNETWORK)) { int c = 3; chemcool_fatal_(&c); } inited = 1; }

    // refresh the per-cell context from the COMMONs that chem_commit_to_fortran
    // and CALC_PHOTO_WRAPPER just wrote (this is the R5 freeze point)
    double tdust_cell = COOLR.tdust; /* this cell's carried dust temperature (CellP.DustTemp) */
    chemcxx_init(CHEMISTRYNETWORK);
    chemcxx_set_tdust(tdust_cell);   /* init would otherwise restore the previous cell's tdust (one-zone carry) */

    int rc = chemcxx_evolve(*timestep, *dl, *yn, *divv, energy, abundances);
    chemcxx_get_lambda(COOLR.lambda, 28, COOLR.lambda_chem, NRATES_CHEM); /* publish rates like the Fortran solver */

    // tdust rides outside the ODE vector (audit §3.1 item 2): publish the
    // driver's accepted value so chem_read_outputs() -> CellP.DustTemp sees it.
    COOLR.tdust = chemcxx_tdust();

    if (rc != 0) {
        // same failure taxonomy and funnel as the Fortran driver:
        // chemcool_fatal -> endrun(78100+code) -> MPI_Abort (audit R7)
        chemcool_fatal_(&rc);
    }
}

#endif /* CHEMCOOL && CHEMCOOL_CXX */
