"""Unit test of jaco_eos_violations (gizmo.test) on synthetic snapshots built from jaco's own EOS (jaco.eos.eos).

Every cell's T, u and P come from jaco's symbolic EOS at an explicit species state; the snapshot fields are written as
GIZMO writes them for a JACO run with STARFORGE defaults and COOLING (COOL_MOLECFRAC_NONEQM): MolecularMassFraction is
H2 per neutral H nucleus, 2 x_H2 / (1 - x_H+), the composition is the Metallicity array as cooling/jaco.cc packs it
(He and metals capped, mass numbers), and the mass density is jaco's sum of species masses.

The clean snapshot must show no violation, including partly ionized molecular gas and He- and metal-rich SN ejecta.
Each negative control evaluates some cells' EOS at a state other than the one the snapshot reports, as a stale EOS
cache or a desynchronised pressure would, and must be flagged in exactly those cells."""

import h5py
import numpy as np
import pytest
import sympy as sp

jaco_eos = pytest.importorskip("jaco.eos.eos")
from jaco.symbols import T as T_SYM, n_  # noqa: E402

from gizmo.test import jaco_eos_violations  # noqa: E402

METALS = ("C", "N", "O", "Ne", "Mg", "Si", "S", "Ca", "Fe")  # Metallicity[2..10]
MASS_NUMBER = np.array([12, 14, 16, 20, 24, 28, 32, 40, 56.0])  # cooling/jaco.cc's mass fraction -> abundance
SPECIES = ("H", "H+", "H_2", "He", "He+", "He++", "e-", "C", "C+") + METALS[1:]
SOLAR = np.array([0.0142, 0.2703, 2.53e-3, 7.41e-4, 6.13e-3, 1.34e-3, 7.57e-4, 7.12e-4, 3.31e-4, 6.87e-5, 1.38e-3])
UNIT_V, UNIT_RHO = 1e5, 1.99e33 / 3.09e18**3
N_HTOT = 100.0  # cm^-3; the checks are per unit mass, so any density

# Cells: T, x_H+, x_H2 (per H nucleus), C+ / C, metallicity [Z, Y, C..Fe] (SN ejecta are above the host's caps on Z
# and Y, with 1.4% of their metal mass untracked, as GIZMO's yields carry it)
EJECTA = np.concatenate([[0.30, 0.45], 0.986 * 0.30 * SOLAR[2:] / SOLAR[2:].sum()])
CELLS = [
    (20.0, 1e-7, 0.4999, 1.0, SOLAR),  # cold molecular
    (300.0, 1e-4, 0.45, 1.0, SOLAR),
    (1000.0, 0.03, 0.30, 1.0, SOLAR),  # partly ionized molecular: x_H+ x_H2 well above the tolerance
    (2500.0, 0.08, 0.20, 1.0, SOLAR),
    (8000.0, 0.2, 1e-6, 1.0, SOLAR),  # warm atomic
    (1.5e4, 0.95, 1e-12, 1.0, SOLAR),
    (1e6, 1.0, 1e-20, 1.0, SOLAR),  # hot, He fully ionized
    (1e6, 1.0, 1e-20, 1.0, EJECTA),  # SN ejecta
    (3e4, 0.9, 1e-12, 1.0, EJECTA),
    (500.0, 0.01, 0.25, 1.0, 0.5 * (SOLAR + EJECTA)),  # enriched molecular
]


def _jaco_state(T, x_Hp, x_H2, f_Cp, Z):
    """Per-H abundances as cooling/jaco.cc packs the composition, the ions set by x_H+ and T (He ionizes after H)"""
    Zc, Y = min(Z[0], 0.25), min(Z[1], 0.35)
    X = 1 - Y - Zc
    y, x_k = 0.25 * Y / X, Z[2:11] / MASS_NUMBER / X
    He_pp = y if T > 1e5 else 0.0
    He_p = y - He_pp if T > 2e4 else 0.0
    x = {"H+": x_Hp, "H_2": x_H2, "H": 1 - 2 * x_H2 - x_Hp, "He": y - He_p - He_pp, "He+": He_p, "He++": He_pp,
         "C": (1 - f_Cp) * x_k[0], "C+": f_Cp * x_k[0]}
    x.update(dict(zip(METALS[1:], x_k[1:])))
    x["e-"] = x_Hp + He_p + 2 * He_pp + x["C+"]
    return x


def _jaco_thermo():
    eos = jaco_eos.EOS(list(SPECIES))
    args = [T_SYM] + [n_(s) for s in SPECIES]
    return [sp.lambdify(args, e, "numpy") for e in (eos.density, eos.internal_energy, eos.pressure)]


def _write_snapshot(path, cells, state_for_eos=None):
    """Snapshot of cells; state_for_eos(k, T, x) -> (T, x) is the state the EOS is evaluated at for cell k, where it
    differs from the one written (the negative controls)"""
    rho_f, u_f, P_f = _jaco_thermo()
    out = {k: [] for k in ("Temperature", "InternalEnergy", "Density", "Pressure", "ElectronAbundance",
                           "MolecularMassFraction", "HII", "Metallicity")}
    for k, (T, x_Hp, x_H2, f_Cp, Z) in enumerate(cells):
        x = _jaco_state(T, x_Hp, x_H2, f_Cp, Z)
        T_eos, x_eos = state_for_eos(k, T, dict(x)) if state_for_eos else (T, x)
        nn = [N_HTOT * x_eos[s] for s in SPECIES]
        rho, u, P = rho_f(T_eos, *nn), u_f(T_eos, *nn), P_f(T_eos, *nn)
        out["Temperature"].append(T)
        out["InternalEnergy"].append(u / UNIT_V**2)
        out["Density"].append(rho / UNIT_RHO)
        out["Pressure"].append(P / (UNIT_RHO * UNIT_V**2))
        out["ElectronAbundance"].append(x["e-"])
        x_H0 = 1 - x["H+"]
        out["MolecularMassFraction"].append(min(1, 2 * x["H_2"] / x_H0) if x_H0 > 0 else 0)  # as jaco_to_gizmo
        out["HII"].append(x["H+"])
        out["Metallicity"].append(np.concatenate([Z, np.zeros(3)]))  # + jet, wind, SN tracers
    with h5py.File(path, "w") as F:
        F.create_group("Header").attrs["UnitVelocity_In_CGS"] = UNIT_V
        g = F.create_group("PartType0")
        for name, v in out.items():
            g.create_dataset(name, data=np.array(v, dtype=np.float32))  # GIZMO writes single precision
    return str(path)


def _flagged(v):
    return {check: v[check][0] * v["N"][0] for check in ("u(T)", "nkT")}


def test_clean_snapshot_is_on_the_eos(tmp_path):
    """Every cell exactly on jaco's EOS, molecular gas partly ionized and SN ejecta included: no violation"""
    v = jaco_eos_violations(_write_snapshot(tmp_path / "clean.hdf5", CELLS))
    assert v["N"][0] == len(CELLS)
    assert _flagged(v) == {"u(T)": 0, "nkT": 0}, v


def test_h2_per_neutral_H_matters(tmp_path):
    """Read as H2 per H nucleus, the per-neutral-H fraction puts the partly ionized molecular cells off the EOS"""
    v = jaco_eos_violations(_write_snapshot(tmp_path / "clean.hdf5", CELLS), h2_per_neutral_H=False)
    assert _flagged(v) == {"u(T)": 3, "nkT": 3}, v


PARTLY_IONIZED_MOLECULAR, HOT, EJECTA_HOT = 2, 6, 7


@pytest.mark.parametrize(
    "label, cells, perturb, expected",
    [
        # pressure out of step with T and u (a P/u desynchronisation): the pressure check alone
        ("P", (0, HOT, EJECTA_HOT), lambda T, x: (T, x), {"u(T)": 0, "nkT": 3}),
        # T not the one that gives u: both checks
        ("T", (0, PARTLY_IONIZED_MOLECULAR), lambda T, x: (T * 1.003, x), {"u(T)": 2, "nkT": 2}),
        # a stale cached composition: the EOS at 5% fewer electrons than the cell reports (T, P and u still agree among
        # themselves), and at 2% less H2
        ("stale ions", (HOT, EJECTA_HOT), lambda T, x: (T, {**x, "e-": 0.95 * x["e-"], "H+": x["H+"] - 0.05 * x["e-"],
                                                           "H": x["H"] + 0.05 * x["e-"]}), {"u(T)": 2, "nkT": 2}),
        ("stale H2", (0, PARTLY_IONIZED_MOLECULAR), lambda T, x: (T, {**x, "H_2": 0.98 * x["H_2"],
                                                                      "H": x["H"] + 0.04 * x["H_2"]}),
         {"u(T)": 2, "nkT": 2}),
    ],
)
def test_negative_controls(tmp_path, label, cells, perturb, expected):
    """The EOS evaluated, in the given cells only, at a state other than the one the snapshot reports"""
    path = _write_snapshot(tmp_path / "perturbed.hdf5", CELLS,
                           lambda k, T, x: perturb(T, x) if k in cells else (T, x))
    if label == "P":
        with h5py.File(path, "r+") as F:
            P = F["PartType0/Pressure"]
            P[list(cells)] = P[list(cells)] * np.float32(1.003)
    v = jaco_eos_violations(path)
    assert _flagged(v) == expected, (label, v)
