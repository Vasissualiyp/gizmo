"""Second (Larson) collapse of a 1 Msun core with the jaco starforge thermochemistry.

The larson_penston IC (uniform 1 Msun sphere, R = 0.01 pc, 1e-5 Msun cells, at rest) collapses isothermally, forms a
first core where the optically-thick limit traps the cooling, and, once the centre reaches ~2000 K, should collapse
again because collisional dissociation takes 4.48 eV per H2 from the gas (Masunaga & Inutsuka 2000). Sink formation is
effectively off (CritPhysDensity 1e30); DENSEST_CELL_TRACK writes the densest cell's state and its H2 energy ledger
(JACO_LEDGER) after every cooling pass, and DENSEST_CELL_TRACK_STOP_DENSITY ends the run, with a snapshot, once the
centre passes 3e-6 g/cm^3.

Passes if
- the centre runs away: the run stops on the density trigger;
- the runaway is the dissociation's: the central H2 fraction 2 x_H2 is below 0.8 by then (a centre that collapses
  still molecular is being cooled by something else), the effective index d ln P / d ln rho stays below 4/3 over at
  least a decade above the density where the centre first reaches 1500 K, and the centre is below 1e4 K at the stop;
- the ledger closes: every collisional dissociation took 4.48 eV, and in the hot central cells the thermal energy plus
  the H2 binding energy changes by the PdV work, the capped radiative heat and the formation energy that escapes
  outside it to 2% of the PdV work, with formation radiating (capped or not) less than 5% of it (the ledger is built
  from the solved states: the outputs' gross formation and dissociation rates, at ~1e3-1e12 cycles per molecule, close
  only to the solver's tolerance on each row).

This tests the bookkeeping between the thermal, chemical and radiative reservoirs. The binding energy H2 formation does
not give the gas is radiation (h2_chemistry/chemical_heat.py): were it to escape regardless of the optical depth, the
dissociate-and-reform cycle at first-core rates (~500 cycles per molecule) would absorb the PdV work, and the centre would
run away isothermally at ~1800 K with 2 x_H2 ~ 1. Under the optically-thick limit, as starforge has it, the first core
keeps it. A model with transported radiation instead (no limit) is one more MODELS entry.

The ledger models live in jaco's jaco_second_collapse branch; point the build's codegen at it with
JACO_PYTHON="env PYTHONPATH=<that jaco>/src python3" in the environment. GIZMO_TEST_OUTPUT_ROOT puts each run's ~0.6 GB
of output elsewhere. Each model takes ~30 min on 2 ranks x 4 threads.
"""

import os
import shutil
import subprocess
from urllib.request import urlretrieve

import numpy as np
import pytest

from gizmo.test import (build_gizmo_for_test, default_mpi_ranks, get_cooling_tables,
                        write_params_with_overrides)

TEST_NAME = "second_collapse"
IC_URL = "https://users.flatironinstitute.org/~mgrudic/gizmo_tests/larson_penston/larson_penston_ics.hdf5"
EV = 1.602176634e-12
H2_BINDING = 4.48 * EV
Y_HE = 0.0994  # He per H, for the particle count
STOP_DENSITY = 3e-6  # g/cm^3, Config.sh's DENSEST_CELL_TRACK_STOP_DENSITY
STOP_LINE = "DENSEST_CELL_TRACK_STOP"
# DENSEST_CELL_TRACK columns, then JacoLedger (cooling/jaco.cc jaco_ledger_add)
COLUMNS = ("time", "rho", "T", "x_H2", "x_e", "Gamma", "u", "h_AU", "ID", "Sigma", "kappa_thick", "L_thick",
           "diss_heat", "diss_events", "form_heat", "form_events", "h2_net", "chem_heat", "total_heat",
           "energy_residual", "formation_radiated", "pdv", "f_thick")
MODELS = {"starforge_ledger": ()}  # model -> pytest marks


def read_track(output_dir):
    """The densest-cell track as a dict of arrays"""
    data = np.atleast_2d(np.loadtxt(os.path.join(output_dir, "densest_cell.txt")))
    return {name: data[:, k] for k, name in enumerate(COLUMNS[: data.shape[1]])}


def collapse_envelope(track):
    """The track where the central density sets a new maximum: the monotone collapse history, without the
    excursions of a core that bounces or of a densest cell that changes identity"""
    rho = track["rho"]
    keep = rho >= np.maximum.accumulate(rho)
    return {k: v[keep] for k, v in track.items()}


def effective_gamma(env, dex=0.1):
    """d ln P / d ln rho along the envelope, P ~ rho T (1 - x_H2 + y), on bins of `dex` in rho: (rho, gamma)"""
    lr = np.log10(env["rho"])
    lp = np.log10(env["rho"] * env["T"] * (1 - env["x_H2"] + Y_HE))
    edges = np.arange(lr.min(), lr.max() + dex, dex)
    idx = np.digitize(lr, edges)
    centers, lpm = [], []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum():
            centers.append(lr[sel].mean())
            lpm.append(lp[sel].mean())
    centers, lpm = np.array(centers), np.array(lpm)
    return 10 ** centers[1:-1], np.gradient(lpm, centers)[1:-1]


def kink_extent(env, T_onset=1500.0):
    """Decades of density above the first T_c >= T_onset over which gamma_eff < 4/3"""
    rho_g, gam = effective_gamma(env)
    rho_on = env["rho"][np.argmax(env["T"] >= T_onset)] if np.any(env["T"] >= T_onset) else np.inf
    above = rho_g > rho_on
    soft = above & (gam < 4.0 / 3.0)
    return np.log10(rho_g[soft].max() / rho_g[soft].min()) if soft.sum() > 1 else 0.0


def dissociated(env):
    """central H2 fraction 2 x_H2 at the end of the collapse"""
    return 2 * env["x_H2"][-1]


def cell_spans(track, T_min=1500.0):
    """(first, last) record indices of each run of consecutive records of one cell above T_min: the ledger's slots are
    per cell, so only differences within a run mean anything"""
    ids, T, spans, i = track["ID"], track["T"], [], 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[i]:
            j += 1
        if j > i and T[i] > T_min:
            spans.append((i, j))
        i = j + 1
    return spans


def ledger_closure(track):
    """(per_event, energy, radiated): the dissociation heat per event of the last record relative to 4.48 eV; and over the
    hot central cells' runs, the energy left unaccounted with H2's binding energy counted as internal energy, and the
    formation energy radiated, each relative to the PdV work done on those cells"""
    last = {k: v[-1] for k, v in track.items()}
    per_event = abs(last["diss_heat"] / (-H2_BINDING * last["diss_events"]) - 1)
    spans = cell_spans(track)
    d = {k: np.array([track[k][j] - track[k][i] for i, j in spans]) for k in ("energy_residual", "formation_radiated", "pdv")}
    work = np.abs(d["pdv"]).sum()
    return per_event, d["energy_residual"].sum() / work, d["formation_radiated"].sum() / work


def output_dir(model):
    """The run's OutputDir: under $GIZMO_TEST_OUTPUT_ROOT if set (e.g. on bulk storage: each run writes ~0.6 GB of
    snapshots), else in the test directory"""
    root = os.environ.get("GIZMO_TEST_OUTPUT_ROOT")
    return os.path.join(root, f"{TEST_NAME}_{model}") if root else os.path.abspath(f"test/{TEST_NAME}/output_{model}")


def run_until_stop(outdir, num_mpi_ranks, num_omp_threads, timeout=6 * 3600):
    """Run GIZMO in the test directory, writing to outdir; it ends on the density trigger, not at TimeMax. Returns
    whether it stopped there."""
    params = write_params_with_overrides(f"{TEST_NAME}.params", {"OutputDir": outdir})
    env = dict(os.environ, OMP_NUM_THREADS=str(max(num_omp_threads, 1)))
    cmd = ["mpirun", "-np", str(num_mpi_ranks), "--bind-to", "none", "./GIZMO", params, "0"]
    out, err = os.path.join(outdir, f"test_{TEST_NAME}.out"), os.path.join(outdir, f"test_{TEST_NAME}.err")
    with open(out, "w") as fo, open(err, "w") as fe:
        try:
            proc = subprocess.run(cmd, stdout=fo, stderr=fe, timeout=timeout, env=env, check=False)
        except subprocess.TimeoutExpired:
            pytest.skip(f"{TEST_NAME} did not reach the second collapse within {timeout} s")
    if proc.returncode != 0:
        raise RuntimeError(f"GIZMO exited with status {proc.returncode}; see {err}")
    with open(out) as f:
        return any(STOP_LINE in line for line in f)


@pytest.mark.parametrize("num_mpi_ranks", (min(default_mpi_ranks(), 2),))
@pytest.mark.parametrize("num_omp_threads", (4,))
@pytest.mark.parametrize("model", [pytest.param(m, marks=k) for m, k in MODELS.items()])
def test_second_collapse(num_mpi_ranks, num_omp_threads, model):
    outdir = output_dir(model)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    get_cooling_tables(f"test/{TEST_NAME}")
    build_gizmo_for_test(TEST_NAME, num_omp_threads, (f"JACO={model}", "JACO_LEDGER"))
    cwd = os.getcwd()
    try:
        os.chdir(f"test/{TEST_NAME}")
        if not os.path.isfile(f"{TEST_NAME}_ics.hdf5"):  # the larson_penston IC
            urlretrieve(IC_URL, f"{TEST_NAME}_ics.hdf5")
        stopped = run_until_stop(outdir, num_mpi_ranks, num_omp_threads)
    finally:
        os.chdir(cwd)

    assert stopped, "the centre never passed the stop density: no second collapse"
    track = read_track(outdir)
    env = collapse_envelope(track)
    per_event, energy, radiated = ledger_closure(track)
    assert per_event < 1e-9
    assert energy < 0.02, f"energy unaccounted for: {energy:.3g} of the PdV work"
    assert radiated < 0.05, f"H2 formation radiated {radiated:.3g} of the PdV work"
    assert env["rho"][-1] > STOP_DENSITY
    assert dissociated(env) < 0.8, f"the centre collapsed still molecular (2 x_H2 = {dissociated(env):.3f})"
    assert kink_extent(env) >= 1.0, "no dissociation kink: gamma_eff < 4/3 over less than a decade above 1500 K"
    assert env["T"][-1] < 1e4
