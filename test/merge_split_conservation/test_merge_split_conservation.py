"""Conservation and memory-safety test for particle merging and splitting (mesh/merge_split.cc).

Builds GIZMO, links merge_split_driver.cc against its objects, and runs the production split_particle_i,
merge_particles_ij and merge_and_split_particles on cells set up by hand. The driver checks, to 1e-13:

  split / merge   mass, momentum, metal mass per species, pending diffusive metal exchange, dust mass,
                  conserved B, the Injected_Sink_Energy reservoir, and H2 and CO mass when the two cells'
                  hydrogen fractions differ (TracAbund are ratios to n_H, so a merge that mixes up the cells'
                  X_H changes the molecular mass)
  split           daughters straddle the parent, each offset within 0.4 of the nearest-neighbour distance
  X_H             particle_hydrogen_massfrac() for the compiled metallicity layout, and not total Z
  non-gas         star-star merges and zero-mass early returns at indices past MaxPartGas write no CellP
  SN spawn        mass moves star -> cell exactly, also when the star holds less than its ejecta; the cell
                  carries only ejecta (no template pending exchange or dust, B scaled to its mass) and is
                  woken; with no room for gas the spawn waits instead of writing CellP past MaxPartGas

CellP is allocated with a canary region past MaxPartGas, so a write for a non-gas index is caught rather than
silently corrupting memory. No simulation is run; the build is the only slow part.

Two configurations: this directory (Stella: individual elements, CHEMCOOL network 17, dust, MHD, diffusion,
SN spawn) and merge_split_conservation_starforge (STARFORGE defaults with SINK_THERMALFEEDBACK).
"""

import os
import shlex
import subprocess
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test, get_cooling_tables

DRIVER_SRC = join(dirname(abspath(__file__)), "merge_split_driver.cc")
REPO = abspath(join(dirname(abspath(__file__)), "..", ".."))


def _link_driver(test_dir):
    """Compile and link the driver against the objects of the build that just finished (runs under the build lock)."""
    build_dir = join(test_dir, "output")
    os.makedirs(build_dir, exist_ok=True)
    dry = subprocess.run(["make", "-n", "-B", "GIZMO"], capture_output=True, text=True, cwd=REPO).stdout.splitlines()
    compile_main = [l for l in dry if " -c core/main.cc" in l]
    link = [l for l in dry if "-o GIZMO" in l]
    assert compile_main and link, "could not read the compile/link commands from make -n"
    obj, exe = join(build_dir, "merge_split_driver.o"), join(build_dir, "merge_split_driver")
    cc = shlex.split(compile_main[-1])
    cc[cc.index("core/main.cc")] = DRIVER_SRC
    cc[cc.index("core/main.o")] = obj
    cc.insert(1, "-I" + REPO)
    ln = shlex.split(link[-1])
    ln[ln.index("core/main.o")] = obj
    ln[ln.index("GIZMO")] = exe
    for cmd in (cc, ln):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, "driver build failed:\n%s\n%s" % (" ".join(cmd), r.stderr[-4000:])


def run_merge_split_checks(test_name, cooling_tables=False):
    """Build test_name's configuration, run the driver, and fail on any failed check. cooling_tables: the build
    uses the tabulated (non-CHEMCOOL) cooling, whose tables the merge's EOS call reads."""
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    test_dir = join(REPO, "test", test_name)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(test_name, post_build=lambda: _link_driver(test_dir))
    finally:
        os.chdir(cwd)
    if cooling_tables:
        get_cooling_tables(test_dir)
    r = subprocess.run([join(test_dir, "output", "merge_split_driver"), f"{test_name}.params"],
                       capture_output=True, text=True, cwd=test_dir)
    lines = r.stdout.splitlines()
    with open(join(test_dir, "output", "checks.txt"), "w") as f:
        f.write(r.stdout)
    done = [l for l in lines if l.startswith("DONE")]
    assert done, "driver did not finish (exit %d):\n%s\n%s" % (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    failed = [l for l in lines if l.startswith("FAIL")]
    assert not failed, "%d of %d merge/split checks failed:\n%s" % (len(failed), sum(l[:4] in ("PASS", "FAIL") for l in lines), "\n".join(failed))


def test_merge_split_conservation():
    run_merge_split_checks("merge_split_conservation")
