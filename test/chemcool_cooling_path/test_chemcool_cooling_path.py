"""CHEMCOOL through the standard cooling path (cooling/cooling.cc, Stella Config without COOLING_OPERATOR_SPLIT, with
SINK_THERMALFEEDBACK). The driver builds a real cell with begrun() on a small Stella box and checks that the cooling path
deposits sink thermal energy, applies unsplit hydro heating, and returns the chemistry's electron abundance.
With CHEMCOOL_LEGACY_COOLING all three fail: the legacy path bypasses them.
"""

import os
import shlex
import subprocess
import tempfile
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test

TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))
NAME = "chemcool_cooling_path"


def _link_driver():
    """Link the driver against the objects of the build that just finished, in place of core/main.o."""
    out = join(TEST_DIR, "output")
    os.makedirs(out, exist_ok=True)
    dry = subprocess.run(["make", "-n", "-B", "GIZMO"], capture_output=True, text=True, cwd=REPO).stdout.splitlines()
    compile_main = [l for l in dry if " -c core/main.cc" in l]
    link = [l for l in dry if "-o GIZMO" in l]
    assert compile_main and link, "could not read the compile/link commands from make -n"
    obj, exe = join(out, f"{NAME}_driver.o"), join(out, f"{NAME}_driver")
    cc = shlex.split(compile_main[-1])
    cc[cc.index("core/main.cc")] = join(TEST_DIR, f"{NAME}_driver.cc")
    cc[cc.index("core/main.o")] = obj
    cc.insert(1, "-I" + REPO)
    ln = shlex.split(link[-1])
    ln[ln.index("core/main.o")] = obj
    ln[ln.index("GIZMO")] = exe
    for cmd in (cc, ln):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, "driver build failed:\n%s\n%s" % (" ".join(cmd), r.stderr[-4000:])


def test_chemcool_cooling_path():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(NAME, post_build=_link_driver)
    finally:
        os.chdir(cwd)
    # outside the test tree: the gizmo/ link to the repo root would otherwise make pytest collect the whole suite
    run = tempfile.mkdtemp(prefix=f"{NAME}_")
    os.makedirs(join(run, "output"))
    os.symlink(REPO, join(run, "gizmo"))  # CHEMCOOL and the stellar tables open their data under gizmo/
    subprocess.run(["python3", join(TEST_DIR, f"make_{NAME}_ics.py"), join(run, f"{NAME}_ics.hdf5")], check=True)
    r = subprocess.run(["mpirun", "-np", "1", join(TEST_DIR, "output", f"{NAME}_driver"), join(TEST_DIR, f"{NAME}.params")],
                       capture_output=True, text=True, cwd=run)
    with open(join(TEST_DIR, "output", "checks.txt"), "w") as f:
        f.write(r.stdout)
    lines = r.stdout.splitlines()
    assert [l for l in lines if l.startswith("DONE")], "driver did not finish (exit %d):\n%s\n%s" % (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    failed = [l for l in lines if l.startswith("FAIL")]
    assert not failed, "%d cooling-path checks failed:\n%s" % (len(failed), "\n".join(failed))
