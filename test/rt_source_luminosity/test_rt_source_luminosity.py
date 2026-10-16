"""rt_get_source_luminosity must not return garbage for a gas source: its callers pass uninitialized arrays."""

import os
import shlex
import subprocess
from os.path import abspath, dirname, join

import pytest

from gizmo.test import build_gizmo_for_test

TEST_NAME = "rt_source_luminosity"
TEST_DIR = dirname(abspath(__file__))
REPO = abspath(join(TEST_DIR, "..", ".."))


def link_driver(test_dir, source, exe_name):
    """Link a driver against the objects of the build that just finished, in place of core/main.o."""
    build_dir = join(test_dir, "output")
    os.makedirs(build_dir, exist_ok=True)
    dry = subprocess.run(["make", "-n", "-B", "GIZMO"], capture_output=True, text=True, cwd=REPO).stdout.splitlines()
    compile_main = [l for l in dry if " -c core/main.cc" in l]
    link = [l for l in dry if "-o GIZMO" in l]
    assert compile_main and link, "could not read the compile/link commands from make -n"
    obj, exe = join(build_dir, exe_name + ".o"), join(build_dir, exe_name)
    cc = [t for t in shlex.split(compile_main[-1]) if t != "-ffast-math"]  # the driver's isfinite() must work
    cc[cc.index("core/main.cc")] = join(test_dir, source)
    cc[cc.index("core/main.o")] = obj
    cc.insert(1, "-I" + REPO)
    ln = shlex.split(link[-1])
    ln[ln.index("core/main.o")] = obj
    ln[ln.index("GIZMO")] = exe
    for cmd in (cc, ln):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, "driver build failed:\n%s\n%s" % (" ".join(cmd), r.stderr[-4000:])
    return exe


def test_rt_source_luminosity():
    if os.environ.get("GIZMO_TEST_SKIP_BUILD_RUN"):
        pytest.skip("builds GIZMO; nothing to validate from external snapshots")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        build_gizmo_for_test(TEST_NAME, post_build=lambda: link_driver(TEST_DIR, "rt_source_luminosity_driver.cc", "rt_source_luminosity_driver"))
    finally:
        os.chdir(cwd)
    r = subprocess.run([join(TEST_DIR, "output", "rt_source_luminosity_driver"), f"{TEST_NAME}.params"], capture_output=True, text=True, cwd=TEST_DIR)
    lines = r.stdout.splitlines()
    assert any(l.startswith("DONE") for l in lines), "driver did not finish (exit %d):\n%s" % (r.returncode, r.stderr[-3000:])
    failed = [l for l in lines if l.startswith("FAIL")]
    assert not failed, "\n".join(failed)
