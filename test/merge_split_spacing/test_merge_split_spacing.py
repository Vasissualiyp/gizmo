"""Split placement: each daughter sits within 0.4 of the distance to its parent's nearest neighbour.

Pairs of over-massive cells sit PAIR_SEPARATION apart inside lattice voids, so each one's nearest neighbour is
its partner, which is splitting too. The nearest-neighbour search in merge_and_split_particles used to skip
neighbours already flagged to split or merge, so the second member of each pair measured its distance to a
lattice node about four times farther away and could place its daughters onto its partner's.
The run is a few steps of a static, uniform-pressure box, so the cells do not move.
"""

import glob
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gizmo.test import build_and_run_test
from make_merge_split_spacing_ics import PAIR_SEPARATION, make_ics

TEST_NAME = "merge_split_spacing"
TEST_DIR = f"test/{TEST_NAME}"


def _wrap(d, box):
    return d - box * np.round(d / box)


def test_merge_split_spacing():
    pairs = make_ics(f"{TEST_DIR}/{TEST_NAME}_ics.hdf5")
    build_and_run_test(TEST_NAME)
    parent_ids = [pid for id_a, _, id_b, _ in pairs.values() for pid in (id_a, id_b)]
    split_snap = None
    for snap in sorted(glob.glob(f"{TEST_DIR}/output/snapshot_*.hdf5")):  # the first output after the split
        with h5py.File(snap, "r") as f:
            ids = f["PartType0/ParticleIDs"][:]
        if all((ids == pid).sum() == 2 for pid in parent_ids):
            split_snap = snap
            break
    assert split_snap, "not every over-massive cell split exactly once"
    with h5py.File(split_snap, "r") as f:
        ids, pos = f["PartType0/ParticleIDs"][:], f["PartType0/Coordinates"][:]
        box = f["Header"].attrs["BoxSize"]

    def at_split(pid):
        """(parent position, daughter offset) at the split: daughters sit symmetrically about the parent, so
        measuring from their midpoint and their half-separation cancels any motion before and after."""
        d1, d2 = pos[ids == pid]
        half = 0.5 * _wrap(d2 - d1, box)
        return d1 + half, np.linalg.norm(half)

    worst_offset, closest = 0.0, np.inf
    for id_a, _, id_b, _ in pairs.values():
        (xa, off_a), (xb, off_b) = at_split(id_a), at_split(id_b)
        r_near = np.linalg.norm(_wrap(xa - xb, box))  # each one's nearest neighbour is its partner
        worst_offset = max(worst_offset, off_a / r_near, off_b / r_near)
        for pa in pos[ids == id_a]:
            for pb in pos[ids == id_b]:
                closest = min(closest, np.linalg.norm(_wrap(pa - pb, box)) / r_near)
    # each offset is capped at 0.4 of the pre-split distance, so no two daughters of a pair come closer than 0.2 of it
    assert worst_offset <= 0.4 * 1.05, f"a split offset its daughters {worst_offset:.3f} x the distance to its nearest neighbour (limit 0.4)"
    assert closest >= 0.2 * 0.95, f"two daughters of neighbouring cells came within {closest:.3f} x their parents' distance (limit 0.2)"
