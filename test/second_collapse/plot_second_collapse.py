"""Central rho-T, H2 fraction and effective adiabatic index of second_collapse runs, from each run's
output/densest_cell.txt, against one-zone adiabats of the same gas (scratch reference, optional .npz).

    python plot_second_collapse.py out.png label=run_dir [label=run_dir ...] [--onezone onezone.npz]
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "..", "python_src")]  # this tree's gizmo, as test/conftest.py does
from test_second_collapse import COLUMNS, collapse_envelope, effective_gamma  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5, fixed order
TRACK_EXTRA = ("Sigma", "kappa_thick", "L_thick")  # written by newer builds only


def read_track(output_dir):
    data = np.atleast_2d(np.loadtxt(f"{output_dir}/densest_cell.txt"))
    cols = COLUMNS if data.shape[1] in (len(COLUMNS), 12) else [c for c in COLUMNS if c not in TRACK_EXTRA]
    return {c: data[:, k] for k, c in enumerate(cols[: data.shape[1]])}


def main(argv):
    out, runs, onezone = argv[0], [], None
    args = argv[1:]
    if "--onezone" in args:
        k = args.index("--onezone")
        onezone = np.load(args[k + 1])
        args = args[:k] + args[k + 2:]
    runs = [a.split("=", 1) for a in args]

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 9), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 1], hspace=0.06))
    ax_T, ax_x, ax_g = axes
    if onezone is not None:
        for key, ls, lab in (("jaco", "--", "one-zone adiabat, 4.48 eV/H$_2$"), ("jaco_nobind", ":", "one-zone adiabat, no binding energy")):
            a, g = onezone[key], onezone[key + "_gamma"]
            ax_T.loglog(a[:, 0], a[:, 1], ls, color="0.55", lw=1.2, label=lab)
            ax_x.semilogx(a[:, 0], 2 * a[:, 2], ls, color="0.55", lw=1.2)
            ax_g.semilogx(a[:, 0], g, ls, color="0.55", lw=1.2)
    for (label, d), c in zip(runs, COLORS):
        env = collapse_envelope(read_track(f"{d}/output"))
        ax_T.loglog(env["rho"], env["T"], color=c, lw=2, label=label)
        ax_x.semilogx(env["rho"], 2 * env["x_H2"], color=c, lw=2)
        rho_g, gam = effective_gamma(env)
        ax_g.semilogx(rho_g, gam, color=c, lw=2)
    ax_g.axhline(4 / 3, color="0.3", lw=0.8)
    ax_g.text(1.5e-17, 4 / 3 + 0.02, "4/3", color="0.3", fontsize=8)
    ax_T.set_ylabel("central T [K]")
    ax_x.set_ylabel("central 2$x_{\\rm H_2}$")
    ax_g.set_ylabel("$d\\ln P/d\\ln\\rho$")
    ax_g.set_xlabel("central $\\rho$ [g cm$^{-3}$]")
    ax_g.set_ylim(0.9, 1.8)
    ax_x.set_ylim(-0.03, 1.03)
    ax_T.set_xlim(1e-17, 1e-4)
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax_T.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1:])
