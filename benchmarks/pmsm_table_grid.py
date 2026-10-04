"""pmsm_table_grid.py -- round 15, after the pre-registration (labelled so in vnet/PLAN.md): how fine a table the PMSM
heads models need.  Compiles each saved heads model (results/pmsm_heads_{size}_s{seed}.sldm) at several grids and
scores W2 like pmsm_heads.py (rms volts against ctrl_exact on the recorded runs 20-23).

  python benchmarks/pmsm_table_grid.py [sizes=medium]
Appends nothing; writes results/pmsm_table_grid.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.dirname(HERE)
REPO = os.path.dirname(LIB)
sys.path.insert(0, os.path.join(LIB, "python"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, HERE)
import pmsm_plant as P  # noqa: E402
import sldrnc  # noqa: E402
from pmsm_heads import rms, rows  # noqa: E402

kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SIZES = kw.get("sizes", "medium").split(",")
GRIDS = [[32, 32, 64], [48, 48, 64], [64, 64, 64], [64, 64, 128], [96, 96, 128], [128, 128, 128]]
RES = os.path.join(LIB, "results")


def main():
    z = np.load(os.path.join(REPO, "results", "pmsm_ssm_data_hs.npz"))
    W2X = [rows(z["U%d" % j][:, :7].astype(np.float64)) for j in range(20, 24)]
    W2Y = [P.ctrl_exact(z["U%d" % j][:, :7].astype(np.float64) * P.X_SCALE) for j in range(20, 24)]
    out = []
    for size in SIZES:
        for seed in (0, 1):
            m = sldrnc.Model.load(os.path.join(RES, "pmsm_heads_%s_s%d.sldm" % (size, seed)))
            net = rms([m.run(x).predictions for x in W2X], W2Y)
            print("%s seed %d: network W2 %.4f V" % (size, seed, net), flush=True)
            for g in GRIDS:
                t = m.compile_table(g)
                e = rms([t.run(x).predictions for x in W2X], W2Y)
                out.append({"size": size, "seed": seed, "grid": g, "network_w2": net, "table_w2": e, "bytes": t.info.table_bytes})
                print("   %-16s W2 %.4f V (%+.1f %%), %.1f MB" % ("x".join(map(str, g)), e, 100 * (e / net - 1), t.info.table_bytes / 1e6), flush=True)
    json.dump(out, open(os.path.join(RES, "pmsm_table_grid.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
