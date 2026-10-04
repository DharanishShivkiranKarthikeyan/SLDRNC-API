"""summarise_modes.py -- the per-mode ranges quoted in the README / whitepaper, from results/hyq_modes.json (trained on
Trot in Lab 2) and results/hyq_modes_lab1.json (trained on Trot in Lab 1).

  python benchmarks/summarise_modes.py
"""
import json
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p))["rows"] if os.path.exists(p) else []


def rng(v):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    lo, hi = min(v), max(v)
    f = (lambda x: "%.3f" % x) if hi >= 0.03 else (lambda x: "%.4f" % x)
    return f(lo) if f(lo) == f(hi) else "%s-%s" % (f(lo), f(hi))


def main():
    lab2, lab1 = load("hyq_modes.json"), load("hyq_modes_lab1.json")
    modes = ["RAW (your network)", "CORRECTION (your network + SLD-RNC)", "FULL small", "FULL medium", "FULL large"]
    cols = [("Lab 2 test", lab2, "test"), ("Lab 2 -> Lab 1", lab2, "other_recording"), ("Lab 1 test", lab1, "test"),
            ("Lab 1 -> Lab 2", lab1, "other_recording")]
    for key, label in (("uncorrected", "without self-correction"), ("corrected", "with self-correction (D = 1)"),
                       ("d50", "with self-correction, 50 ms ahead"), ("corrected_unprotected", "corrected, protection off")):
        print("\n" + label)
        print("  %-38s" % "" + "".join("%-18s" % c[0] for c in cols))
        for m in modes:
            print("  %-38s" % m + "".join("%-18s" % rng([r.get(s + "_" + key) for r in rows if r["mode"] == m]) for _, rows, s in cols))
    print("\nratios (per seed, matched by seed)")
    for _, rows, s in cols:
        raw = {r["seed"]: r[s + "_uncorrected"] for r in rows if r["mode"] == modes[0]}
        if not raw:
            continue
        full_u = [raw[r["seed"]] / r[s + "_uncorrected"] for r in rows if r["mode"].startswith("FULL")]
        full_c = [raw[r["seed"]] / r[s + "_corrected"] for r in rows if r["mode"].startswith("FULL")]
        corr = [raw[r["seed"]] / r[s + "_corrected"] for r in rows if r["mode"] == modes[1]]
        self_c = [r[s + "_uncorrected"] / r[s + "_corrected"] for r in rows if r["mode"].startswith("FULL")]
        prot = [r[s + "_corrected"] / r[s + "_corrected_unprotected"] - 1 for r in rows if r.get(s + "_corrected_unprotected")]
        print("  %-16s RAW/FULL %.2f-%.2f | RAW/FULL+corr %.1f-%.1f | RAW/CORRECTION %.2f-%.2f | FULL self-correction %.2f-%.2f | protection cost %+.0f%% to %+.0f%%" % (
            s if rows is lab2 else s + " (lab1)", min(full_u), max(full_u), min(full_c), max(full_c), min(corr), max(corr), min(self_c), max(self_c),
            100 * min(prot), 100 * max(prot)))
    print("\nsensor fault (trained on Lab 2, Lab 2 test)")
    for m in modes[1:]:
        rows = [r for r in lab2 if r["mode"] == m]
        print("  %-38s none %s | unprotected %s | protected %s" % (m, rng([r["fault_nocorrection"] for r in rows]),
                                                                  rng([r["fault_unprotected"] for r in rows]), rng([r["fault_protected"] for r in rows])))
    print("\nphysical units, corrected (HAA Nm / HFE+KFE N)")
    for _, rows, s in cols:
        for m in modes[1:]:
            rr = [r for r in rows if r["mode"] == m and r.get(s + "_rms_haa_Nm")]
            if rr:
                print("  %-16s %-38s %.2f-%.2f Nm, %.0f-%.0f N" % (s if rows is lab2 else s + " (lab1)", m, min(r[s + "_rms_haa_Nm"] for r in rr),
                      max(r[s + "_rms_haa_Nm"] for r in rr), min(r[s + "_rms_fe_N"] for r in rr), max(r[s + "_rms_fe_N"] for r in rr)))


if __name__ == "__main__":
    main()
