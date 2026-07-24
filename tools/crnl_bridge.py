"""CRNL bridge: does EIR's field relaxation obey the physics of signal restoration?

EIR relaxes discrete logic into a continuous energy field, anneals noise, and
decodes by argmax. CRNL (../CRNL) studies exactly that move as *physics*: when
does a noisy relaxation reliably land in the intended basin? CRNL's two measured
phenomena translate into concrete EIR tuning questions:

  freeze-out   CRNL: dilute the restoring reactions faster than a critical rate and
               the decision freezes half-made. EIR: quench the noise faster than
               some critical schedule and the planner freezes on a wrong decode.
               Mapping: annealing budget `steps` ~ 1/H.

  radix        CRNL: more symbols crowd the basins, so the per-symbol noise margin
               falls. EIR: more discrete options should mean less reliable decode
               at fixed effort.

Test problem: one DiscreteVar with n options, OneHotFactor (n winner-take-all
basins) + ActionBiasFactor giving option 0 a cost advantage `delta`. Option 0 is
the unique global minimum, so a correct decode is unambiguous.

    Q = (P_correct - 1/n) / (1 - 1/n)     0 = random/frozen, 1 = perfect decode

TWO EXPERIMENTS

  --mode budget   sweep annealing budget x n at fixed hyperparameters.
                  Answers: is there a freeze-out-like anneal threshold?

  --mode radix    the CONTROLLED radix test. An earlier version of this bridge
                  swept n at FIXED hyperparameters and reported a decode penalty
                  -- but that conflates a real radix effect with hyperparameters
                  that merely happened to suit small n. Here every n gets its own
                  (lr, noise_start) tuned over a grid, and we report the best each
                  n can do. If the penalty survives per-n tuning it is real; if it
                  vanishes, the earlier claim was a tuning artifact.

Run with the shared Work venv (it has eir installed editable):

    /home/pantelis/Desktop/Projects/Work/venv/bin/python tools/crnl_bridge.py --quick
    /home/pantelis/Desktop/Projects/Work/venv/bin/python tools/crnl_bridge.py --mode radix
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from eir.core import EIRGraph, DiscreteVar
from eir.factors.onehot import OneHotFactor
from eir.factors.bias import ActionBiasFactor
from eir.backends.field import FieldBackend


def decode_reliability(n, delta, steps, lr, noise_start, trials, base_seed,
                       noise_end=0.002, spsa_samples=6):
    """Fraction of runs whose argmax decode is the true minimiser (option 0)."""
    costs = [0.0] + [delta] * (n - 1)
    ok = 0
    for t in range(trials):
        np.random.seed(base_seed + t)          # FieldBackend uses global numpy RNG
        X = DiscreteVar("X", n=n)
        graph = EIRGraph(vars=[X],
                         factors=[OneHotFactor([X]),
                                  ActionBiasFactor([X], costs=costs)])
        res = FieldBackend(steps=steps, lr=lr, noise_start=noise_start,
                           noise_end=noise_end,
                           spsa_samples=spsa_samples).solve(graph)
        ok += (res.assignment["X"] == 0)
    return ok / trials


def quality(p, n):
    """Normalise decode reliability so different n are comparable."""
    return (p - 1.0 / n) / (1.0 - 1.0 / n)


# --------------------------------------------------------------------------- #
# mode: budget -- the freeze-out analogue                                     #
# --------------------------------------------------------------------------- #

def _budget_worker(args):
    n, steps, delta, lr, noise, trials, seed = args
    p = decode_reliability(n, delta, steps, lr, noise, trials, seed)
    return {"n": n, "steps": steps, "p": p, "q": quality(p, n)}


def run_budget(ns, step_grid, delta, lr, noise, trials, seed, jobs):
    tasks = [(n, s, delta, lr, noise, trials, seed) for n in ns for s in step_grid]
    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            res = pool.map(_budget_worker, tasks)
    else:
        res = [_budget_worker(t) for t in tasks]
    by_n = {}
    for r in res:
        by_n.setdefault(r["n"], []).append(r)
    for n in by_n:
        by_n[n].sort(key=lambda r: r["steps"])
    return by_n


# --------------------------------------------------------------------------- #
# mode: radix -- the CONTROLLED test (per-n hyperparameter tuning)             #
# --------------------------------------------------------------------------- #

def _tune_worker(args):
    n, delta, steps, lr, noise, trials, seed = args
    p = decode_reliability(n, delta, steps, lr, noise, trials, seed)
    return {"n": n, "lr": lr, "noise_start": noise, "p": p}


def run_radix_controlled(ns, delta, steps, lrs, noises, trials, seed, jobs):
    """For each n, take the BEST decode reliability over the tuning grid."""
    tasks = [(n, delta, steps, lr, noise, trials, seed)
             for n in ns for lr, noise in itertools.product(lrs, noises)]
    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            res = pool.map(_tune_worker, tasks)
    else:
        res = [_tune_worker(t) for t in tasks]
    by_n = {}
    for r in res:
        by_n.setdefault(r["n"], []).append(r)
    out = {}
    for n, rows in by_n.items():
        best = max(rows, key=lambda r: r["p"])
        untuned = [r for r in rows if r["lr"] == lrs[0] and r["noise_start"] == noises[0]]
        out[n] = {
            "n": n,
            "all": rows,
            "best": best,
            "p_tuned": best["p"],
            "q_tuned": quality(best["p"], n),
            "p_untuned": untuned[0]["p"] if untuned else float("nan"),
            "q_untuned": quality(untuned[0]["p"], n) if untuned else float("nan"),
        }
    return out


# --------------------------------------------------------------------------- #

def figure_budget(by_n, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted(by_n)
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(ns)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for c, n in zip(colors, ns):
        rows = by_n[n]
        ax1.semilogx([r["steps"] for r in rows], [r["q"] for r in rows],
                     "o-", color=c, label=f"n={n}")
        ax2.semilogx([r["steps"] for r in rows], [r["p"] for r in rows],
                     "o-", color=c, label=f"n={n}")
        ax2.axhline(1.0 / n, color=c, lw=0.6, ls=":")
    ax1.set_xlabel("annealing budget (steps ~ 1/H)")
    ax1.set_ylabel("decision quality Q")
    ax1.set_title("EIR freeze-out analogue:\nfast quench → frozen decode")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, which="both", alpha=0.25)
    ax2.set_xlabel("annealing budget (steps)")
    ax2.set_ylabel("P(correct decode)")
    ax2.set_title("Raw decode success\n(dotted = random baseline 1/n)")
    ax2.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote figure -> {out_path}")


def figure_radix(tuned, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted(tuned)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.semilogx(ns, [tuned[n]["p_untuned"] for n in ns], "s--", color="#7f7f7f",
                 label="fixed hyperparameters (uncontrolled)")
    ax1.semilogx(ns, [tuned[n]["p_tuned"] for n in ns], "o-", color="#1f77b4",
                 label="best over per-n tuning (controlled)")
    ax1.semilogx(ns, [1.0 / n for n in ns], "k:", label="random 1/n")
    ax1.set_xlabel("number of options n")
    ax1.set_ylabel("P(correct decode)")
    ax1.set_title("Controlled radix test\ndoes the penalty survive tuning?")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogx(ns, [tuned[n]["q_untuned"] for n in ns], "s--", color="#7f7f7f",
                 label="uncontrolled")
    ax2.semilogx(ns, [tuned[n]["q_tuned"] for n in ns], "o-", color="#1f77b4",
                 label="tuned per n")
    ax2.set_xlabel("number of options n")
    ax2.set_ylabel("decision quality Q")
    ax2.set_title("Normalised quality\n(0 = random, 1 = perfect)")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote figure -> {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["budget", "radix", "both"], default="both")
    p.add_argument("--ns", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 16, 24, 32])
    p.add_argument("--delta", type=float, default=0.1, help="energy margin of option 0")
    p.add_argument("--steps", type=int, nargs="+",
                   default=[4, 8, 16, 32, 64, 128, 256, 512],
                   help="annealing budgets for --mode budget")
    p.add_argument("--tune-steps", type=int, default=384,
                   help="fixed large budget used by the controlled radix test")
    p.add_argument("--lrs", type=float, nargs="+", default=[0.05, 0.1, 0.3])
    p.add_argument("--noises", type=float, nargs="+", default=[0.05, 0.15, 0.4])
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--outdir",
                   default=os.path.join(here, os.pardir, "results"),
                   help="where figures/data land (tools/ stays scripts-only)")
    args = p.parse_args()

    if args.quick:
        args.ns = [2, 8, 32]
        args.steps = [8, 64, 512]
        args.trials = 120
        args.lrs, args.noises = [0.1, 0.3], [0.05, 0.15]

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    results = {}

    if args.mode in ("budget", "both"):
        print(f"[budget] annealing sweep  (delta={args.delta}, "
              f"trials={args.trials}, jobs={args.jobs})")
        by_n = run_budget(args.ns, args.steps, args.delta, args.lrs[1] if
                          len(args.lrs) > 1 else args.lrs[0],
                          0.15, args.trials, args.seed, args.jobs)
        print(f"{'steps':>6}  " + "  ".join(f"n={n:<3}" for n in sorted(by_n)))
        for j, s in enumerate(args.steps):
            print(f"{s:>6}  " + "  ".join(f"{by_n[n][j]['p']:5.2f}"
                                          for n in sorted(by_n)))
        results["budget"] = {str(n): rows for n, rows in by_n.items()}
        figure_budget(by_n, os.path.join(args.outdir, "crnl_bridge_budget.png"))

    if args.mode in ("radix", "both"):
        print(f"\n[radix] CONTROLLED test -- per-n tuning over "
              f"{len(args.lrs)}x{len(args.noises)} grid at steps={args.tune_steps}")
        tuned = run_radix_controlled(args.ns, args.delta, args.tune_steps,
                                     args.lrs, args.noises, args.trials,
                                     args.seed, args.jobs)
        print(f"{'n':>4} {'untuned':>9} {'tuned':>9} {'gain':>7} {'Q_tuned':>9}  best (lr, noise)")
        for n in sorted(tuned):
            r = tuned[n]
            print(f"{n:>4} {r['p_untuned']:>9.3f} {r['p_tuned']:>9.3f} "
                  f"{r['p_tuned']-r['p_untuned']:>+7.3f} {r['q_tuned']:>9.3f}  "
                  f"({r['best']['lr']}, {r['best']['noise_start']})")
        results["radix"] = {str(n): {k: v for k, v in r.items() if k != "all"}
                            for n, r in tuned.items()}
        results["radix_grid"] = {str(n): r["all"] for n, r in tuned.items()}
        figure_radix(tuned, os.path.join(args.outdir, "crnl_bridge_radix.png"))

    data_path = os.path.join(args.outdir, "crnl_bridge_results.json")
    with open(data_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote data -> {data_path}   (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
