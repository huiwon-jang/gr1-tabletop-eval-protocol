#!/usr/bin/env python
"""Scan a finished RoboCasa Kitchen run, quarantine contaminated episodes, and
re-run exactly those slots with one environment per process.

Three subcommands, meant to be run in order:

  scan    judge every mp4, write ``swap_manifest.json`` and append to
          ``swap_cycles.json``. Read-only with respect to the run tree.
  apply   move the flagged mp4s into ``_quarantine_swap/`` and submit a repair
          job covering only the affected tasks. Dry-run unless ``--submit``.
  report  print the before/after success rate and every outcome that moved.

Layout assumed (what ``eval_wam_dit4dit_kitchen.sh`` writes)::

    $EVAL_ROOT/<run>/checkpoint-<step>/nas16/<Task>_Env/
        robocasa_..._env<NN>-episode_<K>-{success,failure}.mp4

Configure with environment variables or flags:

  EVAL_ROOT   parent of the run directories                     (--eval-root)
  GR00T_ROOT  Isaac-GR00T checkout, for the task ordering       (--gr00t-root)
  SUBMIT_CMD  shell template used by ``apply --submit``         (--submit-cmd)

``SUBMIT_CMD`` is formatted with ``{array}``, ``{ckpt}``, ``{step}``,
``{suffix}`` and ``{n}``. See ``scripts/submit_repair_slurm.sh`` for the SLURM
template this was developed against.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess

from .detect import episode_signature, classify

MP4 = re.compile(r"env(\d+)-episode_(\d+)-(success|failure)\.mp4$")
DEFAULT_SUBMIT = (
    "sbatch --array={array}%{n} "
    "--export=ALL,CKPT_CONFIGS_OVERRIDE={ckpt}:{step},MODEL_OUTPUT_DIR={ckpt},"
    "N_ENVS=1,ENV_LAYOUT=5,EVAL_SUFFIX={suffix} "
    "$GR00T_ROOT/run_scripts/eval/robocasa/sbatch_l40s_wan22_kitchen_24task.sh"
)


def eval_root(args):
    r = args.eval_root or os.environ.get("EVAL_ROOT")
    if not r:
        raise SystemExit("set EVAL_ROOT or pass --eval-root")
    return r.rstrip("/")


def run_dir(args):
    d = f"{eval_root(args)}/{args.run}/checkpoint-{args.step}/nas16"
    if not os.path.isdir(d):
        raise SystemExit(f"no such run directory: {d}")
    return d


def task_order(args):
    """Array index -> task name, read from the eval script's own task list.

    The order the eval script iterates is the only definition of the array
    index, so it is parsed rather than duplicated here. The file holds task
    lists for several benchmarks; only the kitchen (panda_omron) entries count.
    """
    root = args.gr00t_root or os.environ.get("GR00T_ROOT")
    if not root:
        raise SystemExit("set GR00T_ROOT or pass --gr00t-root")
    txt = open(f"{root}/run_scripts/eval/robocasa/eval_wam_dit4dit_kitchen.sh").read()
    names = re.findall(r'"robocasa_panda_omron/([A-Za-z0-9_]+)"', txt)
    if len(names) != 24:
        raise SystemExit(f"parsed {len(names)} kitchen tasks, expected 24")
    return {n: i for i, n in enumerate(names)}


# --------------------------------------------------------------------------- scan

def scan(args):
    root = run_dir(args)
    rows, undecodable = [], []
    for td in sorted(glob.glob(f"{root}/*_Env")):
        task = os.path.basename(td)
        for p in sorted(glob.glob(f"{td}/*.mp4")):
            m = MP4.search(p)
            if not m:
                continue
            sig = episode_signature(p)
            reason = classify(sig)
            if reason is None:
                continue
            stats = sig if isinstance(sig, dict) else {"med": -1, "gap": 0.0, "lag1m": 0.0}
            rec = {
                "task": task, "env": int(m[1]), "ep": int(m[2]),
                "out_before": m[3], "mp4": os.path.basename(p),
                "reason": reason, **stats,
            }
            # `undecodable` is recorded but never repaired. The success label is written
            # by the environment into the filename, not read back from the video, so a
            # damaged recording is not evidence of a damaged observation -- re-running
            # discards a valid label and redraws an unseeded noise sample. Measured:
            # swap-alt/sustained-nonphys flipped success->failure 0 times in 46 re-runs,
            # undecodable flipped it 13 times in 44.
            (undecodable if reason == "undecodable" else rows).append(rec)

    json.dump({"run": args.run, "contaminated": rows, "undecodable": undecodable},
              open(f"{root}/swap_manifest.json", "w"), indent=1)
    cycles = _append_cycle(root, rows)

    print(f"[scan] {len(rows)} contaminated (cycle {cycles[-1]['n']})")
    if undecodable:
        print(f"[scan] {len(undecodable)} undecodable -- recorded only, original label kept")
    for r in rows:
        print(f"   {r['task'].replace('_PandaOmron_Env',''):<24} "
              f"env{r['env']}ep{r['ep']:<3} {r['out_before']:<8} "
              f"med={r['med']:>6} gap={r['gap']:>6} lag1m={r['lag1m']:>6} [{r['reason']}]")
    return rows


def _append_cycle(root, rows):
    """Maintain the cycle history that before/after reporting reads.

    Two rules keep it honest. If a previous tree was repaired before this file
    existed, cycle 1 is rebuilt from the quarantined originals -- their
    filenames carry (task, env, episode, original outcome), which is everything
    the 'before' column needs. Without that rebuild the first scan of such a
    tree writes itself as cycle 1 and the original outcomes are lost. And if the
    last recorded cycle has not been re-run yet, this scan replaces it instead
    of appending, so repeated scans without an intervening repair do not inflate
    the cycle count.
    """
    path = f"{root}/swap_cycles.json"
    cycles = json.load(open(path))["cycles"] if os.path.exists(path) else []

    if not cycles and os.path.isdir(f"{root}/_quarantine_swap"):
        prior = []
        for qp in glob.glob(f"{root}/_quarantine_swap/*_Env/*.mp4"):
            m = MP4.search(qp)
            if m:
                prior.append({"task": os.path.basename(os.path.dirname(qp)),
                              "env": int(m[1]), "ep": int(m[2]), "out_before": m[3],
                              "mp4": os.path.basename(qp),
                              "reason": "reconstructed-from-quarantine"})
        if prior:
            cycles = [{"n": 1, "contaminated": prior}]
            print(f"[scan] rebuilt cycle 1 ({len(prior)} entries) from quarantine")

    def refilled(recs):
        return all(glob.glob(f"{root}/{r['task']}/*env{r['env']:02d}-episode_{r['ep']}-*.mp4")
                   for r in recs) if recs else True

    if cycles and not refilled(cycles[-1]["contaminated"]):
        cycles[-1] = {"n": cycles[-1]["n"], "contaminated": rows}
    else:
        cycles.append({"n": len(cycles) + 1, "contaminated": rows})
    json.dump({"cycles": cycles}, open(path, "w"), indent=1)
    return cycles


# -------------------------------------------------------------------------- apply

def apply(args):
    root = run_dir(args)
    man = json.load(open(f"{root}/swap_manifest.json"))
    rows = man["contaminated"]
    if not rows:
        print("[apply] nothing flagged; tree has converged")
        return

    model = os.path.basename(args.ckpt_dir.rstrip("/"))
    if not args.run.startswith(model + "_"):
        raise SystemExit(f"run '{args.run}' does not start with '{model}_' -- check --ckpt-dir")
    suffix = args.run[len(model) + 1:]

    quarantine = f"{root}/_quarantine_swap"
    for r in rows:
        src = f"{root}/{r['task']}/{r['mp4']}"
        if not os.path.exists(src):
            continue
        dst = f"{quarantine}/{r['task']}/{r['mp4']}"
        print(f"[{'move' if args.submit else 'would-move'}] {r['task']}/{r['mp4']}")
        if args.submit:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

    order = task_order(args)
    idx = sorted(order[t] for t in {r["task"] for r in rows})
    cmd = (args.submit_cmd or os.environ.get("SUBMIT_CMD") or DEFAULT_SUBMIT).format(
        array=",".join(map(str, idx)), ckpt=args.ckpt_dir, step=args.step,
        suffix=suffix, n=len(idx))

    print(f"[apply] {len(rows)} episodes across {len(idx)} tasks, array={idx}")
    if args.submit:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())
    else:
        print("[dry-run] would submit:\n  " + cmd)


# ------------------------------------------------------------------------- report

def report(args):
    root = run_dir(args)
    path = f"{root}/swap_cycles.json"
    if not os.path.exists(path):
        raise SystemExit("no swap_cycles.json -- run scan first")
    cycles = json.load(open(path))["cycles"]

    current = {}
    for td in sorted(glob.glob(f"{root}/*_Env")):
        task = os.path.basename(td)
        for p in glob.glob(f"{td}/*.mp4"):
            m = MP4.search(p)
            if m:
                current[(task, int(m[1]), int(m[2]))] = m[3]

    succ = sum(v == "success" for v in current.values())
    total = len(current)
    print(f"[report] {args.run} @ {args.step}")

    # Walk the cycles backwards, undoing each repair to recover the earlier rate.
    rates = [(succ, total)]
    s = succ
    for c in reversed(cycles):
        for r in c["contaminated"]:
            now = current.get((r["task"], r["env"], r["ep"]))
            if now is None:
                continue
            s += (r["out_before"] == "success") - (now == "success")
        rates.append((s, total))
    rates.reverse()

    labels = ["original"] + [f"after cycle {c['n']}" for c in cycles]
    for label, (a, b) in zip(labels, rates):
        print(f"  {label:<16} {a}/{b} = {100 * a / max(b, 1):.2f}%")

    pending = [r for c in cycles for r in c["contaminated"]
               if (r["task"], r["env"], r["ep"]) not in current]
    if pending:
        print(f"  !! {len(pending)} slots still empty -- repair not finished")

    for c in cycles:
        moved = [(r, current.get((r["task"], r["env"], r["ep"])))
                 for r in c["contaminated"]]
        moved = [(r, n) for r, n in moved if n is not None and n != r["out_before"]]
        if moved:
            print(f"  cycle {c['n']}: {len(moved)} outcome(s) changed")
            for r, now in moved:
                print(f"    {r['task'].replace('_PandaOmron_Env',''):<24} "
                      f"env{r['env']}ep{r['ep']:<3} {r['out_before']} -> {now}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scan", "apply", "report"])
    ap.add_argument("run", help="run directory name under EVAL_ROOT")
    ap.add_argument("--step", default="100000")
    ap.add_argument("--eval-root", default="")
    ap.add_argument("--gr00t-root", default="")
    ap.add_argument("--ckpt-dir", default="", help="checkpoint dir (apply)")
    ap.add_argument("--submit-cmd", default="", help="submission template (apply)")
    ap.add_argument("--submit", action="store_true", help="actually move and submit")
    args = ap.parse_args()

    if args.mode == "scan":
        scan(args)
    elif args.mode == "apply":
        if not args.ckpt_dir:
            raise SystemExit("--ckpt-dir is required for apply")
        apply(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
