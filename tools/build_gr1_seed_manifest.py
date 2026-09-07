"""Build the GR-1 episode-seed manifest from the stability sweeps.

Inputs (produced by probe_gr1_reset_stability.py):
  baseline sweeps: every measured slot — offsets -1..9 per env lane (v2 dir covers
                   0..9, the *_m1 dir covers -1). Today's eval wrapper only looks up
                   offsets -1..8, so offset-9 entries are dormant — kept anyway so the
                   manifest stays correct if the episode-numbering convention shifts.
  reserve sweep  : the replacement candidate pool — offsets 10..34 per env lane

Rule v4 (frozen 2026-09-05 by huiwon, judged at the 5.0 s settle checkpoint — the state
the episode actually starts from under GR1_RESET_SETTLE_S=5.0). A slot is REPLACED when:
  1. the pick object ("obj") or the destination container ("container") left the table
     (dropped > VANISH_Z below its sampled placement), or
  2. the destination container is tipped > TILT_DEG, or
  3. ANY critical object (pick object or either container) is LIFTED > LIFT_Z above its
     sampled placement (resting on another object or the robot hand — the wedge
     signature; "낑긴 건 target 물체/placement와 무관하게 교체"), or
  4. the pick object is a bottle/can-family item (wine, bottled water, can, milk) and at
     5.0 s it is fallen (tilt > TILT_DEG) or still falling (angular speed > FALL_W rad/s)
     — a policy trained on upright bottles starts from a grasp it never saw.
Explicitly NOT replaced: a container merely슬라이딩 on the table (17 cm-pushed basket was
solved 3/3), the source container tipping or leaving the table while the pick object stays
(CuttingboardToCardboardbox e4 solved at 100k), and the pick object tipping over.

Replacement slots receive the first STRICT-CLEAN reserve seed from the same env lane
(no critical object tipped/drifted/dropped at 5.0 s; falling back to merely non-replaceable
reserves, then to other lanes, in lane order) — a fully deterministic assignment that every
model shares by reading the same JSON.

Output format (consumed by VideoRecordingWrapper, GR1_EPISODE_SEED_MANIFEST=<path>):
  {task: {str(base_seed): {str(ep_id): alt_seed}}}   with ep_id in -1..8
"""
import argparse, json, math
from pathlib import Path

TILT_DEG, VANISH_Z, LIFT_Z, FALL_W = 30.0, 0.25, 0.03, 0.5
BOTTLE_NOUNS = ("wine", "bottled water", "can", "milk")
STRICT_TILT, STRICT_XY, STRICT_Z = 30.0, 0.05, 0.08
CHECK = "5.0"
BASE = 42          # eval SEED
STRIDE = 100000
EVAL_OFFSETS = range(-1, 10)   # -1..8 = today's eval seed space; 9 = measured spare, kept dormant (2026-09-05 huiwon: bottle/can은 전부 교체 대상)


def _zax(q):
    w, x, y, z = q
    return (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))


def _tilt(qr, qn):
    c = max(-1.0, min(1.0, sum(u * v for u, v in zip(_zax(qr), _zax(qn)))))
    return math.degrees(math.acos(c))


def _metrics(rec, name, cp=CHECK):
    pl = rec["placement"].get(name)
    st = rec["settled"].get(cp, {}).get(name)
    if not pl or not st:
        return None
    p, q = st["qpos"][:3], st["qpos"][3:7]
    drop = pl["pos"][2] - p[2]  # + = fell below placement
    dxy = math.hypot(p[0] - pl["pos"][0], p[1] - pl["pos"][1])
    return drop, dxy, _tilt(pl["quat"], q)


import re as _re


def _obj_noun(rec):
    lang = (rec.get("lang") or "").replace("unlocked_waist: ", "").replace("locked_waist: ", "")
    mm = _re.search(r"pick (?:up )?the ([a-z_ ]+?)(?:,| from| and| in)", lang)
    return mm.group(1).strip() if mm else ""


def _angspeed(rec, name, cp=CHECK):
    st = rec["settled"].get(cp, {}).get(name)
    av = (st or {}).get("angvel")
    return math.sqrt(sum(v * v for v in av)) if av else 0.0


def is_replaceable(rec):
    """Rule v4: reasons this slot must be swapped out (empty list = keep)."""
    reasons = []
    for name in ("obj", "container", "obj_container"):
        m = _metrics(rec, name)
        if m is None:
            continue
        drop, dxy, tl = m
        if name in ("obj", "container") and drop > VANISH_Z:
            reasons.append(f"{name}-vanish({drop * 100:.0f}cm)")
        if name == "container" and tl > TILT_DEG:
            reasons.append(f"container-tilt({tl:.0f}deg)")
        if -drop > LIFT_Z:
            reasons.append(f"{name}-lift(+{-drop * 100:.1f}cm)")
        if name == "obj" and _obj_noun(rec) in BOTTLE_NOUNS:
            w = _angspeed(rec, name)
            if tl > TILT_DEG or w > FALL_W:
                reasons.append(f"bottle-fallen({tl:.0f}deg,w={w:.2f})")
    return reasons


def is_strict_clean(rec):
    """Reserve quality bar: nothing critical moved at all, AND the slot is normal
    under rule v4 itself — signals the plain thresholds don't cover (container lift
    within |dz|, a bottle still falling) must not slip through: a replacement seed
    has to be NORMAL under the same rule it replaces for (2026-09-05, huiwon)."""
    for name in ("obj", "container", "obj_container"):
        m = _metrics(rec, name)
        if m is None:
            continue
        drop, dxy, tl = m
        if tl > STRICT_TILT or dxy > STRICT_XY or abs(drop) > STRICT_Z:
            return False
    return not is_replaceable(rec)


def load(dirs):
    recs = {}
    for d in dirs:
        for j in sorted(Path(d).glob("[0-9]*.json")):
            data = json.loads(j.read_text())
            for r in data["records"]:
                recs[(data["task"], r["env_idx"], r["ep"])] = r
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="baseline probe output dirs (v2 and the offset -1 sweep)")
    ap.add_argument("--reserve", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load(args.baseline)
    resv = load([args.reserve])

    tasks = sorted({t for t, _, _ in base})
    manifest = {}
    n_bad = n_strict = n_loose = n_crosslane = 0
    # per-lane reserve pools, ordered by offset (deterministic)
    pool = {}
    for (t, e, ep), rec in sorted(resv.items()):
        pool.setdefault((t, e), []).append((ep, rec))
    used = set()

    def take(task, lane_pref):
        """First unused strict-clean reserve: own lane, then other lanes in order.
        Falls back to non-replaceable reserves with the same lane order."""
        nonlocal n_strict, n_loose, n_crosslane
        lanes = [lane_pref] + [e for e in range(5) if e != lane_pref]
        for strict in (True, False):
            for lane in lanes:
                for ep, rec in pool.get((task, lane), []):
                    if (task, lane, ep) in used:
                        continue
                    if strict and not is_strict_clean(rec):
                        continue
                    if not strict and is_replaceable(rec):
                        continue
                    used.add((task, lane, ep))
                    if strict:
                        n_strict += 1
                    else:
                        n_loose += 1
                    if lane != lane_pref:
                        n_crosslane += 1
                    return lane, ep
        return None

    for task in tasks:
        for env in range(5):
            for ep in EVAL_OFFSETS:
                rec = base.get((task, env, ep))
                if rec is None:
                    raise SystemExit(f"missing baseline measurement: {task} e{env} ep{ep}")
                reasons = is_replaceable(rec)
                if not reasons:
                    continue
                n_bad += 1
                got = take(task, env)
                if got is None:
                    raise SystemExit(f"reserve pool exhausted for {task} (lane {env})")
                alt_lane, alt_ep = got
                alt_seed = (BASE + alt_lane) * STRIDE + alt_ep
                manifest.setdefault(task, {}).setdefault(str(BASE + env), {})[str(ep)] = alt_seed

    meta = {"_meta": {"rule": "v4@5.0s: obj/container vanish>25cm | container tilt>30deg | "
                              "any-role lift>3cm | bottle/can (wine, bottled water, can, milk) "
                              "fallen>30deg or angspeed>0.5",
                      "frozen": "2026-09-05",
                      "settle_secs": 5.0,
                      "eval_offsets": "-1..9 (9 dormant)",
                      "seed_formula": f"(base={BASE}+env)*{STRIDE}+offset",
                      "replaced": n_bad, "reserve_strict": n_strict,
                      "reserve_loose": n_loose, "cross_lane": n_crosslane}}
    out = dict(meta)
    out.update(manifest)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"replaced {n_bad} slots -> {args.out}")
    print(f"  reserves: strict-clean {n_strict}, loose {n_loose}, cross-lane {n_crosslane}")


if __name__ == "__main__":
    main()
