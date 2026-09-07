# GR-1 Tabletop Eval Protocol

Deterministic, physically-valid episode resets for the 24-task GR-1 tabletop
benchmark in [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).

Two things go wrong with the benchmark as shipped, and both quietly change
scores:

**Episodes are not reproducible.** Upstream seeds only the *first* reset of each
sub-env (`reset_seeds = [seed + i]` in `gr00t/eval/rollout_policy.py`);
episodes 2..N inherit whatever RNG state the process drifted into. Two runs of
the same checkpoint do not evaluate the same episodes, and two models cannot be
compared episode-for-episode.

**Some episodes start broken.** The stock reset settles physics for only ~0.5 s,
so the policy's first observation can be taken mid-fall. Measuring all 1,200
slots of a 24-task run: **16 % start with a critical object tipped, wedged on
the robot, or already off the table.** Those episodes are solved at half the
rate of clean ones (24.1 % vs 49.0 %), so roughly **4 points of any GR-1 score
is reset quality rather than policy quality** — and which slots break is not the
same across runs.

This repo fixes both without forking Isaac-GR00T: three small patches, one
wrapper, and a frozen substitution table.

| | |
|---|---|
| **Settle** | `GR1_RESET_SETTLE_S=5.0` — a fixed, branch-free physics settle at the end of every reset, so the first observation is of a scene at rest. |
| **Deterministic seeding** | every episode gets `(seed + env_idx) * 100000 + episode_id`, so the same slot means the same scene for every model, every run, every node. |
| **Seed manifest** | the 130 slots that are *still* broken after 5 s are swapped for pre-validated replacement seeds, decided once offline and frozen in JSON. |

Measured effect on a fixed checkpoint set (same model and weights; only the
protocol changed): **44.7 % → 48.4 %** mean success, with the gain concentrated
exactly where the manifest acts.

## Why the substitution is decided offline

Never re-derive it during a run. Any accept/reject check evaluated at reset time
sits on a floating-point boundary, and contact-solver jitter across nodes or
library versions can then flip *which* episode a model sees — reintroducing the
incomparability the protocol exists to remove. The manifest is a table lookup:
same JSON in, same episodes out.

## Install

Requires an Isaac-GR00T checkout with the GR-1 tabletop sim environment
installed (`gr00t/eval/sim/robocasa-gr1-tabletop-tasks` submodule initialised
and its venv built — see Isaac-GR00T's own setup for that step).

```bash
git clone https://github.com/huiwon-jang/gr1-tabletop-eval-protocol.git
cd gr1-tabletop-eval-protocol
./scripts/apply_protocol.sh /path/to/Isaac-GR00T
```

`apply_protocol.sh` applies three patches — two to the robocasa submodule
(reset settle, cross-process determinism) and one 20-line hook to
`rollout_policy.py` — and puts `gr1_protocol` on the checkout's import path.
Each patch is checked first and skipped if already applied, so re-running is
safe.

## Run

```bash
GR00T_ROOT=/path/to/Isaac-GR00T \
MODEL_PATH=nvidia/GR00T-N1.5-3B \
    ./scripts/run_gr1_eval.sh          # all 24 tasks; pass 0-23 for one
```

The runner starts a policy server, runs the rollouts in the sim venv, and prints
how many slots were substituted. Every substitution also prints a line you can
audit afterwards:

```
[seed-manifest] PnPWineToCabinetClose base=46 ep=-1: 4599999 -> 4600010
```

To evaluate a different policy, point `MODEL_PATH` at it — the protocol is
model-agnostic. To reproduce pre-protocol numbers, unset the two env vars
(`GR1_RESET_SETTLE_S=0`, no manifest).

⚠ Results measured with and without the protocol are **not comparable**. Report
which one you used.

## The replacement rule

A slot is replaced when, **5 s after reset**, any of these hold:

1. the pick object or the destination container has left the table (fell > 25 cm);
2. the destination container is tipped > 30°;
3. anything is *lifted* > 3 cm above where it was placed — physically only
   possible if it is resting on the robot or another object (the wedge signature);
4. the pick object is a bottle/can-family item (wine, bottled water, can, milk)
   and is tipped > 30° or still falling (angular speed > 0.5 rad/s).

Deliberately **not** replaced: containers that merely slid along the table (a
basket pushed 17 cm was still solved 3/3), a tipped or fallen *source* container
while the pick object stays reachable, and non-bottle objects lying on their
side — a potato is as graspable in any orientation.

Replacements come from a reserve sweep of the same env lane and must clear a
stricter bar than the slots they replace: nothing moved at all, *and* clean
under the rule above. All 130 were re-verified after assignment.

Full derivation, measurements and per-task counts: [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Rebuilding the manifest

The shipped manifest is for the standard shape (24 tasks × 5 env lanes ×
10 episodes, `SEED=42`). Changing thresholds — or the task/object set — means
regenerating it; it is never re-derived at eval time.

```bash
# 1. sweep every slot and record its settled state (needs the sim env)
python tools/probe_gr1_reset_stability.py --task-idx 0 --out-dir sweeps/baseline
python tools/probe_gr1_reset_stability.py --task-idx 0 --ep-list 10:11:...:34 --out-dir sweeps/reserve

# 2. freeze the decision
python tools/build_gr1_seed_manifest.py \
    --baseline sweeps/baseline --reserve sweeps/reserve \
    --out manifest/gr1_episode_seed_manifest.json
```

## Layout

```
gr1_protocol/      seeding + manifest wrapper (the only runtime code)
manifest/          frozen substitution table, 130 slots
patches/           01 settle · 02 determinism · 03 rollout hook
tools/             probe sweep + manifest builder
scripts/           apply_protocol.sh · run_gr1_eval.sh
docs/PROTOCOL.md   rationale, measurements, rule derivation
```

## Compatibility

Patches are cut against Isaac-GR00T at `51d4c89` and the robocasa GR-1 tabletop
tasks submodule pinned there. If upstream moves and a patch stops applying,
`apply_protocol.sh` says which one and stops rather than half-applying; the
patches are small enough to rebase by hand.

The runtime wrapper (`gr1_protocol/`) touches no upstream internals — it only
wraps the env and seeds `reset()` — so it survives upstream churn even when the
patches need a refresh.

## License

Apache-2.0. The robocasa GR-1 tabletop tasks are MIT (NVIDIA GEAR group);
patches here are diffs against that source, not copies of it.
