# RoboCasa Kitchen View-Swap Repair Protocol

Detecting and repairing corrupted observations in the 24-task RoboCasa Kitchen
benchmark in [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), when the
evaluation runs several environments per GPU.

Running 5 environments in one process is how a 1,200-episode kitchen sweep fits
in a working day. It also quietly corrupts a fraction of the episodes.

**Frames get swapped between camera streams.** With 5 concurrent EGL contexts on
one device, a rendered frame is periodically taken from the wrong camera: the
wrist view receives the right-hand view, the left receives the right, and the
next frame swaps back. The signature is an alternation with period 2 — frame *t*
correlates with *t+2* far better than with *t+1* (measured lag-1 0.35–0.47
against lag-2 0.65–0.82). Some frames also arrive channel-swapped or flipped.

**It reaches the policy, not just the video.** The recording wrapper and the
policy read the same observation dict. An episode whose recording shows swapped
views is an episode the policy was asked to act on with swapped views, and it
fails at a much higher rate.

Across 19 evaluated checkpoint trees — 22,800 episodes — **2.36 % of episodes
carry a corrupted observation**. They are not the same episodes from run to run,
so two checkpoints are not compared on equal footing, and a single checkpoint
does not reproduce its own score.

This repo detects those episodes and re-runs *only* them with one environment
per process, preserving the original score alongside the repaired one.

| | |
|---|---|
| **Detect** | A three-statistic rule over the recorded episode. Separates a corrupted render from a fast-moving robot base, which is the discrimination that makes automatic repair possible at all. |
| **Quarantine** | Flagged mp4s move to `_quarantine_swap/`, never deleted. Their filenames carry the original outcome, so the pre-repair score stays reconstructible. |
| **Re-run** | Only the affected `(task, env, episode)` slots, at `N_ENVS=1` with the lane layout unchanged — so the repaired episode is the *same scene*, not a new sample. |
| **Report** | `original → after cycle 1 → …` for every tree, so a repaired number is never silently substituted for the one people already saw. |

## Measured effect

Sixteen checkpoint trees carried through to a converged repair (1,200 episodes
each; the same weights, only the protocol changed):

| | |
|---|---|
| episodes re-run | 433 |
| outcomes that changed | 174 (40 %) |
| mean score | **60.99 % → 61.86 %** |
| per-tree change | median +1.00 pt, max +1.50 pt |

The direction is systematic, not noise: 15 of 16 trees moved up or stayed put.
Corrupted observations make a policy fail, so leaving them in place costs about
**one point of kitchen score** — and costs it unevenly, since which episodes
break differs per run.

After repair, a second scan finds **0.20 %** still flagged, and 38 of those 39
episodes are one unrelated incident (a single eval run that wrote 38 damaged mp4
files). Excluding it, the residual is **1 episode in 19,200** — running one
environment per process removes the swap rather than reducing it.

## The detection rule

Every episode is decoded to 192×64 grayscale and reduced to three numbers:

| | |
|---|---|
| `med` | median absolute inter-frame difference — large for any fast change |
| `gap` | median lag-2 correlation minus lag-1 — positive only under period-2 alternation |
| `lag1m` | adjacent correlation after downsampling to 16×16 — macro-scale continuity |

An episode is flagged when **either**

1. `gap > 0.12` — the period-2 alternation signature of a view swap; or
2. `med > 20` **and** `lag1m < 0.7` — sustained large change that is also
   macro-discontinuous, i.e. not reachable by anything physical.

`lag1m` is what makes the rule usable. A kitchen policy drives a mobile base,
and a fast base swing produces a large `med` that looks alarming per-pixel — but
it stays spatially continuous at macro scale (measured 0.95–0.999), while a
corrupted frame does not (0.34–0.51). Thresholding on frame difference alone
flags every fast episode.

**Deliberately not flagged:** a moving base or a fast arm, however large the
pixel difference, as long as macro continuity holds; and episodes shorter than
40 frames. `terminate_on_success` ends a solved episode after one action chunk,
so a legitimate success can be 16 frames long — judging those as damaged makes
every repair cycle re-flag the same episodes and the loop never converges.

**Flagged but different in kind:** `undecodable`, where ffmpeg reports a damaged
bitstream. Error concealment makes the decoded pixels look plausible, so this is
detected from ffmpeg's stderr rather than from the image. Note that a damaged
*recording* is not evidence of a damaged *observation* — the success label is
written by the environment, not read from the video. In the one tree whose
flagged episodes were all `undecodable`, repair moved the score −0.17 pt, while
every swap-dominated tree moved up. Re-running those episodes resamples an
unseeded noise draw rather than fixing a corrupted input; treat the two classes
separately if that distinction matters to you.

## Install

Requires an Isaac-GR00T checkout with the RoboCasa kitchen sim environment
installed, plus `ffmpeg`, `numpy` and `opencv-python`.

```bash
git clone -b kitchen https://github.com/huiwon-jang/gr1-tabletop-eval-protocol.git
cd gr1-tabletop-eval-protocol
git -C /path/to/Isaac-GR00T apply $(pwd)/patches/01-hole-aware-resume.patch
```

### The patch is not optional

`patches/01-hole-aware-resume.patch` is the one change you cannot skip. The
repair pass re-runs a few episodes with `--n_envs 1` while keeping `ENV_LAYOUT`
at the original lane count, so that episode seeds stay identical. Upstream gates
hole-aware resume on `n_envs > 1` alone, so that pass falls back to global
counting: instead of refilling the quarantined slots it **appends** new episodes
past the per-lane cap. The tree grows beyond 1,200, and the run's original
success rate changes underneath you.

The patch widens the gate to `n_envs > 1 or env_layout > 0` and threads
`--env_layout` through the eval script. Verify after any repair:

```bash
find <run>/checkpoint-<step>/nas16 -path '*_quarantine_swap*' -prune -o \
     \( -name '*-success.mp4' -o -name '*-failure.mp4' \) -print | wc -l   # must be 1200
```

## Run

```bash
export EVAL_ROOT=/path/to/eval/output/robocasa_kitchen/<tag>
export GR00T_ROOT=/path/to/Isaac-GR00T

# 1. judge every episode; writes swap_manifest.json, touches nothing else
python -m kitchen_protocol.repair scan  <run> --step 100000

# 2. quarantine and re-run the flagged slots (drop --submit for a dry run)
python -m kitchen_protocol.repair apply <run> --step 100000 \
       --ckpt-dir /path/to/checkpoints/<model> --submit

# 3. once the repair job finishes
python -m kitchen_protocol.repair report <run> --step 100000
```

```
[report] <run> @ 100000
  original         777/1200 = 64.75%
  after cycle 1    775/1200 = 64.58%
  cycle 1: 2 outcome(s) changed
    PnPCounterToStove        env3ep5   success -> failure
    PnPCounterToStove        env3ep9   success -> failure
```

`apply --submit` shells out to `SUBMIT_CMD`; `scripts/submit_repair_slurm.sh` is
the SLURM template this was developed against. Adjust its header for your
cluster. Two settings are not free to change: `N_ENVS=1`, which is what removes
the swap, and `ENV_LAYOUT` left at the original lane count, which is what keeps
`env_seed = SEED + global_env_idx` — and therefore the scene — identical to the
run being repaired.

## Cycles

A repair pass is itself an evaluation and can in principle produce a fresh
corrupted episode, so scanning again after a repair is part of the protocol, not
a paranoid extra. In practice it converges immediately: of 16 trees carried to a
second scan, **14 came back with zero**. Scan after every repair; stop when a
scan returns zero. Two cycles is the normal case, and a third is only warranted
when the second still finds something.

`swap_cycles.json` records each cycle's entry state, which is what lets `report`
print `original → after cycle 1 → …` rather than a single post-hoc number. If it
is missing — a tree repaired before the file existed — the first scan rebuilds
cycle 1 from the quarantined filenames, which carry the original outcomes. Never
delete `_quarantine_swap/`: it is the only remaining record of the pre-repair
result for such trees.

⚠ Repaired and unrepaired numbers are **not comparable**. Report which one you
used, and prefer reporting both.

## Layout

```
kitchen_protocol/detect.py   the rule: signature + verdict, no cluster deps
kitchen_protocol/repair.py   scan · apply · report
patches/                     hole-aware resume (required)
scripts/                     SLURM submission template
docs/PROTOCOL.md             derivation, measurements, ruled-out alternatives
```

## Compatibility

The detector is pure ffmpeg + numpy and depends only on the recorded mp4 naming
convention (`env<NN>-episode_<K>-{success,failure}.mp4`). The patch is cut
against Isaac-GR00T's `gr00t/eval/rollout_policy.py` and the WAM DiT4DiT kitchen
eval scripts; if upstream moves, `git apply` reports which hunk failed rather
than half-applying.

The array index is read from the eval script's own task list rather than
duplicated here, so a reordered benchmark stays consistent automatically.

## License

Apache-2.0.
