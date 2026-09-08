# The RoboCasa Kitchen view-swap protocol

What breaks when several environments share a GPU, how the detector was derived,
and what was measured. Korean summary: [PROTOCOL.ko.md](PROTOCOL.ko.md).

## 1. The failure

A kitchen sweep is 24 tasks × 5 environment lanes × 10 episodes = 1,200
episodes. Running the 5 lanes concurrently in one process, each with its own EGL
context on one L40S, is what makes the sweep finish in a day.

Under that configuration, some rendered frames are taken from the wrong camera.
The three PandaOmron cameras — `robot0_agentview_left`, `robot0_agentview_right`,
`robot0_eye_in_hand` — periodically exchange content, and the exchange
**alternates every frame**. Measured on flagged episodes:

| statistic | flagged episodes | clean episodes |
|---|---|---|
| lag-1 adjacent correlation | 0.35 – 0.47 | > 0.9 |
| lag-2 correlation | 0.65 – 0.82 | > 0.9 |

Frame *t* resembling *t+2* more than *t+1* is the whole signature: the stream is
alternating between two sources. Fitting the swapped frames against the other
camera streams recovers the mapping — wrist ← right, left ← right, right ← left
— at correlation 0.72–0.84, with some frames additionally channel-swapped (BGR).

This was initially misdiagnosed as the robot's base diverging. Checking only
lag-1 and lag-37, both odd, hides an alternation of period 2 entirely; the
even-lag check is what identifies it.

## 2. It is the observation, not the recording

The recording wrapper and the policy consume the same observation dictionary,
but they consume it differently, and that asymmetry settles the question:

- the mp4 is assembled by iterating `obs.items()` — **positional**;
- the policy looks each camera up **by key name** (`video.res256_image_side_0`
  → `left_view`, and so on).

A defect confined to the recorder could only be positional. The observed
corruption follows the camera content across both paths, so the tensor the
policy received was already wrong. Instrumenting the extraction forward pass
directly — dumping the velocity prediction and VAE-decoding `x0 = latents − v`
during the actual rollout, not from the rendered file — confirms the model is
conditioned on the swapped frames.

## 3. Deriving the detector

The hard part is not finding corrupted frames; it is not flagging a fast robot.
A kitchen policy drives a mobile base, and a base swing produces a per-pixel
frame difference as large as a corrupted frame's.

The separating axis is **macro-scale continuity**. Downsample each frame to
16×16 and correlate adjacent frames:

| | adjacent macro correlation |
|---|---|
| real motion, including fast base swings | 0.95 – 0.999 |
| corrupted frames | 0.34 – 0.51 |

Real motion moves content continuously; a wrong-camera frame is uncorrelated
with its neighbour at every scale. So the rule flags an episode when either the
period-2 signature is present (`gap > 0.12`) or a sustained large difference is
*also* macro-discontinuous (`med > 20` and `lag1m < 0.7`).

### Two false-positive classes that had to be handled

**Short successes.** `terminate_on_success` ends a solved episode after a single
action chunk, so a legitimate success can be 16 frames. An early version treated
any short decode as damage; those episodes were quarantined, re-run, ended early
again, and were re-flagged forever. Episodes under 40 frames are now judged
`short` and left alone. Any repair loop that does not converge should be checked
for this first.

**Concealed decode errors.** ffmpeg's error concealment produces plausible
frames from a damaged bitstream — smooth, coherent, and passing every
pixel-domain test. Damage is therefore detected from ffmpeg's *stderr*, not from
the image. An earlier conclusion that "the codec is fine" came from testing the
concealed pixels.

## 4. What was measured

19 checkpoint trees, 22,800 episodes, scanned at `N_ENVS=5`:

| | |
|---|---|
| episodes flagged | 538 / 22,800 = **2.36 %** |
| flagged after an `N_ENVS=1` repair | 39 / 19,200 = **0.20 %** |
| … excluding one file-corruption incident | **1 / 19,200** |

Of the flagged episodes whose classification was preserved, roughly 62 % were
`swap-alt`, 25 % `sustained-nonphys`, 12 % `undecodable`.

Twenty-one trees carried through to a converged repair:

| | |
|---|---|
| episodes re-run | 566 |
| outcomes changed | 236 (42 %) |
| mean score | 62.25 % → 63.17 % |
| per-tree | median +1.00 pt, max +1.50 pt, **21 of 21 up or unchanged** |

Corrupted observations cost roughly **one point of kitchen score**, in a
direction that is systematic — a policy given scrambled views fails — but spread
over episodes that differ from run to run.

## 5. Alternatives that were ruled out

Fixing the renderer directly would be better than repairing afterwards. Three
approaches were tested on a 250-episode slice:

| variant | persistent-form frames | swap signatures | success |
|---|---|---|---|
| unmodified | 8 | 4 | 132/250 = 52.8 % |
| `make_current` before each render | 11 | 6 | 133/250 = 53.2 % |
| render, verify, retry on mismatch | 5 | 4 | 132/250 = 52.8 % |
| **`N_ENVS=1`** | **2** | **2** | 126/250 = 50.4 % |

Binding the EGL context explicitly before every render did not help, and
verify-and-retry only halved the rate at a large throughput cost. Only reducing
to one environment per process removes the swap. Since the swap affects a small
minority of episodes, re-running that minority at `N_ENVS=1` gets the
correctness of the single-environment configuration at nearly the throughput of
the concurrent one.

Upgrading the simulator (mujoco 3.3.x) was investigated in an isolated kitchen
environment so that in-flight evaluations kept using the existing one; it did
not resolve the swap either.

## 6. The append incident

Worth stating plainly, because it silently rewrote already-reported numbers.

The repair pass runs with `N_ENVS=1` while keeping `ENV_LAYOUT=5`, so that
episode seeds match the original run. Hole-aware resume — the logic that refills
missing episode indices rather than counting forward — was gated on `n_envs > 1`
alone. The repair pass therefore skipped it, appended episodes 10, 11, 12 … past
the per-lane cap, and produced trees with more than 1,200 episodes. Two
already-published scores moved (67.8 → 68.0, 65.1 → 64.8) with no repair having
been intended for them.

The fix is `patches/01-hole-aware-resume.patch`: widen the gate to
`n_envs > 1 or env_layout > 0` and thread `--env_layout` from the eval script.
**Do not run a repair without it.** After any repair, assert the tree holds
exactly 1,200 episodes, excluding `_quarantine_swap/`.

## 7. Cycles and convergence

A repair is an evaluation and can produce a fresh corrupted episode, so the
protocol scans again afterwards. Measured: of 16 trees taken to a second scan,
14 returned zero. The two that did not were one tree with 6 residual episodes
and one unrelated file-corruption incident.

Practically: scan, repair, scan; stop at zero; go to a third cycle only if the
second finds something. Mechanically running three cycles is not necessary.

`swap_cycles.json` stores the entry state of each cycle, which is what makes
`original → after cycle 1 → final` reportable. Do not delete
`_quarantine_swap/` — for trees repaired before that file existed, the
quarantined filenames are the only surviving record of the original outcomes,
and the scanner rebuilds cycle 1 from them.

## 8. Why `undecodable` is detected but never repaired

`undecodable` is a different kind of defect from the swap classes, and treating
it the same way actively damages the score.

A success/failure label is written by the environment into the filename; it is
not read back from the video. A damaged mp4 therefore means the *recording*
failed, which is not evidence that the *observation* was corrupted. Re-running
such an episode discards a probably-valid label and draws a fresh sample — and
because the extraction noise is not seeded, the same scene can produce a
different outcome.

The measured behaviour separates the classes cleanly:

| re-run class | n | success → failure | failure → success | net |
|---|---|---|---|---|
| swap-alt | 33 | **0** | 18 | +18 |
| sustained-nonphys | 13 | **0** | 7 | +7 |
| undecodable | 44 | **13** | 1 | −12 |

Repairing a genuinely corrupted observation never turned a success into a
failure — not once in 46 re-runs. Re-running a merely damaged recording did so
13 times in 44.

There is a mechanism behind the asymmetry. Damaged recordings are enriched in
successes: 82 % of the `undecodable` episodes were successes, against 68 % for
the trees as a whole. `terminate_on_success` ends a solved episode abruptly,
which is precisely when a file's trailer is likely to be truncated. Re-running
that success-enriched set regresses it to the mean, so the score falls.

One tree showed this in isolation. Every episode it flagged was `undecodable`;
repairing them moved it −0.17 pt, while every swap-dominated tree moved up.

**The rule:** `undecodable` episodes are recorded in the manifest under their own
key, left in the tree, and keep their original label. They are never quarantined
and never re-run. With that rule applied, no tree in the measured set moves down.
