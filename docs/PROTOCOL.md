# The GR-1 tabletop reset protocol

How the benchmark breaks, what the fix does, and the measurements behind every
threshold. Korean summary: [PROTOCOL.ko.md](PROTOCOL.ko.md).

## 1. What was measured

Every slot of a standard run — 24 tasks × 5 env lanes × 10 episodes = 1,200
episodes — was reset and its scene recorded at 0.5 / 1 / 2 / 3 / 5 s of free
physics, comparing each object's pose against the placement the sampler had
asked for.

| state at reset | share of slots |
|---|---|
| clean | 66.1 % |
| still moving at first observation | 17.9 % |
| **critical: tipped, slid, wedged or fallen off** | **16.0 %** |

Joining those labels against real evaluations of one checkpoint family
(3,240 episodes, three checkpoints):

| slot state | episodes | success rate |
|---|---|---|
| clean | 2,157 | **49.0 %** |
| still moving | 585 | 42.1 % |
| critical | 498 | **24.1 %** |

The gap is stable across all three checkpoints (22.9 / 23.5 / 25.9 % for
critical slots). With 16 % of episodes solved at half rate, **≈4 points of any
GR-1 score is reset quality, not policy quality** — and because episodes past
the first are unseeded upstream, *which* slots break is not stable across runs.

## 2. The two fixes

### Fixed settle, no branches

`GR1_RESET_SETTLE_S=5.0` runs pure physics at the end of `_reset_internal`
before any observation exists. It contains no accept/reject logic *by design*: a
threshold check at reset time sits on a floating-point boundary, so contact
solver jitter between nodes or library versions could flip which placement is
accepted, and different models would silently see different episodes. A
fixed-duration settle is a pure rollout — same seed in, same settled state out.

One subtlety worth knowing if you port this: the GR-1 branch teleports the
elbow-pitch joints *after* the stock settle loop, and the controller has already
cached its goals from the pre-teleport pose. Stepping without re-syncing drags
the arm ~13 cm back toward stale goals during the settle, so the policy would
start from an arm pose training never shows. The patch force-refreshes the
controller cache from live sim state before pinning goals; measured drift
afterwards is 0.0001 cm.

### Frozen seed manifest

5 s of settling fixes most slots but not all — an object already balanced on
another does not recover. Those slots are replaced by seed substitution:
`manifest/gr1_episode_seed_manifest.json` maps
`{task: {base_seed: {episode_id: replacement_seed}}}`, and the wrapper does a
lookup, nothing more.

Slot identity is `(task, env lane, episode index)`; the nominal seed is
`(SEED + env_idx) * 100000 + episode_id`. Episode ids run **−1..8** for a
10-episode run, because the episode counter is read before it is incremented for
the video slot name — video `episode_k` is seed offset `k−1`. (Offset 9 entries
exist in the manifest and are dormant; they keep the table correct if that
numbering convention ever changes.)

## 3. The replacement rule

Judged at the **5.0 s** checkpoint — the state the episode actually starts from.
A slot is replaced when any holds:

| | condition | rationale |
|---|---|---|
| ① | pick object or destination container fell > 25 cm | it is off the table; the task is unperformable |
| ② | destination container tipped > 30° | nothing can be placed into an upturned container |
| ③ | anything lifted > 3 cm above its placement | only possible while resting on the robot or another object |
| ④ | bottle/can-family pick object (wine, bottled water, can, milk) tipped > 30° or angular speed > 0.5 rad/s | trained grasps assume upright; a fallen bottle is out of distribution |

Explicitly **kept**:

- **containers that merely slid** along the table — a basket pushed 17 cm was
  solved 3/3 by the reference policy; the policy sees where it is;
- **source container tipped or off the table**, as long as the pick object stays
  reachable — one such slot was still solved at 100k;
- **non-bottle objects lying on their side** — a potato or pear is equally
  graspable in any orientation.

Thresholds are frozen in `tools/build_gr1_seed_manifest.py`. Changing one means
regenerating the manifest, never re-judging during an eval.

### Replacements are held to a stricter bar

A replacement seed must be **strict-clean** (no critical object tipped, drifted
or displaced at all) **and** clean under the rule above. The second condition is
not redundant: the plain thresholds do not cover a lifted container, and an
early version of the builder assigned three wedged reserves before that check
was added. All 130 assignments were re-verified after the fix, including
measurement coverage — the verification treats "no data" as a failure rather
than a pass, since a silent skip would look identical to a pass.

Current margins across the 130 assigned seeds:

| metric | median | worst | threshold |
|---|---|---|---|
| tilt | 4.2° | 27.9° | 30° |
| lateral drift | 0.7 cm | 4.9 cm | 5 cm |
| vertical displacement | 2.0 cm | 7.9 cm | 8 cm |
| angular speed (bottles) | 0.000 | 0.297 | 0.5 rad/s |

The worst tilt cases are all croissants — a pick object that is *supposed* to
lie on its side, and not a rule target. The strictest rule-relevant number is
destination-container tilt: median 0.00°, worst 27.6°.

## 4. Effect

Same checkpoints, same weights, same EMA settings; only the protocol changed.
1,200 episodes per point.

| checkpoint | before | after | Δ |
|---|---|---|---|
| 60k | 40.9 % | 44.0 % | +3.1 pp |
| 80k | 44.4 % | 51.1 % | +6.7 pp |
| 100k | 48.7 % | 50.2 % | +1.5 pp |
| **mean** | **44.7 %** | **48.4 %** | **+3.8 pp** |

Decomposed, the picture is what the design predicts:

| | replaced 115 slots | untouched 1,085 slots |
|---|---|---|
| 60k | +2.8 pp | +0.3 pp |
| 80k | +3.2 pp | +3.9 pp |
| 100k | +3.2 pp | −1.8 pp |

The manifest's contribution is **uniform** across checkpoints, and success on
the replaced slots recovers from 13.9 / 16.5 / 20.0 % to 43.5 / 49.6 / 53.0 % —
i.e. to the clean-slot baseline. The per-checkpoint scatter comes entirely from
untouched slots, where settling nudges borderline episodes either way; adjacent
checkpoints disagree on 35 % of individual episodes, so ±2–4 pp there is noise,
not signal. This is also why the +3.8 pp is not a "free improvement": it is the
removal of a contaminating term that was never measuring the policy.

## 5. Auditing a run

Every substitution prints:

```
[seed-manifest] PnPWineToCabinetClose base=46 ep=-1: 4599999 -> 4600010
```

A correct 24-task run fires exactly 115 of these for the eval slots (the other
15 manifest entries are the dormant offset-9 rows). Cross-check against the
manifest rather than trusting the count alone:

```bash
grep -h '\[seed-manifest\]' output/*/rollout.log | wc -l
```

Per-task substitution counts are concentrated where physics is hardest —
`PnPWineToCabinetClose` 25, `PnPBottleToCabinetClose` 17, `TrayToTieredshelf` 11
— which is the expected shape: tall, narrow, top-heavy objects.
