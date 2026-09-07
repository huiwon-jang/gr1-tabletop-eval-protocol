"""Measure GR-1 tabletop reset-state physical stability, replicating the eval's exact seeds.

Question (huiwon 2026-09-05): at env creation some objects are mid-fall, tipped, or gone —
states where the PnP task is unsolvable. How often, and what deterministic fix works?

What this does, per task, per episode seed (the EXACT schedule the eval uses:
episode_seed = (SEED+env_idx)*100000 + ep, env_idx 0..4, ep 0..9  — from
rollout_policy.py env_seed = seed+env_idx and VideoRecordingWrapper
base_seed*seed_stride+episode_id, SEED=42):

  1. gym reset with that seed  -> what the policy sees at t=0 (the env already ran its own
     0.5 s internal settle: tabletop.py _reset_internal steps 10 control steps of zero action).
  2. record every free-joint object: sampled placement pose (intended), pose+velocity at t0.
  3. keep settling WITHOUT any policy action for --settle-secs more, using the same mechanism
     as _reset_internal (sim.step1/_pre_action(zero, policy_step=False)/sim.step2 — the
     controller holds its setpoints, so the robot stays put; only physics acts on objects).
     Snapshot poses at checkpoints.
  4. save the t0 ego image for every episode (evidence for the artifact).

Classification happens in the ANALYSIS step, not here — this only records raw poses so
thresholds can be tuned without re-running 1200 resets.

Runs in the GR-1 sim venv (imports robocasa fork + robosuite). One task per invocation
(--task-index 0..23) so a SLURM array parallelises it; writes are per-task JSON, resumable.
"""
import argparse, json, os, sys, time
from pathlib import Path

TASKS = [  # exact order of eval_wam_dit4dit_kitchen.sh gr1_tabletop TASK_NAMES
    "PnPCupToDrawerClose", "PnPPotatoToMicrowaveClose", "PnPMilkToMicrowaveClose",
    "PnPBottleToCabinetClose", "PnPWineToCabinetClose", "PnPCanToDrawerClose",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA", "PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA", "PosttrainPnPNovelFromCuttingboardToPotSplitA",
    "PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA", "PosttrainPnPNovelFromPlacematToBasketSplitA",
    "PosttrainPnPNovelFromPlacematToBowlSplitA", "PosttrainPnPNovelFromPlacematToPlateSplitA",
    "PosttrainPnPNovelFromPlacematToTieredshelfSplitA", "PosttrainPnPNovelFromPlateToBowlSplitA",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA", "PosttrainPnPNovelFromPlateToPanSplitA",
    "PosttrainPnPNovelFromPlateToPlateSplitA", "PosttrainPnPNovelFromTrayToCardboardboxSplitA",
    "PosttrainPnPNovelFromTrayToPlateSplitA", "PosttrainPnPNovelFromTrayToPotSplitA",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA", "PosttrainPnPNovelFromTrayToTieredshelfSplitA",
]
ROBOT = "GR1ArmsAndWaistFourierHands"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-index", type=int, required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seed", type=int, default=42, help="eval SEED (base)")
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--eps-per-env", type=int, default=10)
    ap.add_argument("--settle-secs", type=float, default=5.0)
    ap.add_argument("--ep-list", default=None,
                    help="comma list of episode ids to probe in EVERY env lane (default 0..eps_per_env-1). "
                         "Used for the RESERVE seed pool (e.g. 10,11,...,34) when building the manifest.")
    ap.add_argument("--out-tag", default=None, help="suffix for the output json name")
    args = ap.parse_args()

    task = TASKS[args.task_index]
    out_dir = Path(args.out_root); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_json = out_dir / f"{args.task_index:02d}_{task}{tag}.json"
    if out_json.exists():
        print(f"[skip] {out_json.name} already exists"); return 0
    img_dir = out_dir / "img" / f"{args.task_index:02d}_{task}"; img_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import random
    import numpy as np
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  (registers gr1_unified/*)
    try:
        import cv2
    except Exception:
        cv2 = None

    env_name = f"gr1_unified/{task}_{ROBOT}_Env"
    env = gym.make(env_name, enable_render=True, seed=args.seed)
    rs = env.unwrapped.env  # robosuite tabletop env

    def obj_states():
        """name -> {qpos(7), linvel(3), angvel(3)} for every free-joint object placed by the sampler."""
        out = {}
        for name, (pos, quat, obj) in rs.object_placements.items():
            for j in obj.joints:
                try:
                    qp = rs.sim.data.get_joint_qpos(j)
                    if np.size(qp) != 7:
                        continue
                    qv = rs.sim.data.get_joint_qvel(j)
                    out[name] = {
                        "qpos": [float(x) for x in np.asarray(qp).ravel()],
                        "linvel": [float(x) for x in np.asarray(qv).ravel()[:3]],
                        "angvel": [float(x) for x in np.asarray(qv).ravel()[3:6]],
                    }
                except Exception:
                    pass
        return out

    def settle(n_control_steps):
        """The exact mechanism _reset_internal uses: physics on, controller holds, no policy."""
        zero = np.zeros(rs.action_spec[0].shape)
        sub = int(rs.control_timestep / rs.model_timestep)
        for _ in range(n_control_steps):
            for _ in range(sub):
                rs.sim.step1(); rs._pre_action(zero, False); rs.sim.step2()

    ctrl_hz = 1.0 / rs.control_timestep
    checkpoints = [0.5, 1.0, 2.0, 3.0, args.settle_secs]
    records = []
    t_start = time.time()
    # ':' separator supported because sbatch --export=A,B=1,2 would split a comma list
    # into separate variables (job 213526 silently probed ONE reserve ep per lane that way).
    ep_ids = ([int(x) for x in args.ep_list.replace(":", ",").split(",")] if args.ep_list
              else list(range(args.eps_per_env)))
    for env_idx in range(args.n_envs):
        for ep in ep_ids:
            episode_seed = (args.seed + env_idx) * 100000 + ep
            random.seed(episode_seed); np.random.seed(episode_seed)
            # ★ the eval wrapper ALSO reseeds the env's INTERNAL rng per episode
            # (VideoRecordingWrapper: robosuite_env.rng = default_rng(episode_seed)); without this
            # the Posttrain tasks' object/layout sampling is SEQUENCE-DEPENDENT and the probe
            # renders a different episode than the eval (caught 2026-09-05: same slot showed a
            # croissant in the probe, an eggplant in the eval).
            rs.rng = np.random.default_rng(episode_seed)
            rs.seed = episode_seed
            obs, _ = env.reset(seed=episode_seed)
            rec = {"env_idx": env_idx, "ep": ep, "seed": episode_seed,
                   "lang": rs.get_ep_meta().get("lang", "")}
            # intended (sampled) placement pose per object
            rec["placement"] = {n: {"pos": [float(x) for x in p], "quat": [float(x) for x in q]}
                                for n, (p, q, o) in rs.object_placements.items()}
            rec["t0"] = obj_states()
            # t0 ego image (what the policy sees first) — cotrain view, matches model input
            if cv2 is not None:
                img = obs.get("video.ego_view_bg_crop_pad_res256_freq20")
                if img is not None:
                    cv2.imwrite(str(img_dir / f"e{env_idx}_ep{ep}_t0.png"),
                                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            prev = 0.0
            snaps = {}
            for cp in checkpoints:
                settle(int(round((cp - prev) * ctrl_hz)))
                snaps[str(cp)] = obj_states()
                prev = cp
            rec["settled"] = snaps
            # post-settle ego render for the same viewpoint (best effort)
            if cv2 is not None:
                try:
                    fr = rs.sim.render(camera_name="egoview", width=1280, height=800)[::-1]
                    h, w = fr.shape[:2]; crop = fr[310:770, 110:1130]
                    r = cv2.resize(crop, (720, 480)); pad = (720 - 480) // 2
                    r = np.pad(r, ((pad, pad), (0, 0), (0, 0)))
                    r = cv2.resize(r, (256, 256))
                    cv2.imwrite(str(img_dir / f"e{env_idx}_ep{ep}_settled.png"),
                                cv2.cvtColor(r, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    rec["render_err"] = str(e)[:120]
            records.append(rec)
            print(f"[{task}] env{env_idx} ep{ep} seed={episode_seed} objs={len(rec['t0'])} "
                  f"({time.time()-t_start:.0f}s)", flush=True)
    out_json.write_text(json.dumps({"task": task, "robot": ROBOT, "settle_secs": args.settle_secs,
                                    "checkpoints": checkpoints, "records": records}))
    print(f"[done] {out_json} ({len(records)} episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
