"""Deterministic, physically-valid episode resets for the GR-1 tabletop benchmark.

Drop this wrapper directly around the raw sim env, inside every recording/step
wrapper, and each episode gets

  1. a **deterministic seed** derived from (env lane, episode index) rather than
     whatever RNG state the process happens to be in, and
  2. a **validated starting scene** — slots whose sampled placement is still
     broken after the fixed reset settle are swapped to a pre-validated
     replacement seed listed in a frozen manifest.

Upstream Isaac-GR00T seeds only the *first* reset of each sub-env
(``reset_seeds = [seed + i]`` in ``gr00t/eval/rollout_policy.py``), so episodes
2..N inherit whatever RNG state the process drifted into. Two runs of the same
checkpoint therefore do not evaluate the same episodes, and two different models
cannot be compared episode-for-episode. This wrapper closes that gap without
touching upstream internals.

Both behaviours are inert unless configured, so importing this module cannot
change an existing eval.

See docs/PROTOCOL.md for the measurement that motivated the manifest and for the
rule that decides which slots get replaced.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

__all__ = ["GR1ProtocolWrapper", "load_manifest", "episode_seed"]

DEFAULT_SEED_STRIDE = 100_000


def load_manifest(path: str | os.PathLike) -> Dict[str, Any]:
    """Read a seed manifest, dropping the ``_meta`` bookkeeping block."""
    with open(path) as fh:
        data = json.load(fh)
    data.pop("_meta", None)
    return data


def episode_seed(base_seed: int, episode_id: int, stride: int = DEFAULT_SEED_STRIDE) -> int:
    """The nominal seed of one slot. Kept as a function so external tooling
    (probe sweeps, manifest builders) can reproduce it without importing gym."""
    return int(base_seed) * int(stride) + int(episode_id)


class GR1ProtocolWrapper(gym.Wrapper):
    """Seed each episode deterministically and honour the seed manifest.

    Args:
        env: the environment to wrap. Place this **closest to the raw env** —
            inside the video recorder and the multi-step wrapper — so that any
            ``reset()`` those issue is seeded here.
        base_seed: this sub-env's lane seed. The convention used by the shipped
            manifest is ``base_seed = SEED + env_idx`` with ``SEED = 42``.
        seed_stride: multiplier separating lanes in seed space. Must match the
            value the manifest was built with (default 100000).
        start_episode_id: id handed to the first episode. The default of ``-1``
            reproduces the numbering the shipped manifest was measured with:
            the counter is read *before* it is incremented for the video slot
            name, so video ``episode_0`` corresponds to seed offset ``-1``.
        manifest: already-loaded manifest, or ``None`` to read the path in
            ``$GR1_EPISODE_SEED_MANIFEST`` (absent → no substitution).
        task_name: the task this env runs, used to look up manifest entries. If
            omitted, the class name of the underlying robosuite env is matched
            against the manifest keys by substring, which is how the task is
            identified in practice.
        verbose: print one ``[seed-manifest]`` line per substitution so a live
            run can be audited for *which* slots were swapped.

    The substitution decision is never re-derived at reset time: it is read from
    the frozen manifest. That is deliberate. Any accept/reject check evaluated
    during the run would sit on a floating-point boundary, and contact-solver
    jitter between nodes or library versions could then flip which episode a
    model sees — silently making two models' scores incomparable.
    """

    def __init__(
        self,
        env,
        base_seed: Optional[int] = None,
        seed_stride: int = DEFAULT_SEED_STRIDE,
        start_episode_id: int = -1,
        manifest: Optional[Dict[str, Any]] = None,
        manifest_path: Optional[str] = None,
        task_name: Optional[str] = None,
        verbose: bool = True,
    ):
        super().__init__(env)
        self.base_seed = base_seed
        self.seed_stride = int(seed_stride)
        self.episode_id = int(start_episode_id) - 1  # incremented on each reset
        self.task_name = task_name
        self.verbose = verbose

        if manifest is None:
            path = manifest_path or os.environ.get("GR1_EPISODE_SEED_MANIFEST")
            manifest = load_manifest(path) if path else {}
        self.manifest = manifest

    # -- internals ---------------------------------------------------------
    def _robosuite_env(self):
        """Descend past gym's OrderEnforcing/PassiveChecker to the robosuite env."""
        unwrapped = getattr(self.env, "unwrapped", None)
        if unwrapped is None:
            return None
        # robocasa's gym adapter keeps the robosuite env at `.env`
        return getattr(unwrapped, "env", unwrapped)

    def _manifest_seed(self, seed: int) -> int:
        if not self.manifest or self.base_seed is None:
            return seed
        task = self.task_name
        if task is None:
            rs = self._robosuite_env()
            cls = type(rs).__name__ if rs is not None else ""
            task = next((t for t in self.manifest if t in cls), None)
            if task is None:
                return seed
        alt = self.manifest.get(task, {}).get(str(int(self.base_seed)), {}).get(str(self.episode_id))
        if alt is None:
            return seed
        if self.verbose:
            print(
                f"[seed-manifest] {task} base={int(self.base_seed)} "
                f"ep={self.episode_id}: {seed} -> {int(alt)}",
                flush=True,
            )
        return int(alt)

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        # robosuite/robocasa keep their own Generator and never see reset(seed=...);
        # PosttrainPnPNovel* draw layout, style and object choices from it, so an
        # unseeded `rng` makes episode CONTENT depend on reset order rather than
        # on the seed. Reseed it explicitly.
        rs = self._robosuite_env()
        if rs is None:
            return
        if hasattr(rs, "seed"):
            rs.seed = seed
        if hasattr(rs, "rng"):
            rs.rng = np.random.default_rng(seed)

    # -- gym API -----------------------------------------------------------
    def reset(self, **kwargs):
        self.episode_id += 1
        if kwargs.get("seed") is None and self.base_seed is not None:
            seed = episode_seed(self.base_seed, self.episode_id, self.seed_stride)
            seed = self._manifest_seed(seed)
            self._seed_everything(seed)
            kwargs["seed"] = seed
        return self.env.reset(**kwargs)
