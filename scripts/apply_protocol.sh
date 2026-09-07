#!/usr/bin/env bash
# Apply the GR-1 reset protocol to an Isaac-GR00T checkout.
#
#   scripts/apply_protocol.sh /path/to/Isaac-GR00T
#
# Applies three patches and installs this package into the checkout:
#   01 robocasa reset settle       -> <repo>/gr00t/eval/sim/robocasa-gr1-tabletop-tasks
#   02 robocasa determinism        -> same submodule
#   03 rollout hook                -> <repo>/gr00t/eval/rollout_policy.py
#
# Every patch is checked before it is applied and skipped if already present,
# so re-running is safe.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-}"

if [ -z "$TARGET" ]; then
    echo "usage: $0 /path/to/Isaac-GR00T" >&2
    exit 2
fi
TARGET="$(cd "$TARGET" && pwd)"
SIM="$TARGET/gr00t/eval/sim/robocasa-gr1-tabletop-tasks"

[ -f "$TARGET/gr00t/eval/rollout_policy.py" ] || {
    echo "[e] $TARGET does not look like an Isaac-GR00T checkout (no gr00t/eval/rollout_policy.py)" >&2
    exit 1
}
[ -d "$SIM/robocasa" ] || {
    echo "[e] GR-1 tabletop tasks not initialised at $SIM" >&2
    echo "    run: git -C '$TARGET' submodule update --init gr00t/eval/sim/robocasa-gr1-tabletop-tasks" >&2
    exit 1
}

apply_patch () {          # apply_patch <git-root> <patch>
    local root="$1" patch="$2" name
    name="$(basename "$patch")"
    if git -C "$root" apply --check "$patch" 2>/dev/null; then
        git -C "$root" apply "$patch"
        echo "  applied  $name"
    elif git -C "$root" apply --reverse --check "$patch" 2>/dev/null; then
        echo "  present  $name (already applied)"
    else
        echo "  FAILED   $name — does not apply to this revision" >&2
        return 1
    fi
}

echo "[1/2] patching"
apply_patch "$SIM"    "$HERE/patches/01-robocasa-reset-settle.patch"
apply_patch "$SIM"    "$HERE/patches/02-robocasa-determinism.patch"
apply_patch "$TARGET" "$HERE/patches/03-isaac-gr00t-rollout-hook.patch"

echo "[2/2] installing gr1_protocol into the checkout"
# A .pth entry keeps the package importable from whichever interpreter runs the
# rollout (the sim venv), without copying files or requiring a build step.
SITE="$(cd "$TARGET" && python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [ -n "$SITE" ] && [ -d "$SITE" ]; then
    echo "$HERE" > "$SITE/gr1_protocol.pth"
    echo "  linked   $SITE/gr1_protocol.pth -> $HERE"
else
    echo "  [w] could not resolve site-packages; add to PYTHONPATH instead:"
    echo "      export PYTHONPATH=\"$HERE:\$PYTHONPATH\""
fi

cat <<EOF

done. To evaluate under the protocol, set these before the rollout:

  export GR1_RESET_SETTLE_S=5.0
  export GR1_PROTOCOL_SEED=42
  export GR1_EPISODE_SEED_MANIFEST="$HERE/manifest/gr1_episode_seed_manifest.json"

See scripts/run_gr1_eval.sh for a complete N1.5 example.
EOF
