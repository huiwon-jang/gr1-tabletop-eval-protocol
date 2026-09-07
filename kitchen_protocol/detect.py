"""Decide whether one recorded episode is contaminated.

The rule is deliberately small and pure: it takes an mp4 path and returns a
signature, and a second function turns that signature into a verdict. Nothing
here touches the cluster, so the thresholds can be re-derived on any machine
that has ffmpeg.

Three statistics are computed from a 192x64 grayscale decode of the episode:

  med    median absolute inter-frame difference. Large for *any* fast change,
         whether that is a real moving base or a corrupted frame.
  gap    median lag-2 correlation minus median lag-1 correlation. The view swap
         alternates with period 2, so frame t correlates with t+2 far better
         than with t+1 and this goes sharply positive. Continuous motion has
         lag-1 >= lag-2 and leaves it at or below zero.
  lag1m  adjacent correlation after downsampling each frame to 16x16. Real
         motion stays spatially continuous at macro scale (measured 0.95-0.999);
         a corrupted frame does not (measured 0.34-0.51). This is the axis that
         separates a fast base swing from a broken render.
"""
import re
import subprocess

import numpy as np

# ffmpeg only reports these on stderr when the bitstream is genuinely damaged.
# Error concealment hides the damage in the decoded pixels, so the pixel-domain
# statistics below cannot be used to detect it -- the stderr line is the signal.
CORRUPT_RE = re.compile(
    r"Invalid NAL|Error splitting|missing picture|decode_slice_header|"
    r"corrupt|Invalid data found"
)

W, H = 192, 64
#: Episodes shorter than this are not judged. ``terminate_on_success`` ends a
#: solved episode after a single action chunk, so a legitimate success can be
#: 16 frames long. Treating those as damaged makes every repair cycle re-flag
#: the same episodes and the loop never converges.
MIN_FRAMES = 40

SWAP_GAP = 0.12          #: gap above this == period-2 alternation, i.e. view swap
SUSTAINED_MED = 20.0     #: sustained large inter-frame change ...
MACRO_CONTINUITY = 0.7   #: ... that is also macro-discontinuous == non-physical


def episode_signature(path):
    """Return ``None`` (bitstream damaged), ``"short"`` (too short to judge), or a dict."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={W}:{H}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    )
    if CORRUPT_RE.search(r.stderr.decode("utf8", "ignore")):
        return None
    n = len(r.stdout) // (W * H)
    if n < MIN_FRAMES:
        return "short"

    g = np.frombuffer(r.stdout[: n * W * H], dtype=np.uint8)
    g = g.reshape(n, H, W).astype(np.float32)
    med = float(np.median(np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))))

    # Sample from one third in, where a rollout is underway but not yet ending.
    a = n // 3
    seg = g[a : min(n, a + 80)]
    lag1 = _median_corr(seg, 1)
    lag2 = _median_corr(seg, 2)

    macro = np.stack([_downsample16(f) for f in seg[:60]])
    return {
        "med": round(med, 2),
        "gap": round(lag2 - lag1, 3),
        "lag1m": round(_median_corr(macro, 1), 3),
    }


def _median_corr(frames, lag):
    return float(np.median([
        np.corrcoef(frames[i].ravel(), frames[i + lag].ravel())[0, 1]
        for i in range(len(frames) - lag)
    ]))


def _downsample16(frame):
    import cv2
    return cv2.resize(frame, (16, 16)).astype(np.float32)


def classify(sig):
    """Map a signature to a reason string, or ``None`` when the episode is clean."""
    if sig is None:
        return "undecodable"
    if sig == "short":
        return None
    if sig["gap"] > SWAP_GAP:
        return "swap-alt"
    if sig["med"] > SUSTAINED_MED and sig["lag1m"] < MACRO_CONTINUITY:
        return "sustained-nonphys"
    return None


def is_contaminated(sig):
    return classify(sig) is not None
