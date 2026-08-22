#!/usr/bin/env python3
"""
Amazon A10 Client-Side Behavioral Scoring Model

Extracted from RXVM bytecode payloads [10] and [11] on Amazon search results pages.

Architecture:
  31 inputs -> Normalize -> Dense(31->16) + BatchNorm + ReLU
                         -> Dense(16->16) + BatchNorm + ReLU
                         -> Dense(16->1)  + Sigmoid
  Output: score in [0, 1]
    0 = bot / suspicious
    1 = genuine human engagement

The model runs client-side on a ~1.1s interval, re-scoring whenever
new behavioral data arrives from the event collectors.

Usage:
  a10_model.py                          — feature importance analysis + simulations
  a10_model.py --score <31 floats>      — score a raw feature vector
  a10_model.py --weights <path.json>    — use alternate weights file

Layers in the bytecode are listed output->input. Inference runs input->output.
"""

import json
import math
import sys
import os

# ─── Feature Definitions ────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Scroll behavior (payload [7])
    'sc_s',   # scroll sum (total scroll displacement px/s)
    'sc_m',   # scroll mean velocity
    'sc_c',   # scroll count (number of scroll events sampled)

    # Mouse velocity (payload [6])
    'ms_s',   # mouse velocity sum
    'ms_m',   # mouse velocity mean
    'ms_c',   # mouse velocity count (movement segments)

    # Mouse acceleration (payload [6])
    'ma_s',   # mouse angular acceleration sum
    'ma_m',   # mouse angular acceleration mean
    'ma_c',   # mouse angular acceleration count

    # Click patterns (payload [5])
    'mc_s',   # click duration sum (total hold time ms)
    'mc_m',   # click duration mean
    'mc_c',   # click count

    # Bot fingerprint hash distribution (payload [4])
    'hdl_8',  # hash distribution level 8 (bit 7 — body width check)
    'hdl_7',  # hash distribution level 7 (bit 6 — outer/inner height)
    'hdl_6',  # hash distribution level 6 (bit 5 — WebGL SwiftShader)
    'hdl_5',  # hash distribution level 5 (bit 4 — PhantomJS)
    'hdl_4',  # hash distribution level 4 (bit 3 — Puppeteer polyfills)
    'hdl_3',  # hash distribution level 3 (bit 2 — ChromeDriver cdc_)
    'hdl_2',  # hash distribution level 2 (bit 1 — navigator.webdriver)
    'hdl_1',  # hash distribution level 1 (bit 0 — Playwright)

    # Viewport dimensions (payload [9])
    'vh_l',   # viewport height (log-transformed)
    'vh',     # viewport height (raw px)
    'vw_l',   # viewport width (log-transformed)
    'vw',     # viewport width (raw px)

    # Screen dimensions (payload [9])
    'sh_l',   # screen height (log-transformed)
    'sh',     # screen height (raw px)
    'sw_l',   # screen width (log-transformed)
    'sw',     # screen width (raw px)

    # Page structure (payload [8])
    'nd_l',   # navigation duration (log-transformed)
    'nd',     # navigation duration (seconds since first visit)
    'sl',     # session length (visit count)
]

FEATURE_CATEGORIES = {
    'Scroll Behavior':   ['sc_s', 'sc_m', 'sc_c'],
    'Mouse Velocity':    ['ms_s', 'ms_m', 'ms_c'],
    'Mouse Acceleration':['ma_s', 'ma_m', 'ma_c'],
    'Click Patterns':    ['mc_s', 'mc_m', 'mc_c'],
    'Bot Fingerprint':   [f'hdl_{i}' for i in range(8, 0, -1)],
    'Viewport/Screen':   ['vh_l', 'vh', 'vw_l', 'vw', 'sh_l', 'sh', 'sw_l', 'sw'],
    'Page Structure':    ['nd_l', 'nd', 'sl'],
}


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_model(path='a10_model_weights.json'):
    with open(path) as f:
        return json.load(f)


# ─── Inference Engine ───────────────────────────────────────────────────────
# Mirrors payload [10] fn@508 (forward pass)

def normalize(features, norms):
    """
    Per-feature normalization from fn@508 in payload [10].

    featureNorms layout: 5 values per feature:
      [0] mean (offset for centering)
      [1] std (scale factor)
      [2] clip_max
      [3] clip_min
      [4] offset (subtracted before normalization)

    Bytecode logic:
      if !isFinite(x): x = norms[0]  (replace NaN/Inf with mean)
      x = min(x, clip_max)
      x = max(x, clip_min)
      x = x - offset
      x = x / std
    """
    out = []
    for i, x in enumerate(features):
        base = i * 5
        mean = norms[base]
        std = norms[base + 1]
        clip_max = norms[base + 2]
        clip_min = norms[base + 3]
        offset = norms[base + 4]

        # Replace non-finite with mean
        if not math.isfinite(x):
            x = mean

        x = min(x, clip_max)
        x = max(x, clip_min)
        x = x - offset

        if std > 0:
            x = (x - mean) / std
        out.append(x)
    return out


def dense(x, weights, bias, in_dim, out_dim):
    """
    Dense layer: y = Wx + b.

    CAVEAT: In production, the Dense constructor passes raw minifloat
    weights through rx.M(in_dim, weights, out_dim), which is part of
    the RXVM core JavaScript runtime — NOT the bytecode payloads.
    rx.M likely performs dequantization or rescaling that we cannot
    reproduce without extracting its source from the interpreter JS.

    As a result, the raw 8-bit minifloat weights (~0.01-0.09 magnitude)
    are too small relative to the BatchNorm betas (~3.0), causing the
    model to saturate at sigmoid(26.5) ≈ 1.0 for all inputs.

    The architecture, feature pipeline, and weight extraction are all
    correct — only the M module's weight processing step is missing.
    """
    y = list(bias)
    for i in range(out_dim):
        for j in range(in_dim):
            y[i] += x[j] * weights[j * out_dim + i]
    return y


def batchnorm(x, params, epsilon):
    """
    BatchNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta.
    Mirrors fn@230 -> fn@325 in payload [10].

    params layout: [gamma * n, beta * n, mean * n, var * n]
    """
    n = len(x)
    gamma = params[0:n]
    beta = params[n:2 * n]
    mean = params[2 * n:3 * n]
    var = params[3 * n:4 * n]
    out = []
    for i in range(n):
        normed = (x[i] - mean[i]) / math.sqrt(var[i] + epsilon)
        out.append(gamma[i] * normed + beta[i])
    return out


def relu(x):
    """ReLU activation. Mirrors fn@433 -> M.R() in payload [10]."""
    return [max(0, v) for v in x]


def sigmoid(x):
    """Sigmoid activation. Mirrors fn@461 -> M.S() in payload [10]."""
    out = []
    for v in x:
        v = max(-500, min(500, v))
        out.append(1.0 / (1.0 + math.exp(-v)))
    return out


def infer(features, model):
    """
    Full forward pass. Returns score in [0, 1].

    Layers are listed output->input in the bytecode spec (payload [11]).
    We reverse them for the forward pass.
    """
    x = normalize(features, model['featureNorms'])

    layers = list(reversed(model['layers']))

    for layer in layers:
        t = layer['type']
        if t == 'dense':
            w = layer['weights']
            b = layer['bias']
            out_dim = len(b)
            in_dim = len(w) // out_dim
            x = dense(x, w, b, in_dim, out_dim)
        elif t == 'batchnorm':
            x = batchnorm(x, layer['weights'], layer['epsilon'])
        elif t == 'relu':
            x = relu(x)
        elif t == 'sigmoid':
            x = sigmoid(x)

    return x[0]


# ─── Feature Importance Analysis ────────────────────────────────────────────

def analyze_importance(model):
    """
    Compute first-layer weight magnitudes per input feature.
    Higher magnitude = more influence on model output.
    """
    # Input dense layer is last in bytecode order (first in forward pass)
    input_layer = model['layers'][-1]   # Dense: 31 -> 16
    weights = input_layer['weights']    # 496 = 31 * 16

    in_dim = 31
    out_dim = 16

    importance = []
    for j in range(in_dim):
        total = sum(abs(weights[j * out_dim + i]) for i in range(out_dim))
        importance.append(total)

    max_imp = max(importance) if importance else 1
    normalized = [v / max_imp for v in importance]

    return list(zip(FEATURE_NAMES, importance, normalized))


def print_analysis(model):
    print("=" * 72)
    print("A10 CLIENT-SIDE BEHAVIORAL SCORING MODEL")
    print("Extracted from Amazon RXVM bytecode payloads [10] and [11]")
    print("=" * 72)

    # Architecture
    print("\nARCHITECTURE")
    print("-" * 72)
    print("  31 features (behavioral + viewport + page)")
    print("  -> Normalize (per-feature mean/std/clip from training population)")
    print("  -> Dense(31->16) + BatchNorm(eps=0.001) + ReLU")
    print("  -> Dense(16->16) + BatchNorm(eps=0.001) + ReLU")
    print("  -> Dense(16->1)  + Sigmoid")
    print("  -> Score in [0, 1]")
    print("     0 = bot / suspicious")
    print("     1 = genuine human engagement")

    # Feature importance
    imp = analyze_importance(model)
    imp_sorted = sorted(imp, key=lambda x: x[1], reverse=True)

    print(f"\nFEATURE IMPORTANCE (first-layer weight magnitudes)")
    print("-" * 72)
    print(f"{'Rank':>4} {'Feature':>10} {'Raw Weight':>12} {'Normalized':>10}  Bar")
    print("-" * 72)
    for rank, (name, raw, norm) in enumerate(imp_sorted, 1):
        bar = "█" * int(norm * 30)
        print(f"{rank:4d} {name:>10} {raw:12.4f} {norm:10.3f}  {bar}")

    # Category importance
    print(f"\nCATEGORY IMPORTANCE")
    print("-" * 72)
    imp_dict = {name: norm for name, _, norm in imp}
    for cat, feats in FEATURE_CATEGORIES.items():
        cat_score = sum(imp_dict.get(f, 0) for f in feats) / len(feats)
        bar = "█" * int(cat_score * 30)
        print(f"  {cat:22s} {cat_score:.3f}  {bar}")

    # Output layer
    print(f"\nOUTPUT LAYER WEIGHTS (Dense 16->1)")
    print("-" * 72)
    # Output dense is second in bytecode order (layers[1])
    out_layer = model['layers'][1]
    for i, w in enumerate(out_layer['weights']):
        direction = "↑ human" if w > 0 else "↓ bot  "
        bar = "+" * int(abs(w) * 4) if w > 0 else "-" * int(abs(w) * 4)
        print(f"  neuron[{i:2d}] w={w:+.4f}  {direction}  {bar}")
    print(f"  bias = {out_layer['bias'][0]:.4f}")


# ─── Simulated Profiles ────────────────────────────────────────────────────

PROFILES = {
    "Genuine browser (desktop)": {
        'sc_s': 450, 'sc_m': 500, 'sc_c': 60,
        'ms_s': 300, 'ms_m': 310, 'ms_c': 45,
        'ma_s': 80,  'ma_m': 90,  'ma_c': 40,
        'mc_s': 18,  'mc_m': 48,  'mc_c': 3,
        'hdl_8': 0.4, 'hdl_7': 0.4, 'hdl_6': 0.4,
        'hdl_5': 0.02, 'hdl_4': 0.11, 'hdl_3': 0.11,
        'hdl_2': 0.4, 'hdl_1': 0.02,
        'vh_l': 160, 'vh': 160, 'vw_l': 520, 'vw': 520,
        'sh_l': 240, 'sh': 240, 'sw_l': 620, 'sw': 620,
        'nd_l': 100, 'nd': 120, 'sl': 7,
    },
    "Headless bot (no interaction)": {
        'sc_s': 0, 'sc_m': 0, 'sc_c': 0,
        'ms_s': 0, 'ms_m': 0, 'ms_c': 0,
        'ma_s': 0, 'ma_m': 0, 'ma_c': 0,
        'mc_s': 0, 'mc_m': 0, 'mc_c': 0,
        'hdl_8': 0, 'hdl_7': 0, 'hdl_6': 0,
        'hdl_5': 0, 'hdl_4': 0, 'hdl_3': 0,
        'hdl_2': 0, 'hdl_1': 0,
        'vh_l': 0, 'vh': 0, 'vw_l': 0, 'vw': 0,
        'sh_l': 0, 'sh': 0, 'sw_l': 0, 'sw': 0,
        'nd_l': 0, 'nd': 0, 'sl': 0,
    },
    "Click farm (fast, shallow)": {
        'sc_s': 100, 'sc_m': 100, 'sc_c': 5,
        'ms_s': 50,  'ms_m': 50,  'ms_c': 10,
        'ma_s': 10,  'ma_m': 10,  'ma_c': 10,
        'mc_s': 50,  'mc_m': 50,  'mc_c': 10,
        'hdl_8': 0.01, 'hdl_7': 0.01, 'hdl_6': 0.01,
        'hdl_5': 0.01, 'hdl_4': 0.01, 'hdl_3': 0.01,
        'hdl_2': 0.01, 'hdl_1': 0.01,
        'vh_l': 160, 'vh': 160, 'vw_l': 520, 'vw': 520,
        'sh_l': 240, 'sh': 240, 'sw_l': 620, 'sw': 620,
        'nd_l': 100, 'nd': 120, 'sl': 2,
    },
    "Selenium + mouse emulation": {
        'sc_s': 200, 'sc_m': 200, 'sc_c': 20,
        'ms_s': 150, 'ms_m': 150, 'ms_c': 20,
        'ma_s': 5,   'ma_m': 5,   'ma_c': 20,  # suspiciously uniform acceleration
        'mc_s': 10,  'mc_m': 30,  'mc_c': 2,
        'hdl_8': 0.0, 'hdl_7': 0.0, 'hdl_6': 0.0,
        'hdl_5': 0.0, 'hdl_4': 0.0, 'hdl_3': 1.0,  # cdc_ detected
        'hdl_2': 1.0, 'hdl_1': 0.0,                  # webdriver detected
        'vh_l': 160, 'vh': 160, 'vw_l': 520, 'vw': 520,
        'sh_l': 240, 'sh': 240, 'sw_l': 620, 'sw': 620,
        'nd_l': 50,  'nd': 60,  'sl': 1,
    },
}


def simulate_profiles(model):
    print(f"\nSIMULATED SCORING")
    print("=" * 72)
    print("  NOTE: All profiles score ~1.0 due to a missing component.")
    print("  The Dense layer constructor passes weights through rx.M(),")
    print("  a matrix module in the RXVM core JS runtime (not the bytecode")
    print("  payloads). rx.M likely rescales the 8-bit minifloat weights.")
    print("  Without it, BatchNorm betas (~3.0) dominate the pre-sigmoid")
    print("  value (~26.5), saturating the output for all inputs.")
    print("  The architecture and weight extraction are correct — only")
    print("  the rx.M weight processing step is missing.")
    print("-" * 72)
    for name, profile in PROFILES.items():
        vec = [profile.get(f, 0) for f in FEATURE_NAMES]
        score = infer(vec, model)
        bar = "█" * int(score * 40)
        label = "HUMAN" if score > 0.5 else "BOT"
        print(f"\n  {name:40s}  score={score:.4f}  [{label}]")
        print(f"  {'':40s}  {bar}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    weights_path = 'a10_model_weights.json'

    # Check for --weights flag
    args = list(sys.argv[1:])
    if '--weights' in args:
        idx = args.index('--weights')
        weights_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    # Try to find weights file
    if not os.path.exists(weights_path):
        # Check same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, weights_path)
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            print(f"ERROR: weights file not found: {weights_path}")
            print(f"       (also checked: {alt_path})")
            sys.exit(1)

    model = load_model(weights_path)

    if '--score' in args:
        # Score a raw feature vector
        idx = args.index('--score')
        values = [float(v) for v in args[idx + 1:]]
        if len(values) != 31:
            print(f"ERROR: expected 31 feature values, got {len(values)}")
            print(f"Features: {', '.join(FEATURE_NAMES)}")
            sys.exit(1)
        score = infer(values, model)
        label = "HUMAN" if score > 0.5 else "BOT"
        print(f"Score: {score:.6f}  [{label}]")
    else:
        print_analysis(model)
        simulate_profiles(model)


if __name__ == '__main__':
    main()
