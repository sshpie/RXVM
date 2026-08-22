# Amazon RXVM: Client-Side Bot Detection System

## Overview

Amazon embeds a behavioral telemetry and bot-detection system in every product page, running silently in the browser. The system is built on a proprietary bytecode virtual machine called **RXVM** (RX Virtual Machine), which executes obfuscated payloads to collect user behavior, fingerprint automation tools, and run a client-side neural network that scores whether the session belongs to a human or a bot.

This document is the product of reverse-engineering that system end to end: the VM itself, its bytecode format, all 15 payloads extracted from a live Amazon search results page, the encryption layer that protects telemetry in transit, the `rxc` cookie format, and the trained neural network weights.

---

## 1. The Virtual Machine

### Bytecode Format

RXVM payloads are base64-encoded binary blobs embedded in page HTML via `rx.ex64("...")` calls. Each payload begins with a two-byte magic number (`0x52 0x58` — ASCII "RX"), followed by a version byte. The rest is a stream of XOR-obfuscated opcodes and operands.

The XOR key is derived from the current byte offset: `key = (offset % 127) + 37`. Critically, each closure body reseeds the key from its own starting offset, meaning that a disassembler that doesn't track closure boundaries will produce garbage for nested functions. This is a deliberate anti-analysis measure — static tools that try to decode the entire blob with a single key will fail partway through.

### Instruction Set

The VM supports roughly 30 opcodes covering the operations needed to express arbitrary JavaScript logic in a compact form:

**Storage**: `STORE_LOCAL`, `STORE_GLOBAL`, `PROP_SET` — local and global variable slots, property assignment.

**Control flow**: `JUMP`, `JMP_TRUE`, `JMP_FALSE` — relative jumps with signed 16-bit offsets.

**Functions**: `MAKE_CLOSURE`, `MAKE_NAMED_CLOSURE` — create function objects with a skip offset that lets the outer walk jump past the body. `RESTORE_ARGS` unpacks arguments into local slots. `RET` / `RET_VAL` return from closures.

**Calls**: `CALL_0` through `CALL_N`, `NEW_0` through `NEW_N` — function calls and constructor invocations with 0 to N arguments.

**Iteration**: `ITER_INIT`, `ITER_NEXT` — for-of style iteration with a jump-on-done target.

**Expressions**: A full set of binary operators (arithmetic, bitwise, comparison, logical, `in`, `instanceof`) encoded inline. Values are typed: varints, i32, f32, a custom minifloat encoding for compact weight storage, inline strings, string table references, arrays, and objects.

**Other**: `LOAD_STRINGS` populates a string table at the top of each payload. `PUSH` / `PUSH_ACC` manage a value stack. `NOP` pads alignment.

### Why Bytecode?

Standard JavaScript minification is reversible — tools like Prettier and de4js can reconstruct readable source from minified code. RXVM's bytecode eliminates that attack surface entirely. The logic is compiled to a binary format that only Amazon's interpreter can execute. The per-closure XOR reseeding adds a second layer: even if you understand the opcode table, you need to correctly track function boundaries and reseed at each one to decode the operand stream. Our v4 disassembler (`rxvm_disasm_v4.py`) handles this by walking the top-level code first, recording all closure `(body_addr, end_addr)` pairs, then disassembling each closure body with a freshly seeded reader.

---

## 2. The Crypto Layer — Payload [0]

The first payload (1344 bytes, 27 strings, 12 closures) is entirely devoted to setting up the encrypted channel through which all collected telemetry flows. No behavioral data is gathered here — this is pure infrastructure.

### Key Derivation (fn@979)

When both a session ID (`sid`) and request ID (`rid`) are available, the payload derives encryption materials:

1. `SHA-256(sid)` → raw digest bytes are saved to `G5` (used as the RC4 fallback key), then imported via `crypto.subtle.importKey` as an AES-128-CBC `CryptoKey` → stored in `G4`.
2. `SHA-256(rid)` → first 16 bytes of the digest become the AES-CBC initialization vector → stored in `G6`.

Both derivations chain through a Promise adapter (fn@627) that bridges standard WebCrypto Promises with IE11's event-based `msCrypto` API.

### Encryption Paths

**AES-CBC (fn@702)** — the primary path. Prepends a 4-byte little-endian Unix timestamp to the plaintext (replay protection), then encrypts via `crypto.subtle.encrypt` with the derived key and IV. Used on all modern browsers.

**RC4 fallback (fn@854)** — identical timestamp-prefix structure, but encrypts with a textbook RC4 implementation (fn@433) using the raw SHA-256 digest bytes as the key. Activated when WebCrypto is unavailable. The RC4 code includes a full KSA (key scheduling algorithm) and PRGA (pseudo-random generation algorithm), with an extracted array-swap helper (fn@402) since RXVM has no destructuring.

### Helper Functions

**TextEncoder wrapper (fn@223)**: `(str) => new TextEncoder().encode(str)` — converts strings to Uint8Array. Used by every downstream crypto function.

**Custom hash — xxH1 mix + murmur3 fmix32 (fn@263)**: A non-standard 32-bit hash function assigned to `L8` and exported as `rx.h`. Seeded with `0xDEADBEEF`. The mixing loop uses xxHash32's `PRIME32_1` (`0x9E3779B1` / 2654435761) as a per-byte multiply-after-XOR — structurally closer to FNV than to Murmur3, which processes 4-byte blocks with rotations. The finalization is Murmur3's `fmix32` avalanche: `h ^= len; h ^= h >>> 16; h = imul(0x85EBCA6B, h); h ^= h >>> 13; h = imul(0xC2B2AE35, h); h ^= h >>> 16; return h >>> 0`. This hybrid was almost certainly chosen for bytecode compactness — no block buffering or rotations needed — while the fmix32 finish ensures adequate bit distribution. Used for hashing session and request identifiers, not for key derivation (that goes through SHA-256).

**Promise/callback adapter (fn@627)**: Bridges Promise-based WebCrypto (`result.then(callback)`) and IE11's msCrypto (`result.oncomplete = wrapper`).

### Exported API

At the end of payload [0], the crypto functions are registered on the `rx` object:
- `rx.ep` → AES-CBC encrypt (fn@702, wrapped via `exec`)
- `rx.ep4` → RC4 fallback encrypt (fn@854)
- `rx.e4` → raw RC4 (fn@433)
- `rx.h` → custom hash (fn@263)

Key derivation fires automatically when `sid` and `rid` are both present. Debug mode (`config.DEBUG`) exports derived keys as `rx.__erk`, `rx.__eri`, and `rx.__er`.

---

## 3. The Statistics Library — Payload [1]

Payload [1] (871 bytes, 7 closures) provides the mathematical primitives that all behavioral payloads use to summarize raw event streams into feature vectors. It exports:

- `rx.asum` (fn@84) — array sum
- `rx.aavg` (fn@116) — array mean
- `rx.astd` (fn@136) — array standard deviation
- `rx.sa` (fn@206) — summary statistics: returns `[count & 0xFFFF, mean & 0xFFFF, stddev & 0xFFFF]`
- `rx.sab` (fn@288) — byte-serialized summary: splits the 16-bit values from `sa` into high/low bytes for compact wire encoding
- `rx.ef16` (fn@356) — IEEE 754 half-precision (float16) encoder: converts a JavaScript number to a 2-byte representation for model scores
- `rx.df16` (fn@600) — float16 decoder: inverse of `ef16`

The float16 codec is notable — it implements full subnormal handling, sign bit extraction, and exponent biasing in ~200 bytes of bytecode. This is how neural network output scores get packed into the telemetry stream without wasting bandwidth.

---

## 4. Cookie Fingerprint — Payload [2]

Payload [2] (194 bytes, 1 closure) generates the `fnp` value used as the RC4 key for the `rxc` cookie. It concatenates `navigator.userAgent` with `rx.sid` (session ID), hashes the result with `rx.h` (the custom hash), and stores:

- `rx.fnp` — the 32-bit hash value
- `rx.fnpb` — the hash split into 4 bytes (little-endian): `[h & 0xFF, (h >>> 8) & 0xFF, (h >>> 16) & 0xFF, (h >>> 24) & 0xFF]`
- `rx.fnpv` — version number (currently 0)

These values end up as the first 5 bytes of every `rxc` cookie (version byte + 4-byte key), meaning the RC4 "encryption" key is deterministically derived from public information. The obfuscation is against casual inspection, not a determined attacker.

---

## 5. State Persistence — Payload [3]

Payload [3] (717 bytes, 7 closures) manages encrypted local state via `localStorage`. It provides:

**`rx.gd` (fn@140)** — reads the `rx` key from `localStorage`, base64-decodes it, extracts an RC4-encrypted blob, decrypts with `rx.e4` using `fnpb` as the key, and JSON-parses the result. Returns the stored state object (defaulting to `{ef: {}}`).

**`rx.sfe` (fn@517)** — the exponential moving average (EMA) updater. Takes a feature name, new value, old value, and smoothing factor (default 0.5). Computes `new_ema = α * new_value + (1 - α) * corrected_old`. Clamps to zero if the absolute value falls below 0.1. This is how behavioral features get smoothed across interactions — raw event spikes are dampened into stable signals.

**State save (fn@299)** — JSON-stringifies the state, TextEncoder-encodes it, RC4-encrypts with `fnpb`, prepends a version header, base64-encodes, and writes back to `localStorage` under the `rx` key. Increments `rx.sv` (state version counter).

**Deferred execution (fn@481, fn@420)** — a `setTimeout`-based batch system that queues EMA updates and flushes them periodically, avoiding per-event writes to `localStorage`.

---

## 6. Bot Fingerprinting — Payload [4]

Payload [4] (845 bytes, 9 closures) runs a battery of automation-detection heuristics. Each test is a separate closure that returns `true` if a bot indicator is detected:

**Playwright (fn@115)**: Checks for `window.playwright`.

**WebDriver (fn@138)**: Checks `navigator.webdriver` or `'webdriver' in window`.

**ChromeDriver (fn@155)**: Iterates `Object.keys(window)` looking for any property starting with `cdc_` — the prefix ChromeDriver injects.

**Puppeteer / Selenium polyfills (fn@206)**: Scans `window` for properties containing `_Array`, `_Symbol`, or `_Proxy` where the value equals the native constructor. These appear when automation frameworks patch globals.

**PhantomJS (fn@368)**: Checks for `_phantom` or `callPhantom` on `window`.

**Headless WebGL (fn@404)**: Creates a canvas, gets a WebGL context, queries `WEBGL_debug_renderer_info` for the unmasked renderer string, and checks for `SwiftShader` — the software renderer used by headless Chrome.

**Window dimensions (fn@603)**: Tests `outerHeight === innerHeight` — in headless browsers, there's no chrome/toolbar, so these are equal.

**Body width (fn@642)**: Tests `body.clientWidth === innerWidth` — another headless tell where there's no scrollbar.

The results are packed into a bitmask (`L1 |= (1 << L2)` per test) and reported via `rx.p()` with metric ID 16, along with per-feature EMA updates via `rx.sfe` for the `hdl_N` features that feed the neural network.

---

## 7. Mouse Timing — Payload [5]

Payload [5] (615 bytes, 6 closures) captures mousedown/mouseup timing on `document`. On each mouseup, it computes the hold duration (`performance.now()` delta) and pushes it to an array. After collecting 15 samples, it computes summary statistics via `rx.sa` and feeds them to the EMA tracker as `mc_c` (click count), `mc_m` (click mean duration), and `mc_s` (click sum). The raw byte-serialized stats also get sent via `rx.p()` with metric ID 32.

Listeners are attached on page load and removed on unload, with an `unload` handler that flushes any partial data via `rx.sab`.

---

## 8. Mouse Movement — Payload [6]

Payload [6] (936 bytes, 5 closures) is the most complex behavioral collector. It attaches a `mousemove` listener and tracks:

- **Velocity**: Euclidean distance between consecutive positions divided by time delta, in pixels/second (computed via `Math.ceil((distance / timeDelta) * 1000)`)
- **Angular acceleration**: The absolute angular difference between consecutive movement directions (`Math.atan2`-based), wrapped to the smaller arc (`min(|Δθ|, 2π - |Δθ|)`)
- **Stale cursor detection**: If the time gap between moves exceeds 200ms, all tracking state resets — this prevents idle time from contaminating velocity estimates

After 255 samples, it computes summary statistics for both velocity and acceleration arrays, producing six EMA-tracked features: `ms_c`, `ms_m`, `ms_s` (velocity count/mean/sum) and `ma_c`, `ma_m`, `ma_s` (acceleration count/mean/sum). Sent with metric ID 33.

This is one of the strongest bot signals. Real human mouse movement has characteristic jitter in both velocity and direction. Automation tools (even those using Bézier curve interpolation) produce unnaturally smooth or periodic acceleration profiles.

---

## 9. Scroll Behavior — Payload [7]

Payload [7] (631 bytes, 5 closures) mirrors the mouse movement collector but for scroll events. It tracks `window.scrollY` changes, computes scroll velocity (pixels/second), and produces `sc_c` (scroll count), `sc_m` (scroll mean velocity), and `sc_s` (scroll sum velocity) via the same EMA pipeline. Metric ID 34.

Same 200ms stale-detection cutoff and 255-sample cap as the mouse movement collector. Same lifecycle pattern of attaching on load, detaching on unload with a flush.

---

## 10. Paint Timing / Navigation Stats — Payload [8]

Payload [8] (360 bytes, 1 closure) measures page load characteristics using `performance.timing.navigationStart` and `localStorage`-based visit counting. It tracks:

- `sl` — "scroll length" / session length: capped visit count across page loads
- `nd` — navigation duration: seconds since first recorded visit
- `nd_l` — log-transformed navigation duration

Values are persisted to `localStorage` under `rx-pnt` as `timestamp|visit_count`, allowing cross-page-load tracking. Metric ID 35.

---

## 11. Viewport / Screen Dimensions — Payload [9]

Payload [9] (319 bytes, 1 closure) captures device geometry on page load:

- `sw` / `sw_l` — `screen.width` and log-transformed
- `sh` / `sh_l` — `screen.height` and log-transformed
- `vw` / `vw_l` — `innerWidth` and log-transformed
- `vh` / `vh_l` — `innerHeight` and log-transformed

All values are EMA-tracked via `rx.sfe`. The float16-encoded screen dimensions are also sent directly via `rx.p()`. Metric ID 48.

The log-transformed variants let the neural network be sensitive to both the raw resolution (distinguishing mobile from desktop) and the relative scale (distinguishing unusual viewport ratios that might indicate headless configurations).

---

## 12. ML Runtime — Payload [10]

Payload [10] (1026 bytes, 11 closures) implements the client-side neural network inference engine. It defines the building blocks:

**Dense layer (fn@160)**: Matrix multiply via an extracted `M.mul` helper, plus bias addition. Stores `weights`, `bias`, and a `forward(input)` method.

**BatchNorm layer (fn@230)**: Extracts gamma, beta, running mean, and running variance from a flat weight array. Forward pass: `gamma * (x - mean) / sqrt(variance + epsilon) + beta`.

**ReLU (fn@433)**: Wraps `M.R()` — max(0, x) element-wise.

**Sigmoid (fn@461)**: Wraps `M.S()` — 1 / (1 + exp(-x)).

**Model (fn@489)**: The full pipeline. `forward(features)` first normalizes inputs (per-feature mean/std/clip from `featureNorms`), converts to a matrix, then chains through all layers in order.

**Specs parser (fn@717)**: Factory function that takes a JSON model specification (layer types, weights, biases, norms) and instantiates the appropriate layer objects. Exported as `rx.ML.specs`.

Debug mode exports the layer constructors as `rx.ML.Dense`, `rx.ML.BatchNorm`, `rx.ML.ReLU`, `rx.ML.Sigmoid`.

---

## 13. ML Scoring — Payload [11]

Payload [11] (2875 bytes) is the largest payload. It contains the trained model weights inline as minifloat-encoded arrays and orchestrates inference:

### Architecture

```
31 inputs → Normalize → Dense(31→16) + BatchNorm + ReLU → Dense(16→16) + BatchNorm + ReLU → Dense(16→1) + Sigmoid → score ∈ [0, 1]
```

### Input Features (31 total)

| Category | Features | Source |
|---|---|---|
| Scroll behavior | `sc_s`, `sc_m`, `sc_c` | Payload [7] |
| Mouse velocity | `ms_s`, `ms_m`, `ms_c` | Payload [6] |
| Mouse acceleration | `ma_s`, `ma_m`, `ma_c` | Payload [6] |
| Click patterns | `mc_s`, `mc_m`, `mc_c` | Payload [5] |
| Bot fingerprint hash | `hdl_1` through `hdl_8` | Payload [4] |
| Viewport dimensions | `vh`, `vh_l`, `vw`, `vw_l` | Payload [9] |
| Screen dimensions | `sh`, `sh_l`, `sw`, `sw_l` | Payload [9] |
| Page structure | `nd`, `nd_l`, `sl` | Payload [8] |

### Scoring Loop

A `setInterval` fires at a configured cadence (1100ms default). On each tick, it checks if `rx.sv` (state version) has changed since the last run — if not, it skips (no new data). Otherwise, it reads the current EMA state via `rx.gd()`, extracts feature values by name, runs `model.forward(features)`, float16-encodes the score, and reports it via `rx.p()` and `rx.pc()` with metric ID 48.

### Output

The sigmoid output maps to: **0 = bot / suspicious**, **1 = genuine human engagement**. The score is shipped to Amazon's servers encrypted via the payload [0] crypto layer.

### Extracted Weights

The model weights are fully extracted in `a10_model_weights.json` (1165 lines). Per-feature normalization stats (mean, std, clip_max, clip_min, offset — 5 values per feature, 155 total), plus all dense layer weights, biases, and batchnorm parameters (gamma, beta, running mean, running variance).

Dense layer weights are stored in the bytecode as 8-bit minifloat arrays (`F[n]` notation) with a 1+4+3 bit layout (sign, exponent with bias 7, mantissa). This gives ~120 distinct positive values, which is very coarse — the 31→16 layer's 496 weights only occupy 43 distinct levels.

### Missing Component: rx.M

The Dense layer constructor in payload [10] passes the raw minifloat weight array through `rx.M(in_dim, weights, out_dim)` before storing the processed result. `rx.M` is a matrix math module that is part of the RXVM **core JavaScript runtime**, not the bytecode payloads. It is loaded as `L0["rx"]["M"]` and provides `M.mul()` (used in the Dense forward pass), plus the matrix infrastructure that `M.R()` (ReLU) and `M.S()` (sigmoid) operate on.

Without `rx.M`, we cannot determine whether the minifloat weights undergo dequantization, rescaling, or other processing at instantiation time. In our Python reimplementation, the raw 8-bit weights (~0.01–0.09 magnitude) are too small relative to the second BatchNorm's beta values (~3.0). This causes the pre-sigmoid activation to be dominated by a constant offset (~26.5), making `sigmoid(26.5) ≈ 1.0` for all inputs regardless of behavioral features.

The architecture, feature pipeline, normalization, and weight extraction are all verified correct. The scoring saturation is specifically caused by the missing `rx.M` weight processing step. To complete the model: extract `rx.M` from the RXVM interpreter's plaintext JavaScript (the non-base64 portion of the `rx` initialization script on Amazon pages).

### Feature Importance

First-layer weight magnitude analysis from `a10_model.py` shows the neural network weighs behavioral signals (mouse, scroll, click patterns) most heavily. Viewport/screen dimensions and page structure features carry moderate weight. The `hdl_N` bot fingerprint features provide binary "this is definitely automation" signals but don't dominate the overall score — the model can flag sophisticated bots that pass all fingerprint checks but have inhuman movement patterns.

---

## 14. Strong Interaction Detection — Payload [12]

Payload [12] (673 bytes, 6 closures) is a simpler behavioral gate that runs alongside the neural network. It attaches `mousemove`, `click`, and `scroll` listeners and tracks:

- **Mouse distance**: Euclidean distance between consecutive positions, accumulated in `G17`
- **Scroll displacement**: Absolute `scrollY` deltas, accumulated in `G16`

When either threshold is met (100px of mouse movement or 100px of scroll), it fires `rx.tag('has-strong-interaction')` and reports via Amazon's `ue`/`uex` telemetry as an `'at'` (action tracking) event. This is a coarser signal than the ML score — a binary "the user physically engaged with the page" flag used for ad attribution and engagement metrics.

---

## 15. Cookie Action Tracking — Payload [13]

Payload [13] (304 bytes, 3 closures) reads the `rx` cookie, extracts a base64-encoded payload from after the `@` delimiter, decodes and parses the first byte as an action level, and calls `rx.tag('rx-highest-action:' + level)`. This lets Amazon's server-side systems track the highest interaction tier observed across the session.

---

## 16. Perplexity / AI Agent Detection — Payload [14]

Payload [14] (1193 bytes, 7 closures) is one of the more interesting payloads. It specifically targets AI browsing agents — particularly Perplexity AI's browser extension/agent:

**DOM scanning (fn@791)**: Queries `document.querySelectorAll('[id^="pplx-agent"]')` and tests each element's ID against a regex matching Perplexity's overlay pattern: `^pplx-agent(?:-[0-9]+_[0-9]+)?-overlay(?:-(?:base|-stop-butt...`.

**CSS scanning**: Also searches `<style>` elements' `textContent` and all stylesheet `cssRules` for the same pattern — catching cases where the agent injects styles rather than visible DOM.

**Cookie-based tracking**: When detected, it sets cookies (`amzn-ctxv-id` and `amzn-css-id`) with randomized TTLs (6–7 days for DOM detection, 83–100 days for CSS detection) to persist the detection signal across sessions. Cookies are set with `Path=/; SameSite=Lax` and `Secure` on HTTPS.

**Periodic polling**: Detection runs on a `setInterval` (default 2000ms) with a `setTimeout` safety cap (default 120000ms), continuously scanning for agents that inject after initial page load.

This payload demonstrates that Amazon is actively monitoring for AI-powered shopping agents and browser extensions that scrape or overlay content on their pages.

---

## 17. The `rxc` Cookie

The `rxc` cookie is the wire format for accumulated behavioral metrics. After base64 decoding:

- **Byte 0**: `fnpv` — version (currently 0)
- **Bytes 1–4**: `fnpb` — the 4-byte RC4 key (derived from `hash(userAgent + sessionId)`, shipped in cleartext)
- **Bytes 5+**: RC4 ciphertext

After RC4 decryption:

- **Bytes 0–3**: Unix timestamp (uint32 LE) — when `ep4` was last called
- **Bytes 4+**: Metric stream — sequential `[metric_id, ...data_bytes]` pairs serialized by the `E()` function

The `rxc_decrypt.py` tool decodes and decrypts these cookies, producing hex dumps and metric stream breakdowns.

---

## 18. System Architecture Summary

```
                         Amazon Page Load
                              │
                    ┌─────────┴──────────┐
                    │  RXVM Interpreter   │
                    │  (base64 → XOR →   │
                    │   bytecode exec)    │
                    └─────────┬──────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                  │
     ┌──────┴──────┐  ┌──────┴──────┐  ┌───────┴───────┐
     │  Payload 0  │  │ Payloads    │  │  Payload 10   │
     │  Crypto     │  │ 1–9, 12–14  │  │  ML Runtime   │
     │  Layer      │  │ Collectors  │  │               │
     └──────┬──────┘  └──────┬──────┘  └───────┬───────┘
            │                │                  │
            │         ┌──────┴──────┐           │
            │         │  rx.sfe()   │     ┌─────┴─────┐
            │         │  EMA State  │────▶│ Payload 11 │
            │         │  (localStorage) │ │ ML Scoring │
            │         └─────────────┘    │ (31→16→16→1)│
            │                            └─────┬──────┘
            │                                  │
            └──────────────┬───────────────────┘
                           │
                    ┌──────┴──────┐
                    │  Encrypted  │
                    │  Telemetry  │
                    ├─────────────┤
                    │ AES-128-CBC │
                    │ (or RC4)    │
                    ├─────────────┤
                    │  rxc cookie │
                    │  (RC4)      │
                    └──────┬──────┘
                           │
                    Amazon Servers
```

### Data Flow

1. **Boot**: RXVM interpreter decodes and executes 15 bytecode payloads.
2. **Key derivation**: SHA-256 of session ID → AES key; SHA-256 of request ID → IV.
3. **Event capture**: Mouse, scroll, click listeners fire; DOM/WebGL fingerprints run.
4. **Aggregation**: Raw events are summarized (count/mean/stddev), EMA-smoothed, and persisted to encrypted localStorage.
5. **Scoring**: Every ~1.1 seconds, the neural network reads the current feature state and produces a [0, 1] human-likelihood score.
6. **Exfiltration**: Scores and metrics are timestamp-prefixed, AES-encrypted (or RC4 on legacy browsers), and shipped to Amazon. A subset is also RC4-encrypted into the `rxc` cookie.

### Tooling

| Tool | Purpose |
|---|---|
| `rxvm_disasm.py` | v2 disassembler — single-pass, no closure reseeding |
| `rxvm_disasm_v4.py` | v4 disassembler — closure body disassembly with XOR key reseeding, boundary enforcement, warnings |
| `rxc_decrypt.py` | Decrypts `rxc=` cookies — extracts version, RC4 key, timestamp, metric stream |
| `a10_model.py` | Standalone Python reimplementation of the A10 neural network — loads extracted weights, runs inference, analyzes feature importance, simulates scoring profiles |
| `a10_model_weights.json` | Full extracted model weights — normalization stats, dense weights/biases, batchnorm parameters |
| `rxvm_v4_full_disasm.txt` | Complete disassembly of all 15 payloads from a live Amazon page |
