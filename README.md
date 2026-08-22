[![Claude Code Friendly](https://img.shields.io/badge/Claude_Code-Friendly-blueviolet?logo=anthropic&logoColor=white)](https://claude.ai/code)

# RXVM — Amazon Client-Side Bot Detection (Reverse Engineering POC)

Reverse engineering of Amazon's RXVM bytecode virtual machine and its
client-side bot detection / behavioral telemetry system.

## Files

| File | Description |
|---|---|
| `RXVM_Paper.pdf` | Full academic research paper (12 pages, peer-review format) |
| `A10_Report.pdf` | Visual technical report — architecture diagrams, feature importance, model analysis |
| `ANALYSIS.md` | Full technical writeup — architecture, all 15 payloads, data flow |
| `rxvm_disasm.py` | RXVM bytecode disassembler (v4) — XOR key reseeding, closure discovery |
| `rxc_decrypt.py` | Amazon `rxc=` cookie decryptor — RC4 decryption, metric stream parsing |
| `a10_model.py` | A10 neural network reimplementation — inference, feature importance, simulation |
| `a10_model_weights.json` | Extracted model weights — norms, dense, batchnorm, all layers |
| `extract_rx_runtime.py` | Utility to extract rx.M matrix module from Amazon page HTML |
| `rx_M_extracted.js` | Deobfuscated rx.M source — confirmed pass-through, no weight rescaling |
| `payload0_crypto_analysis.txt` | Annotated disassembly of the crypto payload with JS equivalents |
| `rxvm_crypto_layer_analysis.docx` | Detailed crypto layer analysis document |
| `rxvm_full_disasm.txt` | Complete disassembly of all 15 payloads from a live Amazon page |

## Quick Start

### Disassemble RXVM payloads from an Amazon page

```bash
python3 rxvm_disasm.py page.html
```

### Disassemble a single base64 payload

```bash
python3 rxvm_disasm.py -b64 "UlgB..."
```

### Decrypt an rxc cookie

```bash
python3 rxc_decrypt.py "AGVoY2hlY2..."
```

### Analyze the neural network

```bash
python3 a10_model.py
```

### Score a custom feature vector

```bash
python3 a10_model.py --score 450 500 60 300 310 45 80 90 40 18 48 3 \
    0.4 0.4 0.4 0.02 0.11 0.11 0.4 0.02 \
    160 160 520 520 240 240 620 620 100 120 7
```

## Known Limitation: Neural Network Scoring

The A10 model reimplementation extracts the correct architecture, features, and weights, but
all profiles currently score ~1.0. The Dense layer constructor in the bytecode passes weights
through `rx.M()`, a matrix module in the RXVM core JS runtime (not the bytecode payloads).
This module likely rescales the 8-bit minifloat weights at instantiation time. Without it,
BatchNorm betas dominate the pre-sigmoid value, saturating the output.

To fix: extract `rx.M` from the RXVM interpreter's plaintext JavaScript on Amazon pages.

## No External Dependencies

All tools are pure Python 3, standard library only. No pip installs required.

## System Overview

```
Amazon Page HTML
    └── rx.ex64("base64...") × 15 payloads
         │
         ├── [0]  Crypto layer (AES-128-CBC + RC4 fallback)
         ├── [1]  Statistics library (sum, mean, stddev, float16 codec)
         ├── [2]  Cookie fingerprint (hash(UA + SID) → fnpb key)
         ├── [3]  State persistence (EMA tracking via encrypted localStorage)
         ├── [4]  Bot fingerprinting (Playwright, WebDriver, ChromeDriver,
         │        Puppeteer, PhantomJS, SwiftShader, headless dimension checks)
         ├── [5]  Mouse click timing (mousedown/mouseup duration)
         ├── [6]  Mouse movement (velocity + angular acceleration)
         ├── [7]  Scroll behavior (scroll velocity)
         ├── [8]  Navigation timing (visit count, session duration)
         ├── [9]  Viewport / screen dimensions
         ├── [10] ML runtime (Dense, BatchNorm, ReLU, Sigmoid layers)
         ├── [11] ML scoring (31→16→16→1 neural net, weights inline)
         ├── [12] Strong interaction detection (mouse/scroll threshold gate)
         ├── [13] Cookie action tracking (highest interaction tier)
         └── [14] AI agent detection (Perplexity browser agent regex scanning)
```

## Responsible Disclosure

Responsibly disclosed to Amazon. Confirmed by Amazon.

| Date | Event |
|------|-------|
| Mar 12, 2026 | Initial discovery during consumer research into A10 algorithm |
| Mar 14, 2026 | Reverse engineering complete, all tools validated across multiple pages |
| Mar 14, 2026 | Responsible disclosure submitted to Amazon Security |
| Mar 16, 2026 | Confirmed by Amazon |

## About

Research by **[Nicholas Michael Kloster](https://github.com/Nicholas-Kloster)** as part of [****](https://) — independent AI infrastructure security research. Tools built collaboratively with Claude (Anthropic).

All analysis was performed on publicly-served JavaScript executing in the researcher's own browser. No exploitation, no circumvention, no access to Amazon's server-side systems.

CISA disclosures: [CVE-2025-4364](https://nvd.nist.gov/vuln/detail/CVE-2025-4364) · [ICSA-25-140-11](https://www.cisa.gov/news-events/ics-advisories/icsa-25-140-11)


## Use with Claude Code

Use Claude Code to run the RXVM disassembler and A10 decryptor against fresh Amazon traffic, or extend the neural network analysis.

```
Read README.md and ANALYSIS.md in this repo (RXVM).
Then:
1. Run rxvm_disasm.py against a fresh Amazon page source and show me what the
   current VM version looks like compared to the documented payload set
2. Help me understand how the A10 neural network (31→16→16→1) classifies
   bot vs. human behavior — which input features matter most?
3. Run rxc_decrypt.py against a current rxc= cookie and walk me through the metric stream
Target URL or cookie: [paste here]
```

```
I want to understand how AI agent detection works in the RXVM system.
Read README.md and ANALYSIS.md, then:
1. Identify which behavioral signals in the A10 model are most likely to misclassify
   headless browser automation (Playwright, Puppeteer) as human
2. Explain the XOR key reseeding mechanism in rxvm_disasm.py and why it matters for obfuscation
3. Map the full detection architecture to MITRE ATT&CK or OWASP categories
```

---
## License

MIT
