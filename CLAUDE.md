# RXVM

Reverse engineering of Amazon's RXVM client-side bot detection system. Includes a bytecode VM disassembler, the extracted A10 neural network (31→16→16→1), an `rxc=` cookie decryptor (RC4), and analysis of the 15-payload behavioral telemetry pipeline. **Responsibly disclosed to Amazon Security; confirmed by Amazon (Mar 16, 2026).**

## Language
Python 3 — **standard library only, no external dependencies, no `pip install` step**.

## Run
There is no `main.py` and no requirements.txt. Each tool is a standalone script:

```
python3 rxvm_disasm.py page.html              # disassemble RXVM payloads from an Amazon page
python3 rxvm_disasm.py -b64 "UlgB..."         # disassemble a single base64 payload
python3 rxc_decrypt.py "AGVoY2hlY2..."        # decrypt an rxc= cookie, parse metric stream
python3 a10_model.py                          # analyze the A10 neural network (architecture, feature importance)
python3 a10_model.py --score <31 floats>      # score a custom feature vector
python3 extract_rx_runtime.py page.html       # extract the rx.M matrix module from page HTML
python3 foxhound.py                           # Firefox on-device ML model extractor + runner
```

## Layout
```
README.md                       # quick-start + system overview + 15-payload architecture
ANALYSIS.md                     # full technical writeup of all 15 payloads
RXVM_Paper.pdf                  # academic paper (peer-review format)
A10_Report.pdf                  # visual report — diagrams, feature importance, model analysis

rxvm_disasm.py                  # bytecode VM disassembler (v4) — XOR key reseeding, closure discovery
rxc_decrypt.py                  # rxc= cookie decryptor — RC4 + metric stream parsing
a10_model.py                    # A10 neural net reimplementation — inference + feature importance
a10_model_weights.json          # extracted norms / dense / batchnorm layer weights
extract_rx_runtime.py           # rx.M matrix module extractor utility
foxhound.py                     # Firefox on-device ML extractor (companion tool)

rx_M_extracted.js               # deobfuscated rx.M source — confirmed pass-through, no rescaling
payload0_crypto_analysis.txt    # annotated disassembly of the crypto payload + JS equivalents
rxvm_crypto_layer_analysis.docx # detailed crypto layer analysis
rxvm_full_disasm.txt            # complete disassembly of all 15 payloads from a live page
```

## Known Limitation
A10 neural-net scoring currently saturates near 1.0 because the Dense-layer constructor passes weights through `rx.M` (a matrix module living in the RXVM core JS runtime, not the bytecode payloads), which likely rescales 8-bit minifloat weights at instantiation. Without that rescale, BatchNorm betas dominate the pre-sigmoid value. Fix: extract `rx.M` from the RXVM interpreter's plaintext JS on Amazon pages.

## Claude Code Notes
- Read `ANALYSIS.md` for the full per-payload technical writeup
- All analysis was performed on publicly-served JavaScript in the researcher's own browser — no exploitation, no circumvention, no access to Amazon's server-side systems
- Built with [Claude Code](https://claude.ai/code)
