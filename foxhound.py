#!/usr/bin/env python3
"""
foxhound.py — Firefox On-Device ML Model Extractor & Runner
Reverse-engineered from Firefox's MLEngine / ml-inference-options Remote Settings.

Pulls quantized ONNX models directly from HuggingFace, runs them outside Firefox,
and dumps architecture info. No browser required.

Usage:
  python3 foxhound.py --list
  python3 foxhound.py --info <feature>
  python3 foxhound.py --download <feature>
  python3 foxhound.py --run <feature> --input "text or image path"
  python3 foxhound.py --run all --input "your query here"
  python3 foxhound.py --dump <feature>
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "foxhound"
REMOTE_SETTINGS = "https://firefox.settings.services.mozilla.com/v1/buckets/main/collections/ml-inference-options/records"
HF_BASE = "https://huggingface.co"

# ── Model registry (mirrors ml-inference-options Remote Settings) ────────────

MODELS = {
    "smart-tab-embedding": {
        "desc": "Tab grouping — sentence embedding for tab similarity clustering",
        "task": "feature-extraction",
        "model": "mozilla/smart-tab-embedding",
        "rev": "v0.1.0",
        "onnx": "onnx/model_quantized.onnx",
        "arch": "BERT (sentence-transformers fine-tune)",
        "input": "Tab title / URL text",
        "output": "384-dim float32 embedding vector",
        "privacy": "Tab titles + URLs processed locally",
    },
    "smart-tab-topic": {
        "desc": "Tab grouping — generates a topic label for a group of tabs",
        "task": "text2text-generation",
        "model": "mozilla/smart-tab-topic",
        "rev": "v0.7.4",
        "onnx_encoder": "onnx/encoder_model_quantized.onnx",
        "onnx_decoder": "onnx/decoder_model_quantized.onnx",
        "arch": "T5-small (seq2seq)",
        "input": "Newline-separated tab titles",
        "output": "Topic label string",
        "privacy": "Tab titles processed locally",
    },
    "moz-image-to-text": {
        "desc": "PDF alt-text — generates image descriptions for accessibility",
        "task": "image-to-text",
        "model": "mozilla/distilvit",
        "rev": "v0.5.0",
        "onnx_encoder": "onnx/encoder_model_quantized.onnx",
        "onnx_decoder": "onnx/decoder_model_merged_quantized.onnx",
        "arch": "ViT encoder + DistilGPT2 decoder (vision-encoder-decoder)",
        "input": "Image (from PDF)",
        "output": "Alt-text description string",
        "privacy": "PDF images processed locally — never leaves device",
    },
    "smart-intent": {
        "desc": "URL bar intent detection — classifies search query intent",
        "task": "text-classification",
        "model": "mozilla/mobilebert-query-intent-detection",
        "rev": "v0.3.1",
        "onnx": "onnx/model_quantized.onnx",
        "arch": "MobileBERT (classification head)",
        "input": "URL bar query string",
        "output": "Intent class label + confidence",
        "privacy": "Every URL bar keystroke classified locally",
    },
    "autofill-classification": {
        "desc": "Form autofill — classifies input fields for smart autofill",
        "task": "text-classification",
        "model": "mozilla/tinybert-uncased-autofill",
        "rev": "v0.1.3",
        "onnx": "onnx/model_quantized.onnx",
        "arch": "TinyBERT (fine-tuned on autofill dataset)",
        "input": "Form field label / placeholder text",
        "output": "Field type classification (name, email, address, ...)",
        "privacy": "Form field context analyzed locally",
    },
    "suggest-intent-classification": {
        "desc": "Firefox Suggest — classifies search intent for suggestions",
        "task": "text-classification",
        "model": "mozilla/mobilebert-uncased-finetuned-LoRA-intent-classifier",
        "rev": "v0.1.5",
        "onnx": "onnx/model_quantized.onnx",
        "arch": "MobileBERT + LoRA adapter",
        "input": "Search query",
        "output": "Intent class (navigational, informational, transactional, ...)",
        "privacy": "Search queries classified before Suggest API call",
    },
    "suggest-NER": {
        "desc": "Firefox Suggest — named entity recognition on search queries",
        "task": "token-classification",
        "model": "mozilla/distilbert-uncased-NER-LoRA",
        "rev": "v0.1.6",
        "onnx": "onnx/model_quantized.onnx",
        "arch": "DistilBERT + LoRA adapter",
        "input": "Search query",
        "output": "Named entities (person, location, org, ...)",
        "privacy": "Entity extraction before Suggest API — strips PII locally",
    },
    "content-classification": {
        "desc": "Content categorization — multi-label IAB taxonomy classification",
        "task": "text-classification",
        "model": "mozilla/content-multilabel-iab-classifier",
        "rev": "v0.1.2",
        "onnx": "onnx/model_quantized.onnx",
        "arch": "BERT (multi-label classification head)",
        "input": "Page text / URL",
        "output": "IAB content category labels",
        "privacy": "Page content categorized locally for ad targeting opt-out",
    },
    "iab-categorizer": {
        "desc": "IAB multi-task inference — content + intent classification",
        "task": "text-classification",
        "model": "mozilla/iab-multitask-inference",
        "rev": "v0.1.3",
        "onnx": "onnx/model_quantized.onnx",
        "arch": "BERT multi-task (shared encoder, multiple heads)",
        "input": "Page text",
        "output": "IAB category + engagement goal",
        "privacy": "Page-level topic profiling done locally",
    },
    "simple-text-embedder": {
        "desc": "Generic embedding utility — used by multiple Firefox features",
        "task": "feature-extraction",
        "model": "Xenova/all-MiniLM-L6-v2",
        "rev": "v0.1.0",
        "onnx": "onnx/model_quantized.onnx",
        "arch": "MiniLM-L6-v2 (sentence-transformers)",
        "input": "Any text",
        "output": "384-dim float32 embedding vector",
        "privacy": "General-purpose local embedding",
    },
}


def hf_url(model_id, rev, path):
    # Version tags in Remote Settings are internal Firefox versions, not HF git tags.
    # Models are always served from the main branch on HuggingFace.
    return f"{HF_BASE}/{model_id}/resolve/main/{path}"


def download(url, dest: Path, desc=""):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return
    label = desc or url.split("/")[-1]
    print(f"  [fetch] {label} ...", end="", flush=True)
    try:
        import requests
        with requests.get(url, stream=True, timeout=120,
                          headers={"User-Agent": "foxhound/1.0"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r  [fetch] {label} ... {pct}%  ", end="", flush=True)
        size = dest.stat().st_size
        print(f"\r  [fetch] {label} ... {size//1024//1024}MB   ")
    except ImportError:
        # Fallback to urllib for small files
        req = urllib.request.Request(url, headers={"User-Agent": "foxhound/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
        print(f" {dest.stat().st_size // 1024}KB")
    except Exception as e:
        if dest.exists():
            dest.unlink()
        print(f"\r  [fetch] {label} ... FAIL: {e}")
        raise


def fetch_model_files(fid):
    m = MODELS[fid]
    model_dir = CACHE_DIR / fid
    model_id = m["model"]
    rev = m["rev"]

    files_needed = []
    if "onnx" in m:
        files_needed.append(("onnx", m["onnx"]))
    if "onnx_encoder" in m:
        files_needed.append(("onnx_encoder", m["onnx_encoder"]))
    if "onnx_decoder" in m:
        files_needed.append(("onnx_decoder", m["onnx_decoder"]))
    files_needed += [
        ("tokenizer", "tokenizer.json"),
        ("tokenizer_config", "tokenizer_config.json"),
        ("config", "config.json"),
    ]
    if m["task"] == "image-to-text":
        files_needed.append(("preprocessor_config", "preprocessor_config.json"))

    paths = {}
    for key, rel_path in files_needed:
        dest = model_dir / rel_path.replace("/", "_")
        try:
            download(hf_url(model_id, rev, rel_path), dest, rel_path)
            paths[key] = dest
        except Exception:
            pass
    return paths


# ── Inference runners ────────────────────────────────────────────────────────

def run_embedding(fid, text):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    paths = fetch_model_files(fid)
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    tokenizer.enable_padding(length=128)
    tokenizer.enable_truncation(max_length=128)

    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attn_mask = np.array([enc.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    sess = ort.InferenceSession(str(paths["onnx"]))
    inp_names = [i.name for i in sess.get_inputs()]
    feeds = {"input_ids": input_ids, "attention_mask": attn_mask}
    if "token_type_ids" in inp_names:
        feeds["token_type_ids"] = token_type_ids

    out = sess.run(None, feeds)
    # Mean-pool last hidden state
    hidden = out[0]  # (1, seq_len, hidden_dim)
    mask = attn_mask[0]
    pooled = (hidden[0] * mask[:, None]).sum(0) / mask.sum()
    # Normalize
    norm = np.linalg.norm(pooled)
    embedding = pooled / norm if norm > 0 else pooled
    return embedding


def run_classification(fid, text):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    import json

    paths = fetch_model_files(fid)
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    tokenizer.enable_truncation(max_length=128)

    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attn_mask = np.array([enc.attention_mask], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    sess = ort.InferenceSession(str(paths["onnx"]))
    inp_names = [i.name for i in sess.get_inputs()]
    feeds = {"input_ids": input_ids, "attention_mask": attn_mask}
    if "token_type_ids" in inp_names:
        feeds["token_type_ids"] = token_type_ids

    raw_out = sess.run(None, feeds)[0]  # (1, seq_len, num_labels) or (1, num_labels)

    # Load label map from config
    labels = {}
    if "config" in paths:
        try:
            cfg = json.loads(paths["config"].read_text())
            labels = cfg.get("id2label", {})
        except Exception:
            pass

    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    if raw_out.ndim == 3:
        # Token-classification (NER): (1, seq_len, num_labels)
        # Return per-token best label, skip [CLS]/[SEP]/padding
        tokens = enc.tokens
        results = []
        for i, (tok, token_logits) in enumerate(zip(tokens, raw_out[0])):
            if tok in ("[CLS]", "[SEP]", "[PAD]") or tok.startswith("##"):
                continue
            probs = softmax(token_logits)
            best_id = int(probs.argmax())
            best_label = labels.get(str(best_id), str(best_id))
            if best_label not in ("O", "0"):  # skip non-entity tokens
                results.append((tok, best_label, float(probs[best_id])))
        return results if results else [("[no entities found]", "O", 1.0)]
    else:
        # Sequence classification: (1, num_labels)
        logits = raw_out[0]
        probs = softmax(logits)
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])
        return [(labels.get(str(i), str(i)), float(p)) for i, p in ranked[:5]]


def run_tab_topic(text):
    """T5 seq2seq: tab titles → topic label."""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    import json

    fid = "smart-tab-topic"
    paths = fetch_model_files(fid)

    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    tokenizer.enable_truncation(max_length=512)

    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attn_mask = np.array([enc.attention_mask], dtype=np.int64)

    enc_sess = ort.InferenceSession(str(paths["onnx_encoder"]))
    encoder_out = enc_sess.run(None, {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
    })[0]

    dec_sess = ort.InferenceSession(str(paths["onnx_decoder"]))

    # Greedy decode
    cfg = json.loads(paths["config"].read_text())
    eos_id = cfg.get("eos_token_id", 1)
    pad_id = cfg.get("decoder_start_token_id", 0)

    dec_ids = [pad_id]
    for _ in range(64):
        dec_input = np.array([dec_ids], dtype=np.int64)
        logits = dec_sess.run(None, {
            "input_ids": dec_input,
            "encoder_attention_mask": attn_mask,
            "encoder_hidden_states": encoder_out,
        })[0]
        next_id = int(logits[0, -1].argmax())
        if next_id == eos_id:
            break
        dec_ids.append(next_id)

    # Decode token ids
    return tokenizer.decode(dec_ids[1:])


def run_image_to_text(image_path):
    """ViT encoder + DistilGPT2 decoder: image → alt text."""
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    import json
    from PIL import Image

    fid = "moz-image-to-text"
    paths = fetch_model_files(fid)

    # Preprocess image (ViT: 224x224, normalize ImageNet)
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    arr = (arr - mean) / std
    pixel_values = arr.transpose(2, 0, 1)[None]  # (1, 3, 224, 224)

    enc_sess = ort.InferenceSession(str(paths["onnx_encoder"]))
    encoder_out = enc_sess.run(None, {"pixel_values": pixel_values})[0]

    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    cfg = json.loads(paths["config"].read_text())

    # GPT2 decoder: use decoder_config for eos/bos
    dec_cfg = cfg.get("decoder", {})
    bos_id = dec_cfg.get("bos_token_id", 50256)
    eos_id = dec_cfg.get("eos_token_id", 50256)

    dec_sess = ort.InferenceSession(str(paths["onnx_decoder"]))
    dec_inp_names = {i.name for i in dec_sess.get_inputs()}

    dec_ids = [bos_id]
    past_key_values = None

    for step in range(64):
        use_cache = "use_cache_branch" in dec_inp_names
        feeds = {
            "input_ids": np.array([[dec_ids[-1]]], dtype=np.int64),
            "encoder_hidden_states": encoder_out,
        }
        if use_cache:
            feeds["use_cache_branch"] = np.array([past_key_values is not None])
            # Inject past KV if available
            if past_key_values:
                for k, v in past_key_values.items():
                    feeds[k] = v
            else:
                # Initialize empty past KV
                for inp in dec_sess.get_inputs():
                    if inp.name.startswith("past_key_values"):
                        shape = [1 if d == "batch_size" else (1 if d is None else d)
                                 for d in inp.shape]
                        feeds[inp.name] = np.zeros(
                            [s if isinstance(s, int) else 0 for s in inp.shape],
                            dtype=np.float32
                        )

        out = dec_sess.run(None, feeds)
        logits = out[0]
        next_id = int(logits[0, -1].argmax())
        if next_id == eos_id:
            break
        dec_ids.append(next_id)

    return tokenizer.decode(dec_ids[1:])


def dump_model(fid):
    """Print ONNX graph summary: inputs, outputs, op counts."""
    try:
        import onnx
    except ImportError:
        print("  pip install onnx for graph dump")
        return
    import onnxruntime as ort

    paths = fetch_model_files(fid)
    m = MODELS[fid]
    onnx_path = paths.get("onnx") or paths.get("onnx_encoder")
    if not onnx_path:
        print("  No ONNX file found")
        return

    model = onnx.load(str(onnx_path))
    graph = model.graph
    print(f"\n  Architecture: {m['arch']}")
    print(f"  ONNX opset:   {model.opset_import[0].version}")
    print(f"  Nodes:        {len(graph.node)}")

    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {op:<30} {cnt}")

    print(f"\n  Inputs:")
    for inp in graph.input:
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name:<40} {shape}")

    print(f"\n  Outputs:")
    for out in graph.output:
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name:<40} {shape}")

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"\n  File size:    {size_mb:.1f} MB (quantized)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_list():
    print(f"\n  Firefox On-Device ML Models ({len(MODELS)} registered)\n")
    print(f"  {'FEATURE':<35} {'TASK':<25} {'ARCH'}")
    print("  " + "-" * 80)
    for fid, m in MODELS.items():
        print(f"  {fid:<35} {m['task']:<25} {m['arch']}")
    print()


def cmd_info(fid):
    m = MODELS[fid]
    print(f"\n  ── {fid} ──")
    print(f"  Description:  {m['desc']}")
    print(f"  HuggingFace:  {m['model']}@{m['rev']}")
    print(f"  Task:         {m['task']}")
    print(f"  Architecture: {m['arch']}")
    print(f"  Input:        {m['input']}")
    print(f"  Output:       {m['output']}")
    print(f"  Privacy note: {m['privacy']}")
    print()


def cmd_download(fid):
    print(f"\n  Downloading {fid}...")
    paths = fetch_model_files(fid)
    print(f"  Cached to {CACHE_DIR / fid}\n")


def cmd_run(fid, text_or_path):
    m = MODELS[fid]
    print(f"\n  Running {fid} ({m['arch']})")
    print(f"  Input: {repr(text_or_path[:80])}")
    print()

    task = m["task"]
    if task == "feature-extraction":
        emb = run_embedding(fid, text_or_path)
        import numpy as np
        print(f"  Embedding shape: {emb.shape}")
        print(f"  First 8 dims:    {np.round(emb[:8], 4).tolist()}")
        print(f"  Norm:            {float(np.linalg.norm(emb)):.6f}")

    elif task == "token-classification":
        results = run_classification(fid, text_or_path)
        print(f"  Named entities:")
        for item in results:
            tok, label, prob = item
            print(f"    {tok:<20} {label:<15} {prob:.4f}")
    elif task == "text-classification":
        results = run_classification(fid, text_or_path)
        print(f"  Classifications:")
        for label, prob in results:
            bar = "█" * int(prob * 30)
            print(f"    {label:<30} {prob:.4f}  {bar}")

    elif task == "text2text-generation":
        result = run_tab_topic(text_or_path)
        print(f"  Topic label: {repr(result)}")

    elif task == "image-to-text":
        result = run_image_to_text(text_or_path)
        print(f"  Alt text: {repr(result)}")

    print()


def cmd_run_all(text_or_path):
    for fid, m in MODELS.items():
        if m["task"] == "image-to-text":
            continue
        if m["task"] == "text2text-generation":
            continue
        print(f"\n  [{fid}]")
        try:
            cmd_run(fid, text_or_path)
        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="foxhound — Firefox on-device ML model extractor & runner"
    )
    ap.add_argument("--list", action="store_true", help="List all registered models")
    ap.add_argument("--info", metavar="FEATURE", help="Show model details")
    ap.add_argument("--download", metavar="FEATURE", help="Download model files")
    ap.add_argument("--run", metavar="FEATURE", help="Run inference (use 'all' for text models)")
    ap.add_argument("--dump", metavar="FEATURE", help="Dump ONNX graph architecture")
    ap.add_argument("--input", metavar="TEXT_OR_PATH", default="", help="Input text or image path")
    args = ap.parse_args()

    if args.list:
        cmd_list()
    elif args.info:
        if args.info not in MODELS:
            print(f"Unknown feature: {args.info}. Use --list to see options.")
            sys.exit(1)
        cmd_info(args.info)
    elif args.download:
        if args.download not in MODELS:
            print(f"Unknown feature: {args.download}")
            sys.exit(1)
        cmd_download(args.download)
    elif args.run:
        if not args.input:
            print("--run requires --input")
            sys.exit(1)
        if args.run == "all":
            cmd_run_all(args.input)
        elif args.run not in MODELS:
            print(f"Unknown feature: {args.run}")
            sys.exit(1)
        else:
            cmd_run(args.run, args.input)
    elif args.dump:
        if args.dump not in MODELS:
            print(f"Unknown feature: {args.dump}")
            sys.exit(1)
        cmd_dump(args.dump)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
