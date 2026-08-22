#!/usr/bin/env python3
"""
Amazon rxc= Cookie Decryptor

The rxc cookie stores accumulated behavioral metrics from RXVM,
encrypted with RC4. The key is embedded in the cookie plaintext.

Cookie format (after base64 decode):
  byte 0:     fnpv (version)
  bytes 1-4:  fnpb (RC4 key, 4 bytes)
  bytes 5+:   RC4 ciphertext (metric data)

Metric data format (after decryption):
  Sequence of [metric_id, ...metric_bytes] pairs
  Serialized by E(): Object.keys(v) → [parseInt(key), ...values]
"""

import base64
import sys
import json


def rc4(key: bytes, data: bytes) -> bytes:
    """Standard RC4 stream cipher."""
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) & 0xFF
        S[i], S[j] = S[j], S[i]

    out = []
    i = j = 0
    for byte in data:
        i = (i + 1) & 0xFF
        j = (j + S[i]) & 0xFF
        S[i], S[j] = S[j], S[i]
        out.append(byte ^ S[(S[i] + S[j]) & 0xFF])
    return bytes(out)


def format_timestamp(ts):
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except:
        return f'invalid ({ts})'


def b64_decode_amazon(s: str) -> bytes:
    """Amazon's custom base64 — standard alphabet, no padding."""
    # Add padding if needed
    missing = len(s) % 4
    if missing:
        s += '=' * (4 - missing)
    return base64.b64decode(s)


def decrypt_rxc(cookie_value: str) -> dict:
    """
    Decrypt an rxc= cookie value.

    Cookie wire format (after base64 decode):
      byte 0:     fnpv (version)
      bytes 1-4:  fnpb (RC4 key — yes, shipped in cleartext)
      bytes 5+:   RC4 ciphertext

    Plaintext format (after RC4 decrypt):
      bytes 0-3:  unix timestamp (uint32 LE) — when ep4 was called
      bytes 4+:   metric stream — serialized by E(v)
    """
    # Strip 'rxc=' prefix if present
    if cookie_value.startswith('rxc='):
        cookie_value = cookie_value[4:]

    # Strip any trailing semicolons or whitespace
    cookie_value = cookie_value.split(';')[0].strip()

    raw = b64_decode_amazon(cookie_value)

    if len(raw) < 6:
        raise ValueError(f"Cookie too short: {len(raw)} bytes")

    version = raw[0]
    key = raw[1:5]
    ciphertext = raw[5:]

    plaintext = rc4(key, ciphertext)

    # First 4 bytes of plaintext = unix timestamp (LE uint32)
    timestamp = None
    metrics_start = 0
    if len(plaintext) >= 4:
        import struct
        timestamp = struct.unpack('<I', plaintext[:4])[0]
        metrics_start = 4

    # Parse metric stream from byte 4 onward
    metric_bytes = plaintext[metrics_start:]

    return {
        'version': version,
        'key': key.hex(),
        'key_bytes': list(key),
        'ciphertext_len': len(ciphertext),
        'plaintext_len': len(plaintext),
        'timestamp': timestamp,
        'timestamp_human': format_timestamp(timestamp) if timestamp else None,
        'plaintext_hex': plaintext.hex(),
        'plaintext_bytes': list(plaintext),
        'metric_bytes': list(metric_bytes),
        'metrics': parse_metrics(metric_bytes),
    }


def parse_metrics(data: bytes) -> list:
    """
    Parse the decrypted metric stream.

    The metrics are serialized by E(v):
      Object.keys(v).forEach(n => { t.push(parseInt(n)); t = t.concat(v[n]) })

    Each metric entry from rx.pc(id, data):
      v[id & 0xFF] = data

    So the stream is: [metric_id, ...data_bytes, metric_id, ...data_bytes, ...]

    Without knowing each metric's length, we dump raw byte groups.
    Known metric IDs from RXVM payloads:
      - IDs correspond to the first arg of rx.p() and rx.pc() calls
      - Payload collection assigns IDs 16, 31-35, 48 etc.
    """
    metrics = []

    if not data:
        return metrics

    # Try to parse as sequential metric entries
    # Each rx.pc call stores v[id & 0xFF] = array_of_bytes
    # E() serializes as: for each key → push(parseInt(key)), concat(values)
    # But we don't know value lengths without the original structure

    # Dump as raw annotated bytes
    i = 0
    while i < len(data):
        metric_id = data[i]
        # Collect remaining bytes until next plausible metric ID
        # or end of stream
        i += 1
        value_bytes = []
        # Heuristic: collect until we see a byte that could be a known metric ID
        # followed by a plausible pattern, or just take all remaining
        while i < len(data):
            value_bytes.append(data[i])
            i += 1

        metrics.append({
            'id': metric_id,
            'raw': value_bytes,
        })
        break  # First pass: treat entire remainder as one blob

    return metrics


def format_output(result: dict) -> str:
    """Pretty print decryption result."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"rxc= COOKIE DECRYPTION")
    lines.append(f"{'='*60}")
    lines.append(f"")
    lines.append(f"Version:         {result['version']}")
    lines.append(f"RC4 Key:         {result['key']} ({result['key_bytes']})")
    lines.append(f"Ciphertext:      {result['ciphertext_len']} bytes")
    lines.append(f"Plaintext:       {result['plaintext_len']} bytes")
    if result.get('timestamp'):
        lines.append(f"Timestamp:       {result['timestamp']} ({result['timestamp_human']})")
    lines.append(f"Metric data:     {len(result.get('metric_bytes', []))} bytes")
    lines.append(f"")
    lines.append(f"{'─'*60}")
    lines.append(f"DECRYPTED BYTES (hex):")
    lines.append(f"{'─'*60}")

    # Format hex dump with ASCII
    data = result['plaintext_bytes']
    for offset in range(0, len(data), 16):
        chunk = data[offset:offset+16]
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        lines.append(f"  {offset:04x}:  {hex_part:<48s}  {ascii_part}")

    lines.append(f"")
    lines.append(f"{'─'*60}")
    lines.append(f"RAW METRIC STREAM:")
    lines.append(f"{'─'*60}")

    for m in result['metrics']:
        lines.append(f"  Metric ID: {m['id']} (0x{m['id']:02x})")
        lines.append(f"  Data ({len(m['raw'])} bytes): {m['raw'][:32]}{'...' if len(m['raw'])>32 else ''}")

    # Attempt to identify known metric IDs
    lines.append(f"")
    lines.append(f"{'─'*60}")
    lines.append(f"KNOWN METRIC ID MAPPING:")
    lines.append(f"{'─'*60}")
    lines.append(f"  16 (0x10) = mouse timing stats (payload 5)")
    lines.append(f"  31 (0x1f) = scroll position hash")
    lines.append(f"  32 (0x20) = mouse movement stats (payload 6)")
    lines.append(f"  33 (0x21) = mouse acceleration stats (payload 6)")
    lines.append(f"  34 (0x22) = scroll velocity stats (payload 7)")
    lines.append(f"  35 (0x23) = paint timing (payload 8)")
    lines.append(f"  48 (0x30) = neural net score (payload 9)")

    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  rxc_decrypt.py <cookie_value>")
        print("  rxc_decrypt.py --file <cookies.txt>")
        print("")
        print("Cookie value is the rxc= value from Amazon,")
        print("either with or without the 'rxc=' prefix.")
        print("")
        print("Example:")
        print("  rxc_decrypt.py 'AGVoY2hlY2...'")
        sys.exit(1)

    if sys.argv[1] == '--file':
        with open(sys.argv[2]) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if 'rxc=' in line:
                    # Extract rxc value from cookie header
                    for part in line.split(';'):
                        part = part.strip()
                        if part.startswith('rxc='):
                            line = part[4:]
                            break
                try:
                    result = decrypt_rxc(line)
                    print(format_output(result))
                except Exception as ex:
                    print(f"FAILED: {ex}")
                print()
    else:
        cookie = sys.argv[1]
        try:
            result = decrypt_rxc(cookie)
            print(format_output(result))
            # Also dump raw JSON
            print(f"\n{'─'*60}")
            print("JSON:")
            print(json.dumps(result, indent=2))
        except Exception as ex:
            print(f"FAILED: {ex}")
            sys.exit(1)


if __name__ == '__main__':
    main()
