#!/usr/bin/env python3
"""
Extract rx.M (matrix math module) from Amazon page source.

rx.M is defined in the plaintext JavaScript that loads BEFORE the
bytecode payloads execute. It's in the rx initialization script,
not in the base64 ex64() blobs.

USAGE:
  1. Open any Amazon product/search page in your browser
  2. View Page Source (Ctrl+U / Cmd+U)
  3. Save the full HTML source to a file
  4. Run: python3 extract_rx_runtime.py saved_page.html

WHAT THIS FINDS:
  - The rx object initialization
  - The M (matrix) module: M(), M.mul(), M.R(), M.S()
  - Any other rx.* runtime methods referenced by bytecode payloads

ALTERNATIVE (faster, from browser console):
  Open DevTools on any Amazon page and run:

    // Dump the matrix module
    console.log(rx.M.toString());
    console.log(rx.M.mul.toString());

    // Or dump everything
    Object.keys(rx).forEach(k => {
        try {
            if (typeof rx[k] === 'function')
                console.log(k + ': ' + rx[k].toString());
        } catch(e) {}
    });

  Copy the output — that's your rx.M source.
"""

import re
import sys
import json


def extract_rx_scripts(html):
    """Find all <script> blocks that reference the rx object."""
    # Match script tags
    scripts = re.findall(
        r'<script[^>]*>(.*?)</script>',
        html,
        re.DOTALL | re.IGNORECASE
    )

    rx_scripts = []
    for i, script in enumerate(scripts):
        # Look for rx object initialization patterns
        indicators = [
            'rx.M',
            'rx.ML',
            'rx.ex64',
            'rx.exec',
            '"M"',
            '.mul(',
            'matrix',
            # The Dense constructor calls M(in_dim, weights, out_dim)
            # and forward calls M.mul(input, weights, bias)
        ]
        score = sum(1 for ind in indicators if ind in script)
        if score > 0:
            rx_scripts.append((i, score, script))

    return rx_scripts


def extract_rx_M_definition(html):
    """Try to find the specific rx.M definition."""
    patterns = [
        # Direct assignment
        r'(rx\.M\s*=\s*(?:function[^{]*\{[^}]*(?:\{[^}]*\}[^}]*)*\}|[^;]+;))',
        # Object literal with M property
        r'(["\'"]M["\'"]:\s*(?:function[^{]*\{[^}]*(?:\{[^}]*\}[^}]*)*\}|[^,}]+))',
        # M.mul definition
        r'((?:rx\.)?M\.mul\s*=\s*(?:function[^{]*\{[^}]*(?:\{[^}]*\}[^}]*)*\}|[^;]+;))',
        # Matrix/vector multiplication functions
        r'(function\s+\w*[Mm](?:at(?:rix)?|ul)\w*\s*\([^)]*\)\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\})',
    ]

    matches = []
    for pat in patterns:
        for m in re.finditer(pat, html):
            matches.append(m.group(1)[:500])  # First 500 chars

    return matches


def extract_ex64_context(html):
    """Find the ex64 function definition to understand the execution model."""
    patterns = [
        r'(ex64\s*[:=]\s*function[^{]*\{[^}]*(?:\{[^}]*\}[^}]*)*\})',
        r'(["\'"]ex64["\'"]:\s*function[^{]*\{[^}]*(?:\{[^}]*\}[^}]*)*\})',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            return m.group(1)[:1000]
    return None


def main():
    if len(sys.argv) < 2:
        print("Extract rx.M (matrix module) from Amazon page source")
        print()
        print("Usage:")
        print("  python3 extract_rx_runtime.py saved_page.html")
        print()
        print("Or from browser DevTools console on any Amazon page:")
        print()
        print('  // Quick extract')
        print('  console.log("M constructor:", rx.M.toString());')
        print('  console.log("M.mul:", rx.M.mul.toString());')
        print('  console.log("M.R:", rx.M.R ? rx.M.R.toString() : "not found");')
        print('  console.log("M.S:", rx.M.S ? rx.M.S.toString() : "not found");')
        print()
        print('  // Full runtime dump')
        print('  JSON.stringify(Object.keys(rx).reduce((o,k) => {')
        print('    try { o[k] = typeof rx[k] === "function" ? rx[k].toString() : typeof rx[k]; }')
        print('    catch(e) { o[k] = "error"; }')
        print('    return o;')
        print('  }, {}), null, 2);')
        sys.exit(1)

    with open(sys.argv[1], 'r', errors='replace') as f:
        html = f.read()

    print(f"Loaded {len(html):,} bytes")
    print()

    # Count ex64 payloads for reference
    payloads = re.findall(r'rx\.ex64\("([A-Za-z0-9+/=]+)"', html)
    print(f"Found {len(payloads)} ex64() bytecode payloads")
    print()

    # Find rx-related scripts
    rx_scripts = extract_rx_scripts(html)
    print(f"Found {len(rx_scripts)} script blocks referencing rx internals")
    print()

    for idx, score, script in sorted(rx_scripts, key=lambda x: -x[1]):
        print(f"{'=' * 72}")
        print(f"Script block #{idx} (relevance score: {score})")
        print(f"{'=' * 72}")

        # Show first/last 1000 chars
        if len(script) > 2500:
            print(script[:1200])
            print(f"\n... [{len(script) - 2400} bytes omitted] ...\n")
            print(script[-1200:])
        else:
            print(script)
        print()

    # Try targeted extraction
    print(f"{'=' * 72}")
    print("TARGETED rx.M EXTRACTION")
    print(f"{'=' * 72}")

    matches = extract_rx_M_definition(html)
    if matches:
        for i, m in enumerate(matches):
            print(f"\nMatch {i + 1}:")
            print(m)
    else:
        print("No direct rx.M definition found via regex.")
        print("This is expected if the code is minified or uses indirect assignment.")
        print()
        print("Use the DevTools console method instead:")
        print('  console.log(rx.M.toString());')

    # Try to find ex64
    ex64 = extract_ex64_context(html)
    if ex64:
        print(f"\nex64 definition found:")
        print(ex64)

    # Output file
    if rx_scripts:
        outpath = sys.argv[1].rsplit('.', 1)[0] + '_rx_scripts.txt'
        with open(outpath, 'w') as f:
            for idx, score, script in sorted(rx_scripts, key=lambda x: -x[1]):
                f.write(f"{'=' * 72}\n")
                f.write(f"Script block #{idx} (score: {score})\n")
                f.write(f"{'=' * 72}\n")
                f.write(script)
                f.write('\n\n')
        print(f"\nFull scripts saved to: {outpath}")


if __name__ == '__main__':
    main()
