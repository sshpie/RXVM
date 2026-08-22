#!/usr/bin/env python3
"""
RXVM Disassembler v4
Reverse-engineered from Amazon's RXVM bytecode VM.
v4: Closure body disassembly with XOR key reseeding.
"""

import base64, struct, sys, re

OPCODES = {
    1:'STORE_LOCAL', 2:'STORE_GLOBAL', 3:'EVAL', 4:'PROP_SET',
    5:'ITER_INIT', 6:'ITER_NEXT', 10:'PUSH', 11:'PUSH_ACC',
    12:'LOAD_STRINGS', 30:'NOT', 42:'NOP', 43:'RESTORE_ARGS',
    44:'RET', 45:'RET_VAL',
    48:'CALL_0', 49:'CALL_1', 50:'CALL_2', 51:'CALL_N',
    52:'NEW_0', 53:'NEW_1', 54:'NEW_2', 55:'NEW_N',
    58:'JUMP', 59:'JMP_TRUE', 60:'JMP_FALSE',
    64:'MAKE_CLOSURE', 65:'MAKE_NAMED_CLOSURE',
}

BINOPS = {
    16:'+', 17:'-', 18:'*', 19:'/', 20:'pow', 21:'%',
    22:'&', 23:'|', 24:'^', 25:'<<', 26:'>>', 27:'>>>',
    28:'&&', 29:'||', 31:'>', 32:'<', 33:'>=', 34:'<=',
    35:'==', 36:'===', 37:'!=', 38:'!==', 39:'in', 40:'instanceof',
}

CONSTANTS = {10:'undefined', 11:'null', 14:'true', 15:'false'}

MAX_EXPR_DEPTH = 16


class Reader:
    def __init__(self, raw, offset=3, key_seed=0):
        self.m = raw
        self.ip = offset
        self.key = (key_seed % 127) + 37

    def eof(self):
        return self.ip >= len(self.m)

    def byte(self):
        if self.ip >= len(self.m):
            raise EOFError
        b = self.m[self.ip] ^ self.key
        self.ip += 1
        return b

    def s16(self):
        lo, hi = self.byte(), self.byte()
        v = lo | (hi << 8)
        return v - 65536 if v > 32767 else v

    def varint(self):
        r, s = 0, 0
        while True:
            b = self.m[self.ip] ^ self.key
            self.ip += 1
            r += (b & 0x7F) << (7 * s)
            s += 1
            if not (b & 0x80):
                break
            if s > 5:
                raise ValueError("varint overflow")
        return r

    def string(self):
        n = self.varint()
        if n > 10000:
            raise ValueError(f"string too long: {n}")
        return ''.join(chr(self.byte()) for _ in range(n))

    def i32(self):
        v = self.byte() | (self.byte()<<8) | (self.byte()<<16) | (self.byte()<<24)
        return v - 4294967296 if v > 2147483647 else v

    def f32(self):
        bs = bytes([self.byte() for _ in range(4)])
        return struct.unpack('<f', bs)[0]

    def minifloat(self):
        b = self.byte()
        sign = -1 if (b & 0x80) else 1
        exp = (b >> 3) & 0xF
        mant = b & 0x7
        if exp == 15: return float('nan')
        if exp == 0: return mant / 8 * sign * 0.015625
        return sign * (1 + mant / 8) * (2 ** (exp - 7))

    def reseed(self, offset):
        self.key = (offset % 127) + 37


class DisassemblerError(Exception):
    """Raised for unrecoverable disassembly failures."""
    pass


class Disassembler:
    def __init__(self, raw):
        if len(raw) < 3:
            raise DisassemblerError(f"payload too short: {len(raw)} bytes (need ≥3)")
        if raw[0] != 0x52 or raw[1] != 0x58:
            raise DisassemblerError(
                f"bad magic: 0x{raw[0]:02x} 0x{raw[1]:02x} (expected 0x52 0x58 \"RX\")")
        self.raw = raw
        self.ver = raw[2]
        self.strtab = []
        self.out = []
        self.closures = []
        self._closure_skip_targets = []  # (body_addr, end_addr) pairs
        self.warnings = []
        self._unknown_count = 0
        self._last_line = None
        self._repeat_count = 0
        self._repeat_suppressed = 0

    def run(self):
        r = Reader(self.raw)
        self._walk(r, 0)

        # Disassemble closure bodies with reseeded keys
        self._disasm_closures()

        payload_len = len(self.raw)

        # Warn on early EOF (stream ended before a natural stopping point)
        if r.ip < payload_len - 1:
            consumed_pct = (r.ip / payload_len) * 100
            if consumed_pct < 90:
                self.warnings.append(
                    f"early stop: consumed {r.ip}/{payload_len} bytes ({consumed_pct:.0f}%)")

        # Flush any trailing repeat suppression
        if self._repeat_suppressed > 0:
            self.out.append(f"     | ... {self._repeat_suppressed} identical lines suppressed")
            self.warnings.append(
                f"degenerate output: {self._repeat_suppressed + 1} consecutive identical lines")

        # Warn on any unknown opcodes
        if self._unknown_count > 0:
            total = len(self.out)
            pct = (self._unknown_count / total * 100) if total else 0
            severity = "likely corrupt" if pct > 50 else "possible corruption"
            self.warnings.append(
                f"unknown opcodes: {self._unknown_count}/{total} lines "
                f"({pct:.0f}%) — {severity}")

        # Warn on empty output from non-trivial payload
        if len(self.out) == 0 and payload_len > 10:
            self.warnings.append(
                f"no instructions decoded from {payload_len}-byte payload")

        # Warn if any closure skip target points past end of payload
        for body, end in self._closure_skip_targets:
            if end > payload_len:
                self.warnings.append(
                    f"closure fn@{body} skip target @{end} is past end of payload "
                    f"({payload_len} bytes) — truncated?")
                break  # one warning is enough

        return self.out

    def _disasm_closures(self):
        """Disassemble all closure bodies with reseeded XOR keys."""
        visited = set()
        queue = list(self._closure_skip_targets)  # (body_addr, end_addr) pairs

        while queue:
            body, end = queue.pop(0)
            if body in visited:
                continue
            visited.add(body)
            if body >= len(self.raw) or end > len(self.raw):
                continue

            r = Reader(self.raw, offset=body)
            r.reseed(body)

            # Save and reset state for this closure's output
            saved_out = self.out
            saved_last = self._last_line
            saved_repeat = self._repeat_count
            saved_suppressed = self._repeat_suppressed
            self.out = []
            self._last_line = None
            self._repeat_count = 0
            self._repeat_suppressed = 0

            old_skip_count = len(self._closure_skip_targets)

            # Walk with boundary enforcement
            self._walk_bounded(r, 1, end)

            closure_lines = self.out

            # Restore
            self.out = saved_out
            self._last_line = saved_last
            self._repeat_count = saved_repeat
            self._repeat_suppressed = saved_suppressed

            # Insert closure disassembly after the MAKE_CLOSURE line
            if closure_lines:
                # Find where to insert (after the line that references fn@{body})
                insert_idx = None
                tag = f'fn@{body}'
                for i, line in enumerate(self.out):
                    if tag in line:
                        insert_idx = i + 1
                        break

                header = f'     | ── fn@{body} {"─" * 40}'
                footer = f'     | ── end fn@{body} {"─" * 37}'

                block = [header] + closure_lines + [footer]
                if insert_idx is not None:
                    self.out[insert_idx:insert_idx] = block
                else:
                    self.out.extend(block)

            # Queue any new closures discovered inside this body
            for pair in self._closure_skip_targets[old_skip_count:]:
                if pair[0] not in visited:
                    queue.append(pair)

    def _walk(self, r, indent):
        """Main execution loop — mirrors f() → m() → e() dispatch."""
        self._walk_bounded(r, indent, len(r.m))

    def _walk_bounded(self, r, indent, end):
        """Walk opcodes but stop at the given byte boundary."""
        while r.ip < end and r.ip < len(r.m):
            try:
                pos = r.ip
                b = r.byte()  # raw opcode byte (not h() dispatched)
                # e(n, r): check o (binops) first, then i (instructions)
                if b in BINOPS:
                    a = self._val(r)
                    b2 = self._val(r)
                    self._emit(pos, 'BINOP', f'{b2} {BINOPS[b]} {a}', indent)
                elif b in OPCODES:
                    self._op(b, r, pos, indent)
                else:
                    self._unknown_count += 1
                    self._emit(pos, f'UNKNOWN_{b}', f'(0x{b:02x})', indent)
            except EOFError:
                self.warnings.append(
                    f"unexpected EOF at byte {r.ip}/{len(r.m)} during opcode parse")
                break
            except Exception as ex:
                self._emit(r.ip, 'ERROR', str(ex), indent)
                break

    def _emit(self, pos, op, detail, indent):
        pad = '  ' * indent
        line_content = f'{op:22s} {detail}'.rstrip()

        # Detect degenerate repeated output
        if line_content == self._last_line:
            self._repeat_count += 1
            if self._repeat_count > 3:
                self._repeat_suppressed += 1
                return  # suppress
        else:
            # Flush suppression notice if we were suppressing
            if self._repeat_suppressed > 0:
                self.out.append(f"     | ... {self._repeat_suppressed} identical lines suppressed")
                self._repeat_suppressed = 0
            self._repeat_count = 0
            self._last_line = line_content

        self.out.append(f'{pad}{pos:04d} | {line_content}')

    def _val(self, r, depth=0):
        """Read one value expression via h() path, return string."""
        if r.eof(): return '<eof>'
        if depth > MAX_EXPR_DEPTH: return '<deep>'
        b = r.byte()
        if b & 0x80:
            op = b & 0x7F
            if op in BINOPS:
                a = self._val(r, depth+1)
                b2 = self._val(r, depth+1)
                return f'({b2} {BINOPS[op]} {a})'
            # Opcodes as sub-expressions (they return a value)
            if op in (48,49,50,51):  # CALL
                func = self._val(r, depth+1)
                argc = r.varint() if op == 51 else (op - 48)
                args = [self._val(r, depth+1) for _ in range(argc)]
                return f'{func}({", ".join(args)})'
            if op in (52,53,54,55):  # NEW
                func = self._val(r, depth+1)
                argc = r.varint() if op == 55 else (op - 52)
                args = [self._val(r, depth+1) for _ in range(argc)]
                return f'new {func}({", ".join(args)})'
            if op == 30:  # NOT
                return f'!({self._val(r, depth+1)})'
            if op == 3:   # EVAL (side effect, returns value)
                return self._val(r, depth+1)
            if op == 64:  # MAKE_CLOSURE
                skip = r.s16()
                body = r.ip
                self.closures.append(body)
                self._closure_skip_targets.append((body, body + skip))
                return f'fn@{body}'
            if op == 65:  # MAKE_NAMED_CLOSURE
                slot = r.varint()
                skip = r.s16()
                body = r.ip
                self.closures.append(body)
                self._closure_skip_targets.append((body, body + skip))
                return f'fn@{body}→L{slot}'
            if op in OPCODES:
                return f'({OPCODES[op]})'
            return f'(op_{op})'
        if b in CONSTANTS:
            return CONSTANTS[b]
        return self._val_from_type(b, r, depth)

    def _val_from_type(self, b, r, depth):
        if depth > MAX_EXPR_DEPTH: return '<deep>'
        if b == 1:  return 'acc'
        if b == 12:
            s = r.string()
            return repr(s[:60])
        if b == 13:
            idx = r.varint()
            if idx < len(self.strtab):
                return f'$"{self.strtab[idx][:40]}"'
            return f'str#{idx}'
        if b == 17: return str(r.varint())
        if b == 18: return str(r.i32())
        if b == 19: return f'{r.f32():.4f}'
        if b == 20: return '[]'
        if b == 21:
            n = r.varint()
            elems = [self._val(r, depth+1) for _ in range(n)]
            if n <= 8:
                s = ', '.join(elems)
                return f'[{s}]'
            s = ', '.join(elems[:8])
            return f'[{s}, ...+{n-8}]'
        if b == 22: return '{}'
        if b == 23:
            n = r.varint() // 2
            pairs = []
            for _ in range(n):
                v = self._val(r, depth+1)
                k = self._val(r, depth+1)
                pairs.append(f'{k}: {v}')
            if n <= 4:
                return '{' + ', '.join(pairs) + '}'
            return '{' + ', '.join(pairs[:4]) + f', ...+{n-4}' + '}'
        if b == 24:
            n = r.varint()
            floats = [r.minifloat() for _ in range(n)]
            if n <= 4: return f'F{floats}'
            return f'F[{n}]'
        if b == 32: return f'L{r.varint()}'
        if b == 33: return f'G{r.varint()}'
        if b == 48:
            key = self._val(r, depth+1)
            obj = self._val(r, depth+1)
            return f'{obj}[{key}]'
        if b == 50: return 'pop()'
        if b == 52:
            v = self._val(r, depth+1)
            return f'typeof({v})'
        return f'?{b}'

    def _op(self, op, r, pos, indent):
        name = OPCODES.get(op) or BINOPS.get(op)

        if op in BINOPS:
            a = self._val(r)
            b = self._val(r)
            self._emit(pos, 'BINOP', f'{b} {BINOPS[op]} {a}', indent)
            return

        if op == 1:  # STORE_LOCAL
            slot = r.byte()
            val = self._val(r)
            self._emit(pos, name, f'L{slot} = {val}', indent)
        elif op == 2:  # STORE_GLOBAL
            slot = r.byte()
            val = self._val(r)
            self._emit(pos, name, f'G{slot} = {val}', indent)
        elif op == 3:  # EVAL
            self._emit(pos, name, self._val(r), indent)
        elif op == 4:  # PROP_SET
            val = self._val(r)
            key = self._val(r)
            obj = self._val(r)
            self._emit(pos, name, f'{obj}[{key}] = {val}', indent)
        elif op == 5:  # ITER_INIT
            src = self._val(r)
            slot = r.byte()
            self._emit(pos, name, f'L{slot} = iter({src})', indent)
        elif op == 6:  # ITER_NEXT
            it = r.byte()
            dst = r.byte()
            jmp = r.s16()
            self._emit(pos, name, f'L{dst} = L{it}.next() else →+{jmp}', indent)
        elif op == 10:
            self._emit(pos, 'PUSH', self._val(r), indent)
        elif op == 11:
            self._emit(pos, 'PUSH_ACC', '', indent)
        elif op == 12:  # LOAD_STRINGS
            n = r.varint()
            strings = []
            for _ in range(n):
                s = r.string()
                strings.append(s)
                self.strtab.append(s)
            preview = [repr(s[:50]) for s in strings[:8]]
            self._emit(pos, name, f'{n} strings: [{", ".join(preview)}{"..." if n>8 else ""}]', indent)
        elif op == 30:
            self._emit(pos, 'NOT', self._val(r), indent)
        elif op == 42:
            self._emit(pos, 'NOP', '', indent)
        elif op == 43:
            self._emit(pos, 'RESTORE_ARGS', f'{r.varint()} → locals', indent)
        elif op == 44:  # RET (no value read)
            self._emit(pos, 'RET', '', indent)
        elif op == 45:  # RET_VAL
            self._emit(pos, 'RET_VAL', self._val(r), indent)
        elif op in (48,49,50,51):
            func = self._val(r)
            argc = r.varint() if op == 51 else (op - 48)
            args = [self._val(r) for _ in range(argc)]
            self._emit(pos, OPCODES[op], f'{func}({", ".join(args)})', indent)
        elif op in (52,53,54,55):
            func = self._val(r)
            argc = r.varint() if op == 55 else (op - 52)
            args = [self._val(r) for _ in range(argc)]
            self._emit(pos, OPCODES[op], f'new {func}({", ".join(args)})', indent)
        elif op == 58:
            self._emit(pos, 'JUMP', f'→+{r.s16()}', indent)
        elif op == 59:
            cond = self._val(r)
            self._emit(pos, 'JMP_TRUE', f'if ({cond}) →+{r.s16()}', indent)
        elif op == 60:
            cond = self._val(r)
            self._emit(pos, 'JMP_FALSE', f'if !({cond}) →+{r.s16()}', indent)
        elif op == 64:
            skip = r.s16()
            body = r.ip
            self.closures.append(body)
            self._closure_skip_targets.append((body, body + skip))
            self._emit(pos, 'MAKE_CLOSURE', f'fn@{body} skip→+{skip}', indent)
            r.ip += skip  # skip over function body
        elif op == 65:
            slot = r.varint()
            skip = r.s16()
            body = r.ip
            self.closures.append(body)
            self._closure_skip_targets.append((body, body + skip))
            self._emit(pos, 'MAKE_NAMED_CLOSURE', f'L{slot} = fn@{body} skip→+{skip}', indent)
            r.ip += skip  # skip over function body
        else:
            self._emit(pos, f'OP_{op}', '?', indent)


def disasm_b64(payload):
    raw = base64.b64decode(payload)
    d = Disassembler(raw)
    d.run()
    return d

def _print_warnings(d):
    """Print any warnings accumulated during disassembly."""
    for w in d.warnings:
        print(f"  ⚠ WARNING: {w}")

def main():
    if len(sys.argv) < 2:
        print("Usage: rxvm_disasm.py <file.html>  |  rxvm_disasm.py -b64 <payload>")
        sys.exit(1)

    if sys.argv[1] == '-b64':
        try:
            d = disasm_b64(sys.argv[2])
        except DisassemblerError as ex:
            print(f"FAIL: {ex}", file=sys.stderr)
            sys.exit(1)
        print(f"v{d.ver} | {len(d.strtab)} strings | {len(d.closures)} closures")
        _print_warnings(d)
        print('\n'.join(d.out))

    else:
        with open(sys.argv[1], 'r', errors='replace') as f:
            content = f.read()
        payloads = re.findall(r'rx\.ex64\("([A-Za-z0-9+/=]+)"', content)
        print(f"{len(payloads)} payloads\n")

        for i, p in enumerate(payloads):
            raw = base64.b64decode(p)
            print(f"{'='*72}")
            print(f"[{i}] {len(raw)} bytes")
            print(f"{'='*72}")
            try:
                d = Disassembler(raw)
                d.run()
                print(f"v{d.ver} | {len(d.strtab)} strings | {len(d.closures)} closures")
                _print_warnings(d)
                print('\n'.join(d.out))
            except DisassemblerError as ex:
                print(f"FAIL: {ex}")
            except Exception as ex:
                print(f"FAIL: {ex}")
            print()

if __name__ == '__main__':
    main()
