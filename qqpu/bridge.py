"""Qwen uses its QPU: a chat session on a merged checkpoint.

Qwen calls the `qpu_run` tool with an OpenQASM 2.0 program, in its native tool-call format. The
host parses the QASM (it plays the compiler / driver: gate names -> instruction words), then feeds
the QPU's host records into the *same* KV cache right after the tool call. From there the model's
own layers write the run (every amplitude of every gate) by greedy decoding until END. The host
reads the final amplitudes from the DeltaNet state, formats them as a tool response, and Qwen
continues. Qwen's layers are muted on QPU tokens, so its context passes through the run unchanged.
"""

import re
import time
from dataclasses import dataclass, field
from fractions import Fraction

import torch

from qqpu.circuit import QPU
from qqpu.hf import Device
import math

from qqpu.machine import (CCX, CX, CZ, END, NQ, NQC, SWAP, Gate, H, S, Sdg, T, Tdg, X, Y, Z,
                          flatten, host_tokens, is_complex, trace, value)

SYSTEM = (
    "You are a helpful assistant with a quantum processing unit (QPU) built into your own "
    "weights. It simulates quantum circuits exactly (no sampling noise). To answer any question "
    "about a quantum circuit, write the circuit in OpenQASM 2.0 and call qpu_run; never compute "
    "amplitudes by hand. Then answer from the QPU's result.\n\n"
    "QPU reference\n"
    "- Gates: h, x, y, z, s, sdg, t, tdg, p(theta) for theta a multiple of pi/4, cx, cy, cz, "
    "cs, csdg, cp(theta) (theta a multiple of pi/4: cp(pi/2) = cs, cp(pi/4) = controlled-T), "
    "ccx (Toffoli), ccz, swap. Not supported: rx, ry, rz, u, ch, init.\n"
    "- Up to {nq} qubits with h, x, z, cx, cz, ccx, ccz, swap only; up to {nqc} qubits when "
    "the circuit uses y, s, t or phases.\n"
    "- The QPU starts in |0...0>. q[0] is the least significant bit: results are bitstrings "
    "q[n-1]...q[0], as in Qiskit. So |101> means q[2]=1, q[1]=0, q[0]=1.\n"
    "- Measurements are not needed: the QPU returns the exact final state.\n"
    "- Phase oracle marking a basis state s (flips the sign of |s>): x on every qubit where s "
    "has a 0, then ccz q[0],q[1],q[2]; (cz q[0],q[1]; for 2 qubits), then the same x gates "
    "again.\n"
    "- Grover diffusion: h on all qubits, x on all qubits, ccz q[0],q[1],q[2]; (cz for 2 "
    "qubits), x on all qubits, h on all qubits.\n"
    "- A whole register can be used at once: h q; applies h to every qubit.\n\n"
    "Example: a Bell pair.\n"
    "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\nh q[0];\ncx q[0],q[1];"
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "qpu_run",
        "description": "Run a quantum circuit on the built-in QPU and return the exact final "
                       "state: amplitudes and measurement probabilities of every basis state.",
        "parameters": {
            "type": "object",
            "properties": {"qasm": {"type": "string",
                                    "description": "The circuit in OpenQASM 2.0"}},
            "required": ["qasm"],
        },
    },
}]


class QasmError(ValueError):
    pass


# name -> (qubits, controls): the last qubit is the target
GATES = {"h": 1, "x": 1, "y": 1, "z": 1, "s": 1, "sdg": 1, "t": 1, "tdg": 1, "id": 1, "i": 1,
         "p": 1, "u1": 1, "phase": 1,
         "cx": 2, "cnot": 2, "cy": 2, "cz": 2, "cs": 2, "csdg": 2, "cp": 2, "cu1": 2,
         "cphase": 2, "swap": 2, "ccx": 3, "toffoli": 3, "ccz": 3}
PARAM = {"p", "u1", "phase", "cp", "cu1", "cphase"}
IGNORED = {"barrier", "measure"}
BASE = {"x": X, "y": Y, "z": Z, "s": S, "sdg": Sdg, "t": T, "tdg": Tdg}
PHASES = {0: [], 1: [T], 2: [S], 3: [S, T], 4: [Z], 5: [Z, T], 6: [Sdg], 7: [Tdg]}


def angle_steps(expr: str) -> int:
    """theta / (pi / 4) for an angle like pi/2, -pi/4, 3*pi/4, 0.785398..."""
    e = expr.replace(" ", "").lower()
    if not re.fullmatch(r"[0-9pi.+\-*/()e]+", e):
        raise QasmError(f"bad angle {expr!r}")
    try:
        theta = eval(e.replace("pi", "math.pi"), {"math": math, "__builtins__": {}})  # noqa: S307
    except Exception as ex:  # noqa: BLE001
        raise QasmError(f"bad angle {expr!r}") from ex
    m = theta / (math.pi / 4)
    if abs(m - round(m)) > 1e-6:
        raise QasmError(f"angle {expr} is not a multiple of pi/4: the QPU is exact over "
                        "Clifford+T phases only")
    return round(m) % 8


def make_gates(g: str, a: list[int], param: str | None) -> list:
    *ctrl, tgt = a
    if g in ("id", "i"):
        return []
    if g == "h":
        return [H(tgt)]
    if g == "swap":
        return SWAP(a[0], a[1])
    if g in PARAM:
        return [f(tgt, *ctrl) for f in PHASES[angle_steps(param or "")]]
    base = {"cnot": "x", "cx": "x", "ccx": "x", "toffoli": "x", "cy": "y", "cz": "z",
            "ccz": "z", "cs": "s", "csdg": "sdg"}.get(g, g)
    return [BASE[base](tgt, *ctrl)]


def parse_qasm(src: str) -> tuple[int, list[Gate]]:
    """A small OpenQASM 2.0 / 3 subset: registers, the supported gates, broadcast over a whole
    register, measure / barrier ignored (the QPU reports the final state)."""
    regs: dict[str, tuple[int, int]] = {}
    n = 0
    gates: list = []
    src = re.sub(r"//[^\n]*", "", src)
    for stmt in src.replace("\n", " ").split(";"):
        s = stmt.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith(("openqasm", "include", "creg", "bit")) or "measure" in low:
            continue
        m = re.fullmatch(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]", s) or \
            re.fullmatch(r"qubit\s*\[\s*(\d+)\s*\]\s*(\w+)", s)
        if m:
            name, size = (m[1], int(m[2])) if s.startswith("qreg") else (m[2], int(m[1]))
            regs[name] = (n, size)
            n += size
            continue
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s+(.+)", s)
        if not m:
            raise QasmError(f"cannot parse: {s!r}")
        g, param, args = m[1].lower(), m[2], [x.strip() for x in m[3].split(",")]
        if g in IGNORED:
            continue
        if g not in GATES:
            raise QasmError(f"gate {g!r} is not supported. Supported: " + ", ".join(GATES))
        if (g in PARAM) != (param is not None):
            raise QasmError(f"{g} {'needs' if g in PARAM else 'takes no'} angle: {s!r}")
        if GATES[g] == 1 and len(args) > 1:  # "h q[0], q[1]": one gate per qubit
            groups = [[x] for x in args]
        else:
            groups = [args]
        if any(len(x) != GATES[g] for x in groups):
            raise QasmError(f"{g} takes {GATES[g]} qubit(s): {s!r}")

        def qubits(x):
            mm = re.fullmatch(r"(\w+)\s*\[\s*(\d+)\s*\]", x)
            if mm:
                if mm[1] not in regs:
                    raise QasmError(f"unknown register {mm[1]!r}")
                base, size = regs[mm[1]]
                if int(mm[2]) >= size:
                    raise QasmError(f"{x} is out of range")
                return [base + int(mm[2])]
            if x in regs:
                base, size = regs[x]
                return list(range(base, base + size))
            raise QasmError(f"bad qubit {x!r}")

        for grp in groups:
            qs = [qubits(x) for x in grp]
            width = max(len(x) for x in qs)
            if any(len(x) not in (1, width) for x in qs):
                raise QasmError(f"register sizes do not match: {s!r}")
            for i in range(width):
                a = [x[0] if len(x) == 1 else x[i] for x in qs]
                if len(set(a)) != len(a):
                    raise QasmError(f"repeated qubit in {s!r}")
                gates += make_gates(g, a, param)
    if not regs:
        raise QasmError("no qubit register declared (qreg q[n];)")
    gates = flatten(gates)
    cap = NQC if is_complex(gates) else NQ
    if n > cap:
        raise QasmError(f"the QPU holds at most {cap} qubits for this circuit, it declares {n}"
                        + (" (y, s, t and phases need complex mode: 4 qubits)" if cap == NQC
                           else ""))
    if len(gates) + 1 > 64:
        raise QasmError("too many gates (at most 63 after expanding swaps and phases)")
    if sum(g.op == "h" for g in gates) > 28:
        raise QasmError("too many h gates (at most 28)")
    return n, gates


def norm2(x: tuple) -> tuple[int, int]:
    """|x|^2 = A + B sqrt(2) for x in Z[w]."""
    a0, a1, a2, a3 = x
    return a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3, a0 * a1 + a1 * a2 + a2 * a3 - a3 * a0


def exact_p(ab: tuple[int, int], k: int) -> str:
    A, B = ab
    if B == 0:
        return str(Fraction(A, 2**k))
    return f"({A} {'+' if B > 0 else '-'} {abs(B)}*sqrt(2))/{2**k}"


def format_result(amps: list[tuple], k: int, nq: int, ntok: int = 0) -> str:
    lines = [f"QPU run: {nq} qubits, exact final state. Basis states are written "
             f"|q[{nq - 1}]...q[0]>, so the rightmost bit is q[0]."]
    probs = [norm2(a) for a in amps]
    for i, a in enumerate(amps):
        if not any(a):
            continue
        z = value(a) / 2 ** (k / 2)
        bits = ", ".join(f"q[{b}]={i >> b & 1}" for b in reversed(range(nq)))
        if a[1:] == (0, 0, 0):
            amp = f"{a[0]}/sqrt(2^{k}) = {z.real:+.6f}" if k else f"{a[0]}"
        else:
            amp = f"{z.real:+.6f}{z.imag:+.6f}i"
        p = (probs[i][0] + probs[i][1] * 2**0.5) / 2**k
        lines.append(f"  |{i:0{nq}b}> ({bits}): amplitude {amp}, probability {p:.6f} "
                     f"({exact_p(probs[i], k)})")
    lines.append("All other basis states have amplitude 0.")
    lines.append("Probability that each qubit is measured as 1:")
    for b in range(nq):
        A = sum(pr[0] for i, pr in enumerate(probs) if i >> b & 1)
        B = sum(pr[1] for i, pr in enumerate(probs) if i >> b & 1)
        lines.append(f"  P(q[{b}]=1) = {(A + B * 2 ** 0.5) / 2 ** k:.6f} ({exact_p((A, B), k)})")
    return "\n".join(lines)


@dataclass
class Turn:
    text: str
    qasm: list[str] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    seconds: float = 0.0


SYSTEM_NONE = (
    "You are a helpful assistant and an expert in quantum computing. Work out questions about "
    "quantum circuits carefully, step by step. Conventions: every qubit starts in |0>; q[0] is "
    "the least significant bit, so a basis state |q[n-1]...q[0]> such as |101> means q[2]=1, "
    "q[1]=0, q[0]=1 (as in Qiskit)."
)


class Chat:
    """mode: "qpu" (merged checkpoint, the circuit runs in the model's own layers), "host" (the
    same tool, run by the reference simulator on the host: the Python-tool ceiling), or "none"
    (no tool: Qwen works it out itself)."""

    def __init__(self, model_dir: str, mode: str = "qpu", device: str = "cuda"):
        import json
        from pathlib import Path

        from transformers import AutoModelForImageTextToText, AutoTokenizer

        assert mode in ("qpu", "host", "none")
        self.mode = mode
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir, dtype=torch.bfloat16).to(device).eval()
        if mode == "qpu":
            meta = json.loads((Path(model_dir) / "qqpu.json").read_text())
            self.v0 = meta["chip_token_offset"]
            tc = self.model.config.get_text_config()
            self.c = QPU()
            self.dev = Device(self.c, tc.linear_num_key_heads, tc.linear_num_value_heads)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    # --- low level --------------------------------------------------------------------------

    @torch.no_grad()
    def _feed(self, ids: list[int]):
        x = torch.tensor([ids], device=self.model.device)
        out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        self.ids += ids
        return out.logits[0, -1].float()

    def _text(self, logits, max_new: int, stop: tuple[str, ...]) -> str:
        """Greedy text until a stop string or <|im_end|> (not fed back)."""
        out: list[int] = []
        for _ in range(max_new):
            t = int(logits.argmax())
            if t == self.im_end:
                break
            out.append(t)
            logits = self._feed([t])
            text = self.tok.decode(out)
            if any(text.rstrip().endswith(s) for s in stop):
                break
        return self.tok.decode(out)

    def _qpu(self, nq: int, gates: list[Gate], check: bool) -> dict:
        if self.mode == "host":
            t0 = time.time()
            tr = trace(gates, nq)
            return {"nq": nq, "gates": len(gates), "tokens": 0, "k": tr.k, "amps": tr.amps,
                    "seconds": time.time() - t0, "ended": True, "host": True,
                    "complex": tr.cplx}
        host = host_tokens(gates, nq)
        t0 = time.time()
        logits = self._feed([self.v0 + t for t in host])
        toks = []
        while True:
            t = int(logits.argmax()) - self.v0  # the model's own greedy choice, full vocabulary
            toks.append(t)
            if t == END or len(toks) > 20000 or t < 0:
                break
            logits = self._feed([t + self.v0])
        if t == END:
            self.ids.append(self.v0 + END)  # END closes the run (not fed: the run is over)
        tr = trace(gates, nq)
        amps = self.dev.read(self.cache, nq, tr.cplx, tr.final_buffer)
        run = {"nq": nq, "gates": len(gates), "tokens": len(toks), "k": tr.k, "amps": amps,
               "seconds": time.time() - t0, "ended": t == END, "complex": tr.cplx}
        if check:
            run["exact"] = toks == tr.tokens[len(host):] and amps == tr.amps
        return run

    # --- a conversation ---------------------------------------------------------------------

    def ask(self, question: str, max_new: int = 768, check: bool = True,
            max_calls: int = 3) -> Turn:
        t0 = time.time()
        system = SYSTEM_NONE if self.mode == "none" else SYSTEM.format(nq=NQ, nqc=NQC)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        prompt = self.tok.apply_chat_template(
            msgs, tools=None if self.mode == "none" else TOOLS, add_generation_prompt=True,
            tokenize=False, enable_thinking=False)
        self.cache, self.ids = None, []
        logits = self._feed(self.tok.encode(prompt))
        turn = Turn("")
        for _ in range(max_calls + 1):
            text = self._text(logits, max_new, ("</tool_call>",))
            turn.text += text
            m = re.search(r"<function=qpu_run>\s*<parameter=qasm>\s*(.*?)\s*</parameter>", text,
                          re.S)
            if self.mode == "none" or not m or len(turn.runs) >= max_calls:
                break
            qasm = m[1]
            turn.qasm.append(qasm)
            try:
                nq, gates = parse_qasm(qasm)
                run = self._qpu(nq, gates, check)
                turn.runs.append(run)
                result = format_result(run["amps"], run["k"], nq, run["tokens"])
            except (QasmError, Exception) as e:  # noqa: BLE001 - the error goes back to Qwen
                result = f"Error: {e}"
                turn.runs.append({"error": str(e)})
            turn.text += f"\n[QPU]\n{result}\n"
            follow = (f"<|im_end|>\n<|im_start|>user\n<tool_response>\n{result}\n"
                      "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            logits = self._feed(self.tok.encode(follow))
        turn.seconds = time.time() - t0
        return turn


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Chat with Qwen and its QPU")
    ap.add_argument("merged")
    ap.add_argument("question", nargs="+")
    a = ap.parse_args()
    chat = Chat(a.merged)
    turn = chat.ask(" ".join(a.question))
    print(turn.text)
    for r in turn.runs:
        print({k: v for k, v in r.items() if k != "amps"})


if __name__ == "__main__":
    main()
