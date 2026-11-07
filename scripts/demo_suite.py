"""The curated demo: every textbook circuit, as OpenQASM, run inside the merged model.

For each circuit: QASM -> host records -> the model writes the whole run (and, for real-mode
circuits, measurement shots sampled with its own tied-logit coins) -> the state read from the
DeltaNet cache is compared with Qiskit. Results go to a JSON file for the write-up.

Run: uv run python scripts/demo_suite.py build/merged-4b-v2 [--shots 32] [--out build/demo.json]
"""

import argparse
import collections
import json
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText

from qqpu import circuits
from qqpu.bridge import parse_qasm
from qqpu.circuit import QPU
from qqpu.hf import Runner
from qqpu.machine import SHOT0, flatten, host_tokens, kind, token_name, trace, value

DESCR = {
    "bell": "Bell pair: H, CNOT",
    "ghz3": "3-qubit GHZ state",
    "ghz5": "5-qubit GHZ state",
    "dj3_const": "Deutsch-Jozsa, constant oracle",
    "dj3_balanced": "Deutsch-Jozsa, balanced oracle x0 xor x2",
    "bv_1011": "Bernstein-Vazirani, secret 1011",
    "grover2_m3": "Grover search, 2 qubits, marked |11>",
    "grover3_m5": "Grover search, 3 qubits, marked |101>, 2 iterations",
    "superdense_10": "Superdense coding, bits 10",
    "superdense_11": "Superdense coding, bits 11",
    "teleport": "Teleportation of |-> (deferred measurement)",
    "swap": "SWAP of |1> into q1",
    "qft3_5": "QFT of |101> on 3 qubits",
    "qpe_t": "Phase estimation of T (phase 1/8)",
    "qpe_s": "Phase estimation of S (phase 1/4)",
    "y_phase": "Y, S, T and controlled-Y mix",
}


def to_qasm(nq, gates) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{nq}];"]
    for g in flatten(gates):
        name = "c" * len(g.ctrl) + g.op
        name = {"cs": "cs", "csdg": "csdg", "ct": "cp(pi/4)", "ctdg": "cp(-pi/4)",
                "cy": "cy"}.get(name, name)
        if name in ("cct", "cctdg", "ccs", "ccsdg", "ccy", "ccy"):
            raise ValueError(name)
        args = ",".join(f"q[{b}]" for b in (*g.ctrl, g.q))
        lines.append(f"{name} {args};")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--shots", type=int, default=32)
    ap.add_argument("--out", default="build/demo.json")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    meta = json.load(open(f"{a.model}/qqpu.json"))
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
    runner = Runner(model, QPU(), offset=meta["chip_token_offset"])
    gen = torch.Generator(device="cuda").manual_seed(2026)
    out = []
    for name, (nq, gates) in circuits.library().items():
        qasm = to_qasm(nq, gates)
        nq2, g2 = parse_qasm(qasm)  # the same path Qwen's tool calls take
        tr = trace(g2, nq2)
        shots = 0 if tr.cplx else a.shots
        host = host_tokens(g2, nq2, shots=shots)
        ref = tr.tokens[len(host_tokens(g2, nq2)):-1]  # the gate part (deterministic)
        t0 = time.time()
        run = runner.run(host, nq2, tr.cplx, tr.final_buffer, gen=gen)
        secs = time.time() - t0
        exact = run.tokens[:len(ref)] == ref and run.amps == tr.amps
        sv = np.array([value(x) for x in run.amps]) / 2 ** (tr.k / 2)
        qk = circuits.qiskit_statevector(nq2, g2)
        shots_out = [t - SHOT0 for t in run.tokens if kind(t) == "shot"]
        cnt = collections.Counter(shots_out)
        probs = np.abs(qk) ** 2
        rec = {
            "name": name, "description": DESCR.get(name, name), "qasm": qasm, "nq": nq2,
            "gates": len(g2), "complex": tr.cplx, "k": tr.k, "model_tokens": len(run.tokens),
            "seconds": round(secs, 1), "exact": exact,
            "matches_qiskit": bool(np.allclose(sv, qk, atol=1e-9)),
            "amplitudes": {f"{i:0{nq2}b}": [round(z.real, 6), round(z.imag, 6)]
                           for i, z in enumerate(sv) if abs(z) > 1e-12},
            "exact_amplitudes": {f"{i:0{nq2}b}": list(x) for i, x in enumerate(run.amps)
                                 if any(x)},
            "probabilities": {f"{i:0{nq2}b}": round(float(p), 6)
                              for i, p in enumerate(probs) if p > 1e-12},
            "shots": shots, "counts": {f"{i:0{nq2}b}": n for i, n in sorted(cnt.items())},
            "trace_head": [token_name(t) for t in run.tokens[:12]],
        }
        out.append(rec)
        print(f"{name:14s} {rec['description']:52s} tokens={len(run.tokens):5d} {secs:5.0f}s "
              f"exact={exact} qiskit={rec['matches_qiskit']} counts={rec['counts']}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"{sum(r['exact'] and r['matches_qiskit'] for r in out)}/{len(out)} exact")


if __name__ == "__main__":
    main()
