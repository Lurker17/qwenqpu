"""Circuit questions in natural language, answered by Qwen with and without its QPU.

Every question asks for one probability; the ground truth comes from a reference circuit run on
the exact reference machine (checked against Qiskit in tests). Conditions:

  none   base Qwen, no tool (it reasons step by step)
  qpu    merged Qwen: it writes OpenQASM, the circuit runs in its own layers
  host   Qwen with the same tool executed by a host simulator (the ceiling)

Run: uv run python scripts/eval_chat.py MODEL_DIR MODE [--out results.jsonl] [--only i,j]
"""

import argparse
import json
import re
import time
from fractions import Fraction

from qqpu import circuits
from qqpu.machine import CCX, CX, CZ, H, S, T, X, Z, trace


def p_state(nq, gates, state):
    return float(trace(gates, nq).probabilities()[state])


def p_qubit(nq, gates, q, v=1):
    p = trace(gates, nq).probabilities()
    return float(sum(x for i, x in enumerate(p) if (i >> q & 1) == v))


def items():
    out = []

    def add(q, truth):
        out.append({"question": q, "truth": truth})

    add("Prepare a Bell pair on two qubits: apply H to q0, then CNOT with control q0 and "
        "target q1. What is the probability of measuring |11>?", p_state(*circuits.bell(), 0b11))
    add("Prepare a 4-qubit GHZ state with H on q0 followed by a chain of CNOTs (q0->q1, "
        "q1->q2, q2->q3). What is the probability of measuring |1111>?",
        p_state(*circuits.ghz(4), 0b1111))
    add("Run Grover's search on 3 qubits for the marked state |101>, with 2 Grover "
        "iterations. What is the probability of measuring 101?",
        p_state(*circuits.grover(3, 0b101, 2), 0b101))
    add("Run Grover's search on 3 qubits for the marked state |011>, with exactly 1 Grover "
        "iteration. What is the probability of measuring 011?",
        p_state(*circuits.grover(3, 0b011, 1), 0b011))
    add("Run Grover's search on 2 qubits for the marked state |10>, with 1 Grover iteration. "
        "What is the probability of measuring 10?", p_state(*circuits.grover(2, 0b10, 1), 0b10))
    nq, g = circuits.deutsch_jozsa(3, "balanced")
    add("Run the Deutsch-Jozsa algorithm with 3 input qubits (q0, q1, q2) and an ancilla q3, "
        "for the balanced function f(x) = x0 XOR x2 (oracle: CNOT q0->q3 and CNOT q2->q3; "
        "the ancilla is prepared with X then H). What is the probability that all three input "
        "qubits are measured as 0?",
        float(sum(trace(g, nq).probabilities()[[0b0000, 0b1000]])))
    g = [H(0), H(1), H(2), CCX(0, 1, 2), H(2)]
    add("Apply H to qubits q0, q1 and q2, then a Toffoli gate with controls q0, q1 and target "
        "q2, then H on q2. What is the probability that q2 is measured as 1?", p_qubit(3, g, 2))
    g = [H(0), H(1), H(2), Z(2, 0, 1), H(0), H(1), H(2)]
    add("On 3 qubits, apply H to every qubit, then a CCZ gate on q0, q1, q2, then H to every "
        "qubit again. What is the probability of measuring |000>?", p_state(3, g, 0))
    g = [H(0), CX(0, 1), H(0), H(1)]
    add("Apply H to q0, then CNOT from q0 to q1, then H to both q0 and q1. What is the "
        "probability of measuring |00>?", p_state(2, g, 0))
    g = [H(0), H(1), H(2), H(3), CZ(0, 1), CZ(2, 3), H(0), H(1), H(2), H(3)]
    add("On 4 qubits, apply H to every qubit, then CZ on (q0, q1) and CZ on (q2, q3), then H "
        "to every qubit again. What is the probability of measuring |0000>?", p_state(4, g, 0))
    nq, g = circuits.teleport(minus=False)
    add("Teleport the state |1> from q0 to q2: make a Bell pair with H on q1 and CNOT q1->q2, "
        "then CNOT q0->q1 and H on q0, then apply the corrections as CNOT q1->q2 and CZ "
        "q0->q2 (deferred measurement). What is the probability that q2 is measured as 1?",
        p_qubit(nq, g, 2))
    g = [H(0), CX(0, 1), CX(1, 2), Z(2), H(0), H(1), H(2)]
    add("Apply H to q0, CNOT q0->q1, CNOT q1->q2, Z on q2, then H on all three qubits. "
        "What is the probability of measuring |000>?", p_state(3, g, 0))
    g = [X(0), H(0), H(1), CZ(0, 1), H(1)]
    add("Apply X to q0, then H to q0 and H to q1, then CZ on q0 and q1, then H on q1. What "
        "is the probability that q1 is measured as 1?", p_qubit(2, g, 1))
    nq, g = circuits.superdense(1, 1)
    add("Superdense coding: make a Bell pair (H on q0, CNOT q0->q1). Alice applies X and then "
        "Z to q0. Bob applies CNOT q0->q1 then H on q0. What is the probability that Bob "
        "measures |11>?", p_state(nq, g, 0b11))
    g = [H(0), CX(0, 1), CX(0, 2), H(0), H(1), H(2)]
    add("Apply H to q0, CNOT q0->q1, CNOT q0->q2, then H to all three qubits. What is the "
        "probability of measuring |000>?", p_state(3, g, 0))
    g = [H(0), H(1), H(2), X(1), Z(2, 0, 1), X(1), H(2), CX(2, 0)]
    add("On 3 qubits: H on q0, q1, q2; X on q1; CCZ on q0, q1, q2; X on q1 again; H on q2; "
        "then CNOT q2->q0. What is the probability of measuring |001>?", p_state(3, g, 0b001))
    # --- complex amplitudes: S, T, controlled phases (QPU v2) ---
    add("On a single qubit, apply H, then T, then H. What is the probability of measuring 1?",
        p_qubit(1, [H(0), T(0), H(0)], 0))
    nq, g = circuits.phase_estimation("t")
    add("Run quantum phase estimation for the T gate acting on its eigenstate |1>: use counting "
        "qubits q0, q1, q2 (q0 least significant) and target q3 prepared with X; controlled-T "
        "from q0, controlled-S (= T^2) from q1, controlled-Z (= T^4) from q2, then the inverse "
        "QFT on q0..q2. What is the probability that the counting register reads 001 "
        "(q2=0, q1=0, q0=1)?", float(sum(trace(g, nq).probabilities()[[0b0001, 0b1001]])))
    nq, g = circuits.qft_basis(3, 5)
    add("Prepare the 3-qubit basis state |101> (q2=1, q1=0, q0=1) and apply the quantum Fourier "
        "transform. What is the probability of measuring |000>?", p_state(nq, g, 0))
    add("Apply H, then S, then H to a single qubit. What is the probability of measuring 1?",
        p_qubit(1, [H(0), S(0), H(0)], 0))
    g = [H(0), CX(0, 1), T(1), CX(0, 1), H(0)]
    add("Apply H to q0, CNOT q0->q1, T on q1, CNOT q0->q1, then H on q0. What is the "
        "probability that q0 is measured as 1?", p_qubit(2, g, 0))
    g = [H(0), H(1), S(1, 0), H(0), H(1)]
    add("Apply H to q0 and q1, then a controlled-S gate (control q0, target q1), then H to q0 "
        "and q1 again. What is the probability of measuring |00>?", p_state(2, g, 0))
    return out


SUFFIX = ("\n\nEnd your reply with one final line of the form `ANSWER: <probability>` with the "
          "probability as a decimal number.")


def parse_answer(text: str):
    m = re.findall(r"ANSWER:\s*\**\s*([-+]?[\d./]+)\s*(%?)", text)
    if not m:
        return None
    num, pct = m[-1]
    try:
        v = float(Fraction(num.rstrip(".")))
    except (ValueError, ZeroDivisionError):
        return None
    return v / 100 if pct else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("mode", choices=["qpu", "host", "none"])
    ap.add_argument("--out")
    ap.add_argument("--only")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--think", action="store_true", help="Qwen's thinking mode on")
    a = ap.parse_args()
    from qqpu.bridge import Chat

    chat = Chat(a.model, a.mode)
    its = items()
    idx = [int(i) for i in a.only.split(",")] if a.only else range(len(its))
    right = 0
    fout = open(a.out, "a") if a.out else None
    for i in idx:
        it = its[i]
        t0 = time.time()
        turn = chat.ask(it["question"] + SUFFIX, max_new=a.max_new, think=a.think)
        ans = parse_answer(turn.text)
        ok = ans is not None and abs(ans - it["truth"]) < 1e-3
        right += ok
        runs = [{k: v for k, v in r.items() if k != "amps"} for r in turn.runs]
        print(f"[{i:2d}] {'OK ' if ok else 'BAD'} truth={it['truth']:.6f} answer={ans} "
              f"calls={len(runs)} exact_runs={sum(bool(r.get('exact')) for r in runs)} "
              f"{time.time() - t0:.0f}s", flush=True)
        if fout:
            fout.write(json.dumps({"i": i, "mode": a.mode, "think": a.think, "model": a.model,
                                   "ok": ok,
                                   "truth": it["truth"], "answer": ans, "runs": runs,
                                   "qasm": turn.qasm, "text": turn.text}) + "\n")
            fout.flush()
    print(f"{a.mode} {a.model}: {right}/{len(idx)} correct")


if __name__ == "__main__":
    main()
