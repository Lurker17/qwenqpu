"""Textbook circuits for the QPU, and conversion to Qiskit for the reference check."""

import random

from qqpu.machine import CCZ, CX, CZ, NQ, OPS, SWAP, Gate, H, X, Z, flatten


def bell():
    return 2, [H(0), CX(0, 1)]


def ghz(n: int):
    return n, [H(0)] + [CX(i, i + 1) for i in range(n - 1)]


def deutsch_jozsa(n: int, oracle: str):
    """n input qubits, ancilla n. oracle: 'const0', 'const1', or 'balanced' (f = x0 ^ x_last)."""
    anc = n
    g = [X(anc), H(anc)] + [H(i) for i in range(n)]
    if oracle == "const1":
        g += [X(anc)]
    elif oracle == "balanced":
        g += [CX(0, anc), CX(n - 1, anc)]
    g += [H(i) for i in range(n)]
    return n + 1, g


def bernstein_vazirani(secret: str):
    """secret bit string, qubit 0 first; the inputs read back the secret."""
    n = len(secret)
    anc = n
    g = [X(anc), H(anc)] + [H(i) for i in range(n)]
    g += [CX(i, anc) for i, b in enumerate(secret) if b == "1"]
    g += [H(i) for i in range(n)]
    return n + 1, g


def _mcz(qs):
    if len(qs) == 2:
        return [CZ(qs[0], qs[1])]
    assert len(qs) == 3
    return [CCZ(*qs)]


def grover(n: int, marked: int, iters: int | None = None):
    """Grover search over n = 2 or 3 qubits for one marked basis state."""
    import math

    iters = iters or max(1, round(math.pi / 4 * math.sqrt(2**n) - 0.5))
    qs = list(range(n))
    flip = [X(i) for i in qs if not marked >> i & 1]
    g = [H(i) for i in qs]
    for _ in range(iters):
        g += flip + _mcz(qs) + flip  # oracle: phase -1 on |marked>
        g += [H(i) for i in qs] + [X(i) for i in qs] + _mcz(qs) + [X(i) for i in qs]
        g += [H(i) for i in qs]  # diffusion (up to a global phase of -1)
    return n, g


def superdense(b0: int, b1: int):
    """Alice sends two classical bits with one qubit of a Bell pair; Bob reads q0 = b0, q1 = b1
    (deterministically)."""
    g = [H(0), CX(0, 1)]
    if b1:
        g.append(X(0))
    if b0:
        g.append(Z(0))
    g += [CX(0, 1), H(0)]
    return 2, g


def teleport(minus: bool = True):
    """Teleport q0's state (|-> or |1>) to q2, with deferred measurement (the classical
    corrections become CX / CZ)."""
    g = [X(0), H(0)] if minus else [X(0)]
    g += [H(1), CX(1, 2)]  # Bell pair between q1 and q2
    g += [CX(0, 1), H(0)]  # Bell measurement basis change
    g += [CX(1, 2), CZ(0, 2)]  # corrections
    return 3, g


def swap_test_basis():
    """SWAP of |1> and |0>: q0 = 1 -> q1 = 1."""
    return 2, [X(0)] + SWAP(0, 1)


def library() -> dict:
    return {
        "bell": bell(),
        "ghz3": ghz(3),
        "ghz5": ghz(5),
        "dj3_const": deutsch_jozsa(3, "const1"),
        "dj3_balanced": deutsch_jozsa(3, "balanced"),
        "bv_1011": bernstein_vazirani("1011"),
        "grover2_m3": grover(2, 3),
        "grover3_m5": grover(3, 5),
        "superdense_10": superdense(1, 0),
        "superdense_11": superdense(1, 1),
        "teleport": teleport(),
        "swap": swap_test_basis(),
    }


def random_circuit(seed: int, nq: int = NQ, n_gates: int = 20, max_h: int = 24):
    rng = random.Random(seed)
    g, nh = [], 0
    while len(g) < n_gates:
        op = rng.choice(OPS[1:])
        if op == "h" and nh >= max_h:
            continue
        q = rng.randrange(nq)
        others = [b for b in range(nq) if b != q]
        nctrl = 0 if op == "h" else rng.choice([0, 0, 1, 1, 2])
        ctrl = tuple(rng.sample(others, min(nctrl, len(others))))
        g.append(Gate(op, q, ctrl))
        nh += op == "h"
    return nq, g


# --- Qiskit -----------------------------------------------------------------------------------


def to_qiskit(nq: int, gates):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(nq)
    for g in flatten(gates):
        if g.op == "h":
            qc.h(g.q)
        elif g.op == "x":
            if not g.ctrl:
                qc.x(g.q)
            else:
                qc.mcx(list(g.ctrl), g.q)
        elif g.op == "z":
            if not g.ctrl:
                qc.z(g.q)
            elif len(g.ctrl) == 1:
                qc.cz(g.ctrl[0], g.q)
            else:
                qc.h(g.q)
                qc.mcx(list(g.ctrl), g.q)
                qc.h(g.q)
    return qc


def qiskit_statevector(nq: int, gates):
    import numpy as np
    from qiskit.quantum_info import Statevector

    return np.asarray(Statevector(to_qiskit(nq, gates)).data)
