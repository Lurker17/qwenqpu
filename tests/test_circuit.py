"""The compiled QPU circuit: exact in Qwenputer's IR evaluator (bf16 checks at every
intermediate), token for token against the reference machine."""

import random
from functools import cache

import pytest
from chipformer.q35.ir import evaluate

from qqpu import circuits
from qqpu.circuit import QPU, amplitudes, matches, predicted
from qqpu.machine import NQC, W, trace


@cache
def qpu() -> QPU:
    return QPU()


def check(tokens, owned, ties=None):
    carry = {}
    m = evaluate(qpu(), tokens[:-1], carry=carry)
    bad = [t for t in owned if not matches(predicted(qpu(), m, t), tokens[t + 1])]
    assert not bad, f"{len(bad)} wrong, first at {bad[0]}: {predicted(qpu(), m, bad[0])}"
    # random bits exactly where the reference drew them, and nowhere else
    ties = ties or {}
    wrong = [t for t in owned if predicted(qpu(), m, t)[1] != ties.get(t, 0)]
    assert not wrong, f"tie masks differ at {wrong[:3]}"
    return carry


def run(gates, nq, init=None, cplx=None, shots=0, seed=0):
    tr = trace(gates, nq, init, cplx, shots, seed)
    carry = check(tr.tokens, tr.owned, tr.ties)
    assert amplitudes(qpu(), carry, nq, tr.cplx, tr.final_buffer) == tr.amps
    return tr


@pytest.mark.parametrize("name", list(circuits.library()))
def test_library(name):
    nq, gates = circuits.library()[name]
    run(gates, nq)


@pytest.mark.parametrize("seed", range(16))
def test_random_real(seed):
    nq, gates = circuits.random_circuit(seed, nq=1 + seed % 5, n_gates=10)
    run(gates, nq)


@pytest.mark.parametrize("seed", range(16))
def test_random_complex(seed):
    nq, gates = circuits.random_circuit(seed, nq=1 + seed % NQC, n_gates=10, cplx=True)
    run(gates, nq, cplx=True)


@pytest.mark.parametrize("seed", range(6))
def test_random_initial_states(seed):
    """Arbitrary signed integer amplitudes stress the adder's carries and the rotations."""
    rng = random.Random(seed)
    cplx = seed % 2 == 1
    nq = 2 + seed % 3
    lim = 1 << (W - 3)
    init = [tuple(rng.randrange(-lim, lim) for _ in range(4 if cplx else 1))
            for _ in range(1 << nq)]
    _, gates = circuits.random_circuit(100 + seed, nq=nq, n_gates=4, max_h=2, cplx=cplx)
    run(gates, nq, init, cplx)


def test_back_to_back_runs():
    """A complex program, then a real one in the same context."""
    nq1, g1 = circuits.library()["qpe_t"]
    t1 = trace(g1, nq1)
    t2 = trace(*reversed(circuits.bell()))
    toks = t1.tokens + t2.tokens
    owned = t1.owned + [len(t1.tokens) + t for t in t2.owned]
    carry = check(toks, owned)
    assert amplitudes(qpu(), carry, 2, False, t2.final_buffer) == t2.amps


@pytest.mark.parametrize("name,shots", [("bell", 12), ("ghz3", 8), ("grover3_m5", 6),
                                        ("dj3_balanced", 6), ("teleport", 5)])
def test_measurement_shots(name, shots):
    """MEAS: random IDX / VAL tokens are ties over exactly the reference's random bits; the
    accept / reject decision and the SHOT outcomes are exact."""
    nq, gates = circuits.library()[name]
    tr = run(gates, nq, shots=shots, seed=7)
    assert len(tr.outcomes) == shots


def test_shots_after_no_gates_and_odd_k():
    run([], 2, shots=3, seed=1)  # k = 0: m = 0, the random VALs are all 0
    run([circuits.H(0), circuits.H(1), circuits.H(2)], 3, shots=4, seed=2)  # k = 3: m = 2
