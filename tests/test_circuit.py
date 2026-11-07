"""The compiled QPU circuit: exact in Qwenputer's IR evaluator (bf16 checks at every
intermediate), token for token against the reference machine."""

import random
from functools import cache

import pytest
from chipformer.q35.ir import evaluate

from qqpu import circuits
from qqpu.circuit import QPU, amplitudes, predicted
from qqpu.machine import W, trace


@cache
def qpu() -> QPU:
    return QPU()


def run(gates, nq, init=None):
    tr = trace(gates, nq, init)
    carry = {}
    m = evaluate(qpu(), tr.tokens[:-1], carry=carry)
    bad = [t for t in tr.owned if predicted(qpu(), m, t) != tr.tokens[t + 1]]
    assert not bad, f"{len(bad)} wrong, first at {bad[0]}"
    assert amplitudes(qpu(), carry, nq) == tr.amps
    return tr


@pytest.mark.parametrize("name", list(circuits.library()))
def test_library(name):
    nq, gates = circuits.library()[name]
    run(gates, nq)


@pytest.mark.parametrize("seed", range(24))
def test_random_circuits(seed):
    nq, gates = circuits.random_circuit(seed, nq=1 + seed % 5, n_gates=12)
    run(gates, nq)


@pytest.mark.parametrize("seed", range(8))
def test_random_initial_states(seed):
    """Arbitrary signed integer amplitudes (no normalisation) stress the adder's carries."""
    rng = random.Random(seed)
    nq = 2 + seed % 4
    lim = 1 << (W - 3)
    init = [rng.randrange(-lim, lim) for _ in range(1 << nq)]
    _, gates = circuits.random_circuit(100 + seed, nq=nq, n_gates=4, max_h=2)
    run(gates, nq, init)


def test_back_to_back_runs():
    """Two programs in one context: the second rewrites the ROM and the state."""
    nq, g1 = circuits.grover(3, 6)
    t1 = trace(g1, nq)
    nq2, g2 = circuits.bell()
    t2 = trace(g2, 3)  # host writes all 8 amplitude rows again
    toks = t1.tokens + t2.tokens
    owned = t1.owned + [len(t1.tokens) + t for t in t2.owned]
    carry = {}
    m = evaluate(qpu(), toks[:-1], carry=carry)
    bad = [t for t in owned if predicted(qpu(), m, t) != toks[t + 1]]
    assert not bad
    assert amplitudes(qpu(), carry, 3) == t2.amps
