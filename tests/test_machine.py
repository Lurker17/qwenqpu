import numpy as np
import pytest

from qqpu import circuits
from qqpu.machine import rot, rot_src, trace, value


@pytest.mark.parametrize("name", list(circuits.library()))
def test_library_matches_qiskit(name):
    nq, gates = circuits.library()[name]
    tr = trace(gates, nq)
    ref = circuits.qiskit_statevector(nq, gates)
    assert np.allclose(tr.statevector(), ref, atol=1e-9)


@pytest.mark.parametrize("seed", range(40))
def test_random_real_matches_qiskit(seed):
    nq, gates = circuits.random_circuit(seed, nq=1 + seed % 5)
    tr = trace(gates, nq)
    assert not tr.cplx
    assert np.allclose(tr.statevector(), circuits.qiskit_statevector(nq, gates), atol=1e-9)


@pytest.mark.parametrize("seed", range(40))
def test_random_complex_matches_qiskit(seed):
    nq, gates = circuits.random_circuit(seed, nq=1 + seed % 4, cplx=True)
    tr = trace(gates, nq, cplx=True)
    assert np.allclose(tr.statevector(), circuits.qiskit_statevector(nq, gates), atol=1e-9)


def test_rotations():
    rng = np.random.default_rng(0)
    w = np.exp(1j * np.pi / 4)
    for _ in range(50):
        x = tuple(int(v) for v in rng.integers(-9, 9, 4))
        for m in range(8):
            assert np.isclose(value(rot(x, m)), w**m * value(x))
            for c in range(4):
                idx, sg = rot_src(c, m)
                assert rot(x, m)[c] == sg * x[idx]


def test_textbook_outcomes():
    lib = circuits.library()
    p = trace(*reversed(lib["grover3_m5"])).probabilities()
    assert p.argmax() == 5 and p[5] > 0.94
    p = trace(*reversed(lib["qpe_t"])).probabilities()
    assert np.isclose(p[0b1001], 1)  # counting register 001 (phase 1/8), target q3 = 1
    p = trace(*reversed(lib["qpe_s"])).probabilities()
    assert np.isclose(p[0b1010], 1)  # 010: phase 1/4
    sv = trace(*reversed(lib["qft3_5"])).statevector()
    assert np.allclose(np.abs(sv), 1 / np.sqrt(8))
