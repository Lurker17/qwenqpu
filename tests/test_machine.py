import numpy as np
import pytest

from qqpu import circuits
from qqpu.machine import trace


@pytest.mark.parametrize("name", list(circuits.library()))
def test_library_matches_qiskit(name):
    nq, gates = circuits.library()[name]
    tr = trace(gates, nq)
    ref = circuits.qiskit_statevector(nq, gates)
    assert np.allclose(tr.statevector(), ref.real, atol=1e-9) and np.allclose(ref.imag, 0)


@pytest.mark.parametrize("seed", range(40))
def test_random_matches_qiskit(seed):
    nq, gates = circuits.random_circuit(seed, nq=1 + seed % 5)
    tr = trace(gates, nq)
    ref = circuits.qiskit_statevector(nq, gates)
    assert np.allclose(tr.statevector(), ref.real, atol=1e-9)


def test_textbook_outcomes():
    lib = circuits.library()
    p = trace(*reversed(lib["grover3_m5"])).probabilities()
    assert p.argmax() == 5 and p[5] > 0.94
    p = trace(*reversed(lib["bv_1011"])).probabilities()
    # inputs read the secret: qubit i = secret[i] -> index bits 0,2,3 set (+ ancilla in |->)
    assert np.isclose(p[[0b01101, 0b11101]].sum(), 1)
    p = trace(*reversed(lib["superdense_10"])).probabilities()
    assert np.isclose(p[0b01], 1)
