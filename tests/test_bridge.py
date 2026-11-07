import numpy as np
import pytest

from qqpu import circuits
from qqpu.bridge import QasmError, format_result, parse_qasm
from qqpu.machine import trace

GROVER = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q;
// oracle for |101>
x q[1];
ccz q[0],q[1],q[2];
x q[1];
h q; x q; ccz q[0], q[1], q[2]; x q; h q;
x q[1]; ccz q[0],q[1],q[2]; x q[1];
h q; x q; ccz q[0],q[1],q[2]; x q; h q;
measure q -> c;
"""


def test_grover_qasm_matches_library():
    nq, gates = parse_qasm(GROVER)
    assert nq == 3
    ref = circuits.qiskit_statevector(*circuits.grover(3, 5))
    assert np.allclose(trace(gates, nq).statevector(), ref.real)


def test_lenient_forms():
    nq, g = parse_qasm("qubit[2] q; h q[0], q[1]; cnot q[0], q[1]; swap q[0],q[1];")
    assert nq == 2 and len(g) == 2 + 1 + 3


@pytest.mark.parametrize("bad", ["qreg q[2]; rz(0.3) q[0];", "qreg q[9]; h q[0];",
                                 "h q[0];", "qreg q[2]; cx q[0],q[0];"])
def test_errors(bad):
    with pytest.raises(QasmError):
        parse_qasm(bad)


def test_format():
    nq, g = circuits.bell()
    tr = trace(g, nq)
    s = format_result(tr.amps, tr.k, nq, 0)
    assert "|11> (q[1]=1, q[0]=1)" in s and "P(q[0]=1) = 0.500000 (1/2)" in s


QFT_QASM = """OPENQASM 2.0; include "qelib1.inc"; qreg q[3];
x q[0]; x q[2];
h q[2]; cp(pi/2) q[1],q[2]; cp(pi/4) q[0],q[2];
h q[1]; cs q[0],q[1];
h q[0];
swap q[0],q[2];"""


def test_qft_qasm_matches_library():
    nq, gates = parse_qasm(QFT_QASM)
    ref = circuits.qiskit_statevector(*circuits.qft_basis(3, 5))
    assert np.allclose(trace(gates, nq).statevector(), ref)


def test_phase_angles():
    from qqpu.bridge import angle_steps
    assert [angle_steps(a) for a in ("pi/4", "pi/2", "-pi/4", "3*pi/4", "pi", "0")] == \
        [1, 2, 7, 3, 4, 0]
    with pytest.raises(QasmError):
        angle_steps("pi/8")


def test_complex_format():
    nq, g = parse_qasm("qreg q[1]; h q[0]; t q[0];")
    tr = trace(g, nq)
    s = format_result(tr.amps, tr.k, nq)
    assert "+0.500000+0.500000i" in s and "P(q[0]=1) = 0.500000 (1/2)" in s


def test_qubit_caps():
    with pytest.raises(QasmError):
        parse_qasm("qreg q[5]; t q[4];")
    parse_qasm("qreg q[5]; h q[4];")
