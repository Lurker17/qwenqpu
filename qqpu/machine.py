"""The QPU: an exact statevector machine, its instruction set and its token trace.

Amplitudes are exact elements of Z[w] / sqrt(2)^k, w = exp(i pi / 4): each amplitude is four
16-bit two's-complement integers (a0, a1, a2, a3) meaning a0 + a1 w + a2 w^2 + a3 w^3, and the
whole state shares the scale 1 / sqrt(2)^k, k = the number of H gates run so far. Clifford+T is
closed over this ring, so the machine never rounds:

    H(q)                     i0' = i0 + i1,  i1' = i0 - i1                       (k += 1)
    X(q) / Y(q)              i0' = i1, i1' = i0  /  i0' = -i i1, i1' = i i0
    Z, S, Sdg, T, Tdg (q)    i1' = w^m i1 with m = 4, 2, 6, 1, 7  (multiplying by w^m rotates
                             the four components, negating the ones that wrap around)

Every gate but H takes a control mask (CNOT, Toffoli, CZ, CCZ, CS, CT, controlled-Y, ...).

Two modes, set by the host per program (CONF record):
  real     only a0 is carried (H, X, Z and controls): one VAL token per amplitude, up to NQ qubits
  complex  all four components: four VAL tokens per amplitude, up to NQC qubits

A gate record rewrites every amplitude, pair by pair (the pairs of its target qubit, lowest index
first, the |0> element first; component 0 first in complex mode):

    GATE(g)  VAL ... VAL  GATE(g + 1)

The state lives in two buffers: gate g reads buffer g mod 2 and writes buffer (g + 1) mod 2, so
every value is computed from the state before the gate. The program sits in a ROM written by the
host (PROG(g) VAL(word)); HALT ends the run: the model writes END after its GATE token. Host
records before the run: CONF(nq, complex), PROG(g) VAL(word) per gate, AMP(i) VAL x components
(initial state, into buffer 0), and the first GATE(0). From there every token is the model's.
"""

import cmath
import math
from dataclasses import dataclass, field

NQ = 5  # qubits in real mode (2^NQ amplitude rows)
NQC = 4  # qubits in complex mode (4 components per amplitude: j up to 64)
NCOMP = 4
W = 16  # bits per component
VAL_MAX = 1 << W
GBITS = 6  # up to 64 gates per program (HALT included)

OPS = ("halt", "h", "x", "z", "y", "s", "sdg", "t", "tdg")
COMPLEX_OPS = {"y", "s", "sdg", "t", "tdg"}
PHASE = {"z": 4, "s": 2, "sdg": 6, "t": 1, "tdg": 7}  # i1' = w^m i1
OPBITS = 4
QBITS = 3

GATE0 = VAL_MAX
PROG0 = GATE0 + (1 << GBITS)
AMP0 = PROG0 + (1 << GBITS)
CONF0 = AMP0 + 128
END = CONF0 + 16
TEXT = END + 1  # every Qwen text token reaches the circuit as this one token (merged model)
CHIP_VOCAB = TEXT


def GATE(g: int) -> int:
    assert 0 <= g < 1 << GBITS
    return GATE0 + g


def PROG(g: int) -> int:
    assert 0 <= g < 1 << GBITS
    return PROG0 + g


def AMP(i: int) -> int:
    assert 0 <= i < 1 << NQ
    return AMP0 + i


def CONF(nq: int, cplx: bool = False) -> int:
    assert 1 <= nq <= (NQC if cplx else NQ)
    return CONF0 + nq + 8 * int(cplx)


def kind(t: int) -> str:
    if t == TEXT:
        return "text"
    if t == END:
        return "end"
    for name, lo in (("conf", CONF0), ("amp", AMP0), ("prog", PROG0), ("gate", GATE0)):
        if t >= lo:
            return name
    return "val"


def token_name(t: int) -> str:
    k = kind(t)
    if k == "val":
        return f"<|q={t:04x}|>"
    if k == "gate":
        return f"<|q_gate {t - GATE0}|>"
    if k == "prog":
        return f"<|q_prog {t - PROG0}|>"
    if k == "amp":
        return f"<|q_amp {t - AMP0}|>"
    if k == "conf":
        x = t - CONF0
        return f"<|q_conf {x & 7}{' c' if x >> 3 else ''}|>"
    if k == "end":
        return "<|q_end|>"
    raise ValueError(t)


# --- instructions ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    op: str
    q: int = 0
    ctrl: tuple[int, ...] = ()

    def __post_init__(self):
        assert self.op in OPS, self.op
        assert 0 <= self.q < NQ and all(0 <= c < NQ for c in self.ctrl)
        assert self.q not in self.ctrl, "a qubit cannot control itself"
        assert len(set(self.ctrl)) == len(self.ctrl)
        assert self.op != "h" or not self.ctrl, "controlled H leaves Z[w] / sqrt(2)^k"

    @property
    def mask(self) -> int:
        return sum(1 << c for c in self.ctrl)

    def encode(self) -> int:
        return OPS.index(self.op) | self.q << OPBITS | self.mask << (OPBITS + QBITS)

    @staticmethod
    def decode(w: int) -> "Gate":
        op = OPS[w & (1 << OPBITS) - 1]
        q = w >> OPBITS & (1 << QBITS) - 1
        mask = w >> (OPBITS + QBITS)
        return Gate(op, q, tuple(b for b in range(NQ) if mask >> b & 1))

    def __str__(self) -> str:
        if self.op == "halt":
            return "halt"
        c = "c" * len(self.ctrl)
        args = ", ".join(f"q{b}" for b in (*self.ctrl, self.q))
        return f"{c}{self.op}({args})"


def H(q):
    return Gate("h", q)


def X(q, *ctrl):
    return Gate("x", q, tuple(ctrl))


def Y(q, *ctrl):
    return Gate("y", q, tuple(ctrl))


def Z(q, *ctrl):
    return Gate("z", q, tuple(ctrl))


def S(q, *ctrl):
    return Gate("s", q, tuple(ctrl))


def Sdg(q, *ctrl):
    return Gate("sdg", q, tuple(ctrl))


def T(q, *ctrl):
    return Gate("t", q, tuple(ctrl))


def Tdg(q, *ctrl):
    return Gate("tdg", q, tuple(ctrl))


def CX(c, t):
    return X(t, c)


def CCX(a, b, t):
    return X(t, a, b)


def CZ(a, b):
    return Z(b, a)


def CCZ(a, b, c):
    return Z(c, a, b)


def SWAP(a, b):
    return [CX(a, b), CX(b, a), CX(a, b)]


def flatten(gates) -> list[Gate]:
    out = []
    for g in gates:
        out += flatten(g) if isinstance(g, list) else [g]
    return out


def is_complex(gates) -> bool:
    return any(g.op in COMPLEX_OPS for g in flatten(gates))


# --- ring arithmetic --------------------------------------------------------------------------


def rot(x: tuple, m: int) -> tuple:
    """w^m * x for x = (x0, x1, x2, x3) in Z[w]."""
    out = [0] * 4
    for i, v in enumerate(x):
        e = i + m
        out[e % 4] += v if (e // 4) % 2 == 0 else -v
    return tuple(out)


def rot_src(c: int, m: int) -> tuple[int, int]:
    """Component c of w^m x is sign * x[idx]: returns (idx, sign)."""
    idx = (c - m) % 4
    return idx, (1 if ((idx + m) // 4) % 2 == 0 else -1)


def add(x, y):
    return tuple(a + b for a, b in zip(x, y))


def value(x: tuple) -> complex:
    w = cmath.exp(1j * math.pi / 4)
    return sum(v * w**i for i, v in enumerate(x))


# --- the pair walk ----------------------------------------------------------------------------


def own_row(a: int, q: int) -> int:
    """The amplitude the a-th amplitude slot of a gate on target q rewrites: pair p = a >> 1
    with a 0 inserted at bit q, then bit q = a & 1."""
    p, e = a >> 1, a & 1
    lo = p & (1 << q) - 1
    return (p >> q) << (q + 1) | e << q | lo


def jidx(q: int, r: int) -> int:
    """The amplitude slot at which a gate on target q rewrites row r (inverse of own_row)."""
    lo = r & (1 << q) - 1
    hi = r >> (q + 1)
    return 2 * (hi << q | lo) + (r >> q & 1)


def to_signed(v: int) -> int:
    v &= VAL_MAX - 1
    return v - VAL_MAX if v >> (W - 1) else v


ZERO = (0, 0, 0, 0)

# The value of one amplitude as (m_own, m_par): w^m_own * own + w^m_par * par (None: absent)
KINDS = {
    "own0": (0, None), "own1": (1, None), "own2": (2, None), "own3": (3, None),
    "own4": (4, None), "own5": (5, None), "own6": (6, None), "own7": (7, None),
    "par0": (None, 0), "par2": (None, 2), "par6": (None, 6),
    "h0": (0, 0), "h1": (4, 0),
}


def kind_of(g: Gate, e: int, active: bool) -> str:
    if not active:
        return "own0"
    if g.op == "h":
        return "h0" if e == 0 else "h1"
    if g.op == "x":
        return "par0"
    if g.op == "y":
        return "par6" if e == 0 else "par2"
    return "own0" if e == 0 else f"own{PHASE[g.op]}"


def step_amp(g: Gate, a: int, amps: list[tuple]) -> tuple:
    """New value of the a-th amplitude slot of gate g, from the state before the gate."""
    own = own_row(a, g.q)
    par = own ^ 1 << g.q
    e = a & 1
    mo, mp = KINDS[kind_of(g, e, own & g.mask == g.mask)]
    v = ZERO
    if mo is not None:
        v = add(v, rot(amps[own], mo))
    if mp is not None:
        v = add(v, rot(amps[par], mp))
    return v


class Overflow(Exception):
    pass


@dataclass
class Trace:
    tokens: list[int]
    owned: list[int]  # positions whose next token the model writes
    amps: list[tuple]  # final amplitudes, in Z[w] components
    k: int  # number of H gates: the state is amps / sqrt(2)^k
    nq: int
    cplx: bool
    gates: list[Gate] = field(default_factory=list)

    def statevector(self):
        import numpy as np

        return np.array([value(a) for a in self.amps]) / 2 ** (self.k / 2)

    def probabilities(self):
        import numpy as np

        return np.abs(self.statevector()) ** 2

    @property
    def final_buffer(self) -> int:
        return len(self.gates) % 2


def ncomp(cplx: bool) -> int:
    return NCOMP if cplx else 1


def host_tokens(gates: list[Gate], nq: int, init: list | None = None,
                cplx: bool | None = None) -> list[int]:
    """What the host feeds: register size and mode, program ROM, initial state, first GATE."""
    gates = flatten(gates)
    cplx = is_complex(gates) if cplx is None else cplx
    assert not (is_complex(gates) and not cplx), "complex gates need complex mode"
    assert nq <= (NQC if cplx else NQ), "too many qubits for this mode"
    assert all(g.q < nq and all(c < nq for c in g.ctrl) for g in gates), "qubit out of range"
    assert len(gates) + 1 <= 1 << GBITS, "program too long"
    init = norm_init(init, nq)
    toks = [CONF(nq, cplx)]
    for i, g in enumerate([*gates, Gate("halt")]):
        toks += [PROG(i), g.encode()]
    for i, a in enumerate(init):
        toks += [AMP(i)] + [x & VAL_MAX - 1 for x in a[:ncomp(cplx)]]
    return toks + [GATE(0)]


def norm_init(init, nq):
    init = init or [1] + [0] * ((1 << nq) - 1)
    assert len(init) == 1 << nq
    return [tuple(a) + (0,) * (4 - len(a)) if isinstance(a, tuple) else (a, 0, 0, 0)
            for a in init]


def trace(gates, nq: int, init: list | None = None, cplx: bool | None = None) -> Trace:
    gates = flatten(gates)
    cplx = is_complex(gates) if cplx is None else cplx
    toks = host_tokens(gates, nq, init, cplx)
    amps = norm_init(init, nq)
    nc = ncomp(cplx)
    owned = []
    k = 0
    for gi, g in enumerate(gates):
        new = list(amps)
        for a in range(1 << nq):
            v = step_amp(g, a, amps)
            if any(not -(1 << W - 1) <= x < 1 << W - 1 for x in v):
                raise Overflow(f"{g}: amplitude {v} does not fit {W} bits")
            assert cplx or v[1:] == (0, 0, 0)
            for c in range(nc):
                owned.append(len(toks) - 1)
                toks.append(v[c] & VAL_MAX - 1)
            new[own_row(a, g.q)] = v
        amps = new
        k += g.op == "h"
        owned.append(len(toks) - 1)
        toks.append(GATE(gi + 1))
    owned.append(len(toks) - 1)  # the HALT gate's token: the model writes END
    toks.append(END)
    return Trace(toks, owned, amps, k, nq, cplx, gates)
