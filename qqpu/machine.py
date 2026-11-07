"""The QPU: an exact statevector machine, its instruction set and its token trace.

State: 2^nq real amplitudes, each a 16-bit two's-complement integer, and a global scale
1/sqrt(2)^k where k is the number of H gates run so far. Every gate of the real Clifford+Toffoli
family keeps amplitudes in Z / sqrt(2)^k, so the machine is exact: no rounding anywhere.

    H(q)                  pair (i0, i1) differing in bit q:  i0' = i0 + i1,  i1' = i0 - i1  (k += 1)
    X(q),  ctrl mask C    i0' = i1, i1' = i0       on pairs whose index has every bit of C set
    Z(q),  ctrl mask C    i1' = -i1                (CZ, CCZ: Z with controls)

so CNOT = X with one control, Toffoli = X with two, CZ / CCZ likewise. {H, Toffoli} is
universal for real-amplitude quantum computation (Shi 2003, Aharonov 2003).

A gate record rewrites every amplitude, pair by pair (the pairs of its target qubit, lowest index
first, the |0> element of each pair first):

    GATE(g)  VAL(a'_own(0))  VAL(a'_own(1))  ...  VAL(a'_own(2^nq - 1))

then GATE(g + 1). The program sits in a ROM written by the host: PROG(g) VAL(word). A HALT gate
ends the run: the model writes END after its GATE token. Host records before the run: CONF(nq)
(the register size), PROG(g) VAL(word) for each gate, AMP(i) VAL(a) for the initial state, and
the first GATE(0). From there every token is the model's.
"""

from dataclasses import dataclass, field

NQ = 5  # qubits the machine holds (2^NQ amplitude rows)
W = 16  # bits per amplitude
VAL_MAX = 1 << W
GBITS = 6  # up to 64 gates per program (HALT included)

OPS = ("halt", "h", "x", "z")
OPBITS = 3
QBITS = 3

GATE0 = VAL_MAX
PROG0 = GATE0 + (1 << GBITS)
AMP0 = PROG0 + (1 << GBITS)
CONF0 = AMP0 + 128
END = CONF0 + 8
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


def CONF(nq: int) -> int:
    assert 1 <= nq <= NQ
    return CONF0 + nq


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
        return f"<|q_conf {t - CONF0}|>"
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
        assert self.op != "h" or not self.ctrl, "controlled H leaves Z / sqrt(2)^k"

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


def Z(q, *ctrl):
    return Gate("z", q, tuple(ctrl))


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


# --- the pair walk ----------------------------------------------------------------------------


def own_row(j: int, q: int) -> int:
    """The amplitude the j-th VAL of a gate on target q rewrites: pair p = j >> 1 with a 0
    inserted at bit q, then bit q = j & 1."""
    p, e = j >> 1, j & 1
    lo = p & (1 << q) - 1
    return (p >> q) << (q + 1) | e << q | lo


def to_signed(v: int) -> int:
    v &= VAL_MAX - 1
    return v - VAL_MAX if v >> (W - 1) else v


def step_value(g: Gate, j: int, amps: list[int]) -> int:
    """The j-th value of gate g, from the amplitudes before the gate (two's complement)."""
    own = own_row(j, g.q)
    par = own ^ 1 << g.q
    e = j & 1
    a, b = amps[own], amps[par]
    active = own & g.mask == g.mask
    if not active or g.op == "z" and e == 0:
        v = a
    elif g.op == "h":
        v = a + b if e == 0 else b - a
    elif g.op == "x":
        v = b
    else:  # z, e = 1
        v = -a
    return v


class Overflow(Exception):
    pass


def run_gate(g: Gate, amps: list[int], nq: int) -> list[int]:
    """The VAL values of one gate record (signed), in emission order."""
    vals = []
    for j in range(1 << nq):
        v = step_value(g, j, amps)
        if not -(1 << W - 1) <= v < 1 << W - 1:
            raise Overflow(f"{g}: amplitude {v} does not fit {W} bits")
        vals.append(v)
    return vals


@dataclass
class Trace:
    tokens: list[int]
    owned: list[int]  # positions whose next token the model writes
    amps: list[int]  # final integer amplitudes
    k: int  # number of H gates: the state is amps / sqrt(2)^k
    nq: int
    gates: list[Gate] = field(default_factory=list)

    def statevector(self):
        import numpy as np

        return np.array(self.amps, dtype=float) / 2 ** (self.k / 2)

    def probabilities(self):
        import numpy as np

        a = np.array(self.amps, dtype=float)
        return a * a / 2.0**self.k


def host_tokens(gates: list[Gate], nq: int, init: list[int] | None = None) -> list[int]:
    """What the host feeds: register size, program ROM, initial state, the first GATE."""
    gates = flatten(gates)
    assert all(g.q < nq and all(c < nq for c in g.ctrl) for g in gates), "qubit out of range"
    assert len(gates) + 1 <= 1 << GBITS, "program too long"
    init = init or [1] + [0] * ((1 << nq) - 1)
    assert len(init) == 1 << nq
    toks = [CONF(nq)]
    for i, g in enumerate([*gates, Gate("halt")]):
        toks += [PROG(i), g.encode()]
    for i, a in enumerate(init):
        toks += [AMP(i), a & VAL_MAX - 1]
    return toks + [GATE(0)]


def trace(gates, nq: int, init: list[int] | None = None) -> Trace:
    gates = flatten(gates)
    toks = host_tokens(gates, nq, init)
    amps = list(init or [1] + [0] * ((1 << nq) - 1))
    owned = []
    k = 0
    for gi, g in enumerate(gates):
        vals = run_gate(g, amps, nq)
        new = list(amps)
        for j, v in enumerate(vals):
            owned.append(len(toks) - 1)
            toks.append(v & VAL_MAX - 1)
            new[own_row(j, g.q)] = v
        amps = new
        k += g.op == "h"
        owned.append(len(toks) - 1)
        toks.append(GATE(gi + 1))
    owned.append(len(toks) - 1)  # the HALT gate's token: the model writes END
    toks.append(END)
    return Trace(toks, owned, amps, k, nq, gates)
