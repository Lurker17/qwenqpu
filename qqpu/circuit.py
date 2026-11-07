"""The QPU as Qwen3.5 layers (Gated DeltaNet heads + SwiGLU MLPs), on Qwenputer's q35 compiler.

Every weight is computed; nothing is trained. Per token, in one forward pass:

1. Record and counter heads: which record we are in (GATE g / AMP i / PROG g / CONF), and j,
   the number of VALs since the record started (a halving head, Qwenputer's counter).
2. ROM head: gate g's instruction word (op, target q, control mask), written by PROG records.
3. The pair walk: own(j, q) (the amplitude the next VAL rewrites) and its partner own ^ 2^q,
   as one-hot queries; whether the controls are satisfied; which of five exclusive cases the
   value is (copy, own + partner, partner - own, partner, -own).
4. Two amplitude heads hold the state, one row per basis state, 16 bits each. Copy A writes a
   VAL's payload at the VAL token itself; copy B one token later. The own amplitude is read
   from A and the partner from B, so both reads see the amplitudes from *before* the gate
   although the record rewrites them in place (see `machine.py` for the order).
5. A 16-bit carry-lookahead adder (two MLP layers) computes the case's value, and the output
   code makes the model's lm_head predict VAL(value), the next GATE, or END.
"""

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Chan, Target
from chipformer.q35.prims import Circ, from_bits

from qgpu.circuit.gadgets import adder_stage2, counter, stage1_digits, thead, thermo
from qqpu.machine import (
    AMP,
    CONF,
    END,
    GATE0,
    GBITS,
    NQ,
    OPBITS,
    OPS,
    PROG,
    QBITS,
    TEXT,
    W,
    kind,
)

NCODE = 17  # every chip token id is < 2^17
CODE_AMP = 128
KAPPA = 2.0**-20
TAU_TEXT = 2.0**-40


def jidx(q: int, r: int) -> int:
    """The VAL index j at which a gate on target q rewrites row r (inverse of own_row)."""
    lo = r & (1 << q) - 1
    hi = r >> (q + 1)
    return 2 * (hi << q | lo) + (r >> q & 1)


class QPU(Circ):
    def __init__(self, target: Target | None = None, nq: int = NQ) -> None:
        super().__init__(target or Target())
        c, one = self, self.one
        R = 1 << nq
        self.nq_max = nq

        def feat(name, fn):
            return c.token_feature(name, fn, full=True)

        def lag(e, k):
            return Chan(e, tuple(1.0 if i == k else 0.0 for i in range(4)))

        is_val = feat("is_val", lambda t: int(kind(t) == "val"))
        is_gate = feat("is_gate", lambda t: int(kind(t) == "gate"))
        is_conf = feat("is_conf", lambda t: int(kind(t) == "conf"))
        pay = [feat(f"pay{b}", lambda t, b=b: int(kind(t) == "val" and t >> b & 1))
               for b in range(W)]
        gb = [feat(f"gb{b}", lambda t, b=b: int(kind(t) == "gate" and (t - GATE0) >> b & 1))
              for b in range(GBITS)]
        cb = [feat(f"cb{b}", lambda t, b=b: int(kind(t) == "conf" and (t - CONF(1) + 1) >> b & 1))
              for b in range(3)]
        amp_at = [feat(f"amp{r}", lambda t, r=r: int(t == AMP(r))) for r in range(R)]
        start = one - is_val  # every other token starts a record (text included)

        # --- record, register size, counter -------------------------------------------
        (rec,) = thead(c, "rec", {0: start}, {0: one},
                       [{"data": {2: is_gate, **{3 + b: gb[b] for b in range(GBITS)}},
                         "ref": one}])
        r_gate, r_g = rec[0], rec[1:1 + GBITS]
        (conf,) = thead(c, "conf", {0: is_conf}, {0: one},
                        [{"data": {2 + b: cb[b] for b in range(3)}, "ref": one}])
        nq_oh = c.onehot_of_bits(conf, "nq", nq + 1)  # nq_oh[n]: the register has n qubits
        jt = thermo(c, counter(c, "j", start, is_val), start, R + 1, "j")

        def j_is(k: int) -> Expr:
            return (one if k == 0 else jt[k - 1]) - jt[k]

        # --- fetch -------------------------------------------------------------------------
        g_oh = c.onehot_of_bits(r_g, "g")
        nprog = 1 << GBITS
        rom_keys = {g: lag(feat(f"prog{g}", lambda t, g=g: int(t == PROG(g))), 1)
                    for g in range(nprog)}
        rom_q = {g: c.and_([r_gate, g_oh[g]], f"romq{g}") for g in range(nprog)}
        wbits = OPBITS + QBITS + nq
        (word,) = thead(c, "rom", rom_keys, rom_q,
                        [{"data": {2 + b: lag(pay[b], 0) for b in range(wbits)}, "ref": one}])
        op_oh = c.onehot_of_bits(word[:OPBITS], "op", len(OPS))
        op_halt, op_h, op_x, op_z = (op_oh[OPS.index(o)] for o in ("halt", "h", "x", "z"))
        q_oh = c.onehot_of_bits(word[OPBITS:OPBITS + QBITS], "q", nq)
        ctrl = word[OPBITS + QBITS:OPBITS + QBITS + nq]

        # j < 2^nq: a value is due; j == 2^nq: the record is done
        due = c.node([t for n in range(1, nq + 1)
                      for t in c.t_and([nq_oh[n], one - jt[(1 << n) - 1]])], "due")
        done = c.node([t for n in range(1, nq + 1)
                       for t in c.t_and([nq_oh[n], jt[(1 << n) - 1], one - jt[1 << n]])],
                      "done")
        live = c.and_([r_gate, one - op_halt, due], "live")

        # --- the pair walk -----------------------------------------------------------------
        own = [c.node([t for q in range(nq) for t in c.t_and([live, q_oh[q], j_is(jidx(q, r))])],
                      f"own{r}") for r in range(R)]
        par = [c.node([t for q in range(nq)
                       for t in c.t_and([live, q_oh[q], j_is(jidx(q, r ^ 1 << q))])],
                      f"par{r}") for r in range(R)]
        active = c.node([t for r in range(R)
                         for t in c.t_and([own[r]] + [one - ctrl[b] for b in range(nq)
                                                      if not r >> b & 1])], "active")
        e = sum((j_is(k) for k in range(1, R, 2)), Expr())  # the pair's |1> element
        case = {
            "copy": c.node(c.t_and([live, one - active])
                           + c.t_and([live, active, op_z, one - e]), "c.copy"),
            "add": c.and_([live, active, op_h, one - e], "c.add"),
            "sub": c.and_([live, active, op_h, e], "c.sub"),
            "x": c.and_([live, active, op_x], "c.x"),
            "neg": c.and_([live, active, op_z, e], "c.neg"),
        }

        # --- amplitude memory --------------------------------------------------------------
        # wr[r]: the payload of the *next* token goes to row r (AMP records, and every VAL the
        # circuit predicts)
        wr = [amp_at[r] + own[r] for r in range(R)]
        data_a = {2 + b: lag(pay[b], 0) for b in range(W)}
        data_b = {2 + b: lag(pay[b], 1) for b in range(W)}
        (a_bits,) = thead(c, "ampA", {r: lag(wr[r], 1) for r in range(R)},
                          {r: own[r] for r in range(R)}, [{"data": data_a, "ref": one}])
        (b_bits,) = thead(c, "ampB", {r: lag(wr[r], 2) for r in range(R)},
                          {r: par[r] for r in range(R)}, [{"data": data_b, "ref": one}])
        self.amp_heads = (c.nodes[-2], c.nodes[-1])

        # --- the value: one 16-bit add over exclusive cases ---------------------------------
        nib = W // 4
        sums = []
        for k in range(nib):
            o = from_bits(a_bits[4 * k:4 * k + 4])
            p = from_bits(b_bits[4 * k:4 * k + 4])
            cin = 1 if k == 0 else 0
            sums.append([(o, [case["copy"]]), (o + p, [case["add"]]),
                         (15 - o + p + cin, [case["sub"]]), (p, [case["x"]]),
                         (15 - o + cin, [case["neg"]])])
        g_, p_, d_ = stage1_digits(c, sums, "val", 31)
        value, _ = adder_stage2(c, g_, p_, d_, nib, "val")

        # --- the next token ----------------------------------------------------------------
        out_gate = c.and_([r_gate, one - op_halt, done], "out.gate")
        out_end = c.and_([r_gate, op_halt], "out.end")
        inc = [sum((g_oh[g] for g in range(nprog - 1) if (g + 1) >> b & 1), Expr())
               for b in range(GBITS)]
        nxt = []
        for i in range(NCODE):
            terms = []
            if i < W:
                terms += c.t_and([live, value[i]])
            if i < GBITS:
                terms += c.t_and([out_gate, inc[i]])
            if GATE0 >> i & 1:
                terms += c.t_and([out_gate])
            if END >> i & 1:
                terms += c.t_and([out_end])
            nxt.append(c.node(terms, f"next.b{i}"))
        self.live, self.out_gate, self.out_end = live, out_gate, out_end
        self.next_bits = nxt
        self.output_code(live + out_gate + out_end)

    def output_code(self, own: Expr) -> None:
        """As in qgpu / Qwenputer: chip tokens carry CODE_AMP * (2 bit - 1) of their id; the exit
        clears it and writes KAPPA * (2 bit - 1) of the predicted token where the circuit owns
        the next token, so lm_head (tied embeddings) picks that token."""
        c, one = self, self.one
        f = c.token_feature
        self.code = [f(f"code{i}", lambda t, i=i: 0 if t == TEXT else
                       CODE_AMP * (2 * (t >> i & 1) - 1), full=True) for i in range(NCODE)]
        self.mu = f("mu", lambda t: int(t != TEXT), full=True)
        self.is_text = f("is_text", lambda t: int(t == TEXT), full=True)
        self.tau = f("tau", lambda t: TAU_TEXT if t == TEXT else -1, full=True)
        mu = self.mu
        owner = c.and_([own, mu], "owner")
        for i, e in enumerate(self.code):
            c.ffn([(-1, one, e), (1, one, -1 * e)], f"code.clear{i}", into=e)
        top = max(n.level for n in c.nodes)
        for i, e in enumerate(self.code):
            terms = (c.t_and([owner, self.next_bits[i]], 2 * KAPPA)
                     if self.next_bits[i].terms else []) + c.t_and([owner], -KAPPA)
            c.ffn(terms, f"exit.code{i}", into=e, min_level=top)
        c.ffn([(1, mu, one)], "exit.tau", into=self.tau, min_level=top)
        self.owner = owner
        self.keep = [next(iter(e.terms)) for e in self.code + [self.tau, mu]]


def dim(e: Expr) -> int:
    (d,) = e.terms
    return d


def predicted(c: QPU, m, t: int) -> int | None:
    """The token the circuit predicts after position t (None where it does not own it)."""
    if m[t, dim(c.owner)] != 1:
        return None
    return sum(int(m[t, dim(b)]) << i for i, b in enumerate(c.next_bits) if b.terms)


def amplitudes(c: QPU, carry: dict, nq: int) -> list[int]:
    """The state from copy A's recurrent state after a run (what the host reads back)."""
    from qqpu.machine import to_signed

    index = {id(n): i for i, n in enumerate(c.nodes)}
    state = carry.get((index[id(c.amp_heads[0])], "S", 0), {})
    out = []
    for r in range(1 << nq):
        vec = state.get(r)
        v = 0 if vec is None else sum(int(vec[2 + b] > 0) << b for b in range(W))
        out.append(to_signed(v))
    return out
