"""The QPU as Qwen3.5 layers (Gated DeltaNet heads + SwiGLU MLPs), on Qwenputer's q35 compiler.

Every weight is computed; nothing is trained. Per token, in one forward pass:

1. Record, register-size and counter heads: which record we are in (GATE g / AMP i / PROG g /
   CONF), the mode (real or complex) and j, the number of VALs since the record started.
2. ROM head: gate g's instruction word (op, target q, control mask), written by PROG records.
3. The pair walk: amplitude slot a and component c of the value due next (a = j, c = 0 in real
   mode; a = j >> 2, c = j & 3 in complex mode), own(a, q) and its partner own ^ 2^q as one-hot
   queries, whether the controls hold, and the value's kind: w^m * own, w^m * partner, or
   own +- partner (H).
4. The state: per component, two buffers (gate g reads buffer g mod 2 and writes the other) and
   two read copies of each (one queried at own, one at the partner): 16 DeltaNet heads, one row
   per basis state. A VAL token's payload is written at the VAL token itself, into the buffer
   and component it belongs to (key: the row chosen one token earlier; write gate: buffer and
   component).
5. Operands: component c of w^m x is +-x[(c - m) mod 4], so each operand is a 4-way select
   with an optional complement; one 16-bit carry-lookahead add finishes the value (a negation
   is complement + 1 through the carry in).
6. The output code makes the model's lm_head predict VAL(value), the next GATE, or END.
"""

from chipformer.dsl.graph import Expr
from chipformer.q35.ir import Chan, Target
from chipformer.q35.prims import Circ, from_bits

from qgpu.circuit.gadgets import adder_stage2, counter, stage1_digits, thead, thermo
from qqpu.machine import (
    AMP0,
    CONF0,
    END,
    GATE0,
    GBITS,
    KINDS,
    NCOMP,
    NQ,
    NQC,
    OPBITS,
    OPS,
    PROG,
    QBITS,
    TEXT,
    W,
    Gate,
    jidx,
    kind,
    kind_of,
    rot_src,
)

NCODE = 17  # every chip token id is < 2^17
CODE_AMP = 128
KAPPA = 2.0**-20
TAU_TEXT = 2.0**-40
JMAX = max(1 << NQ, NCOMP << NQC)  # VALs per gate record, at most


class QPU(Circ):
    def __init__(self, target: Target | None = None) -> None:
        super().__init__(target or Target())
        c, one = self, self.one
        R = 1 << NQ
        RC = 1 << NQC

        def feat(name, fn):
            return c.token_feature(name, fn, full=True)

        def lag(e, k):
            return Chan(e, tuple(1.0 if i == k else 0.0 for i in range(4)))

        is_val = feat("is_val", lambda t: int(kind(t) == "val"))
        is_gate = feat("is_gate", lambda t: int(kind(t) == "gate"))
        is_amp = feat("is_amp", lambda t: int(kind(t) == "amp"))
        is_conf = feat("is_conf", lambda t: int(kind(t) == "conf"))
        pay = [feat(f"pay{b}", lambda t, b=b: int(kind(t) == "val" and t >> b & 1))
               for b in range(W)]
        gb = [feat(f"gb{b}", lambda t, b=b: int(kind(t) == "gate" and (t - GATE0) >> b & 1))
              for b in range(GBITS)]
        ab = [feat(f"ab{b}", lambda t, b=b: int(kind(t) == "amp" and (t - AMP0) >> b & 1))
              for b in range(NQ)]
        cb = [feat(f"cb{b}", lambda t, b=b: int(kind(t) == "conf" and (t - CONF0) >> b & 1))
              for b in range(4)]
        start = one - is_val  # every other token starts a record (text included)

        # --- record, mode, counter ---------------------------------------------------------
        (rec,) = thead(c, "rec", {0: start}, {0: one},
                       [{"data": {2: is_gate, 3: is_amp,
                                  **{4 + b: gb[b] for b in range(GBITS)},
                                  **{4 + GBITS + b: ab[b] for b in range(NQ)}}, "ref": one}])
        r_gate, r_amp = rec[0], rec[1]
        r_g, r_a = rec[2:2 + GBITS], rec[2 + GBITS:2 + GBITS + NQ]
        (conf,) = thead(c, "conf", {0: is_conf}, {0: one},
                        [{"data": {2 + b: cb[b] for b in range(4)}, "ref": one}])
        nq_oh = c.onehot_of_bits(conf[:3], "nq", NQ + 1)  # nq_oh[n]: the register has n qubits
        cplx = conf[3]
        real = one - cplx
        jt = thermo(c, counter(c, "j", start, is_val), start, JMAX + 1, "j")

        def j_is(k: int) -> Expr:
            return (one if k == 0 else jt[k - 1]) - jt[k]

        def j_in4(k: int) -> Expr:  # 4k <= j < 4k + 4
            return (one if k == 0 else jt[4 * k - 1]) - jt[4 * k + 3]

        # --- fetch -------------------------------------------------------------------------
        g_oh = c.onehot_of_bits(r_g, "g")
        nprog = 1 << GBITS
        rom_keys = {g: lag(feat(f"prog{g}", lambda t, g=g: int(t == PROG(g))), 1)
                    for g in range(nprog)}
        rom_q = {g: c.and_([r_gate, g_oh[g]], f"romq{g}") for g in range(nprog)}
        wbits = OPBITS + QBITS + NQ
        (word,) = thead(c, "rom", rom_keys, rom_q,
                        [{"data": {2 + b: lag(pay[b], 0) for b in range(wbits)}, "ref": one}])
        op_oh = c.onehot_of_bits(word[:OPBITS], "op", len(OPS))
        op = dict(zip(OPS, op_oh))
        q_oh = c.onehot_of_bits(word[OPBITS:OPBITS + QBITS], "q", NQ)
        ctrl = word[OPBITS + QBITS:OPBITS + QBITS + NQ]

        # values per record: 2^nq (real) or 4 * 2^nq (complex); j < that: a value is due
        def n_vals(n, cx):
            return (NCOMP if cx else 1) << n

        modes = [(real, False)] + [(cplx, True)]
        due = c.node([t for n in range(1, NQ + 1) for lit, cx in modes
                      if not (cx and n > NQC)
                      for t in c.t_and([nq_oh[n], lit, one - jt[n_vals(n, cx) - 1]])], "due")
        done = c.node([t for n in range(1, NQ + 1) for lit, cx in modes
                       if not (cx and n > NQC)
                       for t in c.t_and([nq_oh[n], lit, jt[n_vals(n, cx) - 1],
                                         one - jt[n_vals(n, cx)]])], "done")
        live = c.and_([r_gate, one - op["halt"], due], "live")

        # --- the pair walk -----------------------------------------------------------------
        def slot_terms(r_of_q, lits):
            """Units: amplitude slot == jidx(q, r_of_q(q)) for the decoded target q."""
            terms = []
            for q in range(NQ):
                r = r_of_q(q)
                if r >= R:
                    continue
                k = jidx(q, r)
                terms += c.t_and([*lits, q_oh[q], real, j_is(k)])
                if r < RC and q < NQC:
                    terms += c.t_and([*lits, q_oh[q], cplx, j_in4(k)])
            return terms

        own = [c.node(slot_terms(lambda q, r=r: r, [live]), f"own{r}") for r in range(R)]
        par = [c.node(slot_terms(lambda q, r=r: r ^ 1 << q, [live]), f"par{r}")
               for r in range(R)]
        active = c.node([t for r in range(R)
                         for t in c.t_and([own[r]] + [one - ctrl[b] for b in range(NQ)
                                                      if not r >> b & 1])], "active")
        e_real = sum((j_is(k) for k in range(1, R, 2)), Expr())
        e_cplx = sum((j_in4(k) for k in range(1, RC, 2)), Expr())
        e = c.node(c.t_and([real, e_real]) + c.t_and([cplx, e_cplx]), "e")
        # component of the value due (cc) and of a VAL token's own payload (vcomp)
        cc = [c.node((c.t_and([real]) if k == 0 else [])
                     + c.t_and([cplx, sum((j_is(4 * a + k) for a in range(RC)), Expr())]),
                     f"cc{k}") for k in range(NCOMP)]
        vcomp = [c.node((c.t_and([is_val, real]) if k == 0 else [])
                        + c.t_and([is_val, cplx,
                                   sum((j_is(4 * a + k + 1) for a in range(RC)), Expr())]),
                        f"vcomp{k}") for k in range(NCOMP)]

        # the value's kind, from (op, e, active)
        kterms: dict[str, list] = {k: [] for k in KINDS}
        kterms["own0"] += c.t_and([live, one - active])
        for o in OPS[1:]:
            for ev in (0, 1):
                k = kind_of(Gate(o), ev, True)
                kterms[k] += c.t_and([live, active, op[o], e if ev else one - e])
        kinds = {k: c.node(t, f"kind.{k}") for k, t in kterms.items() if t}

        def operand_sel(which):
            """(pos[idx], neg[idx]): component idx of the source enters as is / complemented."""
            pos = [[] for _ in range(NCOMP)]
            neg = [[] for _ in range(NCOMP)]
            for k, node in kinds.items():
                m = KINDS[k][which]
                if m is None:
                    continue
                for cmp in range(NCOMP):
                    idx, sign = rot_src(cmp, m)
                    (pos if sign > 0 else neg)[idx] += c.t_and([node, cc[cmp]])
            return ([c.node(t, f"sel{which}.p{i}") for i, t in enumerate(pos)],
                    [c.node(t, f"sel{which}.n{i}") for i, t in enumerate(neg)])

        own_pos, own_neg = operand_sel(0)
        par_pos, par_neg = operand_sel(1)

        # --- amplitude memory --------------------------------------------------------------
        aoh = c.onehot_of_bits(r_a, "a")
        amp_due = c.node(c.t_and([r_amp, real, one - jt[0]]) + c.t_and([r_amp, cplx, one - jt[3]]),
                         "amp_due")
        # wr[r]: the payload of the next token goes to row r (AMP records and every VAL the
        # circuit predicts)
        wr = [c.and_([amp_due, aoh[r]], f"ampwr{r}") + own[r] for r in range(R)]
        rd1 = r_g[0]  # gate g reads buffer g mod 2 ...
        wb = {0: c.node(c.t_and([r_gate, rd1]) + c.t_and([r_amp]), "wb0"),
              1: c.and_([r_gate, one - rd1], "wb1")}  # ... and writes the other (AMP: 0)
        keys = {r: lag(wr[r], 1) for r in range(R)}
        data = {2 + b: lag(pay[b], 0) for b in range(W)}
        q_own = {b: {r: c.and_([own[r], rd1 if b else one - rd1], f"qown{b}.{r}")
                     for r in range(R)} for b in (0, 1)}
        q_par = {b: {r: c.and_([par[r], rd1 if b else one - rd1], f"qpar{b}.{r}")
                     for r in range(R)} for b in (0, 1)}
        own_bits = [[Expr() for _ in range(W)] for _ in range(NCOMP)]
        par_bits = [[Expr() for _ in range(W)] for _ in range(NCOMP)]
        self.amp_heads = {}
        for b in (0, 1):
            for k in range(NCOMP):
                beta = c.and_([is_val, vcomp[k], wb[b]], f"beta{b}.{k}")
                rows = R if k == 0 else RC
                kk = {r: keys[r] for r in range(rows)}
                (ob,) = thead(c, f"amp{b}.{k}.own", kk, {r: q_own[b][r] for r in range(rows)},
                              [{"data": data, "ref": one, "beta": beta}])
                self.amp_heads[b, k] = c.nodes[-1]
                (pb,) = thead(c, f"amp{b}.{k}.par", kk, {r: q_par[b][r] for r in range(rows)},
                              [{"data": data, "ref": one, "beta": beta}])
                for i in range(W):
                    own_bits[k][i] = own_bits[k][i] + ob[i]
                    par_bits[k][i] = par_bits[k][i] + pb[i]

        # --- the value ---------------------------------------------------------------------
        def operand(bits, pos, neg, name):
            return [c.node([t for i in range(NCOMP)
                            for t in c.t_and([pos[i], bits[i][b]])
                            + c.t_and([neg[i], one - bits[i][b]])], f"{name}.b{b}")
                    for b in range(W)]

        xo = operand(own_bits, own_pos, own_neg, "xo")
        xp = operand(par_bits, par_pos, par_neg, "xp")
        cin = sum(own_neg, Expr()) + sum(par_neg, Expr())
        nib = W // 4
        sums = [[(from_bits(xo[4 * k:4 * k + 4]) + from_bits(xp[4 * k:4 * k + 4])
                  + (cin if k == 0 else 0), [])] for k in range(nib)]
        g_, p_, d_ = stage1_digits(c, sums, "val", 32)
        value, _ = adder_stage2(c, g_, p_, d_, nib, "val")

        # --- the next token ----------------------------------------------------------------
        out_gate = c.and_([r_gate, one - op["halt"], done], "out.gate")
        out_end = c.and_([r_gate, op["halt"]], "out.end")
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


def amplitudes(c: QPU, carry: dict, nq: int, cplx: bool, buffer: int) -> list[tuple]:
    """The state from the own-copy heads' recurrent state after a run (the host's read)."""
    from qqpu.machine import to_signed

    index = {id(n): i for i, n in enumerate(c.nodes)}
    out = []
    for r in range(1 << nq):
        comps = []
        for k in range(NCOMP if cplx else 1):
            state = carry.get((index[id(c.amp_heads[buffer, k])], "S", 0), {})
            vec = state.get(r)
            v = 0 if vec is None else sum(int(vec[2 + b] > 0) << b for b in range(W))
            comps.append(to_signed(v))
        out.append(tuple(comps) + (0,) * (4 - len(comps)))
    return out
