"""Run the QPU on a real Hugging Face Qwen3.5 checkpoint (tiny standalone or merged).

The host feeds its records (register size and mode, program ROM, initial state, the first GATE)
in one prefill; from there the model writes every token by greedy decoding until END. The host
then reads the amplitudes from the DeltaNet recurrent state (the final buffer's rows), like
cudaMemcpy.
"""

import time
from dataclasses import dataclass

import torch

from chipformer.q35.ir import Head
from qqpu.circuit import QPU
from qqpu.machine import END, NCOMP, W, to_signed


@dataclass
class Run:
    tokens: list[int]  # the model's tokens (chip ids), END included
    amps: list[tuple]
    seconds: float
    prefill: int


class Device:
    """Where the state lives: for each (buffer, component), the (layer, value head) of its
    own-copy head. emit_layers puts key head j's first value head at j * (n_val // n_key)."""

    def __init__(self, c: QPU, n_key: int, n_val: int, first_layer: int = 0):
        self.where = {}
        for bk, node in c.amp_heads.items():
            heads = [n for n in c.nodes if isinstance(n, Head) and n.level == node.level]
            self.where[bk] = (first_layer + (node.level - 1) // 2,
                              heads.index(node) * (n_val // n_key))

    def read(self, cache, nq: int, cplx: bool, buffer: int) -> list[tuple]:
        comps = []
        for k in range(NCOMP if cplx else 1):
            layer, head = self.where[buffer, k]
            st = cache.get_recurrent_state(layer, state_idx=0)[0, head].float().cpu()
            col = []
            for r in range(1 << nq):
                v = sum(int(st[r, 2 + b] > 0) << b for b in range(W)) if st[r, 0] != 0 else 0
                col.append(to_signed(v))
            comps.append(col)
        while len(comps) < 4:
            comps.append([0] * (1 << nq))
        return [tuple(comps[k][r] for k in range(4)) for r in range(1 << nq)]


class Runner:
    def __init__(self, model, c: QPU, offset: int = 0, first_layer: int = 0):
        """offset: the model id of chip token 0 (0 for the tiny model)."""
        self.model, self.c, self.offset = model, c, offset
        tc = model.config.get_text_config() if hasattr(model.config, "get_text_config") \
            else model.config
        self.dev = Device(c, tc.linear_num_key_heads, tc.linear_num_value_heads, first_layer)

    @torch.no_grad()
    def run(self, host: list[int], nq: int, cplx: bool, buffer: int, max_new: int = 40000,
            cache=None, check: list[int] | None = None) -> Run:
        """host: chip tokens to prefill; decode greedily (full vocabulary) until END. With
        `check` (the reference continuation), stop at the first mismatch."""
        m, off = self.model, self.offset
        t0 = time.time()
        x = torch.tensor([[t + off for t in host]], device=m.device)
        out = m(input_ids=x, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        toks = []
        logits = out.logits[0, -1]
        for i in range(max_new):
            t = int(logits.float().argmax()) - off
            toks.append(t)
            if check is not None and (i >= len(check) or check[i] != t):
                break
            if t == END:
                break
            out = m(input_ids=torch.tensor([[t + off]], device=m.device), past_key_values=cache,
                    use_cache=True)
            cache = out.past_key_values
            logits = out.logits[0, -1]
        if m.device.type == "cuda":
            torch.cuda.synchronize()
        self.cache = cache
        return Run(toks, self.dev.read(cache, nq, cplx, buffer), time.time() - t0, len(host))
