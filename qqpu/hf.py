"""Run the QPU on a real Hugging Face Qwen3.5 checkpoint (tiny standalone or merged).

The host feeds its records (register size, program ROM, initial state, the first GATE) in one
prefill; from there the model writes every token by greedy decoding until END. The host then
reads the amplitudes from the DeltaNet recurrent state (copy A's rows), like cudaMemcpy.
"""

import time
from dataclasses import dataclass

import torch

from chipformer.q35.ir import Head
from qqpu.circuit import QPU
from qqpu.machine import END, W, kind, to_signed


@dataclass
class Run:
    tokens: list[int]  # the model's tokens (chip ids), END included
    amps: list[int]
    seconds: float
    prefill: int


class Device:
    """Where copy A of the amplitude memory lives: (layer, value head)."""

    def __init__(self, c: QPU, n_key: int, n_val: int, first_layer: int = 0):
        node = c.amp_heads[0]
        heads = [n for n in c.nodes if isinstance(n, Head) and n.level == node.level]
        self.layer = first_layer + (node.level - 1) // 2
        self.head = heads.index(node) * (n_val // n_key)

    def read(self, cache, nq: int) -> list[int]:
        st = cache.get_recurrent_state(self.layer, state_idx=0)
        st = st[0, self.head].float().cpu()
        out = []
        for r in range(1 << nq):
            v = sum(int(st[r, 2 + b] > 0) << b for b in range(W)) if st[r, 0] != 0 else 0
            out.append(to_signed(v))
        return out


class Runner:
    def __init__(self, model, c: QPU, offset: int = 0, n_chip: int | None = None,
                 first_layer: int = 0, text_mask: bool = True):
        """offset: the model id of chip token 0 (0 for the tiny model)."""
        self.model, self.c, self.offset = model, c, offset
        tc = model.config.get_text_config() if hasattr(model.config, "get_text_config") \
            else model.config
        self.dev = Device(c, tc.linear_num_key_heads, tc.linear_num_value_heads, first_layer)
        self.n_chip = n_chip

    @torch.no_grad()
    def run(self, host: list[int], nq: int, max_new: int = 20000, cache=None,
            check: list[int] | None = None) -> Run:
        """host: chip tokens to prefill; decode greedily until END. With `check` (the reference
        continuation), stop at the first mismatch."""
        m, off = self.model, self.offset
        lo, hi = off, off + (self.n_chip or 0)
        t0 = time.time()
        x = torch.tensor([[t + off for t in host]], device=m.device)
        out = m(input_ids=x, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        toks = []
        logits = out.logits[0, -1]
        for i in range(max_new):
            if self.n_chip:
                nxt = int(logits[lo:hi].float().argmax()) + lo
            else:
                nxt = int(logits.float().argmax())
            t = nxt - off
            toks.append(t)
            if check is not None and (i >= len(check) or check[i] != t):
                break
            if t == END:
                break
            out = m(input_ids=torch.tensor([[nxt]], device=m.device), past_key_values=cache,
                    use_cache=True)
            cache = out.past_key_values
            logits = out.logits[0, -1]
        if m.device.type == "cuda":
            torch.cuda.synchronize()
        self.cache = cache
        return Run(toks, self.dev.read(cache, nq), time.time() - t0, len(host))


def describe(tokens: list[int]) -> str:
    return " ".join(kind(t) for t in tokens[:8])
