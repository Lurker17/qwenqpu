"""Check a merged checkpoint (qqpu.merge): Qwen's text is unchanged, and the model runs circuits.

1. Text: next-token distributions on plain prompts, merged vs base (KL, top-1 agreement).
2. Circuits inside text, teacher-forced: text, the whole QPU trace, text. At every position the
   circuit owns, the merged model's argmax over the FULL vocabulary must be the next QPU token.
3. Generation: from the host's records, greedy decoding over the full vocabulary must write the
   whole run (every VAL, GATE and the END), and the state read from the DeltaNet cache must equal
   the reference statevector (and Qiskit's).

Run: uv run python scripts/merged_check.py BASE MERGED [circuit names]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from qqpu import circuits
from qqpu.circuit import QPU
from qqpu.hf import Runner
from qqpu.machine import host_tokens, trace

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "A qubit in superposition is described by",
    "Photosynthesis converts light energy into",
    "In Grover's algorithm, the number of iterations needed is roughly",
]


def exact_cuda():
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def logits(model, ids):
    return model(input_ids=torch.tensor([ids], device=model.device)).logits[0].float().cpu()


def main() -> None:
    base_dir, merged_dir = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or list(circuits.library())
    exact_cuda()
    meta = json.loads((merged_dir / "qqpu.json").read_text())
    v0 = meta["chip_token_offset"]
    tok = AutoTokenizer.from_pretrained(base_dir)

    base = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.bfloat16).cuda()
    base_logits = [logits(base, tok.encode(p)) for p in PROMPTS]
    del base
    torch.cuda.empty_cache()
    model = AutoModelForImageTextToText.from_pretrained(merged_dir, dtype=torch.bfloat16).cuda()
    model.eval()
    kls, agree, n = [], 0, 0
    for p, bl in zip(PROMPTS, base_logits):
        ml = logits(model, tok.encode(p))[:, :bl.shape[1]]
        lp, lq = torch.log_softmax(bl, -1), torch.log_softmax(ml, -1)
        kls.append(float((lp.exp() * (lp - lq)).sum(-1).mean()))
        agree += int((bl.argmax(-1) == ml.argmax(-1)).sum())
        n += bl.shape[0]
    print(f"text: mean KL(base || merged) = {np.mean(kls):.2e}, top-1 agreement {agree}/{n}",
          flush=True)

    tc = model.config.get_text_config()
    c = QPU()
    assert c.n_layers() == meta["chip_layers"]
    runner = Runner(model, c, offset=v0)
    all_ok = True
    for name in names:
        nq, gates = circuits.library()[name]
        tr = trace(gates, nq)
        before = tok.encode(f"Running {name} on the QPU inside the model:")
        after = tok.encode(" done.")
        seq = before + [v0 + t for t in tr.tokens] + after
        owned = [len(before) + t for t in tr.owned]
        top = logits(model, seq).argmax(-1).tolist()
        bad = [t for t in owned if top[t] != seq[t + 1]]
        host = host_tokens(gates, nq)
        want = tr.tokens[len(host):]
        run = runner.run(tok.encode("Run:") and [t for t in host], nq, check=want)
        ok_gen = run.tokens == want
        ok_state = run.amps == tr.amps
        ref = circuits.qiskit_statevector(nq, gates).real
        sv = np.array(run.amps, float) / 2 ** (tr.k / 2)
        ok_q = np.allclose(sv, ref, atol=1e-9)
        ok = not bad and ok_gen and ok_state and ok_q
        all_ok &= ok
        print(f"{name:14s} {len(tr.tokens):4d} QPU tokens | teacher-forced {len(owned) - len(bad)}"
              f"/{len(owned)} | generated {len(run.tokens)}/{len(want)} "
              f"{'exact' if ok_gen else 'DIFFERENT'} in {run.seconds:.1f}s "
              f"({len(run.tokens) / run.seconds:.0f} tok/s) | state==ref {ok_state} "
              f"| ==qiskit {ok_q}", flush=True)
    print("ALL OK" if all_ok else "FAILURES")


if __name__ == "__main__":
    main()
