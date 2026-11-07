"""Merge the QPU circuit into a stock Qwen3.5 checkpoint.

Follows qgpu.merge / Qwenputer's chipformer.q35.merge (v2 layout): the circuit's layers go in
front of Qwen's, the residual stream is widened (Qwen's matrices zero-padded, so its text
computation is unchanged), the QPU's tokens are appended to the vocabulary, and Qwen's layers are
muted on QPU tokens. Every Qwen text token carries the circuit features of machine.TEXT.

Run: uv run python -m qqpu.merge SRC DST [--no-mute]
"""

import json
import re
import shutil
from pathlib import Path

import torch

from qqpu.circuit import QPU
from qqpu.machine import CHIP_VOCAB, TEXT, token_name

TEXT_PREFIX = "model.language_model."


def extend_tokenizer(src: Path, dst: Path, first_id: int) -> None:
    from tokenizers import AddedToken, Tokenizer

    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    n = tok.get_vocab_size(with_added_tokens=True)
    names = [token_name(t) for t in range(CHIP_VOCAB)]
    assert len(set(names)) == len(names)
    pad = [AddedToken(f"<|reserved_{i}|>", special=True) for i in range(first_id - n)]
    chip = [AddedToken(s, special=True) for s in names]
    tok.add_special_tokens(pad + chip)
    assert tok.token_to_id(names[0]) == first_id
    dst.mkdir(parents=True, exist_ok=True)
    tok.save(str(dst / "tokenizer.json"))
    for f in ("chat_template.jinja", "preprocessor_config.json",
              "video_preprocessor_config.json", "generation_config.json", "LICENSE"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    tc = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    dec = tc.setdefault("added_tokens_decoder", {})
    for i, t in enumerate(pad + chip):
        dec[str(n + i)] = {"content": t.content, "lstrip": False, "normalized": False,
                           "rstrip": False, "single_word": False, "special": True}
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False),
                                               encoding="utf-8")


def build_merged(src, dst, mute_qwen: bool = True):
    from chipformer.q35 import merge as qm
    from chipformer.q35 import tiny
    from chipformer.q35.emit import Layout, allocate, check_bf16, embedding_rows, emit_layers
    from chipformer.q35.ir import Target
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    cfg, sd = qm.load(src)
    tc = cfg["text_config"]
    d0, v0, inter = tc["hidden_size"], tc["vocab_size"], tc["intermediate_size"]
    c = QPU(Target(tc["linear_num_key_heads"], tc["linear_num_value_heads"], inter))
    keep = set(c.keep)
    alloc = allocate(c, None, keep, inter)
    hidden, m = qm.pick_hidden(d0, 1 + alloc.n_slots)
    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots, over=alloc.over)
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    n_chip = c.n_layers()
    print(f"circuit: {n_chip} layers, {alloc.n_slots} slots; hidden {d0} -> {hidden}", flush=True)

    cfg, sd = qm.widen(cfg, sd, hidden)
    tc = cfg["text_config"]
    n0 = tc["num_hidden_layers"]
    shift = re.compile(r"^(model\.language_model\.layers\.)(\d+)\.")
    sd = {shift.sub(lambda mt: f"{mt[1]}{int(mt[2]) + n_chip}.", k): v for k, v in sd.items()}
    tc["layer_types"] = ["linear_attention"] * n_chip + tc["layer_types"]
    tc["num_hidden_layers"] = n_chip + n0
    vocab = -(-(v0 + CHIP_VOCAB) // 64) * 64
    tc["vocab_size"] = vocab
    mu, tau = c.keep[-1], c.keep[-2]
    if mute_qwen:
        from chipformer.q35.mute import fit_anti_sink, mute

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mute(sd, tc, n0, lay.dim(mu), hidden**0.5, sinks=fit_anti_sink(src, dev), first=n_chip)
    layers = emit_layers(c, lay, tc, 0, TEXT_PREFIX, alloc, keep, chip=c.mu)
    sd.update({k: v.to(torch.bfloat16) for k, v in layers.items()})
    del layers

    full = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    full[:v0, :d0] = sd[TEXT_PREFIX + "embed_tokens.weight"][:, :d0]
    text_row = embedding_rows(c, lay, [TEXT], hidden)[0].to(torch.bfloat16)
    full[:v0, d0:] = text_row[d0:]
    full[v0:v0 + CHIP_VOCAB] = embedding_rows(c, lay, list(range(CHIP_VOCAB)),
                                              hidden).to(torch.bfloat16)
    sd[TEXT_PREFIX + "embed_tokens.weight"] = full
    if "lm_head.weight" in sd and not cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings")):
        raise SystemExit("untied lm_head: not supported")
    bias = "model.visual.merger.linear_fc2.bias"
    if bias in sd:
        sd[bias] = sd[bias].clone()
        sd[bias][d0:] = text_row[d0:].to(sd[bias].dtype)

    norm = sd[TEXT_PREFIX + "norm.weight"].float()
    norm[d0:] = -1.0
    for e in c.code:
        norm[lay.dim(next(iter(e.terms)))] = tiny.code_gain(lay)
    norm[lay.dim(tau)] = qm.TAU_GAIN
    check_bf16("final norm", norm[d0:])
    sd[TEXT_PREFIX + "norm.weight"] = norm.to(torch.bfloat16)

    dst.mkdir(parents=True, exist_ok=True)
    shards, shard, size = [], {}, 0
    for k in sorted(sd):
        shard[k] = sd[k].contiguous()
        size += sd[k].numel() * 2
        if size > 2**31:
            shards.append(shard)
            shard, size = {}, 0
    shards.append(shard)
    index = {"metadata": {"total_size": sum(t.numel() * 2 for t in sd.values())},
             "weight_map": {}}
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, str(dst / name), metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in sh})
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    extend_tokenizer(src, dst, v0)
    meta = {"base": src.name, "chip_token_offset": v0, "chip_vocab": CHIP_VOCAB,
            "chip_layers": n_chip, "qwen_layers": n0, "hidden": hidden,
            "chip_dims_offset": d0, "ballast_dims": m, "slots": alloc.n_slots,
            "muted": mute_qwen, "nq": c.nq_max}
    (dst / "qqpu.json").write_text(json.dumps(meta, indent=2))
    return c, lay


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Merge the QPU circuit into a Qwen3.5 checkpoint")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--no-mute", action="store_true")
    a = ap.parse_args()
    build_merged(a.src, a.dst, mute_qwen=not a.no_mute)


if __name__ == "__main__":
    main()
