# qwenQPU

**Qwen3.5-4B with a quantum circuit simulator inside its weights.** 16 compiled layers sit in front of Qwen's own 32, and Qwen's Gated DeltaNet state holds the qubits.

Qwen writes a circuit as OpenQASM in a tool call, and the host compiles each gate into a 16-bit instruction word. From there the model writes the whole simulation itself, token by token: every amplitude after every gate, and every measurement shot. The run happens in the same KV cache as the conversation. Afterwards the host reads the final statevector back out of the DeltaNet state.

Nothing is trained. The QPU's weights are compiled, in the style of [Qwenputer](https://github.com/divagr18/qwenputer) (a RISC-V CPU in Qwen's weights) and [qwenGPU](https://github.com/vixhal-baraiya/qwenGPU) (a SIMT GPU).

```
user: Make a Bell pair and measure it 32 times.
qwen: <tool_call> qpu_run
        OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];
      </tool_call>
      ── 16 QPU layers write 907 tokens: GATE, VAL, VAL, ... MEAS, IDX, VAL, VAL ...
      |00⟩: 1/√2   |11⟩: 1/√2      shots: 00 ×14, 11 ×18
```

## What the QPU does

| | |
|---|---|
| Qubits | up to 5 with real amplitudes, up to 4 with complex amplitudes |
| Gates | H, X, Y, Z, S, S†, T, T†, with any controls on all but H (CNOT, CZ, Toffoli, CCZ, controlled phases, ...) |
| Arithmetic | exact: every amplitude is an integer in ℤ[ω] / √2ᵏ (ω = e^{iπ/4}), as 16-bit two's-complement tokens. No floats, no rounding |
| Measurement | Born-rule rejection sampling. The model draws an index and two uniform numbers and keeps the index if both fall below \|aᵢ\|. Its coins are logits that tie exactly, broken at random by the sampler |
| Input | OpenQASM 2.0: `h x y z s sdg t tdg`, `cx cy cz cs ccx ccz`, `p`/`u1`/`cp` at multiples of π/4, `swap`, applied to whole registers when no index is given |

## Verified

Each of the 16 textbook circuits below is run on the merged 4B checkpoint (`scripts/merged_check.py`, `scripts/demo_suite.py`) and checked three ways:

- **Teacher-forced:** at every position the QPU owns, the model's argmax over the full vocabulary is the correct next token.
- **Free generation:** greedy decoding writes the whole run with no wrong token.
- **Final state:** the statevector read from the DeltaNet cache equals the reference simulator's and Qiskit's.

| Circuit | Qubits | Gates | Tokens the model wrote | Token-exact | Equals Qiskit | 32 shots |
|---|---|---|---|---|---|---|
| Bell pair | 2 | 2 | 907 | ✓ | ✓ | 00: 14, 11: 18 |
| GHZ | 3 | 3 | 1722 | ✓ | ✓ | 000: 13, 111: 19 |
| GHZ | 5 | 5 | 5352 | ✓ | ✓ | 00000: 18, 11111: 14 |
| Deutsch–Jozsa, constant | 4 | 9 | 3807 | ✓ | ✓ | 0000: 17, 1000: 15 |
| Deutsch–Jozsa, balanced | 4 | 10 | 4580 | ✓ | ✓ | 0101: 20, 1101: 12 |
| Bernstein–Vazirani, s = 1011 | 5 | 13 | 7566 | ✓ | ✓ | 01101: 20, 11101: 12 |
| Grover, 2 qubits | 2 | 12 | 450 | ✓ | ✓ | 11: 32 |
| Grover, 3 qubits, 2 iterations | 3 | 35 | 1836 | ✓ | ✓ | 101: 32 |
| Superdense coding, 10 | 2 | 5 | 496 | ✓ | ✓ | 01: 32 |
| Superdense coding, 11 | 2 | 6 | 429 | ✓ | ✓ | 11: 32 |
| Teleportation of \|−⟩ | 3 | 8 | 1680 | ✓ | ✓ | spread over 8 outcomes |
| SWAP | 2 | 4 | 428 | ✓ | ✓ | 10: 32 |
| QFT of \|101⟩ | 3 | 11 | 364 | ✓ | ✓ | (complex, statevector only) |
| Phase estimation of T | 4 | 16 | 1041 | ✓ | ✓ | (complex, statevector only) |
| Phase estimation of S | 4 | 15 | 976 | ✓ | ✓ | (complex, statevector only) |
| Y, S, T, controlled-Y | 2 | 6 | 103 | ✓ | ✓ | (complex, statevector only) |

Bitstrings are in Qiskit order (q0 rightmost). Full results: [`docs/demo-results.json`](docs/demo-results.json). Grover-3 ends at P(|101⟩) = 121/128 exactly, and 32 of 32 shots land on it.

**Qwen's text is unaffected.** The mean KL between the base and merged next-token distributions on plain text is 4.8e-4, and the top-1 token agrees in 50 of 50 cases. Qwen's own layers are muted on QPU tokens, so its memory passes through a run unchanged.

**What doesn't work yet: Qwen writing every circuit.** On our chat eval (`scripts/eval_chat.py`, 22 questions) the merged model answers all 16 gate-by-gate questions, and all 19 QPU runs are exact. It answers 0 of 6 questions that name an algorithm ("run Grover on…") and leave Qwen to compose the gates. When it fails, the QASM Qwen wrote is wrong; the QPU runs whatever it is given exactly.

## How it works

1. **The machine** (`qqpu/machine.py`) is a statevector simulator designed so every value is a bit or a small integer. A gate is a record: one `GATE` token, then one `VAL` token per amplitude in pair-walk order. Each new amplitude at row i is computed from the old values at rows i and i⊕2^q: their sum or difference for H, a swap for X and Y, a power of ω for the phase gates.
2. **The circuit** (`qqpu/circuit.py`) builds that machine in Qwenputer's IR:
   - Its building blocks are MLP units computing `coef·a·relu(b)`, used as exact step functions, ANDs and a 16-bit carry-lookahead adder.
   - Its memory is Gated DeltaNet heads. One-hot keys write rows, one-hot queries read them, and the decay setting keeps, halves or clears a head (a halving head is a counter).
   - The state is double-buffered by gate parity, which takes 16 heads in one layer.
   - In total: 16 layers, 22 heads, about 3.8k MLP units.
3. **The compiler** (Qwenputer's `chipformer.q35`) schedules every node into the earliest layer where its inputs exist (hence 16 layers) and writes the matrices. Ballast dimensions make every RMSNorm divide by a known power of two, and an evaluator checks that every intermediate value is exact in bf16 before any weight is written.
4. **The merge** (`qqpu/merge.py`, following qwenGPU's recipe):
   - Qwen's hidden size widens from 2560 to 3200, and the 16 layers go in front of its 32.
   - The QPU tokens are added to the vocabulary.
   - The final norm is set so the LM head reads the circuit's output bits as the next token.
5. **The bridge** (`qqpu/bridge.py`) gives Qwen a `qpu_run` tool. It parses the QASM into ROM words, appends them to the same context, and lets the model generate the run.

The full write-up with figures is in [`docs/qwenqpu.html`](docs/qwenqpu.html).

## Run it

Built and tested on one A100 80GB. The merged checkpoint is 16 GB in bf16.

```bash
git clone https://github.com/Lurker17/qwenqpu && cd qwenqpu
uv sync                                              # pulls Qwenputer and qwenGPU at pinned commits
uv run pytest -q                                     # 170 tests: machine, circuit (IR evaluator), bridge

huggingface-cli download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
uv run python -m qqpu.merge models/Qwen3.5-4B build/merged-4b      # compile and merge the QPU
uv run python scripts/merged_check.py models/Qwen3.5-4B build/merged-4b   # text KL + all 16 circuits
```

Ask it something:

```bash
uv run python scripts/demo.py build/merged-4b "Put 3 qubits in a GHZ state and measure 32 shots."
uv run python scripts/demo_suite.py build/merged-4b --shots 32      # the table above
uv run python scripts/eval_chat.py build/merged-4b qpu              # the chat eval (modes: qpu, host, none)
```

Or from Python:

```python
from qqpu.bridge import Chat

chat = Chat("build/merged-4b", mode="qpu")   # "host" runs the same tool on a host simulator instead
turn = chat.ask("Run a Bell pair and measure it 32 times.")
print(turn.text)
for run in turn.runs:
    print(run["tokens"], "tokens written by the model, exact:", run.get("exact"))
```

## Layout

```
qqpu/machine.py     reference machine: ISA, token layout, exact Z[ω] arithmetic, trace
qqpu/circuit.py     the QPU as a Qwenputer circuit graph (16 layers)
qqpu/circuits.py    library circuits, random circuits, Qiskit conversion
qqpu/merge.py       compile and merge into a Qwen3.5 checkpoint
qqpu/hf.py          run on the HF model; read amplitudes from the DeltaNet cache
qqpu/bridge.py      QASM parser, qpu_run tool, chat loop, exact result formatting
scripts/            merged_check, demo, demo_suite, eval_chat
tests/              machine, circuit and bridge tests
```

## Credits

The compiler, the gadgets and the merge recipe are [Qwenputer](https://github.com/divagr18/qwenputer) and [qwenGPU](https://github.com/vixhal-baraiya/qwenGPU), used unchanged at pinned commits. The base model is [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
