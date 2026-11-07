# qwenQPU — working notes

## Status log
- **M1 reference machine** (`qqpu/machine.py`, `qqpu/circuits.py`): exact integer statevector,
  H / X / Z with control masks (CNOT, CCX, CZ, CCZ). Matches Qiskit on 12 textbook circuits + 40
  random ones (`tests/test_machine.py`).
- **M2 compiled circuit** (`qqpu/circuit.py`): 11 layers, 6 DeltaNet heads, ~3.1k MLP units.
  Exact in Qwenputer's IR evaluator (bf16 checks everywhere) on library, random circuits, random
  signed initial states, back-to-back programs (`tests/test_circuit.py`, 98 tests).
- **M3 tiny HF checkpoint** (stock `Qwen3_5ForCausalLM`, bf16, CUDA): all library circuits
  token-exact, state read from the DeltaNet cache equals reference. ~50 tok/s (torch fallback).
- **M4 merged Qwen3.5-2B** (`qqpu/merge.py`, `build/merged-2b`, 6.6 GB): hidden 2048 -> 2432,
  11 QPU layers + 24 Qwen. Text KL 5.35e-4, top-1 agree 50/50. All 12 circuits exact both
  teacher-forced inside text and by free greedy decoding over the full vocab; state == Qiskit.
  ~17 tok/s (torch fallback kernels, python loop).

## Design (one line each)
- Amplitudes: 16-bit two's complement ints, scale 1/sqrt(2)^k (k = #H). Max ~29 H gates.
- A gate record = GATE(g) + 2^nq VALs (pairs of the target qubit, |0> element first) + next GATE.
- Two amplitude heads: A written at the VAL token, B one token later (conv tap lag 2/1) ->
  in-place pair updates read pre-gate values (own from A, partner from B).
- Five exclusive cases (copy, own+par, par-own, par, -own) -> one 16-bit CLA adder.

## Next
- Chat bridge: Qwen writes QASM in a tool call; QPU runs inside the same context; eval vs base.
- Born-rule sampling through the LM sampler (measurement).
- Speed: fla kernels, skip inactive pairs for controlled gates.
- Complex amplitudes (S, T, QFT) via Z[omega].
