# qwenQPU — working notes

A quantum statevector simulator compiled into Qwen3.5 weights (Qwenputer compiler, qwenGPU
merge recipe). Write-up: `docs/qwenqpu.html` (published as an artifact).

## Status (2026-11-07)
- **Machine** (`qqpu/machine.py`): amplitudes in Z[w]/sqrt(2)^k as 16-bit ints. Real mode (1
  component, <= 5 qubits) and complex mode (4 components, <= 4 qubits). Gates H X Y Z S Sdg T
  Tdg, controls on all but H. MEAS: Born-rule rejection sampling with tied-logit coins.
- **Circuit** (`qqpu/circuit.py`): 16 layers, 22 DeltaNet heads (16 in the busiest layer),
  ~3.8k MLP units. Exact in Qwenputer's IR evaluator. `pytest`: 170 tests.
- **Merged Qwen3.5-4B** (`build/merged-4b-v2`, `python -m qqpu.merge`): hidden 2560 -> 3200,
  16 GB. Text KL 4.8e-4, top-1 50/50. All 16 library circuits token-exact (teacher-forced and
  free greedy), state == Qiskit (`scripts/merged_check.py`).
- **Chat** (`qqpu/bridge.py`): Qwen writes OpenQASM in a `qpu_run` tool call (optional shots);
  the run happens in the same KV cache. `scripts/eval_chat.py` (22 questions):
  QPU 16/22 = gate-by-gate 16/16, named algorithms 0/6; 19/19 QPU runs exact.
  Host-simulator tool: 13/16 (v1 set) vs QPU 12/16. No tool (no thinking, 1024 tok): 2/16.
- **Demo** (`scripts/demo_suite.py`): curated circuits as QASM on the merged model, with shots.

## Versions
- tag v1: real amplitudes only (A/B lag trick instead of double buffers), 11 layers.
- main: v2 complex + double buffers + measurement.

## Ideas not done
- Library tool (grover/qft by name) to fix named-algorithm questions.
- Speed: static cache / CUDA graphs (Qwenputer fast.py), skip inactive controlled pairs.
- Overlay (share Qwen dims) to shrink the checkpoint.
- Shots in complex mode (needs |x|^2 in Z[sqrt2]).
