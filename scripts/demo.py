"""Ask Qwen (merged with its QPU) a question; print the conversation, the QASM it wrote and what
its own layers computed.

Run: uv run python scripts/demo.py build/merged-4b-v2 "Run Grover's search ..." [--host]
"""

import argparse
import time

from qqpu.bridge import Chat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--host", action="store_true", help="run the tool on the host instead")
    ap.add_argument("--max-new", type=int, default=1024)
    a = ap.parse_args()
    chat = Chat(a.model, "host" if a.host else "qpu")
    t0 = time.time()
    turn = chat.ask(" ".join(a.question), max_new=a.max_new)
    print(turn.text)
    print("-" * 80)
    for r in turn.runs:
        if "error" in r:
            print("QPU error:", r["error"])
            continue
        mode = "complex" if r["complex"] else "real"
        print(f"QPU run: {r['nq']} qubits ({mode} mode), {r['gates']} gates, {r['tokens']} tokens "
              f"written by the model in {r['seconds']:.1f}s; exact vs reference: "
              f"{r.get('exact')}; shots: {len(r.get('outcomes', []))}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
