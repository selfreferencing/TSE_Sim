# TSE_Sim — proof-carrying simulations of self-replicating markets

Simulations of the **Theory of Strategic Evolution** (Kevin Vallier,
[arXiv:2512.07901](https://arxiv.org/abs/2512.07901)) in which every dynamical
claim the code rests on is a **machine-checked theorem** from the Lean 4
kernel [`TSE_Formal`](https://github.com/selfreferencing/TSE_Formal). The
simulator does not *test* the theory — it *inherits* it: the theorems are
transcribed as runtime `assert`s, so the code self-verifies and the sweeps
merely explore the quantitative regime the theorems bound.

No LLM runs inside the loop. A strategic replicator is a point on a simplex
evolving by deterministic arithmetic; the agents are not minds. The entire
empirical content lives in one function — the fitness/return map — which you
assume (here), fit to data, or measure once. Everything downstream is free.

## What's here

| File | What |
|---|---|
| [`SPEC.md`](SPEC.md) | The frozen contract: equations by name from the kernel, the experiments, and the theorem-asserts (A1–A7, B1–B5). |
| [`sim.py`](sim.py) | One NumPy file. Transcribes the kernel dynamics, bakes the theorems in as asserts, runs the experiments, writes `summary.json`. |
| [`demo.html`](demo.html) | Self-contained interactive artifact — drag κ and μ, watch the market tip from a stable price to perpetual boom–bust live. Open it in any browser. |
| `fig1_contrast.png` | Headline: anchored market settles vs positional market cycles forever. |
| `fig2_phase.png` | 10,000-case phase diagram; empirical settle/cycle boundary vs the proven line `κ = −6μ`. |
| `fig3_barbell.png` | Law 2: a market with two shared constraints collapses to ≤2 types (planners + executors). |
| `summary.json` | The only output that matters for review: the A1–A7 / B1–B5 verdicts and the sweep statistics. |

## Results (all machine-verified)

- **Cycling threshold (Law 7).** The replicator–mutator on biased Rock–Paper–
  Scissors settles when returns are anchored (`κ > −6μ`) and breaks into a
  perpetual limit cycle when they are positional (`κ < −6μ`). Across 10,000
  cases the empirical boundary matches the machine-checked line `κ = −6μ` to
  **99.3%** away from the region where the theory itself says the amplitude is
  undetectable. Trace `−κ/3 − 2μ`, frequency `1/√3`, and mass conservation all
  hold to machine precision.
- **Barbell (Law 2).** A cost-heterogeneous, self-replicating market collapses
  to **at most two active types**. Across 5,000 random economies this holds
  **100%** of the time, with an LP-duality certificate *proving* optimality on
  every one.

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install numpy matplotlib
python sim.py            # full: all experiments + figures + summary.json
python sim.py --smoke    # fast: asserts + tiny sweeps
python sim.py --barbell  # just Law 2
```
A clean run prints `ALL CHECKS PASS (A1-A7, B1-B5)`. Any theorem violation
aborts at the named assert — because it would mean a code bug, not a discovery.

For the interactive demo, just open `demo.html` in a browser.

## Provenance

Kernel: [`selfreferencing/TSE_Formal`](https://github.com/selfreferencing/TSE_Formal)
(73 cold-verified theorems, zero custom axioms). Paper:
[`selfreferencing/TSE_Paper`](https://github.com/selfreferencing/TSE_Paper) /
[arXiv:2512.07901](https://arxiv.org/abs/2512.07901). The constants this
simulation relies on — the boundary `κ = −6μ`, the trace `−κ/3 − 2μ`, the
eigenvalues, the frequency `1/√3`, the ≤2-type sparsity bound — are all proven
there.
