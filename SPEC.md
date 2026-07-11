# SPEC.md — Two-Regime Market Simulation (frozen contract)

Purpose: demonstrate, in one unforgettable figure, that a market of
**self-replicating strategic agents** converges to a stable price when returns
are *anchored* but breaks into **perpetual boom–bust** when returns are
*purely positional* — with the settle/cycle boundary coinciding with a
**machine-checked theorem** from `StrategicEvolutionLean/SEKernel`.

This file is the contract. `sim.py` transcribes it; it does not re-derive it.
Every dynamical claim the simulation rests on is a proven theorem (cited by
name below); those theorems are baked in as runtime **asserts** so the code
self-verifies and needs no LLM-in-the-loop debugging.

---

## 1. The model (transcribed verbatim from the Lean kernel)

Three competing AI strategy types in cyclic (rock–paper–scissors) competition
for a shared resource. State `x = (x₀,x₁,x₂)` on the simplex (population
shares / market composition). Source of truth: `SEKernel/Law7_Hopf.lean`.

- **Biased-RPS payoff** (`biasedRPS κ`):
  ```
  Π(κ) = [[ 0,   -1,  1+κ],
          [1+κ,   0,  -1 ],
          [-1,  1+κ,   0 ]]
  ```
  `κ` = market's *anchoring* parameter. κ>0: returns partly track a real
  productivity signal (the game gains a potential component). κ=0: pure
  zero-sum positional competition. κ<0: super-competitive (the counter to
  each strategy is over-rewarded).
- **Return on compute** (`fitness κ x i`): `fᵢ(x) = (Π(κ) x)ᵢ`.
- **Mean return / shadow price of the shared resource** (`meanFit κ x`):
  `f̄(x) = xᵀ Π(κ) x`.  Closed form (antisymmetric part cancels):
  `f̄(x) = (κ/2)(1 − ‖x‖²)`.
- **Replicator–mutator field** (`rmField κ μ x i`), the dynamics:
  ```
  ẋᵢ = xᵢ (fᵢ(x) − f̄(x)) + μ (1/3 − xᵢ)
  ```
  `μ ≥ 0` = **entry rate** (rate at which new/mutant variants enter toward the
  barycentre) — the "innovation rare" parameter of the paper.

Integrator: classical **RK4**, fixed `dt` (validated by Assert A1). The field
is tangent to the simplex (Assert A1), so no projection is needed; renormalise
only to kill float drift.

Market **price readout** for display: `p(x) = P₀ + f̄(x)` with `P₀ = 1`
(a positive baseline so the level is legible). `p` = baseline plus the average
return on compute in the market.

---

## 2. The theorems this rests on (the free oracles)

From `SEKernel/Law7_Hopf.lean` (all cold-verified, zero custom axioms):

- `rm_center_trace`: the linearisation trace at the barycentre `(⅓,⅓,⅓)` is
  exactly **`−κ/3 − 2μ`**.
- Center eigenvalues (pure replicator, μ=0): **`−κ/6 ± i·√3(κ+2)/6`**
  (`rm_true_hopf`, `rm_center_det`). Real part `−κ/6`; frequency at κ=0 is
  `√3/3 = 1/√3`.
- `firstLyapunov_neg`: the first Lyapunov coefficient is **< 0** — the Hopf is
  **supercritical**, so a *stable* limit cycle emerges on the unstable side.
- `rm_no_hopf` / trace sign: stable focus ⟺ `trace < 0` ⟺ **`κ > −6μ`**;
  unstable focus (→ stable limit cycle, supercritical) ⟺ **`κ < −6μ`**.

**The proven settle/cycle boundary is the line `κ = −6μ`.** The simulation's
job is to (a) show the two regimes as one figure, and (b) confirm the measured
boundary coincides with this line across a 10k-case sweep.

---

## 3. Experiments & outputs

### Exp 1 — Headline figure (`fig1_contrast.png`)
Two runs, same dynamics, differing only in market character:
- **Regime A (anchored):** `κ = +0.20, μ = 0.02`  (trace ≈ −0.11 < 0).
- **Regime B (positional):** `κ = −0.40, μ = 0.02`  (trace ≈ +0.09 > 0).
Both from the same off-centre start `x₀ = (0.5, 0.3, 0.2)`, T = 400, dt = 1e-3.
Figure, two panels:
- **Left — simplex phase portrait.** A spirals *into* the barycentre (stable
  price); B winds *out to a closed limit cycle* (perpetual boom–bust).
- **Right — price vs time** `p(x(t))`. A flat-lines; B oscillates at fixed
  amplitude forever, with no external shock.

### Exp 2 — Phase diagram / the 10,000 cases (`fig2_phase.png`)
Grid: `κ ∈ [−0.5, 0.5]` × `μ ∈ [0, 0.10]`, 100×100 = 10,000 cases (extra
initial-condition reps optional). For each, integrate to T, classify by the
**asymptotic amplitude** `A = limsup ‖x(t) − center‖` over the last window:
`A < ε_settle` → *settle*; else → *cycle*. Plot the settle/cycle map and
**overlay the proven line `κ = −6μ`**. Deliverable metric: fraction of grid
cells whose empirical class matches the theorem's prediction (target ≥ 0.98,
mismatches only within one grid step of the line).

### Exp 3 — Barbell / Law 2 (`fig3_barbell.png`)
The falsifiable-against-data prediction. A cost-heterogeneous market of
self-replicating agents collapses to **at most two active types** (Cor 5.5),
because the portfolio LP has **two** shared constraints (budget + capacity).

Setup (transcribed from `Law2_ESDI.lean` §12.1 LP): `N` candidate types, each
with per-instance return `rᵢ`, cost `cᵢ`, load `ℓᵢ`. Cost-normalised
coordinates `aᵢ = ℓᵢ/cᵢ` (load-per-cost, "weight/capability"), `bᵢ = rᵢ/cᵢ`
(return-per-cost); types drawn on/below a concave ROC frontier (generic
position). Shared per-capita budget `β` and capacity `q`, both scarce.

Dynamics: exponential replicator on shares `x` (always positive, mass-
preserving) with fitness `= exp(η·(rᵢ − μcᵢ − λℓᵢ))`, and price tâtonnement
`μ ← [μ + η_p(Σxᵢcᵢ − β)]₊`, `λ ← [λ + η_p(Σxᵢℓᵢ − q)]₊`. This primal–dual
dynamic converges to the LP saddle point.

Figure: (left) the `(a,b)` plane — all types, the ROC frontier (upper convex
hull), the shadow-price line `b = μ + λa`, the ≤2 survivors circled;
(right) the capability distribution — broad initial mass over `a` collapsing
to **two spikes** (the barbell). Sweep over `M` random economies = the
"generic position" check at scale.

Outputs: `fig1_contrast.png`, `fig2_phase.png`, and `summary.json` (the only
thing that re-enters the model's context — see §5).

---

## 4. Theorem-asserts (baked into `sim.py`; the code self-verifies)

Every run checks these; any failure aborts loudly at the named assert. These
are *proven facts*, so a violation means a code bug, not a discovery.

- **A1 — mass conservation.** `Σ xᵢ = 1` to `1e-9` at every step (rmField is
  tangent to the simplex: `Σ rmField = (f̄−f̄) + μ(1−Σx) = 0`). Used to
  validate `dt`.
- **A2 — simplex invariance.** `xᵢ ≥ −1e-9` throughout.
- **A3 — barycentre is an equilibrium ∀κ,μ.** `‖rmField(κ,μ,center)‖ < 1e-12`
  (row sums of Π(κ) are all κ ⟹ fᵢ(center)=κ/3=f̄).
- **A4 — center trace.** Finite-difference the reduced 2-D Jacobian at the
  barycentre; its trace equals **`−κ/3 − 2μ`** to `1e-4` (checks
  `rm_center_trace`).
- **A5 — conserved quantity at (κ=0, μ=0).** Along a pure-swirl orbit,
  `∏ᵢ xᵢ` is invariant to `1e-6` and `f̄ ≡ 0` (zero-sum) — the closed-orbit
  signature.
- **A6 — frequency at (κ=0, μ=0).** Measured orbit period ≈ `2π√3 ≈ 10.883`
  to within 3% (checks the machine-checked frequency `1/√3`).
- **A7 — boundary.** In Exp 2, the empirical settle/cycle transition lies on
  `κ = −6μ` to within one grid step (checks the trace-sign criterion).

Exp 3 (Law 2) asserts:
- **B1 — mass/simplex.** Exponential replicator keeps `Σx=1`, `x≥0`.
- **B2 — barbell.** Converged support `#{i : xᵢ > 1e-3} ≤ 2` (Cor 5.5).
- **B3 — KKT/complementary slackness.** Supported types have net reduced
  return `rᵢ − μcᵢ − λℓᵢ` equal to the max (equalised) to tol; `μ,λ ≥ 0`; and
  `μ>tol ⇒ Σxᵢcᵢ ≈ β`, `λ>tol ⇒ Σxᵢℓᵢ ≈ q` (checks `KKT.reduced_return`).
- **B4 — shadow-price line.** Survivors satisfy `bᵢ = μ + λ·aᵢ` to tol
  (checks `constraint_role_sparsity` / Prop 2.8).
- **B5 — genericity at scale.** Over `M` random economies, fraction with
  `≤2` survivors is `≥ 0.98` (the "generic position" clause).

`sim.py` prints `ALL CHECKS PASS (A1–A7)` plus `summary.json`, or fails at the
first violated assert with the offending values.

---

## 5. Token discipline (binding)

- One file, `sim.py`, NumPy + matplotlib only. Deterministic (`SEED = 0`).
- Runs happen in the background; **no raw logs, no per-run narration** re-enter
  context.
- The **only** artifact the model reads back is `summary.json` (≈10 lines:
  the A1–A7 verdicts, Exp-1 amplitudes, Exp-2 match-fraction, timings).
- Figures are written to PNG for the human; the model does **not**
  screenshot-iterate on them. A visual problem → one new assert, not a
  reading session.

---

## 6. Provenance

Equations and constants are transcribed from
`StrategicEvolutionLean/SEKernel/Law7_Hopf.lean` (public mirror:
`github.com/selfreferencing/TSE_Formal`). The boundary `κ = −6μ`, the trace
`−κ/3 − 2μ`, the eigenvalues, the frequency `1/√3`, and the supercriticality
sign are all machine-checked there. This is the "proof-carrying simulation":
the simulator explores the quantitative regime that the theorems bound.
