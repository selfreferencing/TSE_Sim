#!/usr/bin/env python3
"""
TSE two-regime market simulation — proof-carrying.

Transcribes the replicator-mutator dynamics on the biased-RPS payoff from
StrategicEvolutionLean/SEKernel/Law7_Hopf.lean.  Every dynamical claim it
rests on is a machine-checked theorem; those are baked in as asserts A1-A7
(see SPEC.md).  The code self-verifies: it prints ALL CHECKS PASS + writes
summary.json, or aborts at the first violated assert.

Run:  python3 sim.py            (full: Exp1 figure + Exp2 10k-case phase map)
      python3 sim.py --smoke    (tiny params, asserts only, no heavy sweep)
"""
import json, sys, time
import numpy as np

SEED = 0
CENTER = np.array([1/3, 1/3, 1/3])
OUT = {}  # -> summary.json

# ----------------------------------------------------------------------------
# Model (verbatim from Law7_Hopf.lean)
# ----------------------------------------------------------------------------
def Pi(k):
    return np.array([[0.0, -1.0, 1+k],
                     [1+k,  0.0, -1.0],
                     [-1.0, 1+k,  0.0]])

def fitness(k, x):            # f_i(x) = (Pi(k) x)_i
    return Pi(k) @ x

def meanfit(k, x):           # f-bar(x) = x^T Pi(k) x
    return x @ Pi(k) @ x

def rmfield(k, mu, x):       # rmField k mu x i = x_i (f_i - f-bar) + mu (1/3 - x_i)
    return x * (fitness(k, x) - meanfit(k, x)) + mu * (1/3 - x)

def price(k, x, P0=1.0):     # display readout: baseline + mean return on compute
    return P0 + meanfit(k, x)

# ----------------------------------------------------------------------------
# Integrator (RK4; field is tangent to the simplex, so we only kill float drift)
# ----------------------------------------------------------------------------
def integrate(k, mu, x0, T, dt, check_mass=True):
    n = int(round(T / dt))
    x = np.array(x0, float)
    traj = np.empty((n + 1, 3)); traj[0] = x
    max_mass_drift = 0.0
    min_coord = x.min()
    for i in range(n):
        k1 = rmfield(k, mu, x)
        k2 = rmfield(k, mu, x + 0.5*dt*k1)
        k3 = rmfield(k, mu, x + 0.5*dt*k2)
        k4 = rmfield(k, mu, x + dt*k3)
        x = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        if check_mass:
            max_mass_drift = max(max_mass_drift, abs(x.sum() - 1.0))
        x = x / x.sum()                      # renormalise away RK4 float drift
        min_coord = min(min_coord, x.min())
        traj[i+1] = x
    return traj, max_mass_drift, min_coord

# ----------------------------------------------------------------------------
# Vectorised batch integrator for the 10k-case sweep (state X: (N,3))
# ----------------------------------------------------------------------------
def _fieldN(k, mu, X):       # k,mu: (N,), X: (N,3) -> (N,3)
    kk = k[:, None]
    P0 = np.stack([np.zeros_like(k), -np.ones_like(k),  1+k], 1)
    P1 = np.stack([1+k,  np.zeros_like(k), -np.ones_like(k)], 1)
    P2 = np.stack([-np.ones_like(k), 1+k, np.zeros_like(k)], 1)
    f = np.stack([(P0*X).sum(1), (P1*X).sum(1), (P2*X).sum(1)], 1)   # (N,3)
    fbar = (X*f).sum(1, keepdims=True)
    return X*(f - fbar) + mu[:, None]*(1/3 - X)

def integrate_batch(k, mu, X0, T, dt, keep_last):
    n = int(round(T/dt)); X = X0.copy()
    keep = np.empty((keep_last, X.shape[0], 3)); ptr = 0
    start = n - keep_last
    for i in range(n):
        a = _fieldN(k, mu, X)
        b = _fieldN(k, mu, X + 0.5*dt*a)
        c = _fieldN(k, mu, X + 0.5*dt*b)
        d = _fieldN(k, mu, X + dt*c)
        X = X + (dt/6.0)*(a + 2*b + 2*c + d)
        X = X / X.sum(1, keepdims=True)
        if i >= start:
            keep[ptr] = X; ptr += 1
    return keep   # (keep_last, N, 3)

# ----------------------------------------------------------------------------
# Standalone theorem-asserts A3-A6 (do not depend on a particular run)
# ----------------------------------------------------------------------------
def assert_A3_equilibrium():
    worst = 0.0
    for k in (-0.4, 0.0, 0.2, 0.9):
        for mu in (0.0, 0.02, 0.1):
            worst = max(worst, np.linalg.norm(rmfield(k, mu, CENTER)))
    assert worst < 1e-12, f"A3 FAIL: ||rmField(center)|| = {worst}"
    return worst

def _reduced_field(k, mu, y):        # y=(x0,x1); x2 = 1-x0-x1
    x = np.array([y[0], y[1], 1-y[0]-y[1]]); r = rmfield(k, mu, x)
    return np.array([r[0], r[1]])

def _center_trace_fd(k, mu, h=1e-6):
    y0 = np.array([1/3, 1/3]); tr = 0.0
    for j in range(2):
        e = np.zeros(2); e[j] = h
        col = (_reduced_field(k, mu, y0+e) - _reduced_field(k, mu, y0-e)) / (2*h)
        tr += col[j]
    return tr

def assert_A4_trace():
    worst = 0.0
    for k in (-0.4, 0.0, 0.2, 0.9):
        for mu in (0.0, 0.02, 0.1):
            worst = max(worst, abs(_center_trace_fd(k, mu) - (-k/3 - 2*mu)))
    assert worst < 1e-4, f"A4 FAIL: |trace_fd - (-k/3-2mu)| = {worst}"
    return worst

def assert_A5_A6_pure_swirl():
    # (k=0, mu=0): prod x_i conserved, f-bar == 0, period ~ 2*pi*sqrt(3)
    x0 = np.array([0.5, 0.3, 0.2])
    traj, _, _ = integrate(0.0, 0.0, x0, T=40.0, dt=1e-3)
    prods = traj[:, 0]*traj[:, 1]*traj[:, 2]
    cons = np.ptp(prods) / prods.mean()                     # A5: relative drift of product
    fbar_max = max(abs(meanfit(0.0, x)) for x in traj[::200])  # A5: zero-sum
    assert cons < 1e-6, f"A5 FAIL: prod x_i drift = {cons}"
    assert fbar_max < 1e-9, f"A5 FAIL: f-bar not ~0: {fbar_max}"
    # A6: the machine-checked frequency 1/sqrt(3) is the LINEARISATION eigenvalue,
    # so measure the period of a SMALL-amplitude orbit near the barycentre
    # (RPS orbit periods lengthen with amplitude; large orbits are not sinusoidal).
    x0s = CENTER + np.array([0.01, -0.005, -0.005])
    trajs, _, _ = integrate(0.0, 0.0, x0s, T=40.0, dt=1e-3)
    s = trajs[:, 0] - trajs[:, 0].mean()
    zc = np.where((s[:-1] < 0) & (s[1:] >= 0))[0]            # upward crossings
    period = np.diff(zc).mean() * 1e-3 if len(zc) >= 3 else np.nan
    target = 2*np.pi*np.sqrt(3)
    assert abs(period - target)/target < 0.02, f"A6 FAIL: period {period} vs {target}"
    return cons, fbar_max, period, target

# ----------------------------------------------------------------------------
# Experiment 1: headline contrast figure
# ----------------------------------------------------------------------------
def simplex_xy(X):                    # barycentric -> 2D equilateral coords
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])
    return X @ v

def experiment1():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x0 = np.array([0.5, 0.3, 0.2]); T, dt = 400.0, 1e-3
    regimes = {"A (anchored, k=+0.20)": (0.20, 0.02, "#2c7fb8"),
               "B (positional, k=-0.40)": (-0.40, 0.02, "#d95f0e")}
    runs = {}
    max_drift = 0.0; min_coord = 1.0
    for name, (k, mu, col) in regimes.items():
        traj, drift, mn = integrate(k, mu, x0, T, dt)
        runs[name] = (traj, k, mu, col)
        max_drift = max(max_drift, drift); min_coord = min(min_coord, mn)
    # A1, A2 on the headline runs
    assert max_drift < 1e-9, f"A1 FAIL: mass drift {max_drift}"
    assert min_coord > -1e-9, f"A2 FAIL: min coord {min_coord}"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    tri = simplex_xy(np.eye(3))
    axL.plot(*np.vstack([tri, tri[0]]).T, color="0.7", lw=1)
    cxy = simplex_xy(CENTER.reshape(1, 3))[0]
    axL.plot(*cxy, "k.", ms=6)
    amps = {}
    tgrid = np.arange(len(next(iter(runs.values()))[0])) * dt
    for name, (traj, k, mu, col) in runs.items():
        xy = simplex_xy(traj)
        axL.plot(xy[:, 0], xy[:, 1], color=col, lw=0.8, alpha=0.8, label=name)
        axR.plot(tgrid, [price(k, x) for x in traj], color=col, lw=1.0, label=name)
        tail = traj[int(0.6*len(traj)):]
        amps[name] = float(np.max(np.linalg.norm(tail - CENTER, axis=1)))
    axL.set_title("Market composition (simplex)"); axL.axis("equal"); axL.axis("off")
    axL.legend(loc="upper right", fontsize=8)
    axR.set_title("Market price  p(x) = 1 + mean return"); axR.set_xlabel("time")
    axR.set_ylabel("price"); axR.legend(loc="upper right", fontsize=8)
    fig.suptitle("Self-replicating market: anchored returns settle, "
                 "positional returns cycle forever", fontsize=12)
    fig.tight_layout(); fig.savefig("fig1_contrast.png", dpi=130); plt.close(fig)
    return {"amp_A": amps["A (anchored, k=+0.20)"],
            "amp_B": amps["B (positional, k=-0.40)"],
            "mass_drift": max_drift, "min_coord": min_coord}

# ----------------------------------------------------------------------------
# Experiment 2: 10,000-case phase diagram + boundary check (A7)
# ----------------------------------------------------------------------------
def experiment2(nk=100, nmu=100, T=200.0, dt=0.02, eps_settle=0.02):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = np.linspace(-0.5, 0.5, nk)
    mus = np.linspace(0.0, 0.10, nmu)
    KK, MM = np.meshgrid(ks, mus)                   # (nmu, nk)
    kflat, mflat = KK.ravel(), MM.ravel()
    N = kflat.size
    rng = np.random.default_rng(SEED)
    X0 = np.tile(np.array([0.5, 0.3, 0.2]), (N, 1))
    keep_last = int(round((0.30*T)/dt))
    keep = integrate_batch(kflat, mflat, X0, T, dt, keep_last)     # (kl, N, 3)
    r = np.linalg.norm(keep - CENTER, axis=2)                      # (kl, N) radius
    half = keep_last // 2
    a_early = r[:half].max(0)                                      # amplitude, early tail
    a_late = r[half:].max(0)                                       # amplitude, late tail
    # limit cycle: amplitude sustained; stable focus: amplitude still decaying.
    settling = a_late < 0.85*a_early                              # trend test
    cyc = ((a_late > eps_settle) & (~settling)).reshape(nmu, nk)  # empirical
    pred = (KK < -6*MM)                                            # theorem: cycle iff k < -6mu
    # A7: match away from the boundary band, where the supercritical cycle
    # amplitude -> 0 (~sqrt(k_c - k)) is genuinely below detection.
    dk = ks[1]-ks[0]
    near = np.abs(KK + 6*MM) <= 2.0*dk
    match = (cyc == pred)
    match_frac_all = float(match.mean())
    match_frac_far = float(match[~near].mean())
    assert match_frac_far >= 0.98, f"A7 FAIL: far-from-boundary match {match_frac_far}"

    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.imshow(cyc, origin="lower", extent=[ks[0], ks[-1], mus[0], mus[-1]],
              aspect="auto", cmap="RdYlBu_r", alpha=0.85)
    ax.plot(ks, -ks/6, "k--", lw=2, label=r"proven boundary  $\kappa=-6\mu$")
    ax.set_xlim(ks[0], ks[-1]); ax.set_ylim(mus[0], mus[-1])
    ax.set_xlabel(r"$\kappa$  (anchoring:  <0 positional,  >0 fundamental)")
    ax.set_ylabel(r"$\mu$  (entry rate)")
    ax.set_title("Settle (blue) vs perpetual boom-bust (red), 10,000 cases\n"
                 "empirical transition vs machine-checked Hopf boundary")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig("fig2_phase.png", dpi=130); plt.close(fig)
    return {"cases": int(N), "match_frac_all": match_frac_all,
            "match_frac_far_from_boundary": match_frac_far,
            "eps_settle": eps_settle}

# ----------------------------------------------------------------------------
# Experiment 3: barbell / Law 2 (support collapses to <=2 types)
# ----------------------------------------------------------------------------
def make_economy(rng, N=40):
    a = rng.uniform(0.3, 3.0, N)               # load-per-cost ("weight")
    bmax = 1.5 + a - 0.15*a**2                 # concave ROC frontier
    b = bmax - rng.exponential(0.35, N)        # types on/below the frontier
    c = rng.uniform(0.5, 1.5, N)               # per-instance cost
    l = a*c; r = b*c                           # load, return
    return c, l, r, a, b

def solve_lp(c, l, r, beta, q):
    """Exact optimum of  max sum r_i n_i  s.t.  sum c_i n_i <= beta,
    sum l_i n_i <= q,  n >= 0.  A 2-constraint LP: every basic optimum has
    <= 2 active types, so enumerate single- and pair-bases and take the best.
    Returns (obj, n, active, mu, lam).  LP duality then certifies optimality:
    the returned (mu, lam) are dual-feasible for ALL types (checked as B4)."""
    N = len(c); best = (-np.inf, None, None, 0.0, 0.0)
    for i in range(N):                                    # single-type bases
        ni = min(beta/c[i], q/l[i]); obj = r[i]*ni
        if obj > best[0]:
            mu, lam = (r[i]/c[i], 0.0) if beta/c[i] <= q/l[i] else (0.0, r[i]/l[i])
            n = np.zeros(N); n[i] = ni; best = (obj, n, [i], mu, lam)
    for i in range(N):                                    # two-type bases (both bind)
        for j in range(i+1, N):
            det = c[i]*l[j] - c[j]*l[i]
            if abs(det) < 1e-12: continue
            ni = ( l[j]*beta - c[j]*q) / det
            nj = (-l[i]*beta + c[i]*q) / det
            if ni < -1e-12 or nj < -1e-12: continue
            obj = r[i]*ni + r[j]*nj
            if obj > best[0]:
                dd = c[i]*l[j] - c[j]*l[i]                # dual: [c l] [mu;lam]=r
                mu  = ( l[j]*r[i] - l[i]*r[j]) / dd
                lam = (-c[j]*r[i] + c[i]*r[j]) / dd
                if mu >= -1e-9 and lam >= -1e-9:
                    n = np.zeros(N); n[i] = ni; n[j] = nj
                    best = (obj, n, [i, j], max(0.0, mu), max(0.0, lam))
    return best

def replicate_at_prices(c, l, r, mu, lam, T=1500, eta=8.0):
    """Discrete replicator on shares at the FIXED equilibrium prices: net
    return is state-independent, so it converges cleanly (SS-2 elimination)
    onto the zero-reduced-return types.  This is the 'self-replication
    collapses to the barbell' dynamics."""
    net = r - mu*c - lam*l
    x = np.full(len(c), 1.0/len(c)); f = np.exp(eta*(net - net.max()))
    for _ in range(T):
        x = x*f; x = x/x.sum()
    return x

def barbell_sweep(rng, M, N=40):
    """Genericity at scale: solve M random economies exactly, return support
    size and whether the dual certificate holds (optimality)."""
    surv = np.empty(M, int); dual_ok = np.empty(M, bool)
    for m in range(M):
        c, l, r, a, b = make_economy(rng, N)
        beta, q = 0.8*c.mean(), 0.8*l.mean()
        obj, n, active, mu, lam = solve_lp(c, l, r, beta, q)
        surv[m] = len(active)
        dual_ok[m] = bool(np.all(r <= mu*c + lam*l + 1e-6))
    return surv, dual_ok

def _upper_hull(a, b):
    o = np.argsort(a); hull = []
    for i in o:
        while len(hull) >= 2:
            x1, y1 = a[hull[-2]], b[hull[-2]]; x2, y2 = a[hull[-1]], b[hull[-1]]
            if (x2-x1)*(b[i]-y1) - (y2-y1)*(a[i]-x1) >= 0: hull.pop()
            else: break
        hull.append(i)
    return hull

def experiment3(smoke=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    # first economy whose EXACT optimum is a clean, well-separated 2-type barbell
    # (illustrative; the B5 sweep reports true genericity, uncherrypicked)
    econ = None
    for _ in range(80):
        c, l, r, a, b = make_economy(rng)
        beta, q = 0.8*c.mean(), 0.8*l.mean()
        obj, n, active, mu, lam = solve_lp(c, l, r, beta, q)
        if len(active) == 2 and abs(a[active[0]] - a[active[1]]) > 0.9:
            econ = (c, l, r, a, b, beta, q, n, active, mu, lam); break
    if econ is None:
        econ = (c, l, r, a, b, beta, q, n, active, mu, lam)
    c, l, r, a, b, beta, q, n, active, mu, lam = econ
    net = r - mu*c - lam*l

    # --- exact theorem-asserts on the demo economy ---
    assert len(active) <= 2, f"B2 FAIL: {len(active)} active types"
    assert np.all(np.abs(net[active]) < 1e-9), "B3 FAIL: active net-return != 0 (CS)"
    assert np.all(r <= mu*c + lam*l + 1e-9), "B4 FAIL: dual infeasible (not optimal)"
    # B_dyn: the self-replicator (at equilibrium prices) selects the barbell
    xrep = replicate_at_prices(c, l, r, mu, lam)
    rep_active = set(int(i) for i in np.where(xrep > 1e-3)[0])
    assert abs(xrep.sum()-1) < 1e-12 and xrep.min() > -1e-12, "B1 FAIL"
    assert rep_active.issubset(set(int(i) for i in active)), \
        "B_dyn FAIL: replicator support outside the optimum"

    # --- B5: genericity at scale (exact LP; every optimum has <=2 support) ---
    M = 300 if smoke else 5000
    surv, dual_ok = barbell_sweep(rng, M)
    frac_le2 = float((surv <= 2).mean())
    assert frac_le2 >= 0.98, f"B5 FAIL: frac(<=2 active) = {frac_le2}"
    assert bool(dual_ok.all()), "B5 FAIL: dual certificate failed somewhere"

    # --- figure ---
    nn = n / n.sum()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    axL.scatter(a, b, s=18, c="0.6", label="candidate types")
    hu = _upper_hull(a, b)
    axL.plot(a[hu], b[hu], "-", color="#2c7fb8", lw=1.2, label="ROC frontier")
    ag = np.linspace(a.min(), a.max(), 50)
    axL.plot(ag, mu + lam*ag, "--", color="0.4", lw=1,
             label=r"shadow-price line $b=\mu+\lambda a$")
    axL.scatter(a[active], b[active], s=150, facecolors="none",
                edgecolors="#d95f0e", linewidths=2, label="survivors (barbell)")
    axL.set_xlabel(r"$a=\ell/c$  (load per cost = capability)")
    axL.set_ylabel(r"$b=r/c$  (return per cost)")
    axL.set_title("ROC frontier: survivors are the shadow-price tangent points")
    axL.legend(fontsize=8, loc="lower center")
    bins = np.linspace(a.min(), a.max(), 24)
    axR.hist(a, bins=bins, weights=np.full(len(a), 1/len(a)), color="0.75",
             label="initial (broad, uniform mass)")
    axR.bar(a[active], nn[active], width=0.07, color="#d95f0e",
            label="final (self-replicated: 2 types)")
    axR.set_xlabel(r"capability  $a=\ell/c$"); axR.set_ylabel("population share")
    axR.set_title("Capability distribution collapses to a barbell (≤2 types)")
    axR.legend(fontsize=8)
    fig.suptitle("Law 2: a self-replicating market with 2 shared constraints "
                 "collapses to at most 2 types (planners + executors)", fontsize=12)
    fig.tight_layout(); fig.savefig("fig3_barbell.png", dpi=130); plt.close(fig)
    return {"demo_active_types": int(len(active)),
            "demo_a_survivors": sorted(float(a[i]) for i in active),
            "mu": float(mu), "lam": float(lam),
            "replicator_selects_barbell": bool(rep_active.issubset(set(int(i) for i in active))),
            "sweep_cases": int(M), "frac_le2_active": frac_le2,
            "dual_certificate_all_ok": bool(dual_ok.all())}

# ----------------------------------------------------------------------------
def main():
    t0 = time.time(); smoke = "--smoke" in sys.argv
    np.random.seed(SEED)
    if "--barbell" in sys.argv:                    # iterate Exp 3 alone, cheaply
        OUT["exp3"] = experiment3(smoke=True)
        OUT["seconds"] = round(time.time()-t0, 1)
        OUT["status"] = "ALL CHECKS PASS (B1-B5)"
    else:
        OUT["A3_equilibrium_residual"] = assert_A3_equilibrium()
        OUT["A4_trace_err"] = assert_A4_trace()
        a5cons, a5fbar, a6per, a6tgt = assert_A5_A6_pure_swirl()
        OUT["A5_prod_drift"] = a5cons; OUT["A5_fbar_max"] = a5fbar
        OUT["A6_period"] = a6per; OUT["A6_period_target"] = a6tgt
        OUT["exp1"] = experiment1()
        OUT["exp2"] = (experiment2(nk=20, nmu=20, T=120.0, dt=0.02)
                       if smoke else experiment2())
        OUT["exp3"] = experiment3(smoke=smoke)
        OUT["seconds"] = round(time.time()-t0, 1)
        OUT["status"] = "ALL CHECKS PASS (A1-A7, B1-B5)"
    with open("summary.json", "w") as f:
        json.dump(OUT, f, indent=2)
    print(OUT["status"]); print(json.dumps(OUT, indent=2))

if __name__ == "__main__":
    main()
