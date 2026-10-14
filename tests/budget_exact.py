"""Exact (hypergeometric) failure probability of the expert-budget rule on a fixed workload (rule level).

Model of Bottler._certify_and_escalate's budget path for ONE pre-specified look and certificate='rectified': the N
items are in descending score order with distinct scores, every answer is usable, no memo; the expert labels a
uniform n-of-N calibration sample.
  k-rule (k = k_used at the k-rule's level): the threshold sits just above the (k+1)-th labelled bad item (everything
    is accepted if at most k sampled items are bad; nothing if k < 0). U_k = unlabelled bad items above it.
  escalations below that threshold = the unlabelled items at or below the (k+1)-th labelled bad item; the budget
    binds iff they are more than `budget`.
  budget threshold (label-free): positions 0..N-1-budget are accepted; X = labelled bad items among them; the claim
    is eps_hat = (M_upper(X) - X)/N at the inversion's level, and the bottled bad count is U_b = B_above - X, so the
    claim fails iff B_above > M_upper(X).
The batch fails iff (not binding and U_k > m) or (binding and B_above > M_upper(X)), m = floor(eps N).

exact_fail() computes these probabilities exactly with a forward recursion over the positions on the state
(c, j, x) = (labelled so far, labelled bad so far, labelled bad above the budget cut so far), with the sequential
sampling probability (n - c)/(N - i) of labelling position i. At the stop (the (k+1)-th labelled bad item at position
p, with c items labelled before it) U_k = #bad before p - k and the escalation count (N - p) - (n - c) are fixed, and
the rest of X (labelled bad items in (p, cut]) is Hypergeom(N - p - 1, bad in (p, cut], n - c - 1). No sampling.
brute_force() enumerates every calibration subset of a small batch through the package's own fixed_sequence and
certified_eps, to tie the recursion to the package code (tests/test_budget_split.py).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import hypergeom

from bottlekit.certify import certified_eps, fixed_sequence, k_used, single_look_design, tuned_looks


def m_upper_table(N: int, n: int, level: float) -> np.ndarray:
    """M_upper(X) for X = 0..n at `level`, read from the package's certified_eps (rectified, k_end = X):
    certified_eps returns max(0, M_upper - X)/N and M_upper >= X, so M_upper = X + round(N * value)."""
    out = np.zeros(n + 1, int)
    for x in range(n + 1):
        s = np.r_[np.ones(x), np.zeros(n - x)]
        b = np.r_[np.ones(x, bool), np.zeros(n - x, bool)]
        out[x] = x + int(round(N * certified_eps(s, b, 0.5, N, level, "rectified")))
    return out


def design(N: int, eps: float, delta: float, budget_set: bool, method: str = "rectified") -> dict:
    """The single-look design the harness applies (v0.9.1): sizes tuned and k at delta/2 with a budget (inversion at
    delta/2 too), else at delta (inversion at delta: the v0.9 budget rule)."""
    lev = delta / 2 if budget_set else delta
    n = tuned_looks(list(single_look_design(N, eps)[0]), [1.0], N, eps, lev, method)[0]
    return {"n": int(n), "k": int(k_used(n, N, eps, lev, method)), "k_level": lev, "inv_level": lev,
            "m": int(math.floor(eps * N))}


def exact_fail(bad, n: int, k: int, budget: int, MU: np.ndarray, m: int) -> dict:
    """Exact P[the claim made fails] of the budget rule on the workload `bad` (bool, descending score order).
    Returns the total and its parts: P[not binding and U_k > m], P[binding and inversion fails], P[binding],
    P[U_k > m] (on every draw) and P[inversion fails] (on every draw)."""
    bad = np.asarray(bad, bool)
    N = len(bad)
    if not 0 <= budget < N - 1:
        raise ValueError("need 0 <= budget < N - 1")
    cut = N - 1 - budget                                  # budget threshold accepts positions 0..cut
    B_above, B_tot = int(bad[:cut + 1].sum()), int(bad.sum())
    badc = np.r_[0, np.cumsum(bad)]                       # badc[i] = bad items before position i
    inv_fails = B_above > MU                              # inversion fails given X = x (index x)
    # P[inversion fails] on every draw: X ~ Hypergeom(N, B_above, n)
    xs = np.arange(n + 1)
    p_inv_any = float(hypergeom.pmf(xs, N, B_above, n)[inv_fails[:n + 1]].sum())
    res = {"fail": 0.0, "fail_not_binding": 0.0, "fail_binding": 0.0, "p_binding": 0.0, "p_k_rule_fails": 0.0,
           "p_inversion_fails": p_inv_any, "B_above": B_above, "B_total": B_tot, "n": n, "k": k, "m": m,
           "budget": budget}
    if k < 0:                                             # nothing certified: every unlabelled item is escalated
        if N - n > budget:
            res["p_binding"] = 1.0
            res["fail_binding"] = res["fail"] = p_inv_any
        return res
    K1 = k + 1
    P = np.zeros((n + 1, K1, K1))                         # P[c, j, x], j <= k labelled bad so far, x <= j
    P[0, 0, 0] = 1.0
    c_idx = np.arange(n + 1)
    for i in range(N):
        pl = np.clip((n - c_idx) / (N - i), 0.0, 1.0)
        lab = P * pl[:, None, None]
        newP = P * (1.0 - pl)[:, None, None]
        if bad[i]:
            inc = 1 if i <= cut else 0
            if k >= 1:                                    # labelled bad, j < k: continue
                if inc:
                    newP[1:, 1:, 1:] += lab[:-1, :-1, :-1]
                else:
                    newP[1:, 1:, :] += lab[:-1, :-1, :]
            S = lab[:, k, :]                              # labelled bad with j == k: the k-rule stops at p = i
            mass_c = S.sum(axis=1)
            Uk = int(badc[i]) - k
            n_esc = (N - i) - (n - c_idx)
            bind = n_esc > budget
            if Uk > m:
                res["p_k_rule_fails"] += float(mass_c.sum())
                res["fail_not_binding"] += float(mass_c[~bind].sum())
            res["p_binding"] += float(mass_c[bind].sum())
            b_rem = int(badc[cut + 1] - badc[i + 1]) if i < cut else 0
            for c in np.nonzero(bind & (mass_c > 0))[0]:
                draws, pop = n - int(c) - 1, N - i - 1
                ys = np.arange(0, min(b_rem, draws) + 1)
                pmf = hypergeom.pmf(ys, pop, b_rem, draws) if pop > 0 else np.ones(1)
                for x in np.nonzero(S[c] > 0)[0]:
                    X = int(x) + inc + ys
                    res["fail_binding"] += float(S[c, x] * pmf[inv_fails[np.minimum(X, n)]].sum())
        else:
            newP[1:] += lab[:-1]
        P = newP
    # at most k labelled bad items: everything is accepted, nothing escalated (never binding)
    for j in range(K1):
        mass = float(P[n, j, :].sum())
        if B_tot - j > m:
            res["p_k_rule_fails"] += mass
            res["fail_not_binding"] += mass
    res["fail"] = res["fail_not_binding"] + res["fail_binding"]
    return res


def brute_force(bad, n: int, eps: float, k_level: float, inv_level: float, budget: int) -> dict:
    """Every calibration subset of a small batch through the package's fixed_sequence and certified_eps (rectified)."""
    bad = np.asarray(bad, bool)
    N = len(bad)
    score = np.linspace(1.0, 0.0, N)                      # descending, distinct
    fail = bind = tot = 0
    for S in itertools.combinations(range(N), n):
        S = list(S)
        is_cal = np.zeros(N, bool)
        is_cal[S] = True
        tau, _ = fixed_sequence(score[S], bad[S], N, eps, k_level, "rectified")
        acc = score >= tau
        esc = int((~acc & ~is_cal).sum())
        if esc > budget:
            bind += 1
            tau_b = float(np.sort(score)[min(N - 1, budget)])
            e = certified_eps(score[S], bad[S], tau_b, N, inv_level, "rectified")
            share = float(((score >= tau_b) & ~is_cal & bad).sum()) / N
            fail += share > e + 1e-12
        else:
            share = float((acc & ~is_cal & bad).sum()) / N
            fail += share > eps + 1e-12
        tot += 1
    return {"fail": fail / tot, "p_binding": bind / tot, "subsets": tot}


def layout(N: int, top: int, gap: int, mid: int) -> np.ndarray:
    """Trap family: `top` bad items, `gap` good, `mid` bad, the rest good (descending score order)."""
    bad = np.zeros(N, bool)
    bad[:top] = True
    bad[top + gap:top + gap + mid] = True
    return bad
