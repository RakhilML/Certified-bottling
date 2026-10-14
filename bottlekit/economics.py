"""Cost model and the AUTO mode rule.

All costs in units of ONE expert single-item call (c_e = 1). The cheap tier costs c_b per item when batched.
  fused   : contexts*N*c_b + n_cal + (1-cov_f)*(N-n_cal)
  student : min(n_build,N)*c_b + n_cal + (1-cov_s)*(N-n_cal)
Voting on everything is worth it iff its EXTRA cheap spend is below the expected coverage gain:
  (contexts*N - min(n_build, N)) * c_b  <  gain * N.
The default coverage-gain prior (0.15) is the median gain measured across 4 development tasks;
it grows when the student is starved (few cheap labels per class), as measured on clinc150 (4 labels/class).
"""

from __future__ import annotations

from .llm import ModelSpec
from .tasks import TaskSpec

# Prompt-size cost estimate (decisions only; measured costs replace it once a batch has run). English-shaped:
# TaskSpec.chars_per_token overrides the ratio for other languages.
CHARS_PER_TOKEN_EST = 3.8    # chars per token of the cost estimate (English text)
HEADER_PAD_CHARS = 200       # the fixed scaffold of a prompt beyond instruction + label block (English, ~200 chars)
ITEM_PAD_TOKENS = 4          # per-item overhead tokens (numbering, separators)
OUT_PAD_TOKENS = 3           # per-answer overhead tokens on top of the label tokens
ECON_REASONING_SHARE = 0.5   # a reasoning expert is assumed to use this share of its reply budget (ModelSpec.max_tokens)
# Student-mode build size and the starved-student gain prior (auto mode). n_build: at least 600 (development tasks)
# and 20 per class (needed on clinc150). Gain prior: 40 labels/class -> no boost, linear to 5 -> x2 (clinc150 only).
N_BUILD_MIN = 600
N_BUILD_PER_CLASS = 20
GAIN_FED_PER_CLASS = 40
GAIN_STARVED_PER_CLASS = 5
GAIN_STARVED_MULT = 2.0
DEFAULT_GAIN_PRIOR = 0.15    # median measured coverage gain of voting on 4 development tasks (optimistic on trec)
DEFAULT_CONTEXTS = 3         # voting contexts (measured on agnews)
COST_BASES = ("usd", "tokens")   # BottleConfig.cost_basis: 'usd' = models.yaml prices; 'tokens' = token volumes


def est_tokens(chars: float, chars_per_token: float | None = None) -> float:
    return chars / (CHARS_PER_TOKEN_EST if chars_per_token is None else chars_per_token)


def _cost(model: ModelSpec, prompt_tokens: float, completion_tokens: float, basis: str) -> float:
    """USD from the model's prices ('usd'), or the raw token volume ('tokens': price 1 per token in and out)."""
    if basis == "tokens":
        return float(prompt_tokens + completion_tokens)
    return model.usd(prompt_tokens, completion_tokens)


def per_item_costs(spec: TaskSpec, cheap: ModelSpec, expert: ModelSpec, avg_chars: float, k: int,
                   warmup: int = 0, basis: str = "usd") -> tuple[float, float]:
    """(cheap batched cost per item, expert single cost per item) in USD (basis 'usd') or tokens (basis 'tokens'),
    from prompt-size estimates."""
    cpt = getattr(spec, "chars_per_token", None)
    header = est_tokens(len(spec.instruction) + len(spec.label_block()) + HEADER_PAD_CHARS, cpt)
    item = est_tokens(min(avg_chars, spec.max_chars), cpt) + ITEM_PAD_TOKENS
    out_tok = spec.max_label_tokens + OUT_PAD_TOKENS
    reply = getattr(expert, "expected_reply_tokens", None)
    reply = (expert.max_tokens or 0) * ECON_REASONING_SHARE if reply is None else reply
    cheap_item = _cost(cheap, header / k + item * (k + warmup) / k, out_tok * (k + warmup) / k, basis)
    expert_item = _cost(expert, header + item, out_tok + reply, basis)
    return cheap_item, expert_item


def unpriced(model: ModelSpec) -> bool:
    """True if the model has no price at all (price_in = price_out = 0, the ModelSpec default, e.g. a local model)."""
    return not (float(getattr(model, "price_in", 0.0) or 0.0) > 0 or float(getattr(model, "price_out", 0.0) or 0.0) > 0)


def gain_prior_for(base: float, n_build: int, n_classes: int, fed: float = GAIN_FED_PER_CLASS,
                   starved: float = GAIN_STARVED_PER_CLASS, mult: float = GAIN_STARVED_MULT) -> float:
    """Voting helps most when the student has few labels per class: prior gain rises from `base` (>= fed labels per
    class, 40) linearly to mult*base (<= starved labels per class, 5; mult 2)."""
    per_class = n_build / max(1, n_classes)
    frac = min(1.0, max(0.0, (fed - per_class) / (fed - starved)))
    return base * (1 + (mult - 1) * frac)          # (mult - 1) * frac == frac exactly at the default mult = 2


def choose_mode(c_b_over_e: float, contexts: int = DEFAULT_CONTEXTS, gain_prior: float = DEFAULT_GAIN_PRIOR,
                N: int | None = None, n_build: int | None = None) -> str:
    if N is None or n_build is None:
        return "fused" if contexts * c_b_over_e < gain_prior else "student"
    extra = (contexts * N - min(n_build, N)) * c_b_over_e
    return "fused" if extra < gain_prior * N else "student"


def expected_cost(mode: str, N: int, c_b: float, n_cal: int, cov: float, contexts: int = DEFAULT_CONTEXTS,
                  n_build: int = N_BUILD_MIN) -> float:
    if mode == "fused":
        return contexts * N * c_b + n_cal + (1 - cov) * (N - n_cal)
    return min(n_build, N) * c_b + n_cal + (1 - cov) * (N - n_cal)


# ------------------------------------------------------------------------------------------------ measured warm/cold
# votes='auto' in pour() compares two PREDICTED costs per certified item, from a
# forecast carried in the bottle (BottleArtifact.forecast) instead of the fixed gain prior:
#   cold : c_exp * (1 - cov_cold)                    (every item the bottle does not bottle costs one expert call)
#   warm : c_vote + c_exp * (1 - cov_warm)           (c_vote = realized cheap-vote USD per item, all contexts)
# and chooses warm iff warm < cold - hysteresis * c_exp. The coverages are bottled shares of the certified population
# measured on an earlier batch (exactly, or by a cross-fitted certified-output student), rescaled to the new batch size
# by the label-free factor below. The choice only selects the NEXT batch's scoring function; it is fixed before that
# batch's calibration draw, whose stream does not depend on it.
DEFAULT_HYSTERESIS = 0.005      # warm iff warm < cold - hysteresis*c_exp (untuned constant)
DEFAULT_GAIN_DECAY = 0.8        # a stale warm gain counts x decay^age (untuned constant)


def rescale_bottled(cov: float, N_old: int, n_old: int, N_new: int, n_new: int) -> float:
    """Label-free N-rescaling of a bottled share: bottled ~= P(score >= tau) * (1 - n/N), so a share measured on a batch
    of N_old items with n_old calibration labels maps to (1 - n_new/N_new) / (1 - n_old/N_old) times itself."""
    a = 1.0 - n_old / max(1, N_old)
    b = 1.0 - n_new / max(1, N_new)
    if a <= 0:
        return float(min(1.0, max(0.0, cov)))
    return float(min(1.0, max(0.0, cov * b / a)))


def measured_votes_pay(cov_cold: float, cov_warm: float, c_vote_item: float, c_exp_call: float,
                       hysteresis: float = DEFAULT_HYSTERESIS) -> tuple[bool, dict]:
    """Warm (cheap votes + bottle) iff its predicted cost per certified item is lower than the cold pour's by more than
    hysteresis * c_exp. Costs in USD per item (any common unit works)."""
    cold = c_exp_call * (1.0 - cov_cold)
    warm = c_vote_item + c_exp_call * (1.0 - cov_warm)
    use = bool(warm < cold - hysteresis * c_exp_call)
    return use, {"cost_cold": cold, "cost_warm": warm, "cov_cold": cov_cold, "cov_warm": cov_warm,
                 "c_vote_item": c_vote_item, "c_exp_call": c_exp_call, "hysteresis": hysteresis * c_exp_call,
                 "vote_over_expert": c_vote_item / c_exp_call if c_exp_call > 0 else float("inf"),
                 "rule": "warm iff c_vote + c_exp*(1-cov_warm) < c_exp*(1-cov_cold) - hysteresis*c_exp"}


# ------------------------------------------------------------------------------------------------ can the votes pay?
# (forecast v2) A warm pour can bottle at most (i) the non-calibration items whose fused answer agrees
# with the expert and (ii) the errors the certificate lets through. With a = vote-expert agreement measured on an
# earlier batch's uniform calibration sample (an ESTIMATE, se ~ sqrt(a(1-a)/n)) and e_new = the planned design's
# label-free worst-case expected error share (k+1)(N-n)/((n+1)N):
#     E[cov_warm] <= cap = (1 - n_new/N_new) * a + e_new
# so even the best warm pour costs c_vote + c_exp*(1 - cap). If that is not below the cold cost c_exp*(1 - cov_cold) by
# at least hysteresis*c_exp, the votes cannot pay and the pour goes cold. This is an estimated bound, not a guarantee
# (a is estimated); it only selects the NEXT batch's scoring function, fixed before that batch's calibration draw.
def votes_can_pay(c_vote: float, c_exp: float, cov_cold: float, a: float, n_new: int, N_new: int, e_new: float,
                  hysteresis: float = DEFAULT_HYSTERESIS) -> tuple[bool, dict]:
    """(can_pay, info). can_pay is False iff c_vote >= c_exp*(cap - cov_cold) + hysteresis*c_exp, i.e. even a warm pour
    that bottles cap (the agreement ceiling) is not cheaper than the cold pour by the hysteresis margin."""
    lab = 1.0 - n_new / max(1, N_new)
    cap = float(min(1.0, max(0.0, max(0.0, lab) * float(a) + float(e_new))))
    margin = c_exp * (cap - cov_cold) + hysteresis * c_exp
    can = bool(c_vote < margin)
    return can, {"cap": cap, "a": float(a), "e_new": float(e_new), "unlabelled_share": lab, "cov_cold": float(cov_cold),
                 "c_vote_item": c_vote, "c_exp_call": c_exp, "cost_cold": c_exp * (1.0 - cov_cold),
                 "cost_warm_best": c_vote + c_exp * (1.0 - cap), "hysteresis": hysteresis * c_exp,
                 "rule": "votes can pay iff c_vote < c_exp*(cap - cov_cold) + hysteresis*c_exp, "
                         "cap = (1 - n_new/N_new)*a + e_new"}
