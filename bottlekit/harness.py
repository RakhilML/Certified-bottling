"""Bottler: certified bottling of a repetitive LLM classification workload.

  1. PLAN      auto mode: vote on every item ('fused') only if the extra cheap batched passes cost less than the
               expected expert calls they save (gain prior rises when the student would be starved of labels).
  2. LABEL     randomized batched labeling (never similarity-sorted), optional warm-up for cold starts,
               position-balanced voting contexts with label-free position-reliability weights.
  3. DISTILL   TF-IDF(+dense) student, cross-fitted so every item is scored by a model that never saw it.
  4. FUSE      fused: answer = weighted vote; score = vote support + student support for that answer.
               student: answer/score = student prediction/confidence.
  5. CERTIFY   expert labels a uniform calibration sample of the WHOLE workload; exact finite-population
               fixed-sequence test with a closed-form threshold. Default design: ONE pre-specified look at the
               full delta of ~1.25*sqrt(N/eps) items (sawtooth-tuned). design='staged' / custom looks: pre-specified
               looks with Bonferroni shares, dominance stop, the look that bottles most is selected.
               RECTIFIED by default: calibration items are answered by the expert, so only the unlabelled bottled
               items can be wrong, and the test bounds exactly those (certificate='hypergeometric' bounds all bad
               items above the threshold, calibration items included: stricter, never better).
               certificate['risk']: the exact worst-case law of the output error count of the rule actually used
               (expected share, full tail, delta spent; report-only), see certify.risk_profile.
  6. ESCALATE  everything below the threshold (and every unusable answer) goes to the expert.
  7. REUSE     the certified output trains a BottleArtifact; `pour` answers the NEXT workload with the artifact
               alone (cold pour: no cheap-LLM calls) or fused with fresh cheap votes (warm pour, chosen by the same
               economics rule), certified on that workload's own calibration sample, and folds the new expert answers
               back in (flywheel: each pour bottles more). With embeddings (emb= or cfg.embedder) the bottle is a
               TF-IDF + dense student and pours score new items with their own embeddings.
  8. RECUR     (cfg.memo=True, opt-in; assumes a DETERMINISTIC expert: identical prompt -> identical answer)
               a pour answers items whose exact expert request is already in the bottle with an EXPERT answer
               (same prompt fingerprint) from memory ('memo'), decided from the texts and the bottle only, before any
               vote or calibration draw; only the remaining population R is scored, calibrated and certified (explicit
               allowance m). Within a batch the expert is called once per unique prompt, and calibration answers are
               propagated to non-calibration copies of the same prompt after the threshold is fixed.
  9. HEALTH    (cfg.health=True, report-only) after everything above is fixed: agreement of each answer source
               with the expert on the calibration sample (exact intervals; unbiased for fixed-size designs), a valid
               lower bound on the bottled answers' precision from the batch's own law, label-free drift statistics
               (pours) in diagnostics['health'], and one learning-curve row in the exported bottle's `history`.
Returns answers + per-item provenance + certificate + per-run cost ledger + diagnostics (+ artifact).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import pickle
import time
import warnings
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import hypergeom as _hypergeom

from .certify import (BATCH_SPEND_MAX_ALPHA, DEFAULT_DESIGN_C, DEFAULT_MIN_TAX, MAX_MIN_TAX, NONE, STREAM_DELTAS, Certificate, RiskTooLarge,
                      StreamLaw, batch_spent_alpha, batch_tail, report_deltas,
                      certified_eps, certify,
                      certify_sequential, default_design, disagreement, expected_eps_closed_form, inversion_profile,
                      k_used, min_batch_for_tax, risk_profile, sequential_certified_eps, single_look_design,
                      single_look_size, spend, tune_size, tuned_looks)
from .certify import certify as _fc_certify                        # forecast-only aliases (the same functions,
from .certify import certify_sequential as _fc_certify_sequential  # not counted as the pipeline's own draws)
from .economics import (COST_BASES, DEFAULT_GAIN_DECAY, DEFAULT_GAIN_PRIOR, DEFAULT_HYSTERESIS, GAIN_FED_PER_CLASS,
                        GAIN_STARVED_MULT, GAIN_STARVED_PER_CLASS, N_BUILD_MIN, N_BUILD_PER_CLASS, choose_mode,
                        gain_prior_for, measured_votes_pay, per_item_costs, rescale_bottled, unpriced, votes_can_pay)
from .labeling import (CAL_STREAM_KEY, CROSSFIT_STREAM_KEY, EARLY_FRAC, label_batched, random_order, single_max_tokens,
                       sub_seed, vote_label)
from .labeling import FORECAST_STREAM_KEY, HEALTH_STREAM_KEY, PARTITION_STREAM_KEY  # noqa: F401  (stream registry: labeling.STREAM_KEYS)
from .students import CROSS_FIT_FOLDS, DEFAULT_C, DRIFT_LOW_PERCENTILE, TfidfLR, check_student, cross_fit, drift_stats
from .tasks import EXPERT_SIDE_EXTRAS, PARSER_VERSION, TaskSpec, parse_label, single_messages, spec_extras, spec_to_plain

EXPERT_TEMPERATURE = 0.0   # every expert request uses it; part of the memo fingerprint

MODES = ("auto", "fused", "student")
CERTIFICATES = ("rectified", "hypergeometric")   # finite-population tests for the realized workload (never 'binomial')
DESIGNS = ("single", "staged")                    # default calibration design when cfg.looks is None
MEMO_ALLOWANCES = ("novel", "batch")             # allowance m for the certified population R of a memo pour
CHUNKINGS = ("cold", "auto", "off")              # right-sized pours: split a large pour into chained chunks
CHUNK_COLD_RATIO = 4.0                           # 'cold' rule: split only cold pours with N >= 4 x memory (fixed
                                                 # in advance; never tuned on the test data)
FORECAST_V2 = ("off", "canpay", "pour1", "full")  # votes='auto' forecast v2 parts (see BottleConfig.forecast_v2)
# FORECAST_STREAM_KEY (7: every forecast cross-fit, (seed, 7, bottle version)) and PARTITION_STREAM_KEY (13: a chunked
# pour's partition) are imported from labeling's stream registry (1 cal, 2 cross-fit, 7 forecast, 11 contexts, 13).
XFIT_MAX_FOLDS = CROSS_FIT_FOLDS                 # forecast cross-fit folds when the work budget allows (5)
XFIT_BUDGET = 2                                  # forecast_folds='auto': training rows K*M - R < 2 bottle fits (M rows)
DEFAULT_MAX_MEMORY = 200_000                     # bottle memory cap (newest rows kept); eviction disables the next-
                                                 # bottle cross-fit for evicted batches (it records why)
PHASE_CHEAP_VOTE, PHASE_CHEAP_BUILD = "cheap_vote", "cheap_build"            # ledger phases
PHASE_EXPERT_CAL, PHASE_EXPERT_ESC = "expert_calibration", "expert_escalation"
EXPERT_PHASE_PREFIX = "expert"                   # every expert ledger phase starts with this (cost accounting)
CHEAP_WARN_SHARE = 0.5                           # warn when the cheap model labels fewer than half of the items
COLD_START_GAP = 0.04                            # diagnostic: cold start if reliability bin 0 < mean(others) - gap
MEMO_WARN_RATIO = 2.0                            # memo_allowance='batch': warn if m/N_R > ratio * eps
MEAN_CAP_SLACK = 1e-12                           # float slack of the mean-risk cap check (report only)
V09 = {"since": "0.9"}                           # metadata of optional BottleConfig fields (see config_hash)
HEALTH_LEVEL = 0.95                              # default of BottleConfig.health_level (and of cp_interval /
                                                 # finite_interval): the conventional two-sided level of a REPORT
                                                 # interval, deliberately not tied to delta (see BottleConfig)
HISTORY_CAP = 500                                # default of BottleConfig.history_cap (rows kept in a bottle's history)
HEALTH_MAX_ROWS = 20_000                         # default of BottleConfig.health_max_rows (drift-statistics subsample)
HISTORY_DIGITS = 6                               # decimals kept for the floats of a learning-curve row (storage and
                                                 # display precision of a report-only row; diagnostics keep full floats)
CHUNK_SPENDS = ("chunk", "batch")                # chunked pours: each chunk at (eps, delta) ('chunk') or the
                                                 # batch's delta spent across the chunks ('batch')
EXPERT_PROVENANCE = ("expert_calibration", "expert_escalation", "expert_calibration_dup", "memo")
SRC_EXPERT, SRC_BOTTLE = "expert", "bottle"
F16_MAX = float(np.finfo(np.float16).max)      # bottles store embeddings as float16 (65504)


def check_emb_values(emb, what: str = "emb", store: bool = True) -> None:
    """Fail BEFORE any LLM call on embeddings the pipeline cannot use: NaN/inf (the student cannot fit them) or, when
    a bottle will store them (store=True), values beyond the float16 range (the export would fail after every expert
    call of the batch was paid for). Valid embeddings pass unchanged."""
    a = np.asarray(emb)
    if a.dtype.kind not in "fiu":
        a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{what} contains NaN or infinite values")
    if store and a.size and float(np.abs(a).max()) > F16_MAX:
        raise ValueError(f"{what} has values up to {float(np.abs(a).max()):.4g}; bottles store embeddings as float16 "
                         f"(max {F16_MAX:.0f}): L2-normalise the rows")


def expert_key(spec: TaskSpec, text: str) -> str:
    """The exact user content of the expert request for `text` (what the expert actually sees besides the
    fingerprinted system prompt). Memo lookups and within-batch dedupe use this key, never a looser normalisation."""
    return single_messages(spec, text)[-1]["content"]


def src_of(provenance: str) -> str:
    """'expert' iff the final answer is the expert's own answer for this exact prompt (calibration, escalation,
    propagated calibration duplicate, or memo); '+fallback' answers and bottle answers are 'bottle'."""
    return SRC_EXPERT if provenance in EXPERT_PROVENANCE else SRC_BOTTLE


def cal_stream_key(seed: int, artifact=None) -> tuple:
    """sub_seed key of a batch's calibration draw: (seed, 1) for run() (artifact None), (seed, 1, version + 1,
    |memory|) for a pour of `artifact`. One definition for Bottler and BottleSession's lineage guard."""
    if artifact is None:
        return (int(seed), CAL_STREAM_KEY)
    return (int(seed), CAL_STREAM_KEY, int(artifact.version) + 1, len(artifact.texts))


class ExpertFailure(RuntimeError):
    """The expert produced no usable answer for too many calibration items (outage, wrong model, bad parser)."""


class UnknownModelError(KeyError, ValueError):
    """A BottleConfig model alias (cheap / expert) that the client's models file does not define."""

    def __str__(self):
        return str(self.args[0]) if self.args else ""


_CHUNK_TOKEN = object()     # only _pour_chunked can build a _ChunkDesign (pour() rejects any other object)


@dataclass(frozen=True)
class _ChunkDesign:
    """Planned design of one chunk of a batch-spent chunked pour (cfg.chunk_spend='batch'): one look of n items at
    the common chunk level alpha (>= delta), from the label-free plan. Built by _pour_chunked only; pour() passes it to
    _certify_and_escalate, which certifies the chunk with looks=[n] (no re-tuning) at level alpha, so its closed-form
    rule applies k_used(n, N, eps, alpha) = the planned k (checked after the chunk against the plan)."""
    n: int
    N: int
    k: int
    alpha: float
    delta_batch: float
    j: int
    J: int
    token: object = field(default=None, repr=False, compare=False)


@dataclass
class BottleConfig:
    cheap: str | None
    expert: str
    mode: str = "auto"                 # 'fused' | 'student' | 'auto'
    k: int = 20                        # real items per cheap call
    contexts: int = 3                  # voting contexts (fused mode)
    warmup: int = 0                    # throwaway items per call (cold-start fix for weak labelers)
    balanced: bool = True              # position-balanced contexts
    weighted: bool = True              # label-free position-reliability weights
    fusion: str = "prob"               # fused score: 'prob' = vote + P_student(vote) ; 'agree' = vote + [student top == vote]*conf
    n_build: int | None = None         # student mode: items labelled by the cheap model (None: max(600, 20*classes))
    eps: float = 0.05
    delta: float = 0.1
    calibration: str = "sequential"    # 'sequential' | 'fixed'
    certificate: str = "rectified"     # 'rectified' (bound the errors that reach the output: unlabelled bottled items)
                                       # | 'hypergeometric' (bound every bad item above tau, calibration included)
    n_cal: int = 300                   # fixed calibration size
    looks: tuple | None = None         # explicit sequential looks (None: the `design` below)
    spending: tuple | None = None      # Bonferroni weights per look for custom looks (None: uniform)
    design: str = "single"             # looks=None: 'single' = ONE full-delta look of round(design_c*sqrt(N/eps)) items
                                       # (<= N/2) | 'staged' = default_design(N): (100, 300, 900, ...) weights 1:4:4
    design_c: float = DEFAULT_DESIGN_C  # size constant of the single look (1.25, fixed in advance; label-free)
    tune: bool = True                  # sawtooth-tuned calibration sizes (exact tests are discrete)
    patience: int | None = None        # optional heuristic stop; the dominance stop is always on
    target_coverage: float | None = None
    dense: bool = False                # unused (kept for backward compatibility): pass emb= or set `embedder`
    gain_prior: float = DEFAULT_GAIN_PRIOR  # expected coverage gain of voting (auto mode): 0.15 = the median measured
                                       # on 4 development tasks; the measured forecast replaces it from pour 2 on
    expert_budget: int | None = None   # max escalation calls; if exceeded, escalate the least certain items only and
                                       # report the certified eps actually achieved (test inversion). When set, the
                                       # k-rule runs at delta/2 and the inversion at delta/2 (Bonferroni), so
                                       # the label-dependent switch between them stays within delta
    max_expected_eps: float | None = None  # optional, STRICTER only: also cap the worst-case expected share of
                                       # unlabelled bottled answers != expert, by lowering each look's k to
                                       # k_mean_cap(n, N, cap * delta share) (label-free). None: unchanged (default)
    max_expert_none: float = 0.2       # abort (ExpertFailure) if more calibration answers than this are unusable
    fallback: bool = True              # if the expert gives no usable answer, keep the bottle's answer (flagged)
    export: bool = True                # build a reusable BottleArtifact from the certified output
    seed: int = 0
    embedder: object = None            # optional callable list[str] -> (n, d) array (sync or async): a FIXED,
                                       # label-free text encoder, called once per batch when emb= is not given, before
                                       # any calibration draw. Never tune or fine-tune it on the batch's expert labels.
    embedder_id: str | None = None     # name of the embedder (recorded in artifacts; a mismatch is detected on pour)
    memo: bool = False                 # recurrence-aware bottling (opt-in). ASSUMES A DETERMINISTIC EXPERT for
                                       # identical prompts (temperature 0, same model; a cache_path makes it literal).
                                       # On: pours answer recurring items that the bottle holds with an expert answer
                                       # under the same prompt fingerprint from memory and certify only the rest; within
                                       # a batch the expert is called once per unique prompt and calibration answers
                                       # reach identical prompts. Inert without recurrence. False: no memo.
    memo_allowance: str = "novel"      # allowance m for the certified population R of a memo pour: 'novel' =
                                       # floor(eps*N_R) (R alone meets eps; the batch share is <= eps*N_R/N) |
                                       # 'batch' = floor(eps*N) (the whole batch's allowance moves onto R: cheaper,
                                       # but up to floor(eps*N)/N_R of the novel items may disagree)
    forecast: bool = True              # after each certified batch, record a warm/cold forecast in the exported
                                       # bottle (measured: exact cold counterfactual of warm pours, realized coverage of
                                       # cold pours) and use it in pour(votes='auto'). Report-only for the batch that
                                       # produces it (outputs are identical on/off). False: no forecast; votes='auto'
                                       # uses the gain-prior rule.
    forecast_run: bool = False         # (EXPERIMENTAL, opt-in) also record a cross-fitted certified-output forecast at run()
                                       # (lazily resolved by the first votes='auto' decision) and use it for the first
                                       # pour and for the stale gain after a cold first pour. Off by default: on the
                                       # 8-seed default-mode replay its warm arm was biased low (-0.066 bottled share)
                                       # and it chose cold where warm won (emotion +2.2, banking77 +3.3 pts vs the prior
                                       # rule). Off: the first pour from a run() bottle uses the gain-prior rule.
    chunking: str = "cold"             # right-sized pours: split a pour much larger than the bottle's memory into
                                       # label-free geometric chunks certified in sequence, each chunk's certified output
                                       # updating the bottle before the next (intra-pour flywheel); the whole batch keeps
                                       # P[share != expert > eps] <= delta_batch <= delta (exact convolution of the
                                       # per-chunk worst-case laws). Only with certificate='rectified', design='single',
                                       # calibration='sequential', no custom looks, no expert_budget, memo=False,
                                       # export=True; otherwise (and whenever the plan has one chunk) exactly the single
                                       # pour.
                                       #  'cold' (default): the pour's warm/cold choice is made ONCE, before planning (votes
                                       #    False -> cold, True -> warm, 'auto' -> the votes='auto' rule); only a COLD
                                       #    pour with N >= chunk_cold_ratio x memory is split, and every chunk is cold.
                                       #    Warm pours and smaller pours are exactly the single pour.
                                       #  'auto' (opt-in): every eligible pour is planned; each chunk decides its
                                       #    own votes. Lost on warm and 2x pours (agnews warm -4 to -7 pt). N_min is
                                       #    the exact 355 (eps .05).
                                       #  'off': never split.
    chunk_cold_ratio: float = CHUNK_COLD_RATIO  # 'cold' rule threshold N >= ratio x memory (4.0, fixed in advance)
    chunk_tax: float = DEFAULT_MIN_TAX  # N_min = smallest batch whose single-look calibration share is <= chunk_tax
                                       # (certify.min_batch_for_tax = min_certified_size; 355 at eps=.05, delta=.1);
                                       # in (0, 0.45]: no chunk is smaller
    chunk_growth: float = 1.0          # chunk j = max(N_min, round(chunk_growth * planned memory before chunk j))
    max_chunks: int | None = None      # optional cap on the number of chunks (None: no cap)
    forecast_v2: str = "full"          # votes='auto' forecast v2 (needs forecast=True; inputs are recorded in every
                                       # bottle whatever this is; only the decision reads it):
                                       #  'off'    the measured-forecast decision without (1)-(4)
                                       #  'canpay' + (1) vote-expert agreement a (measured on each voted batch's uniform
                                       #           calibration sample, carried through cold pours with its age) and
                                       #           (2) the can-pay ceiling: warm can bottle at most (1-n/N)*a + e_new, so
                                       #           go cold when even that cannot pay (economics.votes_can_pay); warm-arm
                                       #           forecasts are clipped at that ceiling
                                       #  'pour1'  + (3) the run-time cross-fitted COLD arm for the first pour (warm arm
                                       #           still behind forecast_run)
                                       #  'full'   + (4) the next-bottle cold arm for pours: a cross-fit of the batch's rows
                                       #           in the NEW bottle's memory (other memory rows always in training)
    forecast_folds: int | str = "auto" # folds of the forecast cross-fits (3)/(4): 'auto' = the largest K in [2, 5] whose
                                       # training rows K*M - N_rows stay STRICTLY below 2 bottle fits of the memory M,
                                       # i.e. K = 2 (a batch's rows are part of the memory: N_rows <= M) | an int >= 2
    # ------------------------------------------------------------------------------------------- generality knobs
    # Generality knobs. Every default gives the standard outputs; a field still at its
    # default is left out of session.config_hash (and bottles/fingerprints carry no trace of it), so checkpoints
    # and bottles made with the defaults resume and load unchanged. Validity: every knob is label-free, so ANY value fixed before the batch
    # keeps the guarantee; choosing a value from this batch's calibration labels or outcomes does not. "tuned" below
    # says whether the default was chosen on the 4 development tasks (agnews, emotion, esci, banking77).
    student: dict | None = field(default=None, metadata=V09)
    #   TfidfLR settings of the run() student and of new bottles (students.STUDENT_KEYS): C (4.0), char (True),
    #   dense_weight (1.0), word_ngrams ((1, 2)), char_ngrams ((3, 5)), char_analyzer ('char_wb'), token_pattern,
    #   coef_budget (8e6), feature_clip ((5000, 400000)), word_div (3), min_df ('auto'), max_iter (2000). None: these defaults.
    #   Chosen once on the development tasks. Valid for any fixed value (run() still cross-fits).
    crossfit_folds: int = field(default=CROSS_FIT_FOLDS, metadata=V09)
    #   folds of run()'s cross-fit (and of the forecast_run proxy). 5; not tuned. Any K >= 2 is valid; K = 1 is not
    #   allowed (the student would score its own training items).
    max_memory: int = field(default=DEFAULT_MAX_MEMORY, metadata=V09)
    #   memory cap of bottles this Bottler exports (newest rows kept). 200,000; not tuned (RAM). No validity effect.
    #   Non-default values replace the bottle's own cap on export; the default keeps the bottle's stored cap.
    n_build_min: int = field(default=N_BUILD_MIN, metadata=V09)
    n_build_per_class: int = field(default=N_BUILD_PER_CLASS, metadata=V09)
    #   student mode: n_build = max(n_build_min, n_build_per_class * classes) when cfg.n_build is None. 600 (development
    #   tasks) and 20 (clinc150). No validity effect.
    gain_fed_per_class: float = field(default=GAIN_FED_PER_CLASS, metadata=V09)
    gain_starved_per_class: float = field(default=GAIN_STARVED_PER_CLASS, metadata=V09)
    gain_starved_mult: float = field(default=GAIN_STARVED_MULT, metadata=V09)
    #   auto mode: the gain prior rises from gain_prior (>= fed labels per class, 40) linearly to mult x gain_prior
    #   (<= starved labels per class, 5; mult 2). Measured on clinc150 only. No validity effect.
    early_frac: float = field(default=EARLY_FRAC, metadata=V09)
    #   cold-start zone share of a cheap batch (position-balanced contexts) and reliability bins round(1/early_frac).
    #   0.25 (a 4B model on agnews, a development task). No validity effect (cheap labels only).
    forecast_hysteresis: float = field(default=DEFAULT_HYSTERESIS, metadata=V09)
    #   votes='auto': warm iff its predicted cost < cold - hysteresis * c_exp. 0.005; untuned constant.
    forecast_gain_decay: float = field(default=DEFAULT_GAIN_DECAY, metadata=V09)
    #   a stale warm gain counts x decay^age. 0.8; untuned constant.
    forecast_fit_budget: float = field(default=XFIT_BUDGET, metadata=V09)
    #   forecast_folds='auto': training rows K*M - R stay below budget x one bottle fit. 2; set from the measured
    #   overhead, not from outcomes.
    forecast_max_folds: int = field(default=XFIT_MAX_FOLDS, metadata=V09)
    #   upper bound of the 'auto' forecast folds. 5; not tuned.
    forecast_max_rows: int | None = field(default=None, metadata=V09)
    #   cap on the PRE-batch memory rows a forecast cross-fit trains on (a uniform subsample from the key-7 stream);
    #   None: all rows (the cost grows with the memory). Decision only: no validity effect.
    forecast_canpay_z: float = field(default=0.0, metadata=V09)
    #   can-pay ceiling: uses a + z*se(a) instead of the point estimate a (z > 0 is conservative toward warm). 0.0.
    cost_basis: str = field(default="usd", metadata=V09)
    #   'usd': decisions from models.yaml prices (a model with no price makes mode='auto' choose student and
    #   votes='auto' go cold; bottlekit warns). 'tokens': token volumes (price 1 per token in and out). No validity
    #   effect (it only picks the scoring function before the draw).
    # ---- health report (report-only: never read by a decision; REPORT_ONLY_FIELDS, so never in config_hash)
    health: bool = field(default=True, metadata=V09)
    #   after each certified batch: agreement with the expert on the calibration sample (exact intervals), a valid lower
    #   bound on the precision of the bottled answers, label-free drift statistics (pours) and one learning-curve row
    #   appended to the exported bottle's `history`. Computed after tau, the answers, the escalation and the export are
    #   fixed; nothing feeds any decision. False: no health block and no new history row (outputs identical either way).
    health_level: float = field(default=HEALTH_LEVEL, metadata=V09)
    #   two-sided level of the agreement intervals. 0.95, the conventional level of a descriptive interval; deliberately
    #   NOT 1 - delta, so it is never read as part of the certificate.
    history_cap: int = field(default=HISTORY_CAP, metadata=V09)
    #   rows kept in BottleArtifact.history (oldest dropped). A storage bound only (a row is a few hundred bytes).
    health_max_rows: int | None = field(default=HEALTH_MAX_ROWS, metadata=V09)
    #   drift statistics read at most this many new items and memory rows (a seeded, label-free subsample above it;
    #   None = all). A compute bound for the nearest-memory cosine (O(rows x memory)).
    health_low_percentile: float = field(default=DRIFT_LOW_PERCENTILE, metadata=V09)
    #   the low-tail percentile the drift statistics report (conf_pXX / cos_pXX). Part of the statistic's definition;
    #   never compared with a threshold.
    # ---- batch-spent chunk plans
    chunk_spend: str = field(default="chunk", metadata=V09)
    #   how a chunked pour spends delta. 'chunk' (default): every chunk is certified at (eps, delta), so the whole batch's
    #   tail delta_batch is only ~0.01-0.05 and most of delta goes unspent. 'batch': every chunk's k is raised to a
    #   common per-chunk level alpha* >= delta (same calibration sizes), chosen label-free by exact search so that the
    #   whole batch still has P[share != expert > eps] <= delta_batch <= delta (certify.batch_spent_alpha); each chunk
    #   then holds at (eps, alpha*) on its own, and its certificate says so. Opt-in; validity holds for both values.
    chunk_alpha_max: float = field(default=BATCH_SPEND_MAX_ALPHA, metadata=V09)
    #   chunk_spend='batch': ceiling on the common per-chunk level alpha*. 0.5: above it a chunk's own claim is no
    #   better than a coin flip. A value <= delta raises nothing (the 'chunk' plan). Never binds the whole-batch bound.

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.calibration not in ("sequential", "fixed"):
            raise ValueError("calibration must be 'sequential' or 'fixed'")
        if self.certificate not in CERTIFICATES:
            raise ValueError(f"certificate must be one of {CERTIFICATES}")
        if self.fusion not in ("agree", "prob"):
            raise ValueError("fusion must be 'agree' or 'prob'")
        if self.design not in DESIGNS:
            raise ValueError(f"design must be one of {DESIGNS}")
        if not self.design_c > 0:
            raise ValueError("design_c must be > 0")
        if not (0 < self.eps < 1 and 0 < self.delta < 1):
            raise ValueError("need 0 < eps < 1 and 0 < delta < 1")
        if self.expert_budget is not None and int(self.expert_budget) < 0:
            raise ValueError("expert_budget must be >= 0 (or None); a negative budget indexed the score order from "
                             "the top and escalated almost every item")
        if self.calibration == "fixed" and int(self.n_cal) < 1:
            raise ValueError("n_cal must be >= 1 with calibration='fixed'")
        if self.k < 1 or self.contexts < 1:
            raise ValueError("k and contexts must be >= 1")
        if self.max_expected_eps is not None and not self.max_expected_eps > 0:
            raise ValueError("max_expected_eps must be > 0 (or None)")
        if self.memo_allowance not in MEMO_ALLOWANCES:
            raise ValueError(f"memo_allowance must be one of {MEMO_ALLOWANCES}")
        if self.chunking not in CHUNKINGS:
            raise ValueError(f"chunking must be one of {CHUNKINGS}")
        if not 0 < self.chunk_tax <= MAX_MIN_TAX:
            raise ValueError(f"chunk_tax must be in (0, {MAX_MIN_TAX}] (the domain of certify.min_certified_size)")
        if not self.chunk_cold_ratio > 0:
            raise ValueError("chunk_cold_ratio must be > 0")
        if not self.chunk_growth > 0:
            raise ValueError("chunk_growth must be > 0")
        if self.max_chunks is not None and int(self.max_chunks) < 1:
            raise ValueError("max_chunks must be >= 1 (or None)")
        if self.forecast_v2 not in FORECAST_V2:
            raise ValueError(f"forecast_v2 must be one of {FORECAST_V2}")
        if not (self.forecast_folds == "auto" or (isinstance(self.forecast_folds, int)
                                                   and not isinstance(self.forecast_folds, bool)
                                                   and self.forecast_folds >= 2)):
            raise ValueError("forecast_folds must be 'auto' or an int >= 2")
        check_student(self.student)
        if int(self.crossfit_folds) < 2:
            raise ValueError("crossfit_folds must be >= 2 (cross-fitting is required for validity)")
        if int(self.max_memory) < 1 or int(self.n_build_min) < 1 or int(self.n_build_per_class) < 0:
            raise ValueError("max_memory and n_build_min must be >= 1, n_build_per_class >= 0")
        if not self.gain_fed_per_class > self.gain_starved_per_class >= 0 or not self.gain_starved_mult >= 1:
            raise ValueError("need gain_fed_per_class > gain_starved_per_class >= 0 and gain_starved_mult >= 1")
        if not 0 < self.early_frac <= 1:
            raise ValueError("early_frac must be in (0, 1]")
        if not (self.forecast_hysteresis >= 0 and 0 <= self.forecast_gain_decay <= 1 and self.forecast_fit_budget > 0
                and int(self.forecast_max_folds) >= 2 and self.forecast_canpay_z >= 0):
            raise ValueError("need forecast_hysteresis >= 0, 0 <= forecast_gain_decay <= 1, forecast_fit_budget > 0, "
                             "forecast_max_folds >= 2, forecast_canpay_z >= 0")
        if self.forecast_max_rows is not None and int(self.forecast_max_rows) < 1:
            raise ValueError("forecast_max_rows must be >= 1 (or None)")
        if self.cost_basis not in COST_BASES:
            raise ValueError(f"cost_basis must be one of {COST_BASES}")
        if not 0 < float(self.health_level) < 1:
            raise ValueError("health_level must be in (0, 1)")
        if int(self.history_cap) < 1:
            raise ValueError("history_cap must be >= 1")
        if self.health_max_rows is not None and int(self.health_max_rows) < 1:
            raise ValueError("health_max_rows must be >= 1 (or None)")
        if not 0 < float(self.health_low_percentile) < 100:
            raise ValueError("health_low_percentile must be in (0, 100)")
        if self.chunk_spend not in CHUNK_SPENDS:
            raise ValueError(f"chunk_spend must be one of {CHUNK_SPENDS}")
        if not 0 < self.chunk_alpha_max < 1:
            raise ValueError("chunk_alpha_max must be in (0, 1)")


@dataclass
class BottleResult:
    answers: list
    provenance: list
    certificate: dict
    ledger: dict
    diagnostics: dict = field(default_factory=dict)
    artifact: "BottleArtifact | None" = field(default=None, repr=False)

    def coverage(self) -> float:
        return sum(p == "bottle" for p in self.provenance) / max(1, len(self.provenance))

    def memo_share(self) -> float:
        """Share of items answered from the bottle's expert memory (provenance 'memo'; not counted in coverage())."""
        return sum(p == "memo" for p in self.provenance) / max(1, len(self.provenance))

    def to_json(self) -> str:
        d = {k: getattr(self, k) for k in ("answers", "provenance", "certificate", "ledger", "diagnostics")}
        return json.dumps(d, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


class BottleArtifact:
    """A reusable bottle: a student distilled from CERTIFIED outputs (bottle answers + expert answers).
    It keeps its training memory (texts, labels[, embeddings]) so it can be refitted, updated with new expert answers,
    and saved without pickling sklearn internals. Each reuse is certified on the new workload itself (see Bottler.pour).

    Dense bottles: with `emb` (one row per text: a fixed, label-free embedding of that text) the student is TF-IDF +
    dense columns, and predict(texts, emb=...) scores new items with their own embeddings. `emb` stays row-aligned
    with the memory through filtering, dedupe (newest wins) and the max_memory cap. It is stored as float16 and the
    student is fitted on the stored values, so a saved and reloaded bottle predicts exactly like the original.
    Without embeddings for the new items, predict falls back to a TF-IDF-only student of the same memory, i.e. exactly
    the bottle a TF-IDF artifact would be (see dense_status).

    Provenance: `src[j]` is 'expert' iff memory label j is the expert's own answer for that text (calibration,
    escalation, propagated duplicate or memo; never '+fallback' or a bottle answer), and `fingerprint` identifies the
    expert request (system prompt, label block, max_chars, model identity) under which those answers were obtained.
    src follows its label through dedupe (newest wins) and the cap; old pickles load with src='bottle' and no
    fingerprint, which turns the memo off. Neither field changes the student.

    History (report-only): `history` is the bottle's learning curve, a list of small plain dicts, one per
    certified batch of its lineage (appended by Bottler when cfg.health is on, which then keeps the last
    cfg.history_cap rows, default 500). It is carried through updated(), with_embeddings() and save/load (older
    pickles load with []). Nothing in bottlekit reads it for a decision: not predict(), not pour(votes='auto'), not
    the chunk plan, not the session's lineage digest."""

    def __init__(self, spec: TaskSpec, texts=(), labels=(), C: float = DEFAULT_C, max_memory: int = DEFAULT_MAX_MEMORY,
                 version: int = 0, emb: np.ndarray | None = None, embedder_id: str | None = None, src=None,
                 fingerprint: str | None = None, forecast: dict | None = None, student: dict | None = None,
                 history: list | None = None):
        self.spec, self.C, self.max_memory, self.version = spec, C, max_memory, version
        # TfidfLR settings beyond C (BottleConfig.student without 'C'); None: the default student
        st = check_student(student)
        if "C" in st:
            self.C = C = st.pop("C")
        self.student_cfg = st or None
        # warm/cold forecast for the NEXT pour (plain dict; written by Bottler after certification, read only by
        # pour(votes='auto'); predict() never reads it). None: no forecast (older pickles, hand-built bottles).
        self.forecast = forecast
        # learning curve, report-only: kept as given (plain dicts); the Bottler trims it to cfg.history_cap
        self.history = [dict(r) for r in (history or []) if isinstance(r, dict)]
        texts, labels = list(texts), list(labels)
        src = [SRC_BOTTLE] * len(texts) if src is None else list(src)
        if len(src) != len(texts):
            raise ValueError(f"src has {len(src)} entries; expected one per memory text ({len(texts)})")
        if any(x not in (SRC_EXPERT, SRC_BOTTLE) for x in src):
            raise ValueError(f"src entries must be {SRC_EXPERT!r} or {SRC_BOTTLE!r}")
        if emb is not None:
            emb = np.asarray(emb)
            if emb.ndim != 2 or emb.shape[0] != len(texts):
                raise ValueError(f"emb has shape {emb.shape}; expected ({len(texts)}, d): one row per memory text")
        keep = [(t, l, i) for i, (t, l) in enumerate(zip(texts, labels)) if l is not None and l != NONE]
        keep = list({t: (t, l, i) for t, l, i in keep}.values())[-max_memory:]   # dedupe by text, newest wins
        self.texts, self.labels = [t for t, _, _ in keep], [l for _, l, _ in keep]
        self.src = [src[i] for _, _, i in keep]                                   # aligned with the newest label
        self.fingerprint = fingerprint
        self.emb, self.embedder_id = None, None
        if emb is not None:
            e = emb[np.asarray([i for _, _, i in keep], dtype=np.int64)].astype(np.float16)
            if not np.isfinite(e).all():
                raise ValueError("emb contains non-finite values (after float16 storage)")
            self.emb, self.embedder_id = e, embedder_id
        self._tfidf = None
        self.student = (self._new_student(None if self.emb is None else self.emb.astype(np.float32))
                        .fit(self.texts, self.labels, list(range(len(self.texts))))) if self.texts else None

    def _new_student(self, dense=None) -> TfidfLR:
        """An unfitted student of this bottle's class (C, student settings, optional dense columns)."""
        return TfidfLR(C=self.C, dense=dense, **(self.student_cfg or {}))

    def student_kwargs(self) -> dict:
        """TfidfLR settings of this bottle (C and any student settings), e.g. for cross_fit(student=...)."""
        return {"C": self.C, **(self.student_cfg or {})}

    # ------------------------------------------------------------------ dense helpers
    @property
    def dense(self) -> bool:
        """True if the bottle carries memory embeddings (TF-IDF + dense student)."""
        return self.emb is not None

    @property
    def emb_dim(self) -> int | None:
        return None if self.emb is None else int(self.emb.shape[1])

    def check_emb(self, emb, n: int, embedder_id: str | None = None) -> np.ndarray:
        """Validate embeddings of n new items against this bottle (rows; width and embedder id if the bottle is
        dense). Raises ValueError on a mismatch."""
        emb = np.asarray(emb)
        if emb.ndim != 2 or emb.shape[0] != n:
            raise ValueError(f"emb has shape {emb.shape}; expected ({n}, d): one embedding row per text")
        if self.emb is not None:
            if emb.shape[1] != self.emb_dim:
                raise ValueError(f"emb width {emb.shape[1]} != bottle embedding width {self.emb_dim} (bottle embedder "
                                 f"{self.embedder_id!r}); re-embed the memory with with_embeddings() or set "
                                 "BottleConfig.embedder")
            if embedder_id is not None and self.embedder_id is not None and embedder_id != self.embedder_id:
                raise ValueError(f"embedder mismatch: batch {embedder_id!r} vs bottle {self.embedder_id!r}; re-embed "
                                 "the memory with with_embeddings() or set BottleConfig.embedder")
        return emb

    def dense_status(self, emb) -> str:
        """'dense' (bottle and new items both have embeddings), 'dense_missing' (only one side has them: the TF-IDF
        fallback answers), or 'off' (no embeddings anywhere)."""
        if self.emb is not None and emb is not None:
            return "dense"
        return "dense_missing" if (self.emb is not None or emb is not None) else "off"

    def _tfidf_student(self):
        """TF-IDF-only student of the same memory (fitted lazily): the fallback when new items have no embeddings."""
        if self.emb is None:
            return self.student
        if self._tfidf is None and self.texts:
            self._tfidf = self._new_student().fit(self.texts, self.labels, list(range(len(self.texts))))
        return self._tfidf

    def with_embeddings(self, emb, embedder_id: str | None = None) -> "BottleArtifact":
        """Same memory, provenance and version with new memory embeddings (upgrade a TF-IDF bottle, or switch
        embedder); emb=None gives the TF-IDF-only bottle."""
        return BottleArtifact(self.spec, self.texts, self.labels, self.C, self.max_memory, self.version, emb, embedder_id,
                              self.src, self.fingerprint, copy.deepcopy(self.forecast), self.student_cfg,
                              copy.deepcopy(self.history))

    # ------------------------------------------------------------------ expert memory (recurrence)
    def memo_table(self, spec: TaskSpec | None = None) -> tuple[dict, int]:
        """{expert request key: expert answer} over memory items with src='expert' (key = expert_key(spec, text)).
        Keys whose stored expert answers disagree (evidence of a non-deterministic expert, or of distinct texts that
        map to one prompt with different answers) are dropped. Returns (table, number of conflicting keys dropped).
        Only meaningful under a matching fingerprint (see Bottler.pour)."""
        spec = spec or self.spec
        tab, conflict = {}, set()
        for t, l, s in zip(self.texts, self.labels, self.src):
            if s != SRC_EXPERT:
                continue
            k = expert_key(spec, t)
            if k in tab and tab[k] != l:
                conflict.add(k)
            tab.setdefault(k, l)
        for k in conflict:
            del tab[k]
        return tab, len(conflict)

    # ------------------------------------------------------------------ use
    def predict(self, texts: list[str], query: list | None = None, emb: np.ndarray | None = None):
        """(top label, its probability[, probability of query[i]]) for each text. `emb`: embeddings of THESE texts;
        used only if the bottle has memory embeddings, otherwise the TF-IDF student answers (see dense_status)."""
        if emb is not None:
            emb = self.check_emb(emb, len(texts))
        if self.student is None:
            out = [None] * len(texts), np.zeros(len(texts))
            return (*out, np.zeros(len(texts))) if query is not None else out
        if emb is not None and self.emb is not None:
            pred, conf, q = self.student.predict(texts, list(range(len(texts))), query,
                                                 dense=np.asarray(emb, np.float32))
        else:
            pred, conf, q = self._tfidf_student().predict(texts, list(range(len(texts))), query)
        return (pred, conf, q) if query is not None else (pred, conf)

    def updated(self, texts, labels, emb: np.ndarray | None = None, embedder_id: str | None = None, src=None,
                fingerprint: str | None = None, *, max_memory: int | None = None,
                student: dict | None = None) -> "BottleArtifact":
        """Next version: memory + (texts, labels). Embeddings are kept only if both the memory and the new batch have
        them; otherwise the next version is TF-IDF-only ('dense_missing': Bottler.pour reports and warns).
        src: provenance of the new labels (default 'bottle'); fingerprint: the expert request fingerprint of the new
        batch (None: unchanged). If it differs from this bottle's, the old memory's expert answers were obtained under
        another prompt or model, so they are demoted to src='bottle' (no memo for them).
        max_memory / student (optional): replace this bottle's memory cap / student settings (None: keep them)."""
        texts, labels = list(texts), list(labels)
        src = [SRC_BOTTLE] * len(texts) if src is None else list(src)
        fp = self.fingerprint if fingerprint is None else fingerprint
        old_src = self.src if fp == self.fingerprint else [SRC_BOTTLE] * len(self.texts)
        new_emb = None
        if emb is not None:
            emb = self.check_emb(emb, len(texts), embedder_id)
            if self.emb is not None:
                new_emb = np.vstack([self.emb.astype(np.float32), np.asarray(emb, np.float32)])
        eid = (self.embedder_id or embedder_id) if new_emb is not None else None
        # the forecast is copied (Bottler replaces it with the new batch's forecast right after the update). If no new
        # forecast is written, the copy keeps the parent's artifact_version, so votes='auto' treats it as stale.
        C, st = (self.C, self.student_cfg) if student is None else (DEFAULT_C, student)
        return BottleArtifact(self.spec, self.texts + texts, self.labels + labels, C,
                              self.max_memory if max_memory is None else int(max_memory),
                              self.version + 1, new_emb, eid, old_src + src, fp, copy.deepcopy(self.forecast), st,
                              copy.deepcopy(self.history))   # history: report-only learning curve

    def save(self, path: str | Path):
        d = {"spec": spec_to_plain(self.spec),       # core keys (+ non-default optional TaskSpec fields only)
             "texts": self.texts, "labels": self.labels, "C": self.C, "max_memory": self.max_memory,
             "version": self.version, "src": list(self.src), "fingerprint": self.fingerprint,
             "forecast": _plain(self.forecast)}
        if self.student_cfg:          # student settings: only when set (default bottles keep their exact bytes)
            d["student"] = _plain(self.student_cfg)
        if self.history:              # learning curve: only when present (default bottles keep their bytes)
            d["history"] = _plain(self.history)
        if self.emb is not None:      # float16 bytes + shape (plain types only in the pickle)
            d.update({"emb": self.emb.astype("<f2").tobytes(), "emb_shape": tuple(int(x) for x in self.emb.shape),
                      "emb_dtype": "<f2", "embedder_id": self.embedder_id})
        Path(path).write_bytes(pickle.dumps(d))

    @classmethod
    def load(cls, path: str | Path) -> "BottleArtifact":
        d = pickle.loads(Path(path).read_bytes())
        emb = None
        if d.get("emb") is not None:  # bottles saved without embeddings load as TF-IDF bottles
            emb = np.frombuffer(d["emb"], dtype=d.get("emb_dtype", "<f2")).reshape(d["emb_shape"])
        # bottles saved without provenance load with src='bottle' everywhere and no fingerprint (memo off)
        # bottles saved without a forecast: votes='auto' then uses the gain-prior rule for their next pour
        # bottles saved without a history (learning curve): it starts empty
        return cls(TaskSpec(**d["spec"]), d["texts"], d["labels"], d["C"], d["max_memory"], d["version"],
                   emb, d.get("embedder_id"), d.get("src"), d.get("fingerprint"), d.get("forecast"), d.get("student"),
                   d.get("history") or [])


class Bottler:
    _dedupe_expert = True     # (cfg.memo only) one expert call per unique prompt per batch (tests can switch it off)
    _propagate = True         # (cfg.memo only) calibration answers reach non-calibration copies after tau

    def __init__(self, client, spec: TaskSpec, cfg: BottleConfig):
        self.client, self.spec, self.cfg = client, spec, cfg
        # deterministic forecast cross-fit results, kept OUTSIDE the bottles (a decision never mutates its
        # input artifact): {artifact (weak key): {(block digest, K, version, memory, eps, cap, method): result}}
        self._xfit_cache = weakref.WeakKeyDictionary()

    # ------------------------------------------------------------------------------------------------ helpers
    async def _expert(self, texts, idx, phase, known: dict | None = None, stats: dict | None = None):
        """Expert answers for texts[idx]. known=None: one call per item. With `known` (a per-batch dict
        expert_key -> usable expert answer; cfg.memo): items whose prompt is already known are answered from it,
        the rest are called ONCE per unique prompt and the answer is broadcast to every copy (deterministic expert);
        usable answers are added to `known`. Unusable answers are returned for EVERY routed copy (each one counts in
        the unusable count F) and are not cached, so a later call retries the prompt."""
        mt = single_max_tokens(self.client, self.cfg.expert, self.spec)
        if known is None:
            res = await self.client.map([self.client.chat(self.cfg.expert, single_messages(self.spec, texts[i]),
                                                          max_tokens=mt, temperature=EXPERT_TEMPERATURE,
                                                          phase=phase) for i in idx])
            if stats is not None:
                stats["calls"] = stats.get("calls", 0) + len(idx)
            out = {i: (parse_label(self.spec, r.text) or NONE) for i, r in zip(idx, res)}
            self._note_unusable(res, out.values())
            return out
        keys = {i: expert_key(self.spec, texts[i]) for i in idx}
        todo: dict = {}
        for i in idx:
            if keys[i] not in known and keys[i] not in todo:
                todo[keys[i]] = i
        reps = list(todo.values())
        res = await self.client.map([self.client.chat(self.cfg.expert, single_messages(self.spec, texts[r]),
                                                      max_tokens=mt, temperature=EXPERT_TEMPERATURE,
                                                      phase=phase) for r in reps])
        got = {k: (parse_label(self.spec, r.text) or NONE) for k, r in zip(todo, res)}
        self._note_unusable(res, got.values())
        for k, a in got.items():
            if a != NONE:
                known[k] = a
        if stats is not None:
            stats["calls"] = stats.get("calls", 0) + len(reps)
            stats["deduped"] = stats.get("deduped", 0) + len(idx) - len(reps)
        return {i: (got[keys[i]] if keys[i] in got else known[keys[i]]) for i in idx}

    def _require_models(self, *aliases) -> None:
        """Fail BEFORE any LLM call, with the known aliases, when a configured model alias is not in the client's
        models file (otherwise: a bare KeyError, possibly after the cheap votes were already paid for)."""
        models = getattr(self.client, "models", None)
        if not isinstance(models, dict):
            return
        for a in aliases:
            if a is not None and a not in models:
                raise UnknownModelError(f"model alias {a!r} is not in the models file (known aliases: "
                                        f"{sorted(models)}); add it to models.yaml or fix BottleConfig.cheap/expert")

    def _note_unusable(self, results, answers) -> None:
        """Remember the last unusable expert reply (request error or unparsed text) for the ExpertFailure message.
        Report-only: no output depends on it."""
        for r, a in zip(results, answers):
            if a == NONE:
                err = getattr(r, "error", None)
                self._last_unusable = (f"request error: {err}" if err else
                                       f"unparsed reply {str(getattr(r, 'text', ''))[:80]!r}")[:200]

    def _warn_cheap(self, labels, what: str) -> None:
        """Warn (outputs unchanged) when the cheap model gave no usable label for most items. Without it an outage
        or a parser mismatch shows only as low coverage and ledger errors."""
        n = len(labels)
        miss = sum(a is None for a in labels)
        if n and miss / n > CHEAP_WARN_SHARE:
            warnings.warn(f"cheap model {self.cfg.cheap!r} gave no usable {what} for {miss / n:.0%} of items (endpoint "
                          "down, wrong model id or label parser?); those items fall back to the student or the expert",
                          stacklevel=3)

    def stream_risk(self, certificates, Ns=None) -> dict:
        """Stream-level tail over consecutive certificates of this Bottler's run()/pour() calls (report-only; see the
        module function stream_risk for the claim and the conditions the caller must ensure)."""
        return stream_risk(certificates, self.cfg.eps, self.cfg.delta, Ns)

    def fingerprint(self) -> str:
        """Hash of the exact expert request apart from the item text: the system prompt (instruction + label block),
        max_chars, temperature, reply budget and the expert alias's model identity (ids, base_url, extra_body,
        system_prefix). Bottles record it; a pour uses its memo only under an identical fingerprint."""
        ms = self.client.models.get(self.cfg.expert)
        ident = {"v": 1, "system": single_messages(self.spec, "")[0]["content"], "labels": self.spec.labels,
                 "max_chars": self.spec.max_chars, "user_limit": self.spec.single_limit,
                 "temperature": EXPERT_TEMPERATURE, "parser": PARSER_VERSION,
                 "max_tokens": single_max_tokens(self.client, self.cfg.expert, self.spec),
                 "ids": sorted(str(x) for x in (getattr(ms, "ids", None) or [])),
                 "base_url": getattr(ms, "base_url", None), "extra_body": getattr(ms, "extra_body", None),
                 "system_prefix": getattr(ms, "system_prefix", None)}
        extras = spec_extras(self.spec, EXPERT_SIDE_EXTRAS)
        if extras:     # non-default expert prompt / parser hooks; absent by default (default fingerprints)
            ident["spec_extras"] = extras
        return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:32]

    def _n_build(self, N: int) -> int:
        nb = (self.cfg.n_build if self.cfg.n_build is not None
              else max(self.cfg.n_build_min, self.cfg.n_build_per_class * len(self.spec.labels)))
        return min(nb, N)

    def _price_note(self, em) -> dict:
        """cost_basis='usd' with an unpriced expert: every cost ratio is infinite, so mode='auto' picks student
        mode and votes='auto' goes cold. Warn and flag it ({'price_warning': 'no_prices'}); outputs are unchanged."""
        if self.cfg.cost_basis != "usd" or not unpriced(em):
            return {}
        warnings.warn(f"expert model {self.cfg.expert!r} has no price (price_in = price_out = 0): mode='auto' and "
                      "votes='auto' will never choose cheap votes. Set prices in models.yaml or use "
                      "BottleConfig(cost_basis='tokens')", stacklevel=3)
        return {"price_warning": "no_prices"}

    def _bottle_kw(self) -> dict:
        """Non-default bottle settings of this configuration (max_memory, student) for exported bottles;
        {} by default, so exported bottles keep the default construction (and a pour keeps the bottle's own settings)."""
        cfg, kw = self.cfg, {}
        if int(cfg.max_memory) != DEFAULT_MAX_MEMORY:
            kw["max_memory"] = int(cfg.max_memory)
        if cfg.student is not None:
            kw["student"] = dict(cfg.student)
        return kw

    # ------------------------------------------------------------------------------------------------ embeddings
    def _embedder_id(self) -> str | None:
        return self.cfg.embedder_id or getattr(self.cfg.embedder, "embedder_id", None)

    async def _call_embedder(self, texts) -> np.ndarray:
        out = self.cfg.embedder(list(texts))
        if inspect.isawaitable(out):
            out = await out
        out = np.asarray(out, np.float32)
        if out.ndim != 2 or out.shape[0] != len(texts):
            raise ValueError(f"embedder returned shape {out.shape} for {len(texts)} texts")
        if not np.isfinite(out).all():
            raise ValueError("embedder returned non-finite values")
        check_emb_values(out, "embedder output", store=bool(self.cfg.export))
        return out

    async def _resolve_emb(self, texts, emb, diag) -> np.ndarray | None:
        """Embeddings of this batch: the caller's `emb`, else cfg.embedder (once per batch, BEFORE any calibration
        draw), else None. They must be a fixed, label-free function of each text, so scores stay fixed before the
        calibration labels are seen. An embedder failure falls back to TF-IDF only (flagged; validity unaffected)."""
        if emb is not None:
            emb = np.asarray(emb)
            if emb.ndim != 2 or emb.shape[0] != len(texts):
                raise ValueError(f"emb has shape {emb.shape}; expected ({len(texts)}, d): one embedding row per text")
            check_emb_values(emb, store=bool(self.cfg.export))
            return emb
        if self.cfg.embedder is None:
            return None
        try:
            return await self._call_embedder(texts)
        except Exception as e:  # noqa: BLE001 - an embedding outage must not stop the pipeline
            diag["embedder_error"] = f"{type(e).__name__}: {e}"[:300]
            warnings.warn(f"embedder failed ({diag['embedder_error']}); continuing with TF-IDF features only",
                          stacklevel=3)
            return None

    def plan(self, texts) -> tuple[str, dict]:
        cfg = self.cfg
        if cfg.mode != "auto":
            return cfg.mode, {"reason": "fixed by config"}
        N = len(texts)
        cm, em = self.client.models[cfg.cheap], self.client.models[cfg.expert]
        avg = float(np.mean([len(t) for t in texts])) if texts else 0
        cb, ce = per_item_costs(self.spec, cm, em, avg, cfg.k, cfg.warmup, cfg.cost_basis)
        ratio = cb / ce if ce > 0 else float("inf")
        nb = self._n_build(N)
        gain = gain_prior_for(cfg.gain_prior, nb, len(self.spec.labels), cfg.gain_fed_per_class,
                              cfg.gain_starved_per_class, cfg.gain_starved_mult)
        mode = choose_mode(ratio, cfg.contexts, gain, N=N, n_build=nb)
        return mode, {"cheap_batched_usd_per_item": cb, "expert_usd_per_item": ce, "ratio": ratio, "n_build": nb,
                      "gain_prior": gain,
                      "rule": f"fused iff ({cfg.contexts}*N - n_build)*ratio < gain_prior*N", **self._price_note(em)}

    # ------------------------------------------------------------------------------------------------ cold start
    async def run(self, texts: list[str], emb: np.ndarray | None = None) -> BottleResult:
        cfg, N = self.cfg, len(texts)
        snap = self.client.ledger.snapshot()
        if N == 0:
            return BottleResult([], [], {"certified": False, "N": 0}, {}, {"N": 0})
        if cfg.cheap is None:
            raise ValueError("run() needs a cheap model; use pour() to reuse an artifact without one")
        self._require_models(cfg.cheap, cfg.expert)
        emb_diag: dict = {}
        emb = await self._resolve_emb(texts, emb, emb_diag)
        mode, plan_info = self.plan(texts)
        diag = {"mode": mode, "plan": plan_info, "N": N}
        groups = None
        if cfg.memo:   # group-aware cross-fitting: copies of one prompt share a fold (quality only; label-free)
            keys = [expert_key(self.spec, t) for t in texts]
            n_unique = len(set(keys))
            if n_unique < N:
                groups = keys
                diag["n_duplicate_items"] = N - n_unique
        if emb is not None or cfg.embedder is not None:
            diag.update({"dense": "dense" if emb is not None else "dense_missing", "embedder_id": self._embedder_id(),
                         **emb_diag})
        if mode == "fused":
            v = await vote_label(self.client, cfg.cheap, self.spec, texts, k=cfg.k, contexts=cfg.contexts,
                                 warmup=cfg.warmup, seed=cfg.seed, balanced=cfg.balanced, weighted=cfg.weighted,
                                 phase=PHASE_CHEAP_VOTE, early_frac=cfg.early_frac)
            vote, vscore = v["answers"], np.asarray(v["scores"], float)
            self._warn_cheap(vote, "votes")
            diag["position_reliability"] = v["reliability"]
            if v["reliability"]:
                rel = v["reliability"]
                diag["cold_start_detected"] = bool(rel[0] < np.mean(rel[1:]) - COLD_START_GAP)
            s_pred, s_conf, s_q = cross_fit(texts, vote, list(range(N)), seed=sub_seed(cfg.seed, CROSSFIT_STREAM_KEY),
                                            dense=emb, query=vote, groups=groups, folds=cfg.crossfit_folds,
                                            student=cfg.student)
            final = [a if a is not None else s for a, s in zip(vote, s_pred)]
            if cfg.fusion == "prob":    # vote share of the answer + student probability of the answer
                support = np.where([a is not None for a in vote], s_q, s_conf)
            else:                       # vote share + student confidence when the student's top label agrees
                support = np.array([float(s == f) for s, f in zip(s_pred, final)]) * s_conf
            score = vscore + support
            # s_pred: the CROSS-FITTED student's answers (read only by the health report; never the exported bottle)
            fc = {"kind": "run", "vote": vote, "vscore": vscore, "s_pred": s_pred}
        else:
            fc = {"kind": "run"}
            nb = self._n_build(N)
            B = random_order(N, cfg.seed)[:nb]
            ctx = await label_batched(self.client, cfg.cheap, self.spec, texts, B, cfg.k, warmup=cfg.warmup,
                                      seed=cfg.seed, phase=PHASE_CHEAP_BUILD)
            diag["n_build"] = nb
            self._warn_cheap([ctx.labels[i] for i in B], "labels")
            s_pred, s_conf = cross_fit(texts, ctx.labels, B, seed=sub_seed(cfg.seed, CROSSFIT_STREAM_KEY), dense=emb,
                                       groups=groups, folds=cfg.crossfit_folds, student=cfg.student)
            final, score = s_pred, np.asarray(s_conf, float)
            fc["s_pred"] = s_pred          # health report only (the cross-fitted student IS the answer here)
        res = await self._certify_and_escalate(texts, final, score, diag, emb=emb, fc_inputs=fc)
        res.ledger = self.client.ledger.since(snap)
        self._forecast_costs(res)
        return res

    # ------------------------------------------------------------------------------------------------ reuse
    def _votes_pay(self, texts, artifact: "BottleArtifact | None" = None, dense_next: bool | None = None
                   ) -> tuple[bool, dict]:
        """votes='auto'. With a matching forecast in the bottle (cfg.forecast): warm iff its MEASURED predicted
        cost per item is lower (economics.measured_votes_pay). Otherwise the gain-prior rule: votes iff
        contexts*c_cheap/c_expert < gain_prior. Either way the choice is a function of earlier batches and of the new
        texts' lengths only, fixed before this batch's calibration draw (whose stream does not depend on it)."""
        cfg = self.cfg
        if cfg.cheap is None:
            return False, {"reason": "no cheap model configured"}
        cm, em = self.client.models[cfg.cheap], self.client.models[cfg.expert]
        avg = float(np.mean([len(t) for t in texts])) if texts else 0
        cb, ce = per_item_costs(self.spec, cm, em, avg, cfg.k, cfg.warmup, cfg.cost_basis)
        ratio = cb / ce if ce > 0 else float("inf")
        prior = cfg.contexts * ratio < cfg.gain_prior
        note = self._price_note(em)
        why = {"ratio": ratio, "rule": f"votes iff {cfg.contexts}*ratio < {cfg.gain_prior}", "basis": "prior", **note}
        if not cfg.forecast or artifact is None:
            return prior, why
        try:
            use, info = self._measured_votes(texts, artifact, dense_next)
        except Exception as e:  # noqa: BLE001 - a forecast problem falls back to the prior rule
            return prior, {**why, "forecast_error": f"{type(e).__name__}: {e}"[:300]}
        if use is None:
            return prior, {**why, "forecast": info}
        return use, {**info, "prior_rule_choice": bool(prior), "prior_ratio": ratio, **note}

    def _measured_votes(self, texts, artifact: "BottleArtifact", dense_next) -> tuple[bool | None, dict]:
        """(use_votes, info) from artifact.forecast, or (None, reason) when no matching, complete forecast exists."""
        cfg = self.cfg
        f = artifact.forecast
        if not f:
            return None, {"reason": "the bottle carries no forecast"}
        if f.get("artifact_version") != artifact.version:
            # inherited from an ancestor whose child batch recorded no forecast (forecast failed, forecast=False,
            # all-memo pour): its arms describe an older bottle, so the prior rule decides
            return None, {"reason": f"stale forecast: recorded for bottle version {f.get('artifact_version')}, "
                                    f"this bottle is version {artifact.version}"}
        if f.get("key") != self._forecast_key():
            return None, {"reason": "forecast recorded under another configuration (models, contexts, eps, ...)"}
        if dense_next is not None and bool(f.get("dense")) != bool(dense_next):
            return None, {"reason": f"forecast dense={bool(f.get('dense'))} but this pour dense={bool(dense_next)}"}
        if cfg.forecast_v2 != "off":
            return self._measured_votes_v2(texts, artifact, f)
        arms = self._forecast_arms(artifact, f)
        if arms.get("cold") is None or arms.get("warm") is None:
            return None, {"reason": "forecast lacks an arm", **arms}
        c_vote, vb, c_exp, eb = self._unit_costs(texts, f)
        if not c_exp > 0:
            return None, {"reason": "expert cost unknown", **arms}
        N_new = len(texts)
        n_new = self._planned_n(N_new)
        cov_c = rescale_bottled(arms["cold"], f["N"], f["n_cal"], N_new, n_new)
        cov_w = rescale_bottled(arms["warm"], f["N"], f["n_cal"], N_new, n_new)
        use, info = measured_votes_pay(cov_c, cov_w, c_vote, c_exp, cfg.forecast_hysteresis)
        info.update({"basis": "measured", "forecast_source": f.get("source"),
                     "forecast_version": f.get("artifact_version"),
                     "arms_at_source": {"cold": arms["cold"], "warm": arms["warm"]},
                     "arms_basis": {"cold": arms.get("basis_cold"), "warm": arms.get("basis_warm")},
                     "cost_basis": {"vote": vb, "expert": eb}, "N_source": f["N"], "n_source": f["n_cal"],
                     "N_new": N_new, "n_new": n_new})
        return use, info

    def _unit_costs(self, texts, f: dict) -> tuple[float, str, float, str]:
        """(c_vote per item, basis, c_exp per call, basis): measured unit costs of the forecast, rescaled to this batch's
        text lengths (estimated from prompt sizes when none were measured)."""
        cfg = self.cfg
        costs = f.get("costs") or {}
        cm, em = self.client.models[cfg.cheap], self.client.models[cfg.expert]
        avg = float(np.mean([len(t) for t in texts])) if texts else 0.0
        bs = cfg.cost_basis
        vk, ek = (("vote_tokens_per_item", "expert_tokens_per_call") if bs == "tokens"
                  else ("vote_usd_per_item", "expert_usd_per_call"))
        cb_new, ce_new = per_item_costs(self.spec, cm, em, avg, cfg.k, cfg.warmup, bs)
        if costs.get(vk) is not None:     # measured, rescaled to this batch's text lengths
            cb_old, _ = per_item_costs(self.spec, cm, em, costs["vote_avg_chars"], cfg.k, cfg.warmup, bs)
            c_vote, vb = costs[vk] * (cb_new / cb_old if cb_old > 0 else 1.0), costs["vote_basis"]
        else:
            c_vote, vb = cfg.contexts * cb_new, "estimated"
        if costs.get(ek) is not None:
            _, ce_old = per_item_costs(self.spec, cm, em, costs["expert_avg_chars"], cfg.k, cfg.warmup, bs)
            c_exp, eb = costs[ek] * (ce_new / ce_old if ce_old > 0 else 1.0), costs["expert_basis"]
        else:
            c_exp, eb = ce_new, "estimated"
        return c_vote, vb, c_exp, eb

    # ------------------------------------------------------------------------------------------------ forecast v2
    def _measured_votes_v2(self, texts, artifact: "BottleArtifact", f: dict) -> tuple[bool | None, dict]:
        """cfg.forecast_v2 != 'off'. Arms: the measured-forecast arms, with (3) the run-time cross-fitted cold arm for the first pour
        ('pour1', 'full') and (4) the next-bottle cross-fitted cold arm for later pours ('full'). Then (2) the can-pay
        ceiling: if even a warm pour bottling cap = (1 - n/N)*a + e_new cannot beat the cold arm by the hysteresis,
        choose cold ('can_pay_bound'); otherwise clip the warm arm at cap and decide as without v2. No usable arm pair:
        (None, reason), i.e. the gain-prior rule. Inputs are earlier batches only; nothing here draws from a
        calibration stream or calls a model."""
        cfg = self.cfg
        arms = self._forecast_arms(artifact, f)          # measured-forecast arms (resolves a forecast_run pending block if opted in)
        arms, xinfo = self._v2_arms(artifact, f, arms)
        c_vote, vb, c_exp, eb = self._unit_costs(texts, f)
        base = {"forecast_v2": cfg.forecast_v2, "forecast_source": f.get("source"),
                "forecast_version": f.get("artifact_version"),
                "arms_at_source": {"cold": arms.get("cold"), "warm": arms.get("warm")},
                "arms_basis": {"cold": arms.get("basis_cold"), "warm": arms.get("basis_warm")},
                "cost_basis": {"vote": vb, "expert": eb}, "N_source": f.get("N"), "n_source": f.get("n_cal"), **xinfo}
        if not c_exp > 0:
            return None, {"reason": "expert cost unknown", **base}
        N_new = len(texts)
        n_new = self._planned_n(N_new)
        base.update({"N_new": N_new, "n_new": n_new})

        def resc(c):
            return None if c is None else rescale_bottled(c, f["N"], f["n_cal"], N_new, n_new)
        cov_c, cov_w = resc(arms.get("cold")), resc(arms.get("warm"))
        ceil = self._can_pay(f, c_vote, c_exp, cov_c, N_new, n_new)
        if ceil is not None and ceil.get("can_pay") is False:
            return False, {**base, "basis": "can_pay_bound", "cov_cold": cov_c, "cov_warm": cov_w,
                           "cost_cold": ceil["cost_cold"], "cost_warm_best": ceil["cost_warm_best"],
                           "c_vote_item": c_vote, "c_exp_call": c_exp, "hysteresis": ceil["hysteresis"],
                           "vote_over_expert": c_vote / c_exp, "can_pay": ceil}
        clipped = False
        if ceil is not None and cov_w is not None and cov_w > ceil["cap"]:
            cov_w, clipped = ceil["cap"], True
        if cov_c is None or cov_w is None:
            return None, {**base, "reason": "forecast lacks an arm",
                          **{k: arms.get(k) for k in ("cold", "warm", "basis_cold", "basis_warm")},
                          "cov_cold": cov_c, "cov_warm": cov_w, "can_pay": ceil, "vote_over_expert": c_vote / c_exp}
        use, info = measured_votes_pay(cov_c, cov_w, c_vote, c_exp, cfg.forecast_hysteresis)
        info.update({**base, "basis": "measured", "warm_clipped_at_cap": clipped, "can_pay": ceil})
        return use, info

    def _planned_e(self, N: int, n: int) -> float:
        """Label-free worst-case expected share of unlabelled bottled answers != expert of the planned design for a
        batch of N with n calibration labels: (k+1)(N-n)/((n+1)N), k = the closed-form rule's k at the last look."""
        cfg, meth = self.cfg, self.cfg.certificate
        if N <= 0 or n >= N:
            return 0.0
        alpha = float(cfg.delta)
        if cfg.calibration == "sequential":
            if cfg.looks:
                looks, spending = sorted({min(int(l), N) for l in cfg.looks if int(l) > 0}), cfg.spending
            elif cfg.design == "single":
                looks, spending = single_look_design(N, cfg.eps, cfg.design_c)
            else:
                looks, spending = default_design(N)
            looks = list(looks)
            spending = spending if spending and len(spending) == len(looks) else None
            alpha = float(cfg.delta) * float(spend(looks, spending)[-1])
        k = k_used(int(n), int(N), cfg.eps, alpha, meth, max_expected_eps=cfg.max_expected_eps)
        return float(expected_eps_closed_form(int(n), int(N), int(k)))

    def _can_pay(self, f: dict, c_vote: float, c_exp: float, cov_cold, N_new: int, n_new: int) -> dict | None:
        """(2) The can-pay ceiling from the measured vote-expert agreement (None when no agreement was measured on
        this lineage, e.g. student-mode runs or old bottles). can_pay is None when there is no cold arm to compare."""
        ag = f.get("agree")
        if not ag or ag.get("a") is None:
            return None
        e_new = self._planned_e(N_new, n_new)
        a = float(ag["a"])
        z = float(self.cfg.forecast_canpay_z)
        a_raw = a
        if z > 0 and int(ag.get("n") or 0) > 0:   # (opt-in) upper confidence value of the estimated agreement
            a = float(min(1.0, a + z * math.sqrt(a * (1 - a) / int(ag["n"]))))
        if cov_cold is None:
            cap = float(min(1.0, max(0.0, (1.0 - n_new / max(1, N_new)) * a + e_new)))
            info = {"cap": cap, "a": a, "e_new": e_new, "can_pay": None}
        else:
            can, info = votes_can_pay(c_vote, c_exp, cov_cold, a, n_new, N_new, e_new, self.cfg.forecast_hysteresis)
            info["can_pay"] = can
        if z > 0:
            info.update({"a_raw": a_raw, "canpay_z": z})
        a = a_raw
        n_a = int(ag.get("n") or 0)
        info.update({"a_n": n_a, "a_age": int(ag.get("age", 0)), "a_basis": ag.get("basis"),
                     "a_se": float(math.sqrt(a * (1 - a) / n_a)) if n_a > 0 else None})
        return info

    def _v2_arms(self, artifact, f: dict, arms: dict) -> tuple[dict, dict]:
        """Arms with parts (3)/(4) of forecast v2 applied (any failure keeps the measured-forecast arms and records why)."""
        cfg, src = self.cfg, f.get("source")
        out, info = dict(arms), {}
        if cfg.forecast_v2 not in ("pour1", "full"):
            return out, info
        if src == "run":
            if out.get("cold") is not None:                # forecast_run=True already resolved a cold arm
                return out, info
        elif cfg.forecast_v2 != "full" or src not in ("pour_warm", "pour_cold"):
            return out, info
        blk = f.get("xfit")
        if not blk:
            info["xfit"] = "unavailable (no cross-fit block recorded: old bottle or a batch without calibration)"
            return out, info
        try:
            r = self._resolve_xfit(artifact, blk)
        except Exception as e:  # noqa: BLE001 - a cross-fit problem keeps the measured-forecast arms
            info["xfit_error"] = f"{type(e).__name__}: {e}"[:300]
            return out, info
        info["xfit"] = {k: r.get(k) for k in ("cold", "tau", "folds", "n_rows", "n_cal_rows", "n_other_rows",
                                              "stream")}      # deterministic only (wall time: pour's wall_time_s)
        if src == "run":
            out["cold"], out["basis_cold"] = r["cold"], f"crossfit cold ({r['folds']} folds)"
            return out, info
        lagged = out.get("cold")
        info["cold_lagged"] = lagged
        out["cold"], out["basis_cold"] = r["cold"], f"next-bottle crossfit ({r['folds']} folds)"
        if src == "pour_cold" and out.get("warm") is not None:
            g = f.get("gain") or {}
            age = int(g.get("age", 1))
            if g.get("value") is not None:
                eff = float(g["value"]) * cfg.forecast_gain_decay ** age
                why = f"next-bottle cold + stale gain {float(g['value']):+.4f} x {cfg.forecast_gain_decay}^{age}"
            else:                                       # gain resolved from a forecast_run block by _forecast_arms
                eff = float(out["warm"]) - float(lagged)
                why = f"next-bottle cold + stale gain {eff:+.4f}"
            out["warm"], out["basis_warm"] = float(min(1.0, max(0.0, r["cold"] + eff))), why
        return out, info

    def _xfit_folds(self, n_rows: int, M: int) -> int:
        """Folds of a forecast cross-fit over n_rows batch rows in a memory of M rows. 'auto': the largest K in [2, 5]
        with training rows K*M - n_rows STRICTLY below XFIT_BUDGET * M (one bottle fit = M rows). The batch's rows are
        part of the memory (n_rows <= M), so this is K = 2: 2M - n_rows training rows + n_rows predicted rows = 2M rows
        of work. (With <= instead of <, a run() bottle (n_rows = M) would take K = 3: 2.15-2.45x one
        bottle fit measured.)"""
        ff = self.cfg.forecast_folds
        if ff == "auto":
            M = max(1, int(M))
            b = float(self.cfg.forecast_fit_budget)
            if b.is_integer():
                K = (int(b) * M + int(n_rows) - 1) // M          # largest K with K*M - n_rows < budget*M (exact ints)
            else:
                K = math.ceil(b + int(n_rows) / M) - 1
            K = min(int(self.cfg.forecast_max_folds), max(2, K))
        else:
            K = int(ff)
        return max(2, min(K, n_rows))

    def _resolve_xfit(self, artifact: "BottleArtifact", blk: dict) -> dict:
        """(3)/(4) Cross-fitted cold arm of the bottle that holds a batch. The batch's rows are found in the bottle's
        memory by 64-bit text hash (exact identity: every batch text must map to exactly one memory row, else
        _XfitUnavailable, e.g. after max_memory eviction or a hash collision). Rows are split into K folds by a
        permutation from sub_seed(seed, 7, version) (version 0 = run()); every OTHER memory row (the
        pre-batch memory) is always in training. Each fold is scored by the bottle's student class (TF-IDF, or TF-IDF
        + dense with the bottle's stored embeddings) fitted on the other rows, so no row is scored by a model that saw
        its text. The fold scores (top label, its probability) are certified on the batch's own calibration rows with
        their RAW expert answers at the batch's level alpha (_fc_certify: fixed-sample test, no draw); cold arm =
        share of non-calibration rows above tau x (1 - n/N). The result is deterministic (no wall time). It is cached
        in this Bottler (self._xfit_cache: weakly keyed by the artifact, then by a digest of the block, K, the version,
        the memory size and the certificate settings), never in the artifact: a decision leaves its input bottle (and
        its pickle) unchanged. No LLM call; the only random stream is the key-7 permutation."""
        cfg = self.cfg
        if int(blk.get("version", -1)) != int(artifact.version):
            raise _XfitUnavailable(f"block recorded for bottle version {blk.get('version')}, not {artifact.version}")
        mem = artifact.texts
        M = len(mem)
        if int(blk.get("memory", -1)) != M:
            raise _XfitUnavailable(f"memory size {M} != {blk.get('memory')} recorded with the block")
        hashes = [int(h) for h in blk["hashes"]]
        want = set(hashes)
        row_of: dict = {}
        coll = 0
        for j, t in enumerate(mem):
            h = _text_hash(t)
            if h in want:
                if h in row_of:
                    coll += 1
                row_of[h] = j
        if coll:
            raise _XfitUnavailable(f"{coll} hash collisions between memory texts")
        miss = sum(h not in row_of for h in want)
        if miss:
            raise _XfitUnavailable(f"{miss} batch texts are not in the bottle's memory (max_memory eviction?)")
        local: dict = {}
        rows: list = []
        for h in hashes:                     # one row per distinct batch text, in order of first occurrence
            if h not in local:
                local[h] = len(rows)
                rows.append(row_of[h])
        R = len(rows)
        cap = cfg.forecast_max_rows                # optional cap on the pre-batch memory rows (M - R of them)
        K = self._xfit_folds(R, M if cap is None else min(M, R + int(cap)))
        store, ck = self._xfit_store(artifact), self._xfit_key(artifact, blk, K)
        if store is not None and ck in store:
            return dict(store[ck])
        cal_loc: dict = {}
        for i, a in blk["cal"].items():
            cal_loc.setdefault(local[hashes[int(i)]], a)
        cr = sorted(cal_loc)
        if not cr or R < 2:
            raise _XfitUnavailable("no calibration rows")
        in_batch = set(rows)
        other = [j for j in range(M) if j not in in_batch]
        if cap is not None and len(other) > int(cap):   # uniform label-free subsample from the key-7 stream
            sub = np.random.default_rng(sub_seed(int(blk["seed"]), FORECAST_STREAM_KEY, int(blk["version"]), 1))
            other = sorted(other[int(i)] for i in sub.permutation(len(other))[:int(cap)])
        E = None if artifact.emb is None else artifact.emb.astype(np.float32)
        key = [int(blk["seed"]), FORECAST_STREAM_KEY, int(blk["version"])]
        perm = np.random.default_rng(sub_seed(*key)).permutation(R)
        pred: list = [None] * R
        conf = np.zeros(R)
        for fold in np.array_split(perm, K):
            if len(fold) == 0:
                continue
            fs = set(int(x) for x in fold)
            tr = other + [rows[r] for r in range(R) if r not in fs]
            model = artifact._new_student(E).fit([mem[j] for j in tr], [artifact.labels[j] for j in tr], tr)
            fl = sorted(fs)
            p, c, _ = model.predict([mem[rows[r]] for r in fl], [rows[r] for r in fl])
            for q, r in enumerate(fl):
                pred[r], conf[r] = p[q], float(c[q])
        N, n = int(blk["N"]), int(blk["n_cal"])
        fin = [a if (a is not None and a != NONE) else NONE for a in pred]
        valid = np.array([a != NONE for a in fin], bool)
        sc = np.where(valid, conf, -np.inf)
        cap_kw = {"max_expected_eps": cfg.max_expected_eps} if cfg.max_expected_eps is not None else {}
        mkw = {"m": int(blk["m"])} if blk.get("m") is not None else {}
        cert = _fc_certify(sc[cr], [disagreement(fin[r], cal_loc[r]) for r in cr], N, cfg.eps, float(blk["alpha"]),
                           blk["meth"], **cap_kw, **mkw)
        noncal = np.ones(R, bool)
        noncal[cr] = False
        frac = float(np.mean((sc[noncal] >= cert.tau) & valid[noncal])) if noncal.any() else 0.0
        out = {"cold": frac * max(0.0, 1.0 - n / N), "tau": float(cert.tau) if np.isfinite(cert.tau) else None,
               "folds": int(K), "n_rows": R, "n_cal_rows": len(cr), "n_other_rows": len(other), "stream": key}
        if cap is not None:
            out["other_rows_cap"] = int(cap)
        if store is not None:
            store[ck] = out
        return dict(out)

    def _xfit_store(self, artifact) -> dict | None:
        """This Bottler's cross-fit cache for `artifact` (None: the artifact cannot be weakly referenced; no cache)."""
        cache = self.__dict__.get("_xfit_cache")
        if cache is None:                    # a Bottler built without __init__ (tests, old subclasses)
            cache = self._xfit_cache = weakref.WeakKeyDictionary()
        try:
            return cache.setdefault(artifact, {})
        except TypeError:
            return None

    def _xfit_key(self, artifact, blk: dict, K: int) -> tuple:
        """Cache key of a cross-fit: a digest of the block's content (a replaced forecast never hits a stale entry), K,
        the bottle version and memory size, and the certificate settings the result depends on."""
        canon = (blk.get("v"), [int(h) for h in blk["hashes"]],
                 sorted((int(i), str(a)) for i, a in blk["cal"].items()),
                 *(blk.get(k) for k in ("N", "n_cal", "alpha", "meth", "m", "seed", "version", "memory")))
        dig = hashlib.blake2b(repr(canon).encode("utf-8"), digest_size=16).hexdigest()
        return (dig, int(K), int(artifact.version), len(artifact.texts), float(self.cfg.eps),
                self.cfg.max_expected_eps, self.cfg.certificate,
                *(() if self.cfg.forecast_max_rows is None else (int(self.cfg.forecast_max_rows),)))

    # ------------------------------------------------------------------------------------------------ forecast
    def _forecast_key(self) -> dict:
        """Configuration a forecast is valid for (models, voting, certificate): a forecast recorded under another key
        is not used (the prior rule decides)."""
        cfg = self.cfg
        return {"cheap": cfg.cheap, "expert": cfg.expert, "contexts": int(cfg.contexts), "k": int(cfg.k),
                "warmup": int(cfg.warmup), "fusion": cfg.fusion, "eps": float(cfg.eps), "delta": float(cfg.delta),
                "certificate": cfg.certificate, "calibration": cfg.calibration,
                "design": "custom" if cfg.looks else cfg.design, "design_c": float(cfg.design_c),
                **({} if cfg.early_frac == EARLY_FRAC else {"early_frac": float(cfg.early_frac)})}

    def _planned_n(self, N: int) -> int:
        """Calibration labels the design plans for a population of N (label-free; the last pre-specified look)."""
        cfg, meth = self.cfg, self.cfg.certificate
        if N <= 0:
            return 0
        if cfg.calibration == "fixed":
            return int(tune_size(min(cfg.n_cal, N), N, cfg.eps, cfg.delta, meth) if cfg.tune else min(cfg.n_cal, N))
        if cfg.looks:
            looks, spending = sorted({min(int(l), N) for l in cfg.looks if int(l) > 0}), cfg.spending
        elif cfg.design == "single":
            looks, spending = single_look_design(N, cfg.eps, cfg.design_c)
        else:
            looks, spending = default_design(N)
        looks = list(looks)
        spending = spending if spending and len(spending) == len(looks) else None
        if cfg.tune:
            looks = tuned_looks(looks, spend(looks, spending), N, cfg.eps, cfg.delta, meth)
        return int(looks[-1])

    def _prop_set(self, texts, labs: dict, is_cal: np.ndarray) -> set:
        """Items that propagation (cfg.memo) answers from a calibration duplicate: a function of the calibration
        answers and texts only, so it is the same for every scoring arm with the same labelled set."""
        if not (self.cfg.memo and self._propagate and labs):
            return set()
        known: dict = {}
        for i, a in labs.items():
            if a != NONE:
                known.setdefault(expert_key(self.spec, texts[i]), a)
        return {i for i in range(len(texts)) if not is_cal[i] and expert_key(self.spec, texts[i]) in known}

    async def _arm_outcome(self, arm_final, arm_score, ctx: dict) -> dict:
        """EXACT outcome of scoring this batch with (arm_final, arm_score) instead: the same population, look sizes,
        method, allowance m, calibration stream (hence the same sample) and RAW expert answers (NONE kept, before any
        +fallback), through the same certificate functions, propagation set and budget rule. Raises _ArmUnavailable
        if that arm would need labels this batch did not buy (a staged design stopping at another look)."""
        cfg, N, cal = self.cfg, ctx["N"], ctx["cal"]
        k_level = ctx.get("k_level", cfg.delta)   # delta/2 under an expert budget (the batch's own k-rule level)
        fin = [a if (a is not None and a != NONE) else NONE for a in arm_final]
        valid = np.array([a != NONE for a in fin], bool)
        sc = np.where(valid, np.asarray(arm_score, float), -np.inf)
        if ctx["trivial"]:
            tau, labs = -np.inf, {}
        elif cfg.calibration == "sequential":
            async def from_cal(idx):
                miss = [i for i in idx if i not in cal]
                if miss:
                    raise _ArmUnavailable(f"needs {len(miss)} calibration labels this batch did not buy")
                return {i: cal[i] for i in idx}
            # the SAME function and stream key as the batch's own certificate (re-derives the same permutation;
            # no new draw), called through a private alias so it is not counted as one of the pipeline's draws
            cert, labs = await _fc_certify_sequential(sc, fin, from_cal, N, cfg.eps, k_level, ctx["looks"],
                                                      cfg.target_coverage, seed=ctx["s_cal"], patience=cfg.patience,
                                                      method=ctx["meth"], spending=ctx["spending"], **ctx["cap_kw"],
                                                      **ctx["mkw"])
            tau = cert.tau
        else:
            idx, labs = sorted(cal), dict(cal)
            cert = _fc_certify(sc[idx], [disagreement(fin[i], cal[i]) for i in idx], N, cfg.eps, k_level,
                               ctx["meth"], **ctx["cap_kw"], **ctx["mkw"])
            tau = cert.tau
        if labs and float(np.mean([v == NONE for v in labs.values()])) > cfg.max_expert_none:
            raise _ArmUnavailable("that arm's calibration labels would raise ExpertFailure")
        is_cal = np.zeros(N, bool)
        is_cal[list(labs)] = True
        free = ~is_cal
        free[list(self._prop_set(ctx["texts"], labs, is_cal))] = False
        accept = (sc >= tau) & valid
        capped = False
        if not ctx["trivial"] and cfg.expert_budget is not None and int((~accept & free).sum()) > cfg.expert_budget:
            accept = (sc >= float(np.sort(sc)[min(N - 1, cfg.expert_budget)])) & valid
            capped = True
        b = int((accept & free).sum())
        return {"bottled": b / N, "bottled_count": b, "tau": float(tau) if np.isfinite(tau) else None,
                "certified": bool(tau < np.inf), "n_cal": len(labs), "budget_capped": capped}

    async def _forecast(self, art, base_artifact, fc: dict, ctx: dict, prov, diag) -> dict:
        """Forecast record for the NEXT pour from `art` (plain types only):
          run       : measured unit costs only ('unavailable': the next pour uses the gain-prior rule), or with
                      cfg.forecast_run a 'pending' cross-fitted certified-output forecast (resolved lazily by
                      votes='auto': a 5-fold cross-fit of the certified answers, seed sub_seed(seed, 7))
          pour_warm : cold arm = EXACT counterfactual (bottle alone on this batch's own calibration sample), warm arm
                      = realized; gain = warm - cold (age 0)
          pour_cold : cold arm = realized; the warm gain is carried from the parent forecast (age + 1, 'stale')."""
        cfg, N, kind = self.cfg, int(ctx["N"]), fc["kind"]
        n_real = int(sum(p == "bottle" for p in prov))
        f = {"v": 1, "source": kind, "key": self._forecast_key(), "artifact_version": int(art.version), "N": N,
             "N_batch": int(ctx["N_tot"]), "n_cal": int(len(ctx["cal"])),
             "avg_chars": float(np.mean([len(t) for t in ctx["texts"]])) if N else 0.0,
             "dense": bool(diag.get("dense") == "dense") if kind != "run" else bool(art.dense),
             "status": "measured", "arms": {}, "gain": None}
        parent = getattr(base_artifact, "forecast", None) if base_artifact is not None else None
        if parent and parent.get("key") == f["key"] and parent.get("costs"):
            f["parent_costs"] = dict(parent["costs"])
        tau = ctx["tau"]
        realized = {"bottled": n_real / N, "bottled_count": n_real, "tau": None if tau is None else float(tau),
                    "n_cal": f["n_cal"], "basis": "realized"}
        if ctx.get("chunk_level") is not None and kind != "run":
            # batch-spent chunk: its own threshold used the chunk level alpha* > delta. A forecast describes an
            # ordinary (eps, delta) pour of this bottle, so the realized arm is replaced by its EXACT standard-design
            # counterfactual on the same calibration sample (same n and labels, the k of level delta): chunk-level
            # coverage never reaches a later votes='auto' decision
            f["chunk_level"] = float(ctx["chunk_level"])
            f["arms_design"] = "standard (eps, delta) design on this chunk's calibration sample"
            try:
                realized = await self._arm_outcome(ctx["final"], ctx["score"], ctx)
                realized["basis"] = "exact (standard design)"
            except _ArmUnavailable as e:
                realized = {"bottled": None, "basis": "unavailable", "reason": str(e)}
        if kind == "run" and not cfg.forecast_run:
            # no run-time forecast (default): only the measured unit costs are recorded; the first pour from this
            # bottle uses the gain-prior rule, and so does a pour after a cold first pour (no warm gain measured yet)
            f["status"] = "unavailable"
            f["reason"] = "run-time forecast off (cfg.forecast_run=False)"
        elif kind == "run":
            cert = ctx["cert"]
            plan = list(getattr(cert, "plan", None) or [])
            jl = cert.chosen_look if (cert.chosen_look is not None and cert.chosen_look < len(plan)) else len(plan) - 1
            alpha = float(plan[jl]["alpha"]) if plan else float(cfg.delta)
            vote, vscore = fc.get("vote"), fc.get("vscore")
            f["status"] = "pending"
            f["pending"] = {"texts": list(ctx["texts"]), "cal": {int(i): str(a) for i, a in ctx["cal"].items()},
                            "vote": None if vote is None else [None if v is None else str(v) for v in vote],
                            "vscore": None if vscore is None else [float(x) for x in vscore],
                            "N": N, "n_cal": f["n_cal"], "alpha": alpha, "seed": int(cfg.seed),
                            "meth": ctx["meth"], "fusion": cfg.fusion}
        elif kind == "pour_warm":
            try:
                cold = await self._arm_outcome(fc["b_pred"], fc["b_conf"], ctx)
                cold["basis"] = "exact"
            except _ArmUnavailable as e:
                cold = {"bottled": None, "basis": "unavailable", "reason": str(e)}
            f["arms"] = {"cold": cold, "warm": realized}
            if cold.get("bottled") is not None and realized.get("bottled") is not None:
                f["gain"] = {"value": float(realized["bottled"] - cold["bottled"]), "age": 0, "basis": "exact"}
        else:
            f["arms"] = {"cold": realized}
            f["gain"] = self._carry_gain(parent, f["key"])
        # forecast v2 inputs, recorded for every batch whatever cfg.forecast_v2 is (read only by votes='auto'):
        # the vote-expert agreement of this batch (or carried) and the slim cross-fit block. Report-only, as above.
        try:
            ag = self._agreement(kind, fc, ctx, parent, f["key"])
            if ag is not None:
                f["agree"] = ag
            blk = self._xfit_block(art, ctx)
            if blk is not None:
                f["xfit"] = blk
        except Exception as e:  # noqa: BLE001 - never breaks the measured forecast
            f["v2_error"] = f"{type(e).__name__}: {e}"[:300]
        return f

    @staticmethod
    def _agreement(kind: str, fc: dict, ctx: dict, parent: dict | None, key: dict) -> dict | None:
        """(1) a = share of calibration items with a usable RAW expert answer on which this batch's fused answer equals
        it (run() fused mode and warm pours; uniform calibration sample, so an unbiased estimate for the batch). A
        cold pour carries its parent's agreement one pour older. Unusable expert answers are excluded from both
        counts (they are disagreements for the certificate, which e_new covers)."""
        voted = kind == "pour_warm" or (kind == "run" and fc.get("vote") is not None)
        if voted:
            final, cal = ctx["final"], ctx["cal"]
            use = [i for i, a in cal.items() if a != NONE]
            if not use:
                return None
            agree = int(sum(final[i] == cal[i] for i in use))
            return {"a": agree / len(use), "agree": agree, "n": len(use), "n_cal": len(cal), "age": 0,
                    "basis": "measured", "source": kind}
        if kind == "pour_cold" and parent and parent.get("key") == key and parent.get("agree"):
            g = dict(parent["agree"])
            g["age"], g["basis"] = int(g.get("age", 0)) + 1, "carried"
            return g
        return None

    def _xfit_block(self, art, ctx: dict) -> dict | None:
        """Slim block for the lazy cross-fits (3)/(4): 64-bit text hashes of the certified population (the bottle's
        memory holds the texts and certified labels), the RAW calibration answers by batch index (n_cal entries; exact
        even when a text recurs in the batch), N, n, the level alpha of the look used, the method, the allowance m,
        the Bottler seed and the bottle version and memory size it describes. No texts."""
        if ctx["trivial"] or not ctx["cal"]:
            return None
        cert = ctx["cert"]
        plan = list(getattr(cert, "plan", None) or [])
        jl = cert.chosen_look if (cert.chosen_look is not None and cert.chosen_look < len(plan)) else len(plan) - 1
        alpha = float(plan[jl]["alpha"]) if plan else float(self.cfg.delta)
        return {"v": 1, "hashes": [_text_hash(t) for t in ctx["texts"]],
                "cal": {int(i): str(a) for i, a in ctx["cal"].items()}, "N": int(ctx["N"]), "n_cal": len(ctx["cal"]),
                "alpha": alpha, "meth": ctx["meth"], "m": ctx["mkw"].get("m"), "seed": int(self.cfg.seed),
                "version": int(art.version), "memory": len(art.texts)}

    @staticmethod
    def _carry_gain(parent: dict | None, key: dict) -> dict | None:
        """Warm gain carried through a cold pour: the parent's gain (or its unresolved run-time forecast), one pour
        older ('stale'; decayed by DEFAULT_GAIN_DECAY per age at decision time)."""
        if not parent or parent.get("key") != key:
            return None
        g = parent.get("gain")
        if g is None:
            pend = parent.get("pending") if parent.get("source") == "run" else None
            return None if pend is None else {"value": None, "age": 1, "basis": "stale", "origin": "crossfit",
                                              "pending": pend}
        out = {"value": g.get("value"), "age": int(g.get("age", 0)) + 1, "basis": "stale",
               "origin": g.get("origin", g.get("basis"))}
        if g.get("value") is None and g.get("pending") is not None:
            out["pending"] = g["pending"]
        return out

    def _forecast_arms(self, artifact, f: dict) -> dict:
        """Bottled-share forecasts of both arms at the SOURCE batch size (resolving a pending run forecast)."""
        src = f.get("source")
        xfit = bool(self.cfg.forecast_run)       # cross-fitted (run-time) forecasts are used only when opted in
        if src == "run":
            if not xfit or f.get("pending") is None:
                why = ("unavailable (run-time forecast off: cfg.forecast_run=False)" if not xfit
                       else "unavailable (no run-time forecast recorded)")
                return {"cold": None, "warm": None, "basis_cold": why, "basis_warm": why}
            r = self._resolve_pending(artifact, f.get("pending"))
            f["status"] = "resolved"
            return {"cold": r.get("cold"), "warm": r.get("warm"), "basis_cold": "crossfit",
                    "basis_warm": "crossfit" if r.get("warm") is not None else "unavailable (no votes in run)"}
        arms = f.get("arms") or {}
        cold = (arms.get("cold") or {}).get("bottled")
        if src == "pour_warm":
            return {"cold": cold, "warm": (arms.get("warm") or {}).get("bottled"),
                    "basis_cold": (arms.get("cold") or {}).get("basis"), "basis_warm": "realized"}
        g = f.get("gain")
        if not g:
            return {"cold": cold, "warm": None, "basis_cold": "realized", "basis_warm": "unavailable (no warm gain)"}
        val = g.get("value")
        if val is None and g.get("pending") is not None and xfit:
            r = self._resolve_pending(artifact, g["pending"])
            val = None if (r.get("warm") is None or r.get("cold") is None) else float(r["warm"] - r["cold"])
        if val is None or cold is None:
            why = ("unavailable (stale gain is a run-time forecast; cfg.forecast_run=False)"
                   if (val is None and g.get("pending") is not None and not xfit) else "unavailable")
            return {"cold": cold, "warm": None, "basis_cold": "realized", "basis_warm": why}
        age = int(g.get("age", 1))
        dec = self.cfg.forecast_gain_decay
        warm = float(min(1.0, max(0.0, cold + val * dec ** age)))
        return {"cold": cold, "warm": warm, "basis_cold": "realized",
                "basis_warm": f"stale: gain {val:+.4f} x {dec}^{age}"}

    def _resolve_pending(self, artifact, pend: dict | None) -> dict:
        """Cross-fitted CERTIFIED-OUTPUT forecast of a run() batch: every batch item held in the bottle is scored by a
        student trained (5 folds, seed sub_seed(seed, 7)) on the other items' certified answers, i.e. a held-out
        proxy of the exported bottle. Cold arm: (top label, its probability); warm arm: this run's votes fused with
        it exactly as a warm pour would. Each arm is certified on the run's own calibration sample (raw expert
        answers) at the run's level; bottled share = share of non-calibration items above tau * (1 - n/N). Cached
        in pend['resolved'] (the heavy inputs are then dropped). Uses only past batches: no LLM call, no
        calibration stream."""
        if pend is None:
            return {}
        if "resolved" in pend:
            return pend["resolved"]
        cfg = self.cfg
        texts, cal = pend["texts"], pend["cal"]
        pos = {t: j for j, t in enumerate(artifact.texts)}
        row_of: dict = {}
        mem: list = []
        for t in texts:                     # one row per distinct batch text held in memory (its certified answer)
            j = pos.get(t)
            if j is not None and t not in row_of:
                row_of[t] = len(mem)
                mem.append(j)
        out: dict = {"n_rows": len(mem)}
        cal_rows: dict = {}
        for i, a in cal.items():
            r = row_of.get(texts[int(i)])
            if r is not None and r not in cal_rows:
                cal_rows[r] = a
        cr = sorted(cal_rows)
        out["n_cal_rows"] = len(cr)
        if len(mem) >= 2 and cr:
            T = [artifact.texts[j] for j in mem]
            L = [artifact.labels[j] for j in mem]
            E = None if artifact.emb is None else artifact.emb[np.asarray(mem, dtype=np.int64)].astype(np.float32)
            vote, vscore = pend.get("vote"), pend.get("vscore")
            v_row, s_row = None, None
            if vote is not None:
                v_row, s_row = [None] * len(T), np.zeros(len(T))
                for i, t in enumerate(texts):           # newest batch item per text (the bottle keeps the newest)
                    r = row_of.get(t)
                    if r is not None:
                        v_row[r], s_row[r] = vote[i], float(vscore[i])
            cf = cross_fit(T, L, list(range(len(T))), seed=sub_seed(int(pend["seed"]), FORECAST_STREAM_KEY), dense=E,
                           query=v_row, folds=cfg.crossfit_folds, student=artifact.student_kwargs())
            pred, conf = cf[0], np.asarray(cf[1], float)
            arms = {"cold": (pred, conf)}
            if v_row is not None:
                q = np.asarray(cf[2], float)
                ans = [v if v is not None else p for v, p in zip(v_row, pred)]
                if pend.get("fusion", "prob") == "prob":
                    sup = np.where([v is not None for v in v_row], q, conf)
                else:
                    sup = np.array([float(p == a) for p, a in zip(pred, ans)]) * conf
                arms["warm"] = (ans, s_row + sup)
            N, n = int(pend["N"]), int(pend["n_cal"])
            noncal = np.ones(len(T), bool)
            noncal[cr] = False
            cap_kw = {"max_expected_eps": cfg.max_expected_eps} if cfg.max_expected_eps is not None else {}
            for name, (ans, sc) in arms.items():
                fin = [a if (a is not None and a != NONE) else NONE for a in ans]
                valid = np.array([a != NONE for a in fin], bool)
                s = np.where(valid, np.asarray(sc, float), -np.inf)
                cert = _fc_certify(s[cr], [disagreement(fin[r], cal_rows[r]) for r in cr], N, cfg.eps,
                                   float(pend["alpha"]), pend["meth"], **cap_kw)
                frac = float(np.mean((s[noncal] >= cert.tau) & valid[noncal])) if noncal.any() else 0.0
                out[name] = frac * max(0.0, 1.0 - n / N)
                out[name + "_tau"] = float(cert.tau) if np.isfinite(cert.tau) else None
        pend["resolved"] = out
        for k in ("texts", "cal", "vote", "vscore"):
            pend.pop(k, None)
        return out

    def _forecast_costs(self, res: BottleResult) -> None:
        """Attach measured unit costs to the forecast this batch just recorded (after its ledger is known):
        cheap-vote USD per certified ITEM (all contexts; this batch's votes, else carried from the parent forecast)
        and expert USD per CALL, each with the batch's average text length (rescaled to a new batch by length)."""
        art = res.artifact
        f = getattr(art, "forecast", None) if art is not None else None
        if not self.cfg.forecast or not f or f.get("artifact_version") != art.version or "costs" in f:
            return
        try:
            led, N, avg = res.ledger or {}, max(1, int(f["N"])), float(f.get("avg_chars", 0.0))
            parent = f.pop("parent_costs", None) or {}
            c: dict = {}
            tok = self.cfg.cost_basis == "tokens"       # also record token volumes (only in that mode)
            cv = led.get(PHASE_CHEAP_VOTE)
            if cv and cv.get("calls"):
                c.update(vote_usd_per_item=float(cv["usd"]) / N, vote_avg_chars=avg, vote_basis="measured")
                if tok:
                    c["vote_tokens_per_item"] = float(cv.get("prompt_tokens", 0) + cv.get("completion_tokens", 0)) / N
            elif parent.get("vote_usd_per_item") is not None:
                c.update(vote_usd_per_item=parent["vote_usd_per_item"], vote_avg_chars=parent["vote_avg_chars"],
                         vote_basis="carried")
                if tok and parent.get("vote_tokens_per_item") is not None:
                    c["vote_tokens_per_item"] = parent["vote_tokens_per_item"]
            ex = [v for k, v in led.items() if k.startswith(EXPERT_PHASE_PREFIX)]
            calls = sum(int(v.get("calls", 0)) for v in ex)
            if calls:
                c.update(expert_usd_per_call=float(sum(float(v["usd"]) for v in ex)) / calls, expert_avg_chars=avg,
                         expert_basis="measured")
                if tok:
                    c["expert_tokens_per_call"] = float(sum(float(v.get("prompt_tokens", 0))
                                                            + float(v.get("completion_tokens", 0)) for v in ex)) / calls
            elif parent.get("expert_usd_per_call") is not None:
                c.update(expert_usd_per_call=parent["expert_usd_per_call"],
                         expert_avg_chars=parent["expert_avg_chars"], expert_basis="carried")
                if tok and parent.get("expert_tokens_per_call") is not None:
                    c["expert_tokens_per_call"] = parent["expert_tokens_per_call"]
            f["costs"] = c
        except Exception as e:  # noqa: BLE001
            res.diagnostics["forecast_error"] = f"costs: {type(e).__name__}: {e}"[:300]

    async def pour(self, artifact: BottleArtifact, texts: list[str], votes: str | bool = "auto",
                   emb: np.ndarray | None = None, *, _n_memo: int = 0, _memo_diag: dict | None = None,
                   _chunk: bool = False, _chunk_design: "_ChunkDesign | None" = None) -> BottleResult:
        """Answer a NEW workload with a previously built bottle, certified on this workload's own uniform calibration
        sample (same finite-population guarantee; no iid assumption about how the new workload relates to the old
        one); expert answers are folded back into the returned bottle (flywheel).
          votes=False : COLD pour, the bottle alone (zero cheap-LLM calls)
          votes=True  : WARM pour, cheap votes on the new batch fused with the bottle (score = vote share of the answer
                        + bottle probability of the answer). The bottle never saw this batch, so no cross-fitting.
          votes='auto': warm iff the forecast the bottle carries predicts a lower cost (cfg.forecast, forecast_v2;
                        see _votes_pay), else the gain-prior rule contexts*ratio < gain_prior.
        The calibration permutation is a fresh stream keyed by (cfg.seed, bottle version + 1, bottle memory size): never
        the stream of a batch the bottle learned from (premise of the per-batch guarantee when the same items recur in
        the same order). For a hand-built version-0 bottle made from earlier bottlekit outputs, use a new cfg.seed.
        Measured: the bottle alone loses the LLM signal on tasks a TF-IDF student learns poorly (sst2 0.787 -> 0.102),
        while warm pours beat the bottle alone and match or beat a fresh cold start on the development tasks.
          emb         : embeddings of the new items (else cfg.embedder, else none). A dense bottle scores them with its
                        TF-IDF + dense student (cold and warm); without them it falls back to its TF-IDF student and
                        the next bottle is TF-IDF-only (diagnostics 'dense': 'dense_missing'). With cfg.embedder, a
                        bottle without (matching) memory embeddings is re-embedded first. The bottle is trained only on
                        earlier batches and is updated only after this batch is certified and escalated.
        Recurrence (cfg.memo=True, opt-in; assumes a deterministic expert): items whose exact expert request the
        bottle holds with an expert answer under the same prompt fingerprint are answered from memory (provenance
        'memo', no votes, no score, no expert call); only the rest, R, is voted on, scored and certified, with the
        explicit allowance m = floor(eps*N_R) ('novel', default) or floor(eps*N) ('batch'). Errors on memo items are
        0 under the assumption, so P[share of all N answers != expert > eps] <= delta still holds.
        Right-sized pours: a pour much larger than the bottle's memory is split by a label-free plan (chunk_plan) into
        chunks poured in sequence, each an ordinary pour of the bottle updated by the previous chunk; the batch keeps
        P[share != expert > eps] <= delta_batch <= delta (see _pour_chunked). A plan with one chunk is exactly the
        single pour. cfg.chunking='cold' (default): the warm/cold choice is made ONCE for the whole pour,
        before planning and before any embedding, vote or calibration draw (votes=False/True as given; 'auto' -> the
        same _votes_pay call a single pour makes); only a cold pour with N >= cfg.chunk_cold_ratio x memory is split,
        and every chunk is then poured cold. Warm and smaller pours run the single pour with that decision (an
        'auto' pour equals the forced pour it chose). cfg.chunking='auto' (opt-in): every eligible pour is planned and
        each chunk decides its own votes."""
        cfg, N = self.cfg, len(texts)
        if _chunk_design is not None:
            # private: only _pour_chunked builds a _ChunkDesign (batch-spent plan); a public pour cannot set a design
            cd = _chunk_design
            if not (_chunk and isinstance(cd, _ChunkDesign) and cd.token is _CHUNK_TOKEN and not _n_memo
                    and cfg.chunk_spend == "batch" and int(cd.N) == N and 1 <= int(cd.n) <= N
                    and float(cfg.delta) <= float(cd.alpha) <= self._chunk_alpha_max()):
                raise ValueError("_chunk_design is private to chunked pours (Bottler._pour_chunked)")
        snap = self.client.ledger.snapshot()
        if N == 0:
            return BottleResult([], [], {"certified": False, "N": 0}, {}, {"N": 0}, artifact)
        if not (isinstance(votes, str) and votes == "auto") and bool(votes) and cfg.cheap is None:
            raise ValueError("a warm pour needs cfg.cheap")
        self._require_models(cfg.expert, cfg.cheap if (isinstance(votes, str) or bool(votes)) else None)
        chunk_info = None
        decided = None            # ('cold' rule) the pour-level (use_votes, why), made once before planning
        decided_wall = None       # its wall time (outside the deterministic decision record)
        if not _chunk and not _n_memo:
            M0 = len(artifact.texts)
            if cfg.chunking == "cold":
                # label-free gate from (N, memory size at entry, cfg) only: eligible configuration and N >= ratio x M0
                gate = self._cold_gate(N, M0)
                if gate is None:
                    # the pour-level warm/cold decision: a function of earlier batches (the bottle's forecast) and the
                    # new texts' lengths only, made BEFORE the plan and before any embedding, vote or calibration draw,
                    # with exactly the inputs a single pour gives _votes_pay; never re-decided per chunk
                    t_dec = time.perf_counter()
                    decided = self._pour_votes(texts, artifact, votes, emb)
                    decided_wall = ({"votes_decision": float(time.perf_counter() - t_dec)}
                                    if isinstance(votes, str) and votes == "auto" else None)
                    if decided[0]:
                        chunk_info = {"chunks": 1, "rule": "cold", "N": N, "M0": M0,
                                      "ratio": float(cfg.chunk_cold_ratio), "decision": "warm",
                                      "reason": "warm pour: never split ('cold' rule)"}
                    else:
                        chunk_info = self.chunk_plan(N, M0)
                        chunk_info["decision"] = "cold"
                        if chunk_info.get("chunks", 1) >= 2:
                            return await self._pour_chunked(artifact, texts, False, emb, chunk_info, snap,
                                                            votes_decision=decided[1])
                else:
                    chunk_info = gate
            else:
                # label-free plan from (N, memory size at entry, eps, delta, cfg) only, before any vote, embedding or draw
                chunk_info = self.chunk_plan(N, M0)
                if chunk_info.get("chunks", 1) >= 2:
                    return await self._pour_chunked(artifact, texts, votes, emb, chunk_info, snap)
        # RECURRENCE (before any embedding, vote or calibration draw): items whose exact expert request is in the
        # bottle's memory with an EXPERT answer under the same prompt fingerprint are answered from memory. Mo and
        # R are functions of the batch texts and the bottle only (fixed before this batch); no expert answer of this
        # batch can influence them.
        n_memo = int(_n_memo)
        if n_memo:                     # inner call of _pour_with_memo: `texts` is the population R
            memo_diag = dict(_memo_diag or {})
        else:
            memo_idx, memo_diag = self._memo_lookup(artifact, texts)
            if memo_idx:
                return await self._pour_with_memo(artifact, texts, votes, emb, memo_idx, memo_diag, snap)
        wall = None
        if decided is not None:
            use_votes, why = decided
            wall = decided_wall
        else:
            t_dec = time.perf_counter()
            use_votes, why = self._pour_votes(texts, artifact, votes, emb)
            if isinstance(votes, str) and votes == "auto":
                wall = {"votes_decision": float(time.perf_counter() - t_dec)}
        diag = {"mode": "pour_warm" if use_votes else "pour", "N": N, "votes_decision": why,
                "artifact_version": artifact.version, "artifact_memory": len(artifact.texts), **memo_diag}
        if wall is not None:     # wall time lives OUTSIDE the deterministic decision record
            diag["wall_time_s"] = wall
        emb = await self._resolve_emb(texts, emb, diag)          # before any vote or calibration draw
        eid = self._embedder_id()
        if (emb is not None and cfg.embedder is not None and artifact.texts
                and (not artifact.dense or artifact.emb_dim != emb.shape[1]
                     or (eid is not None and artifact.embedder_id is not None and eid != artifact.embedder_id))):
            try:   # bottle without (matching) memory embeddings: re-embed its memory with the configured embedder
                artifact = artifact.with_embeddings(await self._call_embedder(artifact.texts), eid)
                diag["dense_memory_reembedded"] = True
            except Exception as e:  # noqa: BLE001
                diag["embedder_error"] = f"{type(e).__name__}: {e}"[:300]
        if emb is not None:
            try:
                emb = artifact.check_emb(emb, N, eid)           # width / embedder mismatch -> ValueError
            except ValueError as e:
                if "embedder_error" not in diag:
                    raise                                       # caller-supplied mismatch: fail loudly
                # the configured embedder could not re-embed the bottle's memory: documented fallback to TF-IDF
                diag["dense_fallback"] = f"{e}"[:300]
                warnings.warn(f"dense bottle unusable after embedder failure ({e}); using TF-IDF features only",
                              stacklevel=2)
                emb = None
        diag.update({"dense": artifact.dense_status(emb), "bottle_dense": artifact.dense,
                     "embedder_id": artifact.embedder_id if artifact.dense else eid})
        if diag["dense"] == "dense_missing":
            warnings.warn("dense_missing: " + ("the new items have no embeddings" if artifact.dense else
                                               "the bottle has no memory embeddings") +
                          "; the TF-IDF student answers and the next bottle is TF-IDF-only", stacklevel=2)
        if use_votes:
            v = await vote_label(self.client, cfg.cheap, self.spec, texts, k=cfg.k, contexts=cfg.contexts,
                                 warmup=cfg.warmup, seed=cfg.seed, balanced=cfg.balanced, weighted=cfg.weighted,
                                 phase=PHASE_CHEAP_VOTE, early_frac=cfg.early_frac)
            vote, vscore = v["answers"], np.asarray(v["scores"], float)
            self._warn_cheap(vote, "votes")
            b_pred, b_conf, b_q = artifact.predict(texts, query=vote, emb=emb)
            final = [a if a is not None else s for a, s in zip(vote, b_pred)]
            if cfg.fusion == "prob":
                support = np.where([a is not None for a in vote], b_q, b_conf)
            else:
                support = np.array([float(s == f) for s, f in zip(b_pred, final)]) * np.asarray(b_conf)
            score = vscore + support
            diag["position_reliability"] = v["reliability"]
            # the bottle alone (cold counterfactual of the forecast); `vote` is read only by the health report
            fc = {"kind": "pour_warm", "b_pred": b_pred, "b_conf": b_conf, "vote": vote}
        else:
            final, conf = artifact.predict(texts, emb=emb)
            score = np.asarray(conf, float)
            fc = {"kind": "pour_cold", "b_pred": final, "b_conf": score}   # b_*: read only by the health report
        # Fresh calibration stream for this batch: keyed by the bottle's lineage (next version, memory size), so it is
        # never the stream of any batch the bottle learned from (run: (seed, 1); a pour of version v: (seed, 1, v+1, .)).
        res = await self._certify_and_escalate(texts, final, score, diag, base_artifact=artifact, emb=emb,
                                               cal_stream=cal_stream_key(cfg.seed, artifact)[2:], n_memo=n_memo,
                                               fc_inputs=fc, **({"chunk_design": _chunk_design}
                                                                if _chunk_design is not None else {}))
        if chunk_info is not None:
            res.diagnostics["chunking"] = {k: v for k, v in chunk_info.items() if k not in ("designs", "risk")}
        res.ledger = self.client.ledger.since(snap)
        self._forecast_costs(res)
        return res

    def _pour_votes(self, texts, artifact: BottleArtifact, votes, emb) -> tuple[bool, dict]:
        """(use_votes, why) of a pour: votes=True/False as given; 'auto' -> _votes_pay on the bottle's forecast (or the
        gain-prior rule) and the new texts' lengths, with dense_next from emb/cfg.embedder and the bottle. A function
        of earlier batches and of the texts only (no embedding call, vote, draw or label)."""
        cfg = self.cfg
        if isinstance(votes, str) and votes == "auto":
            dense_next = ((emb is not None or cfg.embedder is not None)
                          and (artifact.dense or cfg.embedder is not None))
            return self._votes_pay(texts, artifact, dense_next=dense_next)
        use_votes = bool(votes)
        if use_votes and cfg.cheap is None:
            raise ValueError("a warm pour needs cfg.cheap")
        return use_votes, {"reason": "fixed by caller"}

    # ------------------------------------------------------------------------------------------------ right-sized pours
    def _chunk_ineligible(self) -> list[str]:
        """Configurations without an exact per-chunk law or without a flywheel: never split (label-free)."""
        cfg = self.cfg
        # certificate='hypergeometric' is the stricter alternative test, kept for reproduction: never split (the
        # convolution would hold for it too: same closed-form k-rule with its own smaller k_c)
        return [w for w, bad in (("certificate != 'rectified'", cfg.certificate != "rectified"),
                                 ("calibration != 'sequential'", cfg.calibration != "sequential"),
                                 ("custom looks", bool(cfg.looks)), ("design != 'single'", cfg.design != "single"),
                                 ("expert_budget set", cfg.expert_budget is not None), ("memo=True", bool(cfg.memo)),
                                 ("export=False", not cfg.export)) if bad]

    def _cold_gate(self, N: int, M0: int) -> dict | None:
        """chunking='cold': None if the pour may be split once it is decided cold (eligible configuration and
        N >= chunk_cold_ratio x M0), else the no-split record for diagnostics['chunking']. Label-free: (N, M0, cfg)."""
        cfg = self.cfg
        why = self._chunk_ineligible()
        if why:
            return {"chunks": 1, "rule": "cold", "reason": "not eligible: " + ", ".join(why)}
        r = float(cfg.chunk_cold_ratio)
        if not int(N) >= r * int(M0):
            return {"chunks": 1, "rule": "cold", "N": int(N), "M0": int(M0), "ratio": r,
                    "reason": f"'cold' rule: N = {int(N)} < {r:g} x memory ({int(M0)}); the single pour runs"}
        return None

    def chunk_min_size(self) -> int:
        """N_min of this configuration: the smallest batch whose single-look calibration share is <= cfg.chunk_tax
        (certify.min_batch_for_tax = the exact scan certify.min_certified_size; label-free, cached; 355 at eps=.05,
        delta=.1)."""
        cfg = self.cfg
        return min_batch_for_tax(float(cfg.eps), float(cfg.delta), cfg.certificate, float(cfg.design_c),
                                 float(cfg.chunk_tax), bool(cfg.tune))

    def _chunk_design(self, N: int) -> tuple[int, int, int]:
        """(n, N, k) of the ordinary single-look certificate of a batch of N under this configuration: exactly the
        look size and k that _certify_and_escalate / certify_sequential apply (single_look_design + tuned_looks at the
        full delta; k_used with the optional mean-risk cap). Label-free."""
        cfg = self.cfg
        n = single_look_size(int(N), cfg.eps, cfg.delta, cfg.certificate, cfg.design_c, cfg.tune)
        n = min(n, int(N))
        k = k_used(n, int(N), cfg.eps, cfg.delta, cfg.certificate, max_expected_eps=cfg.max_expected_eps)
        return int(n), int(N), int(k)

    def _chunk_alpha_max(self) -> float:
        """Ceiling of the common per-chunk level of a batch-spent plan: cfg.chunk_alpha_max, never below delta (a
        ceiling <= delta leaves only the delta-level k-vector, i.e. the 'chunk' plan). Label-free."""
        return max(float(self.cfg.delta), float(self.cfg.chunk_alpha_max))

    def chunk_plan(self, N: int, M0: int) -> dict:
        """Label-free plan of a pour of N items into a bottle holding M0 memory items (cfg.chunking).
        Sizes: with N_min = chunk_min_size() and M = M0, repeat c = max(N_min, round(chunk_growth*M)); if the rest
        is < c + N_min (or the max_chunks-th chunk is reached) the chunk takes the whole rest; M += c. The plan is used
        only if it has >= 2 chunks, the configuration has an exact per-chunk law (certificate='rectified',
        design='single', calibration='sequential', no custom looks, no expert_budget, memo=False, export=True; the
        'hypergeometric' certificate is never split) and the whole-batch tail
        delta_batch = batch_tail(...)['delta_batch'] <= delta. A function of (N, M0, eps, delta, cfg) only: fixed
        before any vote, embedding, calibration draw or label of this batch. Returns {'chunks': 1, 'reason'} when
        the pour is not split (the single pour then runs unchanged).
        chunking='cold': the plan of a COLD pour (pour() calls it only after deciding cold), and only when
        N >= chunk_cold_ratio x M0 (else one chunk). M0 is the memory size at entry: label-free for THIS batch, but a
        function of earlier batches (BottleArtifact drops rows whose label is unusable and dedupes texts).
        cfg.chunk_spend='batch': after the delta-level plan passes, every chunk's k is raised (same sizes,
        partition and n_j) to k_used(n_j, N_j, eps, alpha*) at the common level alpha* of certify.batch_spent_alpha
        (largest k-vector with delta_batch <= delta, exact re-check); 'designs', 'risk' and 'delta_batch' then describe
        the raised plan, with 'alpha_chunk', 'designs_chunk_level', 'delta_batch_chunk_level' and 'k_raise' recorded.
        If no k can be raised within delta the plan is exactly the delta-level one ('spend_note')."""
        cfg = self.cfg
        N, M0 = int(N), int(M0)
        if cfg.chunking == "off":
            return {"chunks": 1, "reason": "chunking='off'"}
        if cfg.chunking == "cold":
            gate = self._cold_gate(N, M0)
            if gate is not None:
                return gate
        why = self._chunk_ineligible()
        if why:
            return {"chunks": 1, "reason": "not eligible: " + ", ".join(why)}
        try:
            n_min = self.chunk_min_size()
            sizes, M, rem = [], M0, N
            while rem > 0:
                c = max(n_min, int(round(cfg.chunk_growth * M)))
                last = cfg.max_chunks is not None and len(sizes) + 1 >= int(cfg.max_chunks)
                if rem < c + n_min or last:
                    c = rem
                sizes.append(int(c))
                M += c
                rem -= c
            base = {"N": N, "M0": M0, "N_min": int(n_min), "tax": float(cfg.chunk_tax), "growth": float(cfg.chunk_growth),
                    "max_chunks": cfg.max_chunks, "sizes": sizes}
            if cfg.chunking == "cold":
                base.update({"rule": "cold", "ratio": float(cfg.chunk_cold_ratio)})
            if len(sizes) < 2:
                return {"chunks": 1, **base, "reason": "one chunk: N < first chunk + N_min"}
            designs = [self._chunk_design(c) for c in sizes]
            bt = batch_tail(designs, N, cfg.eps, deltas=report_deltas(cfg.delta))
            single = self._chunk_design(N)
            base.update({"designs": designs, "risk": bt, "delta_batch": float(bt["delta_batch"]),
                         "n_cal_planned": int(sum(d[0] for d in designs)), "n_cal_single": int(single[0])})
            if not bt["delta_batch"] <= cfg.delta:
                return {**base, "chunks": 1, "reason": f"delta_batch {bt['delta_batch']:.4f} > delta"}
            if cfg.chunk_spend == "batch":
                # spend the batch's delta across the chunks: every chunk's k raised to the common level alpha*
                # at the SAME n_j (label-free; exact search + exact re-check on the final k-vector)
                alpha, dsb, info = batch_spent_alpha([(n, c) for n, c, _ in designs], N, cfg.eps, cfg.delta,
                                                     cfg.certificate, max_expected_eps=cfg.max_expected_eps,
                                                     alpha_max=self._chunk_alpha_max())
                btb = batch_tail(dsb, N, cfg.eps, deltas=(0.5, cfg.delta, 0.05, 0.01))
                if not (info.get("ok") and btb["delta_batch"] <= cfg.delta):   # cannot happen (re-checked inside)
                    raise AssertionError(f"batch-spent plan has delta_batch {btb['delta_batch']!r} > delta")
                if [int(d[2]) for d in dsb] == [int(d[2]) for d in designs]:
                    # no chunk's k can be raised within delta: exactly the 'chunk' plan (and pour)
                    base["spend_note"] = "chunk_spend='batch': no chunk's k can be raised within delta (chunk-level plan)"
                    return {**base, "chunks": len(sizes), "reason": "split"}
                base.update({"spend": "batch", "alpha_chunk": float(alpha),
                             "designs_chunk_level": [list(d) for d in designs],
                             "delta_batch_chunk_level": float(bt["delta_batch"]),
                             "designs": [tuple(int(x) for x in d) for d in dsb], "risk": btb,
                             "delta_batch": float(btb["delta_batch"]),
                             "k_raise": [int(b[2]) - int(a[2]) for a, b in zip(designs, dsb)]})
            return {**base, "chunks": len(sizes), "reason": "split"}
        except Exception as e:  # noqa: BLE001 - a planning problem never blocks a pour: no split
            return {"chunks": 1, "reason": f"planning error ({type(e).__name__}: {e})"[:300]}

    async def _pour_chunked(self, artifact: BottleArtifact, texts: list[str], votes, emb, plan: dict, snap,
                            votes_decision: dict | None = None):
        """Right-sized pour: the items are split by a uniform permutation from sub_seed(seed, 13, version+1, M0) into
        the planned chunk sizes; chunk j is an ORDINARY pour (its own votes decision, scores, fresh calibration stream
        (seed, 1, v_j+1, |memory_j|), certificate at (eps, delta), escalation) of the bottle exported by chunk j-1,
        whose memory holds only certified outputs. Answers and provenance are stitched back into input order; the
        returned bottle is the last chunk's.
        Guarantee. The plan (sizes, partition, every chunk's n_j and k_j) depends only on (N, M0, eps, delta, cfg,
        seed), so it is fixed before any label of this batch. Chunk j's scores are fixed before its calibration draw,
        which is a fresh uniform draw, so P[U_j > u | F_{j-1}] <= g_j(u) (the single-batch worst-case law), and by
        conditional stochastic dominance U = sum_j U_j has P[U > floor(eps*N)] <= delta_batch (certify.batch_tail),
        checked <= delta before chunk 1 starts. Each chunk ALSO holds at (eps, delta) on its own; these are separate
        statements, not one simultaneous claim at delta. Unusable expert answers add sum_j F_j / N.
        Batch-spent plan (cfg.chunk_spend='batch'): chunk j is certified with one look of the planned n_j at the
        common level alpha* (private _ChunkDesign), so its rule applies the planned, raised k_j; g_{n,N,k} never uses
        alpha, so the same convolution bounds the batch at delta_batch <= delta. Each chunk then holds at (eps, alpha*)
        on its own (its certificate says delta = alpha*). The k each chunk APPLIED is compared with the plan; any
        mismatch (or a chunk certificate not at alpha*) withdraws the whole-batch claim.
        chunking='cold': pour() decided the whole pour COLD before planning and passes votes=False, so every chunk is a
        cold pour (no per-chunk votes decision); `votes_decision` is that pour-level decision (diagnostics)."""
        cfg, N = self.cfg, len(texts)
        if emb is not None:
            emb = np.asarray(emb)
            if emb.ndim != 2 or emb.shape[0] != N:
                raise ValueError(f"emb has shape {emb.shape}; expected ({N}, d): one embedding row per text")
        v0, M0 = int(artifact.version), int(len(artifact.texts))
        part_key = [int(cfg.seed), PARTITION_STREAM_KEY, v0 + 1, M0]
        perm = np.random.default_rng(sub_seed(*part_key)).permutation(N)
        sizes = [int(c) for c in plan["sizes"]]
        spent = plan.get("spend") == "batch"
        alpha = float(plan["alpha_chunk"]) if spent else None
        art = artifact
        answers, prov = [None] * N, [None] * N
        certs, diags, chunk_ix = [], [], []
        off = 0
        for j, c in enumerate(sizes):
            idx = np.sort(perm[off:off + c])
            off += c
            # batch-spent plan: the chunk is certified with the PLANNED (n_j, alpha*) (k_j follows from them)
            ckw = {}
            if spent:
                n_j, N_j, k_j = (int(x) for x in plan["designs"][j])
                ckw["_chunk_design"] = _ChunkDesign(n_j, N_j, k_j, alpha, float(plan["delta_batch"]), j, len(sizes),
                                                    token=_CHUNK_TOKEN)
            r = await self.pour(art, [texts[int(i)] for i in idx], votes=votes,
                                emb=None if emb is None else emb[idx], _chunk=True, **ckw)
            for jj, i in enumerate(idx.tolist()):
                answers[i], prov[i] = r.answers[jj], r.provenance[jj]
            certs.append(r.certificate)
            diags.append(r.diagnostics)
            chunk_ix.append(idx.tolist())
            if r.artifact is None:          # export=True is required by the plan; defensive
                raise RuntimeError("a chunk returned no bottle (export must be on for chunked pours)")
            art = r.artifact
        # the realized per-chunk design must be the planned one (label-free; a mismatch would void delta_batch)
        realized = []
        for c, cd in zip(sizes, certs):
            if spent:   # the k the chunk's rule APPLIED (cert.plan), compared with the inflated plan (never re-derived)
                k_r = int((cd.get("chunk_design") or {}).get("k", -2))
            else:
                looks = (cd.get("risk") or {}).get("looks") or []
                k_r = int(looks[0]["k"]) if len(looks) == 1 else self._chunk_design(c)[2]   # k is label-free either way
            realized.append((int(cd.get("n_cal", -1)), int(c), k_r))
        planned = [tuple(int(x) for x in d) for d in plan["designs"]]
        consistent = realized == planned
        if spent:   # ... and every chunk certificate is at the planned common level alpha*
            consistent = consistent and all(cd.get("delta") == alpha for cd in certs)
        bt = plan["risk"]
        F = int(sum(int(cd.get("n_expert_unusable", 0)) for cd in certs))
        eps_cert = float(cfg.eps + F / N)
        dbatch = float(plan["delta_batch"])
        risk = {"basis": "convolution", "chunks": bt["chunks"], "expected_eps_bound": bt["expected_eps_bound"],
                "tail": bt["tail"], "eps_at_delta": bt["eps_at_delta"], "delta_spent": dbatch, "eps": cfg.eps,
                "delta": cfg.delta, "overflow": bt["overflow"], "n_expert_unusable": F, "unusable_share": F / N,
                "scope": bt["scope"] + "; final share <= U/N + unusable_share"}
        J = len(sizes)
        chunk_level = (f"{alpha:.4f} (batch-spent: every chunk's k raised to this common level, chosen label-free so "
                       f"that the whole batch stays <= delta)") if spent else f"{cfg.delta}"
        if consistent:
            guarantee = (f"whole batch: P[share of final answers != expert > {eps_cert:.4f}] <= {dbatch:.4f} "
                         f"(<= delta = {cfg.delta}; exact convolution of the {J} chunks' worst-case laws, plan fixed "
                         f"before any label{f'; eps {cfg.eps} + {F}/{N} expert-routed items without a usable expert answer' if F else ''})"
                         f"; separately, each chunk j: P[share of chunk j's answers != expert > eps + F_j/N_j] <= "
                         f"{chunk_level}; E[#bottled answers != expert]/N <= {bt['expected_eps_bound']:.2%} "
                         "(worst case of this procedure)")
        else:   # cannot happen unless the per-chunk design code drifts from _chunk_design (tests pin it)
            lv = (f"its own certificate's level (delta_j; planned {alpha:.4f})" if spent else f"{cfg.delta}")
            guarantee = (f"each chunk j: P[share of chunk j's answers != expert > eps + F_j/N_j] <= {lv} "
                         "(per-chunk only: the realized chunk designs differ from the plan, so no whole-batch claim)")
            warnings.warn("chunked pour: realized chunk designs differ from the plan; whole-batch claim withdrawn",
                          stacklevel=2)
        cert = {"tau": None, "taus": [cd.get("tau") for cd in certs],
                "certified": bool(all(cd.get("certified") for cd in certs)), "accept_all": False,
                # a batch-spent pour whose plan did not hold makes per-chunk claims at alpha* only: report that level,
                # so no consumer (session rows, sum of deltas) books the batch at cfg.delta
                "eps": cfg.eps, "delta": alpha if (spent and not consistent) else cfg.delta,
                "method": f"chunked pour: {J} chained single-look {cfg.certificate} pours; whole batch by convolution",
                "n_cal": int(sum(int(cd.get("n_cal", 0)) for cd in certs)), "N": N, "bad_at_tau": None,
                "looks": [], "chosen_look": None, "design": "chunked", "chunked": True,
                "plan": sizes, "chunk_designs": [list(d) for d in planned], "plan_consistent": bool(consistent),
                "delta_batch": dbatch if consistent else None, "batch_claim": bool(consistent),
                "chunks": certs, "partition_stream": part_key,
                "cal_stream": [cd.get("cal_stream") for cd in certs],
                "n_expert_unusable": F, "eps_certified": eps_cert,
                "n_cal_single_pour": plan.get("n_cal_single"),
                "scope": certs[0].get("scope") if certs else None,
                "risk": risk if consistent else {"basis": "unavailable", "reason": "plan mismatch"},
                "guarantee": guarantee}
        if spent:   # batch-spent plan: the common chunk level and the delta-level plan for reference
            cert.update({"chunk_spend": "batch", "alpha_chunk": alpha,
                         "chunk_designs_chunk_level": plan.get("designs_chunk_level"),
                         "delta_batch_chunk_level": plan.get("delta_batch_chunk_level")})
        modes = {d.get("mode") for d in diags}
        # 'mode' keeps its meaning (warm or cold): the chunks' common mode, 'pour_mixed' if votes='auto' chose both
        diag = {"mode": modes.pop() if len(modes) == 1 else "pour_mixed", "chunked": True, "N": N,
                "artifact_version": v0, "artifact_memory": M0,
                **({"votes_decision": votes_decision} if votes_decision is not None else {}),
                "chunking": {k: v for k, v in plan.items() if k not in ("designs", "risk")},
                "chunks": [{"N": int(c), "mode": d.get("mode"), "coverage_bottle": d.get("coverage_bottle"),
                            "artifact_version": d.get("artifact_version"), "artifact_memory": d.get("artifact_memory"),
                            "votes_decision": d.get("votes_decision"), "wall_time_s": d.get("wall_time_s"),
                            "dense": d.get("dense"),
                            "n_cal": int(cd.get("n_cal", 0)), "tau": cd.get("tau"), "forecast": d.get("forecast")}
                           for c, d, cd in zip(sizes, diags, certs)],
                "chunk_index": chunk_ix,
                "coverage_bottle": float(np.mean([p == "bottle" for p in prov])),
                "expert_calls_frac": float(np.mean([p != "bottle" for p in prov])),
                "n_fallback": int(sum(int(d.get("n_fallback", 0)) for d in diags)),
                "n_unusable_bottle_answers": int(sum(int(d.get("n_unusable_bottle_answers", 0)) for d in diags)),
                "expert_none_rate_calibration": (float(sum(float(d.get("expert_none_rate_calibration", 0.0)) *
                                                           int(cd.get("n_cal", 0)) for d, cd in zip(diags, certs))
                                                       / max(1, sum(int(cd.get("n_cal", 0)) for cd in certs)))),
                "next_bottle_dense": art.dense}
        for key in ("dense", "bottle_dense", "embedder_id", "forecast", "forecast_error"):
            if key in diags[-1]:
                diag[key] = diags[-1][key]
        if cfg.health:   # report-only: per-chunk health + stratified batch aggregate; nothing feeds a decision
            try:
                hs = [d.get("health") or {} for d in diags]
                diag["health"] = _chunked_health(hs, sizes, N, int(sum(p == "bottle" for p in prov)),
                                                 risk if consistent else None, float(cfg.delta),
                                                 level=float(cfg.health_level))
                # tag this pour's learning-curve rows (one per chunk, appended by each chunk's export)
                for r in art.history:
                    vo = r.get("version_out")
                    if isinstance(vo, int) and v0 + 1 <= vo <= v0 + J:
                        r["chunk"], r["chunks"] = vo - v0 - 1, J
            except Exception as e:  # noqa: BLE001
                diag["health"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        return BottleResult(answers, prov, cert, self.client.ledger.since(snap), diag, art)

    # ------------------------------------------------------------------------------------------------ risk profile
    def _risk(self, cert, tau_b, score, final, cal, s_cal, looks, spending, meth, n_unusable, N,
              level: float | None = None) -> dict:
        """Exact risk profile of the PROCEDURE that produced this certificate (report-only: scores, sample, tau and
        answers are untouched). U = bottled answers != expert among items the expert did not label; the final
        disagreement share is <= U/N + F/N with F = n_unusable (observed after the run, reported separately and
        never folded into the expectation).
          * budget-capped (tau_b set): the bottled set is fixed by the budget, not by the k-rule, so only the test
            inversion is reported: eps_at_delta[d] = M_upper(d) - K_end (over N), a data-dependent (1-d) confidence
            bound for each d separately (the same inversion as eps_achieved). NO expectation, NO fixed-eps' tail.
          * not certified: nothing is bottled, so U = 0.
          * otherwise: risk_profile over EVERY pre-specified look (cert.plan: the n labelled at that look and the k
            its closed-form rule actually applies, recorded by certify / certify_sequential, incl. any mean-risk cap).
            One look: 'exact'. Staged / custom looks: 'union bound' over all pre-specified looks (the selected look's
            error count is <= its own U_j <= max_j U_j, whatever the dominance stop did). certificate='hypergeometric'
            applies the same closed-form k-rule with its own (smaller) k_c, so the same law bounds its output count.
        level (batch-spent chunk): the certificate's own level alpha* (reported as the profile's 'delta', with
        eps_at_delta also at alpha*); None = cfg.delta (every other certificate, unchanged)."""
        cfg = self.cfg
        deltas = report_deltas(cfg.delta)
        d_level = cfg.delta
        if level is not None:          # a batch-spent chunk also reports at its own level alpha*
            d_level = float(level)
            deltas = (deltas[0], d_level) + tuple(deltas[1:])
        if tau_b is not None:
            seq = cfg.calibration == "sequential"
            eps_at = inversion_profile(deltas, score, final, cal, tau_b, N, meth, order_seed=s_cal if seq else None,
                                       looks=looks if seq else None, spending=spending if seq else None)
            return {"basis": "inversion", "looks": None, "expected_eps_bound": None, "expected_eps_closed_form": None,
                    "tail": None, "eps_at_delta": eps_at,
                    "delta_spent": None, "eps": cfg.eps, "delta": cfg.delta, "n_expert_unusable": int(n_unusable),
                    "unusable_share": n_unusable / N,
                    "scope": "budget-capped: eps_at_delta[d] is a (1-d) confidence bound on U/N from the hypergeometric "
                             "test inversion (each d separately); no expectation or fixed-eps' tail is claimed",
                    **({"max_expected_eps": cfg.max_expected_eps, "mean_cap_met": None}
                       if cfg.max_expected_eps is not None else {})}
        plan = list(getattr(cert, "plan", None) or [])
        if not plan:
            return {"basis": "unavailable"}
        ks = [int(p["k"]) for p in plan] if cert.certified else [-1] * len(plan)
        try:
            r = risk_profile([int(p["n_cal"]) for p in plan], N, ks, cfg.eps, d_level, deltas=deltas,
                             n_unusable=n_unusable)
        except RiskTooLarge as e:      # huge N x k: the exact tail table is skipped; one look keeps its exact mean
            if len(plan) == 1:
                cf = expected_eps_closed_form(int(plan[0]["n_cal"]), N, ks[0])
                r = {"basis": "closed form", "expected_eps_bound": cf, "expected_eps_closed_form": cf, "tail": None,
                     "eps_at_delta": None, "delta_spent": None, "note": str(e)}
            else:
                r = {"basis": "unavailable", "error": str(e)}
            r.update({"eps": cfg.eps, "delta": d_level, "n_expert_unusable": int(n_unusable),
                      "unusable_share": n_unusable / N})
        r["looks"] = [{"n_cal": int(p["n_cal"]), "alpha": float(p["alpha"]), "k": int(p["k"])} for p in plan]
        if not cert.certified:
            r["basis"] = "not certified"
        if cfg.max_expected_eps is not None:
            r["max_expected_eps"] = cfg.max_expected_eps
            r["mean_cap_met"] = bool(r["expected_eps_bound"] <= cfg.max_expected_eps + MEAN_CAP_SLACK)
        return r

    def _risk_budget_split(self, cert, tau_b, score, final, cal, s_cal, looks, spending, meth, n_unusable, N,
                           k_level: float, inv_level: float) -> dict:
        """Risk profile of a batch with an expert budget (binding or not), made on EVERY such batch with the
        same Bonferroni split as its guarantee. For each report level d (0.5, delta, 0.05, 0.01):
          eps_at_delta[d] = the k-rule's worst-case quantile at d/2 (its exact or union-bound law; budget not binding)
                          | the budget test inversion at d/2 (binding; at d = delta this is eps_achieved).
        Proof: U = bottled answers != expert. Let U_k be the count the k-rule's threshold bottles and U_b the count
        the label-free budget threshold tau_b bottles; both are defined on every draw. U = U_k when the budget does
        not bind and U = U_b when it binds, so P[U > N eps_at[d]] <= P[U_k > q_k(d/2)] + P[U_b > N inv(d/2)] <= d,
        whatever the labels select. No expectation, no fixed-eps' tail and no delta_spent are reported for ANY budget
        batch: U_b has no label-free law, and showing the k-rule's law only on the label-dependent event 'budget not
        binding' is biased (measured: mean U/N 0.035 against the k-rule's bound 0.030 on a benign workload).
        `looks` (non-binding batches only) is the k-rule's label-free design at k_level, read by the stream ledger."""
        cfg = self.cfg
        lv = sorted({float(x) for x in report_deltas(cfg.delta)}, reverse=True)
        half = [d / 2 for d in lv]
        out = {"basis": "budget split", "budget_binding": tau_b is not None, "looks": None,
               "expected_eps_bound": None, "expected_eps_closed_form": None, "tail": None, "eps_at_delta": None,
               "delta_spent": None, "eps": cfg.eps, "delta": cfg.delta,
               "delta_split": {"k_rule": float(k_level), "budget_inversion": float(inv_level)},
               "n_expert_unusable": int(n_unusable), "unusable_share": n_unusable / N}
        if cfg.max_expected_eps is not None:
            out.update({"max_expected_eps": cfg.max_expected_eps, "mean_cap_met": None})
        claim = ("each eps_at_delta[d] holds at level d over the whole procedure (k-rule at d/2 + budget inversion at "
                 "d/2), whichever part the labels select; no expectation, fixed-eps' tail or delta_spent is claimed "
                 "under an expert budget (the budget threshold's count has no label-free law, and the k-rule's law "
                 "would be shown only on the label-dependent event that the budget does not bind)")
        if tau_b is not None:
            seq = cfg.calibration == "sequential"
            inv = inversion_profile(half, score, final, cal, tau_b, N, meth, order_seed=s_cal if seq else None,
                                    looks=looks if seq else None, spending=spending if seq else None)
            out["eps_at_delta"] = {f"{d:g}": inv[f"{h:g}"] for d, h in zip(lv, half)}
            out["part"] = "budget inversion"
            out["scope"] = ("expert budget binding: eps_at_delta[d] is the hypergeometric test inversion at the budget "
                            "threshold at level d/2 (a data-dependent bound on U/N); " + claim)
            return out
        plan = list(getattr(cert, "plan", None) or [])
        if not plan:
            return {"basis": "unavailable", "note": "expert budget set; no k-rule design recorded"}
        ks = [int(p["k"]) for p in plan] if cert.certified else [-1] * len(plan)
        out["looks"] = [{"n_cal": int(p["n_cal"]), "alpha": float(p["alpha"]), "k": int(p["k"])} for p in plan]
        out["part"] = "k-rule" if cert.certified else "k-rule (nothing certified: U = 0)"
        try:
            r = risk_profile([int(p["n_cal"]) for p in plan], N, ks, cfg.eps, k_level, deltas=half,
                             n_unusable=n_unusable)
            out["eps_at_delta"] = {f"{d:g}": r["eps_at_delta"][f"{h:g}"] for d, h in zip(lv, half)}
            out["k_rule_basis"] = r["basis"]
        except RiskTooLarge as e:      # huge N x k: no exact quantiles (no precision bound either)
            out["note"] = str(e)
        out["scope"] = ("expert budget set, not binding in this batch: eps_at_delta[d] is the k-rule's worst-case "
                        "quantile at level d/2 (design: looks); " + claim)
        return out

    # ------------------------------------------------------------------------------------------------ recurrence
    def _memo_lookup(self, artifact: BottleArtifact, texts) -> tuple[dict, dict]:
        """{batch index: expert answer from memory} for the items of `texts` whose expert request key is held with
        src='expert' by a bottle whose fingerprint equals this Bottler's. Empty unless cfg.memo. Label-free: a
        function of the texts and the bottle only."""
        if not self.cfg.memo:
            return {}, {}
        if artifact.fingerprint is None:
            return {}, {"memo": "off: bottle has no prompt fingerprint (an older bottle or one built by hand)"}
        if artifact.fingerprint != self.fingerprint():
            return {}, {"memo": "off: prompt fingerprint mismatch (instruction, labels, max_chars or expert model "
                                "changed)"}
        tab, n_conflict = artifact.memo_table(self.spec)
        out = {}
        if tab:
            for i, t in enumerate(texts):
                a = tab.get(expert_key(self.spec, t))
                if a is not None:
                    out[i] = a
        d = {"memo": "on", "n_memo": len(out)}
        if n_conflict:
            d["memo_conflicting_keys_dropped"] = n_conflict
        return out, d

    async def _pour_with_memo(self, artifact, texts, votes, emb, memo_idx: dict, memo_diag: dict, snap):
        """Pour where some items are answered from expert memory: run the ordinary pour on the remaining population
        R only (votes, embeddings, scores, calibration, certificate with an explicit allowance, escalation), then put
        the memo answers back in place. Memo items are not re-added to the bottle (their text, expert answer and
        provenance are already in memory)."""
        N = len(texts)
        R = [i for i in range(N) if i not in memo_idx]
        if emb is not None:
            emb = np.asarray(emb)
            if emb.ndim != 2 or emb.shape[0] != N:
                raise ValueError(f"emb has shape {emb.shape}; expected ({N}, d): one embedding row per text")
        texts_R = [texts[i] for i in R]
        if R:
            res = await self.pour(artifact, texts_R, votes=votes, emb=None if emb is None else emb[R],
                                  _n_memo=len(memo_idx), _memo_diag=memo_diag)
        else:
            res = self._all_memo_result(artifact, memo_diag, N)
        answers, prov = [None] * N, [None] * N
        for i, a in memo_idx.items():
            answers[i], prov[i] = a, "memo"
        for j, i in enumerate(R):
            answers[i], prov[i] = res.answers[j], res.provenance[j]
        res.answers, res.provenance = answers, prov
        d = res.diagnostics
        d.update({"N": N, "N_population": len(R), "n_memo": len(memo_idx), "memo_share": len(memo_idx) / N,
                  "coverage_bottle": float(np.mean([p == "bottle" for p in prov])),
                  "expert_calls_frac": d.get("expert_calls", 0) / N})
        res.ledger = self.client.ledger.since(snap)
        return res

    def _all_memo_result(self, artifact, memo_diag, N: int) -> BottleResult:
        """Every item of the batch is answered from expert memory: nothing to vote on, score or certify."""
        cfg = self.cfg
        cert = {"tau": None, "certified": True, "accept_all": False, "eps": cfg.eps, "delta": cfg.delta,
                "method": "none: every item answered from expert memory (no certificate needed)", "n_cal": 0,
                "N": N, "N_population": 0, "n_memo": N, "bad_at_tau": None, "looks": [], "chosen_look": None,
                "no_calibration_needed": True,
                "allowance_m": 0, "memo_allowance": cfg.memo_allowance, "eps_population": None,
                "n_expert_unusable": 0, "eps_certified": float(cfg.eps),
                "guarantee": (f"P[share of final answers != expert > {cfg.eps}] <= {cfg.delta} (every answer is a "
                              "stored expert answer for the identical prompt; assumes a deterministic expert)"),
                "assumption": "deterministic expert: identical prompt -> identical answer"}
        diag = {"mode": "pour_memo", "artifact_version": artifact.version, "artifact_memory": len(artifact.texts),
                **memo_diag, "expert_calls": 0, "n_fallback": 0, "n_unusable_bottle_answers": 0}
        art = None
        if cfg.export:
            e0 = np.zeros((0, artifact.emb_dim), np.float32) if artifact.dense else None
            art = artifact.updated([], [], emb=e0, embedder_id=artifact.embedder_id, src=[],
                                   fingerprint=self.fingerprint(), **self._bottle_kw())
            diag["next_bottle_dense"] = art.dense
        return BottleResult([], [], cert, {}, diag, art)

    # ------------------------------------------------------------------------------------------------ certify + escalate
    async def _certify_and_escalate(self, texts, final, score, diag, base_artifact: BottleArtifact | None = None,
                                    emb: np.ndarray | None = None, cal_stream: tuple | None = None, n_memo: int = 0,
                                    fc_inputs: dict | None = None, chunk_design: "_ChunkDesign | None" = None):
        """Certify the population `texts` (N items) and escalate. n_memo > 0: these N items are the population R of
        a batch of N + n_memo items whose other answers come from expert memory (0 errors under a deterministic
        expert); the certificate then uses the explicit allowance m (cfg.memo_allowance) for R, look sizes from
        (N_R, eps_R = max(eps, m/N_R), delta) only, and reports shares over the whole batch.
        chunk_design (batch-spent chunk of a chunked pour; set by _pour_chunked only): ONE look of the planned
        n (no re-tuning) at the planned common chunk level alpha instead of delta, so the closed-form rule applies
        k_used(n, N, eps, alpha) (recorded in certificate['chunk_design'] and checked against the plan by the caller).
        The chunk certificate states delta = alpha. Its forecast (for later votes='auto' decisions) is computed at the
        standard (eps, delta) design on the same calibration sample, so it never carries chunk-level coverage."""
        cfg, N = self.cfg, len(texts)
        n_memo = int(n_memo)
        N_tot = N + n_memo
        restricted = n_memo > 0
        if restricted:   # label-free: (N, N_R, eps, allowance rule) only
            m_allow = int(math.floor(cfg.eps * (N_tot if cfg.memo_allowance == "batch" else N)))
            eps_design = max(cfg.eps, m_allow / N)
        else:
            m_allow, eps_design = None, cfg.eps
        trivial = restricted and m_allow >= N         # every error in R is within the allowance: no calibration
        mkw = {} if m_allow is None else {"m": m_allow}   # plain call signatures when nothing is restricted
        recur = cfg.memo and self._dedupe_expert
        known: dict | None = {} if recur else None    # per-batch expert_key -> usable expert answer
        estats: dict = {}
        final = [a if (a is not None and a != NONE) else NONE for a in final]
        valid = np.array([a != NONE for a in final], bool)
        score = np.where(valid, score, -np.inf)            # unusable answers are never bottled
        # Calibration stream: run() uses (seed, 1); pour() passes cal_stream = (bottle version
        # + 1, bottle memory size), so a pour never reuses the permutation of a batch its bottle was trained on (a
        # re-pour of the same items in the same order would otherwise calibrate exactly on memorized expert answers).
        # The ONE s_cal drives the sequential order, the fixed draw and the budget inversion.
        cal_key = (CAL_STREAM_KEY,) if cal_stream is None else (CAL_STREAM_KEY, *(int(x) for x in cal_stream))
        # numpy SeedSequence treats trailing zero entropy words as absent, so a pour key starting (1, 0, ...) could
        # collide with run()'s (1,). The version component must therefore stay >= 1.
        assert cal_stream is None or int(cal_stream[0]) >= 1, "cal_stream[0] (bottle version + 1) must be >= 1"
        s_cal = sub_seed(cfg.seed, *cal_key)
        # Look sizes and delta shares are fixed from (N, eps, delta, method[, design_c]) only, before the calibration
        # permutation is drawn: no score, answer or label reaches them. The SAME looks/spending feed certify_sequential
        # and the budget inversion (sequential_certified_eps).
        if cfg.looks:
            looks, spending = sorted({min(int(l), N) for l in cfg.looks if int(l) > 0}), cfg.spending
            design = "custom"
        elif cfg.design == "single":
            looks, spending = single_look_design(N, eps_design, cfg.design_c)
            looks, design = list(looks), "single"
        else:
            looks, spending = default_design(N)
            looks, design = list(looks), "staged"
        spending = spending if spending and len(spending) == len(looks) else None
        meth = cfg.certificate
        # Expert budget: Bonferroni split. With a budget the batch claim is EITHER the k-rule's
        # (budget not binding) OR the test inversion at the label-free budget threshold tau_b (binding), and WHICH one
        # is made depends on the calibration labels (the escalation count below the k-rule's tau). Each claim holds at
        # its own level, the label-dependent switch does not: with both at delta an adversarial workload
        # reaches P[violation] ~ 0.15 at delta = 0.1 (measured). So the k-rule runs at delta/2 and the inversion at
        # delta/2. Both failure events are defined on EVERY draw (tau_b depends on the scores only), so
        #   P[the claim made fails] <= P[k-rule fails] + P[inversion at tau_b fails] <= delta/2 + delta/2 = delta
        # whichever path the labels select. The split and the sizes (tuned at delta/2) are fixed from (cfg, N) only.
        # No budget (or a trivial memo population, which needs no calibration and never applies the budget): no split,
        # every level below is delta.
        budget_split = cfg.expert_budget is not None and not trivial
        split_level = cfg.delta / 2 if budget_split else cfg.delta
        if cfg.tune:   # sizes from (N, eps, delta, method[, m]) only: label-free
            looks = tuned_looks(looks, spend(looks, spending), N, eps_design, split_level, meth, **mkw)
        level = cfg.delta              # the certificate's level (delta; a batch-spent chunk: its planned alpha)
        if chunk_design is not None:
            if (restricted or cfg.looks or cfg.design != "single" or cfg.calibration != "sequential"
                    or cfg.expert_budget is not None):
                raise ValueError("a batch-spent chunk design needs the single-look sequential design without memo "
                                 "and without an expert budget")
            # the plan's (n_j, alpha*): label-free, fixed before this batch; n_j is NOT re-tuned at alpha*
            looks, spending, design = [int(chunk_design.n)], None, "single"
            level = float(chunk_design.alpha)
        # the k-rule's level and the budget inversion's level (each delta/2 under a budget; else level / delta)
        k_level, inv_level = (split_level, split_level) if budget_split else (level, cfg.delta)
        self._last_unusable = None
        expert_fn = lambda idx: self._expert(texts, idx, PHASE_EXPERT_CAL, known, estats)  # noqa: E731
        # optional stricter-only mean-risk cap (label-free k cap); omitted entirely by default
        cap_kw = {"max_expected_eps": cfg.max_expected_eps} if cfg.max_expected_eps is not None else {}
        if trivial:
            cert, cal = Certificate(-np.inf, cfg.eps, cfg.delta, f"none: allowance m={m_allow} >= population {N} "
                                    "(every item of R may be bottled; no calibration needed)", 0, N, 0), {}
        elif cfg.calibration == "sequential":
            cert, cal = await certify_sequential(score, final, expert_fn, N, cfg.eps, k_level, looks,
                                                 cfg.target_coverage, seed=s_cal, patience=cfg.patience,
                                                 method=meth, spending=spending, **cap_kw, **mkw)
        else:
            n_cal = (tune_size(min(cfg.n_cal, N), N, cfg.eps, split_level, meth, **mkw) if cfg.tune
                     else min(cfg.n_cal, N))
            cal_idx = sorted(np.random.default_rng(s_cal).choice(N, size=n_cal, replace=False).tolist())
            cal = await expert_fn(cal_idx)
            cert = certify(score[cal_idx], [disagreement(final[i], cal[i]) for i in cal_idx], N, cfg.eps, k_level,
                           meth, **cap_kw, **mkw)
        none_rate = float(np.mean([v == NONE for v in cal.values()])) if cal else 0.0
        diag["expert_none_rate_calibration"] = none_rate
        if none_rate > cfg.max_expert_none:
            raise ExpertFailure(f"expert {cfg.expert!r} gave no usable answer for {none_rate:.0%} of calibration items "
                                f"(limit {cfg.max_expert_none:.0%}); check the endpoint, model and label parser. "
                                f"Last unusable reply: {getattr(self, '_last_unusable', None)}")
        is_cal = np.zeros(N, bool)
        is_cal[list(cal)] = True
        # Propagation (cfg.memo): AFTER tau is fixed, every non-calibration item whose exact expert prompt equals a
        # calibration item's takes that (usable) expert answer. tau and the certificate are unchanged; under a
        # deterministic expert this only turns some unlabelled answers into the expert's own answer, so the output
        # errors stay <= the unlabelled bad items above tau that the certificate bounds.
        prop: dict = {}
        if cfg.memo and self._propagate and cal:
            cal_known = {}
            for i, a in cal.items():
                if a != NONE:
                    cal_known.setdefault(expert_key(self.spec, texts[i]), a)
            if cal_known:
                for i in range(N):
                    if not is_cal[i]:
                        a = cal_known.get(expert_key(self.spec, texts[i]))
                        if a is not None:
                            prop[i] = a
        accept = (score >= cert.tau) & valid
        esc = [i for i in np.nonzero(~accept & ~is_cal)[0].tolist() if i not in prop]
        cert_d = cert.as_dict()
        cert_d["cal_stream"] = [int(cfg.seed), *cal_key]          # sub_seed key of the calibration draw (audit)
        cert_d["design"] = design if cfg.calibration == "sequential" else "fixed"
        if design == "single" and cfg.calibration == "sequential":
            cert_d["design_c"] = cfg.design_c
        if chunk_design is not None:
            k_app = int(cert.plan[0]["k"]) if cert.plan else -2
            cert_d["chunk_design"] = {"n_cal": int(looks[0]), "N": int(N), "k": k_app, "k_planned": int(chunk_design.k),
                                      "alpha": level, "spend": "batch", "chunk": int(chunk_design.j),
                                      "chunks": int(chunk_design.J), "delta_batch": float(chunk_design.delta_batch)}
        split_note = ""
        if budget_split:   # the batch claim is at delta overall: k-rule at delta/2 + budget inversion at delta/2
            split_note = (f"expert_budget={cfg.expert_budget}: k-rule at {k_level:g} + budget inversion at "
                          f"{inv_level:g}, a Bonferroni split of delta = {cfg.delta:g}, so the claim made holds at "
                          f"{cfg.delta:g} whichever part the labels select")
            cert_d.update({"delta": cfg.delta, "expert_budget": cfg.expert_budget, "budget_capped": False,
                           "delta_split": {"k_rule": k_level, "budget_inversion": inv_level},
                           "guarantee": (f"P[share of final answers != expert > {cfg.eps}] <= {cfg.delta} (the "
                                         f"budget did not bind, so this claim comes from the k-rule; the probability is over the whole procedure, "
                                         f"not conditional on that branch; {split_note})")})
        tau_b = None
        if not trivial and cfg.expert_budget is not None and len(esc) > cfg.expert_budget:
            # Budget threshold from ALL scores (independent of the calibration labels): items strictly below the
            # budget-th lowest score are escalated; the eps actually achieved is certified by test inversion at
            # inv_level (delta/2: see budget_split above).
            tau_b = float(np.sort(score)[min(N - 1, cfg.expert_budget)])
            accept = (score >= tau_b) & valid
            esc = [i for i in np.nonzero(~accept & ~is_cal)[0].tolist() if i not in prop]
            # restricted: M_upper bounds the bad items of the population R; the share is over all N_tot answers
            dkw = {"denom": N_tot} if restricted else {}
            if cfg.calibration == "sequential":
                eps_hat = sequential_certified_eps(score, final, cal, s_cal, looks, tau_b, N, inv_level,
                                                   method=meth, spending=spending, **dkw)
            else:
                cal_idx = sorted(cal)
                eps_hat = certified_eps(score[cal_idx], [disagreement(final[i], cal[i]) for i in cal_idx], tau_b, N,
                                        inv_level, meth, **dkw)
            cert_d.update({"budget_capped": True, "expert_budget": cfg.expert_budget,
                           "tau_budget": tau_b if np.isfinite(tau_b) else None, "tau_unbudgeted": cert_d["tau"],
                           "eps_achieved": eps_hat, "budget_exceeded_by_unusable": max(0, len(esc) - cfg.expert_budget),
                           "guarantee": (f"P[share of final answers != expert > {eps_hat:.4f}] <= {cfg.delta} "
                                         f"(budget-capped: the budget inversion's claim; {split_note})")})
        exp_esc = await self._expert(texts, esc, PHASE_EXPERT_ESC, known, estats) if esc else {}
        answers_out, prov, n_fb = [], [], 0
        for i in range(N):
            if is_cal[i]:
                a, p = cal[i], "expert_calibration"
            elif i in prop:
                a, p = prop[i], "expert_calibration_dup"
            elif accept[i]:
                a, p = final[i], "bottle"
            else:
                a, p = exp_esc[i], "expert_escalation"
            if a == NONE and cfg.fallback and final[i] != NONE:
                a, p, n_fb = final[i], p + "+fallback", n_fb + 1
            answers_out.append(a)
            prov.append(p)
        # Items routed to the expert (calibration AND escalation) whose expert answer is unusable keep the bottle's
        # answer (+fallback) or none at all: they are output errors the certificate does not cover (it bounds the
        # unlabelled bottled items). Observed after the fact, so they are ADDED to the certified eps.
        n_unusable = int(sum(cal[i] == NONE for i in cal) + sum(exp_esc[i] == NONE for i in exp_esc))
        eps_base = cert_d["eps_achieved"] if cert_d.get("budget_capped") else cfg.eps
        eps_cert = float(eps_base + n_unusable / N_tot)
        cert_d.update({"n_expert_unusable": n_unusable, "eps_certified": eps_cert,
                       "scope": ("errors that reach the output (calibration items are answered by the expert)"
                                 if meth == "rectified" else "all bad items above the threshold, calibration included")})
        if n_unusable:
            cert_d["guarantee"] = (f"P[share of final answers != expert > {eps_cert:.4f}] <= {level} "
                                   f"(eps {eps_base:.4f} + {n_unusable}/{N_tot} expert-routed items without a usable "
                                   f"expert answer{'; budget-capped' if cert_d.get('budget_capped') else ''}"
                                   f"{'; ' + split_note if split_note else ''})")
        n_dedup = int(estats.get("deduped", 0))
        if restricted or prop or n_dedup:
            # Recurrence acted on this batch: record the population, the allowance and the assumption.
            g = cert_d["guarantee"]
            notes = []
            if restricted:
                cert_d.update({"N": N_tot, "N_population": N, "n_memo": n_memo, "allowance_m": m_allow,
                               "memo_allowance": cfg.memo_allowance, "eps_population": m_allow / N,
                               "eps_design": eps_design, "no_calibration_needed": bool(trivial),
                               "eps_bound_tight": float((m_allow + n_unusable) / N_tot) if not cert_d.get(
                                   "budget_capped") else eps_cert})
                notes.append(f"{n_memo} of {N_tot} items answered from expert memory; the other {N} certified with "
                             f"allowance m={m_allow} (eps_R={m_allow / N:.4f})")
                if m_allow / N > MEMO_WARN_RATIO * cfg.eps:   # errors concentrate on novel items
                    cert_d["eps_population_warning"] = True
                    warnings.warn(f"memo_allowance={cfg.memo_allowance!r}: the {N} novel items may carry up to "
                                  f"{m_allow / N:.1%} disagreement (> 2*eps); use memo_allowance='novel' to bound "
                                  f"them at eps", stacklevel=2)
            if prop:
                notes.append(f"{len(prop)} copies of calibration prompts got the calibration answer")
            if n_dedup:
                notes.append(f"{n_dedup} expert-routed copies shared one call per prompt")
            cert_d.update({"n_dup_propagated": len(prop), "n_expert_dedup": n_dedup,
                           "assumption": "deterministic expert: identical prompt -> identical answer"})
            cert_d["guarantee"] = g + " [" + "; ".join(notes) + "; assumes a deterministic expert]"
        try:   # report-only: a failure here must never break or change a run
            if budget_split:   # every budget batch, binding or not: the same split as the batch claim
                risk = self._risk_budget_split(cert, tau_b, score, final, cal, s_cal, looks, spending, meth,
                                               n_unusable, N, k_level, inv_level)
            else:
                risk = self._risk(cert, tau_b, score, final, cal, s_cal, looks, spending, meth, n_unusable, N,
                                  **({"level": level} if chunk_design is not None else {}))
        except Exception as e:  # noqa: BLE001
            risk = {"basis": "unavailable", "error": f"{type(e).__name__}: {e}"[:300]}
        budget_note = cfg.expert_budget is not None and tau_b is None   # only a trivial memo population
        if budget_note and risk.get("basis") in ("exact", "union bound", "closed form"):
            risk["scope"] = ((risk.get("scope") + "; ") if risk.get("scope") else "") + (
                "worst case of the k-rule only: an expert budget is set but did not bind in this batch; the budget "
                "fallback, used when it binds, is not covered by this profile")
        if restricted and risk.get("basis") == "budget split" and risk.get("eps_at_delta") is not None:
            risk["population_N"], risk["batch_N"] = int(N), int(N_tot)
            risk["scope"] += (f"; eps_at_delta is a share of the certified population ({N} of {N_tot} items; memo "
                              f"items excluded); as a share of the batch multiply by {N}/{N_tot}")
        if restricted and risk.get("expected_eps_bound") is not None:
            # memo items were removed from the certified population: the profile is for the N_R certified items;
            # as a share of the whole batch it is smaller by N_R / N_tot (the population share is also a valid bound)
            risk["population_N"], risk["batch_N"] = int(N), int(N_tot)
            risk["expected_eps_bound_batch"] = float(risk["expected_eps_bound"]) * N / N_tot
            risk["scope"] = ((risk.get("scope") + "; ") if risk.get("scope") else "") + (
                f"computed for the certified population ({N} of {N_tot} items; memo items excluded); as a share of "
                f"the batch multiply by {N}/{N_tot}")
        cert_d["risk"] = risk
        cert_d["guarantee"] += _risk_sentence(risk, N, budget_note)
        if chunk_design is not None:
            cert_d["guarantee"] += (f" [chunk {int(chunk_design.j) + 1} of {int(chunk_design.J)} of a batch-spent "
                                    f"chunked pour: held at the common chunk level alpha* = {level:.4f} >= delta = "
                                    f"{cfg.delta}; the whole batch is held at delta_batch = "
                                    f"{float(chunk_design.delta_batch):.4f} <= delta (see the pour's certificate)]")
        diag.update({"coverage_bottle": float(np.mean([p == "bottle" for p in prov])),
                     "expert_calls_frac": float(np.mean([p != "bottle" for p in prov])),
                     "n_fallback": n_fb, "n_unusable_bottle_answers": int((~valid).sum())})
        if cfg.memo:
            diag["expert_calls"] = int(estats.get("calls", 0))
            if prop or n_dedup:
                diag["expert_calls_frac"] = diag["expert_calls"] / N_tot
        art = None
        if cfg.export:   # AFTER certification and escalation: the bottle never scores the batch it learns from
            eid = self._embedder_id()
            src = [src_of(p) for p in prov]
            fp = self.fingerprint()
            bkw = self._bottle_kw()        # {} by default: the default construction
            art = (base_artifact.updated(texts, answers_out, emb=emb, embedder_id=eid, src=src, fingerprint=fp, **bkw)
                   if base_artifact is not None
                   else BottleArtifact(self.spec, texts, answers_out, emb=emb, embedder_id=eid, src=src, fingerprint=fp,
                                       **bkw))
            diag["next_bottle_dense"] = art.dense
        if art is not None and cfg.forecast and fc_inputs is not None:
            # warm/cold forecast for the NEXT pour. Report-only: it runs after tau, answers, escalation and
            # export are fixed, makes no LLM call and draws from no calibration stream; a failure never alters the run.
            try:
                fctx = {"texts": texts, "cal": cal, "N": N, "N_tot": N_tot, "looks": looks, "spending": spending,
                        "meth": meth, "s_cal": s_cal, "mkw": mkw, "cap_kw": cap_kw, "trivial": trivial, "cert": cert,
                        "final": final,
                        "tau": cert_d.get("tau_budget") if cert_d.get("budget_capped") else cert_d.get("tau")}
                if chunk_design is not None:   # forecast at the standard (eps, delta) design, never at alpha*
                    fctx.update({"chunk_level": level, "score": score})
                if budget_split:               # counterfactual arms apply the same k-rule level (delta/2)
                    fctx["k_level"] = k_level
                art.forecast = await self._forecast(art, base_artifact, fc_inputs, fctx, prov, diag)
                diag["forecast"] = _forecast_summary(art.forecast)
            except Exception as e:  # noqa: BLE001
                diag["forecast_error"] = f"{type(e).__name__}: {e}"[:300]
        if cfg.health:
            # health report. Report-only: it runs after tau, the answers, the escalation, the export and the
            # forecast are fixed, reads the calibration answers already bought, makes no LLM call, draws from no shared
            # random stream and writes only diagnostics['health'] and one row of the exported bottle's history.
            try:
                h = self._health(texts=texts, final=final, cal=cal, fc=fc_inputs or {}, prov=prov, cert_d=cert_d,
                                 N=N, N_tot=N_tot, restricted=restricted, trivial=trivial, design=design,
                                 plan_len=len(getattr(cert, "plan", None) or []), base_artifact=base_artifact,
                                 emb=emb)
                diag["health"] = h
                if art is not None:
                    row = _history_row(h, art, base_artifact, _next_row(art.history))
                    art.history = (art.history + [row])[-int(cfg.history_cap):]
            except Exception as e:  # noqa: BLE001 - health is non-fatal and never changes a batch
                diag["health"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        return BottleResult(answers_out, prov, cert_d, {}, diag, art)

    # ------------------------------------------------------------------------------------------------ health
    def _health(self, *, texts, final, cal: dict, fc: dict, prov, cert_d: dict, N: int, N_tot: int, restricted: bool,
                trivial: bool, design: str, plan_len: int, base_artifact, emb) -> dict:
        """Health report of one certified population (report-only; see _certify_and_escalate).
        Agreement with the expert on the calibration sample (usable expert answers only; an item whose answer is
        missing counts as a disagreement), each with a two-sided Clopper-Pearson interval `ci` at cfg.health_level
        (the reported interval: conservative under sampling without replacement) and, when no sampled expert answer
        is unusable, the exact finite-population (hypergeometric) interval `ci_finite` for the share of the N items
        (tighter; exact, so its coverage is at the nominal level rather than above it):
          final   : the pre-escalation answer (the bottle in a cold pour; the fused answer in run() and warm pours)
          bottle  : pours: the bottle alone (warm pours: the cold counterfactual the forecast uses)
          vote    : run() fused and warm pours: the cheap-vote answer
          student : run(): the CROSS-FITTED student (the exported bottle is never scored on its own training batch)
        Unbiased (and the intervals exact) only for a FIXED-SIZE uniform sample: the single look or calibration='fixed'.
        A staged or custom multi-look design stops on a label-dependent rule, so it is flagged unbiased=False.
        Memo-restricted pours: everything is over the certified population R (scope 'population').
        precision: on the event {U <= u_delta} of the batch's own worst-case law (P >= 1 - delta, the same law and the
        same delta as the batch guarantee, on which that guarantee also holds), at most u_delta of the A bottled
        answers disagree with the expert, so their precision is >= 1 - u_delta/A. Not a separate guarantee. The
        disagreement rate among calibration items above tau is NOT reported (tau was chosen with those labels).
        drift (pours): label-free statistics of the batch against the bottle that scored it (students.drift_stats)."""
        cfg = self.cfg
        level = float(cfg.health_level)
        kind = fc.get("kind") or ("run" if base_artifact is None else "pour")
        use = sorted(i for i, a in cal.items() if a != NONE)
        n_none = len(cal) - len(use)
        fixed_size = (cfg.calibration == "fixed") or plan_len == 1
        unbiased = bool(use) and fixed_size and not trivial
        h: dict = {"v": 1, "kind": kind, "report_only": True, "level": level,
                   "scope": ("population (the certified items R; memo items excluded)" if restricted else "batch"),
                   "N": int(N), "n_cal": int(len(cal)), "n_usable": int(len(use)), "n_expert_none": int(n_none),
                   "design": "fixed" if cfg.calibration == "fixed" else design, "unbiased": bool(unbiased)}
        if not cal:
            h["unbiased_note"] = "no calibration sample (nothing to estimate from)"
        elif not fixed_size:
            h["unbiased_note"] = (f"{design} design: the number of calibration items depends on a label-dependent "
                                  "stop, so the rates and intervals are indicative only")
        else:
            h["unbiased_note"] = ("fixed-size uniform calibration sample drawn after the answers were fixed: unbiased "
                                  "over the items with a usable expert answer" + (
                                      f" ({n_none} unusable expert answer(s) excluded)" if n_none else ""))
        arms = {"final": final}
        if kind == "pour_warm":
            arms["bottle"], arms["vote"] = fc.get("b_pred"), fc.get("vote")
        elif kind == "pour_cold":
            arms["bottle"] = fc.get("b_pred", final)
        elif kind == "run":
            if fc.get("vote") is not None:
                arms["vote"] = fc["vote"]
            if fc.get("s_pred") is not None:
                arms["student"] = fc["s_pred"]
        agree = {}
        for name, ans in arms.items():
            if ans is None or len(ans) != N:
                continue
            k = int(sum(1 for i in use if ans[i] is not None and ans[i] != NONE and ans[i] == cal[i]))
            agree[name] = _agree_entry(k, len(use), N, n_none, level)
        h["agree"] = agree
        h["precision"] = _precision_entry(cert_d.get("risk") or {}, int(sum(p == "bottle" for p in prov)), N,
                                          float(cfg.delta))
        h["bottled"] = float(sum(p == "bottle" for p in prov)) / max(1, N_tot)
        if base_artifact is not None and fc.get("b_pred") is not None:
            try:
                prev = next((r.get("conf_mean") for r in reversed(base_artifact.history or [])
                             if r.get("conf_mean") is not None), None)
                d = drift_stats(base_artifact, texts, fc.get("b_conf"), fc.get("b_pred"), low_percentile=cfg.health_low_percentile,
                                emb=emb if base_artifact.dense else None,
                                seed=sub_seed(int(cfg.seed), HEALTH_STREAM_KEY, int(base_artifact.version) + 1,
                                              len(base_artifact.texts)),
                                max_rows=cfg.health_max_rows)
                d["conf_mean_prev"] = prev
                h["drift"] = d
            except Exception as e:  # noqa: BLE001
                h["drift"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        h["summary"] = _health_sentence(h)
        return h


class _ArmUnavailable(RuntimeError):
    """A forecast arm cannot be evaluated exactly from this batch's calibration labels."""


class _XfitUnavailable(RuntimeError):
    """A forecast cross-fit cannot map its batch onto the bottle's memory (eviction, collision, version)."""


def _text_hash(text: str) -> int:
    """64-bit text identity used by the slim forecast block (blake2b of the UTF-8 text)."""
    return int.from_bytes(hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest(), "little")


def _plain(o):
    """Plain Python types (for the pickle): numpy scalars/arrays -> Python numbers/lists, tuples -> lists."""
    if isinstance(o, dict):
        return {(k.item() if isinstance(k, np.generic) else k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def _forecast_summary(f: dict | None) -> dict | None:
    """Compact view of a forecast for diagnostics (no pending inputs)."""
    if not f:
        return None
    arms = {k: {kk: v.get(kk) for kk in ("bottled", "tau", "basis")} for k, v in (f.get("arms") or {}).items()}
    g = f.get("gain")
    ag = f.get("agree")
    return {"source": f.get("source"), "status": f.get("status"), "N": f.get("N"), "n_cal": f.get("n_cal"),
            "arms": arms, "gain": None if not g else {k: g.get(k) for k in ("value", "age", "basis")},
            "agree": None if not ag else {k: ag.get(k) for k in ("a", "n", "age", "basis")}}


def _risk_sentence(risk: dict, N: int, budget_note: bool = False) -> str:
    """One sentence appended to certificate['guarantee'] (see Bottler._risk)."""
    F, b = int(risk.get("n_expert_unusable") or 0), risk.get("basis")
    extra = f"; plus {F}/{N} = {F / N:.2%} expert-unusable items (observed, not part of the expectation)" if F else ""
    if b == "inversion":
        return "; no expected-error bound under a budget cap (inversion tail in certificate['risk'])"
    if b == "budget split":
        return ("; no expected-error bound under an expert budget (certificate['risk']['eps_at_delta'][d] holds at "
                "level d: k-rule at d/2 + budget inversion at d/2)" + extra)
    if b == "not certified":
        return "; nothing bottled, so no bottled answer can disagree with the expert" + extra
    what = ("worst case of the k-rule; the budget fallback is not covered" if budget_note
            else "worst case of this procedure")
    if b in ("exact", "union bound"):
        return f"; E[#bottled answers != expert]/N <= {risk['expected_eps_bound']:.2%} ({b} {what}){extra}"
    if b == "closed form":
        return f"; E[#bottled answers != expert]/N <= {risk['expected_eps_bound']:.2%} (exact closed form, {what}){extra}"
    return ""


# ------------------------------------------------------------------------------------------------ health
def cp_interval(k: int, n: int, level: float = HEALTH_LEVEL) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval for k successes in n trials (exact binomial; conservative for a uniform
    sample drawn WITHOUT replacement, checked exhaustively in tests/test_health.py)."""
    k, n = int(k), int(n)
    if n <= 0:
        raise ValueError("cp_interval needs n >= 1")
    a = (1.0 - float(level)) / 2.0
    lo = 0.0 if k <= 0 else float(_beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k >= n else float(_beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def finite_interval(k: int, n: int, N: int, level: float = HEALTH_LEVEL) -> tuple[float, float]:
    """Exact two-sided interval for the share M/N of a finite population of N items, from k successes in a uniform
    sample of n items drawn without replacement: every M with P[X <= k; M] > a and P[X >= k; M] > a, a = (1-level)/2
    (hypergeometric; both tails are monotone in M, so bisection). Tighter than Clopper-Pearson when n/N is not small."""
    k, n, N = int(k), int(n), int(N)
    if not 0 <= k <= n <= N or n <= 0:
        raise ValueError("finite_interval needs 0 <= k <= n <= N and n >= 1")
    a = (1.0 - float(level)) / 2.0
    lo_M, hi_M = k, N - (n - k)                         # the M values the observed sample allows
    L, R = lo_M, hi_M                                   # largest M with P[X <= k; M] > a (non-increasing in M)
    while L < R:
        mid = (L + R + 1) // 2
        if float(_hypergeom.cdf(k, N, mid, n)) > a:
            L = mid
        else:
            R = mid - 1
    upper = L
    L, R = lo_M, hi_M                                   # smallest M with P[X >= k; M] > a (non-decreasing in M)
    while L < R:
        mid = (L + R) // 2
        if (float(_hypergeom.sf(k - 1, N, mid, n)) if k > 0 else 1.0) > a:
            R = mid
        else:
            L = mid + 1
    return L / N, upper / N


def _agree_entry(k: int, n: int, N: int, n_none: int, level: float) -> dict:
    if n <= 0:
        return {"k": 0, "n": 0, "rate": None, "ci": None, "ci_finite": None}
    lo, hi = cp_interval(k, n, level)
    out = {"k": int(k), "n": int(n), "rate": k / n, "ci": [lo, hi], "ci_finite": None}
    if n_none == 0 and n <= N:
        flo, fhi = finite_interval(k, n, N, level)
        out["ci_finite"] = [flo, fhi]
    return out


def _precision_entry(risk: dict, A: int, N: int, delta: float) -> dict:
    """Lower bound on the precision of the A bottled answers from the batch's own risk profile: u_delta = the
    eps_at_delta[delta] share x N (P[U > u_delta] <= delta, U = bottled answers != expert); precision >= 1 - u_delta/A
    on that event. None when there is nothing bottled or the profile reports no eps_at_delta.
    Basis 'exact' / 'union bound' / 'convolution': the worst-case law of the procedure; u_delta <= floor(eps N) because
    that law certified eps at delta, so the batch guarantee also holds on this event. Basis 'inversion' (budget cap):
    the same data-dependent test inversion as the batch's eps_achieved, at the same delta. Either way it is the event
    the batch's own claim is made on: no new claim and no new delta.
    Basis 'budget split' (any batch with an expert budget): the same Bonferroni split as the batch guarantee,
    the k-rule's law at delta/2 (u_delta <= floor(eps N)) when the budget does not bind, the budget inversion at
    delta/2 (u_delta = eps_achieved x N) when it binds; P[U > u_delta] <= delta/2 + delta/2 whatever the labels select."""
    v = (risk.get("eps_at_delta") or {}).get(f"{float(delta):g}")
    basis = risk.get("basis")
    out = {"bottled": int(A), "delta": float(delta), "basis": basis, "u_delta": None, "precision_lower": None}
    if v is None or A <= 0:
        out["note"] = "nothing bottled" if A <= 0 else "the risk profile reports no eps_at_delta"
        return out
    u = int(round(float(v) * int(N)))
    src = ("the same test inversion and delta as this batch's budget-capped eps_achieved" if basis == "inversion" else
           "the same Bonferroni split and delta as this batch's guarantee: the k-rule's law at delta/2 if the budget "
           "does not bind, the budget inversion at delta/2 (eps_achieved) if it binds" if basis == "budget split" else
           "the same worst-case law and delta as this batch's guarantee, which also holds on this event")
    out.update({"u_delta": u, "precision_lower": max(0.0, 1.0 - u / A),
                "statement": (f"with probability >= 1 - {float(delta):g} ({src}; not a separate claim) at most {u} of "
                              f"the {A} bottled answers differ from the expert")})
    return out


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * float(x):.1f}%"


def _health_sentence(h: dict) -> str:
    """One line for diagnostics['health']['summary'] (never part of the guarantee string)."""
    if h.get("error"):
        return f"health unavailable ({h['error']})"
    ag = h.get("agree") or {}
    parts = []
    names = {"final": "final pre-escalation answers", "bottle": "bottle alone", "vote": "cheap vote",
             "student": "cross-fitted student"}
    for key in ("bottle", "final", "vote", "student"):
        e = ag.get(key)
        if e and e.get("rate") is not None:
            ci = e.get("ci")
            parts.append(f"{names[key]} {_pct(e['rate'])}" + ("" if ci is None else f" [{_pct(ci[0])}, {_pct(ci[1])}]"))
    s = "Health (report-only): "
    if parts:
        how = "unbiased" if h.get("unbiased") else f"indicative only: {h.get('design')} design"
        s += (f"agreement with the expert on {h.get('n_usable', 'the')} calibration items ({how}; "
              f"{100 * float(h.get('level', HEALTH_LEVEL)):g}% intervals): " + "; ".join(parts) + ". ")
    else:
        s += "no calibration agreement (no usable calibration sample). "
    pr = h.get("precision") or {}
    if pr.get("precision_lower") is not None:
        s += (f"Bottled answers: precision >= {_pct(pr['precision_lower'])} ({pr['bottled']} bottled; at most "
              f"{pr['u_delta']} differ from the expert with probability >= {1 - pr['delta']:g}, the same event as "
              "the batch's own claim, not a separate guarantee). ")
    d = h.get("drift") or {}
    if d and not d.get("error"):
        bits = []
        if d.get("conf_mean") is not None:
            bits.append(f"mean bottle confidence {d['conf_mean']:.3f}" + (
                f" (previous batch {d['conf_mean_prev']:.3f})" if d.get("conf_mean_prev") is not None else ""))
        if d.get("label_tv") is not None:
            bits.append(f"label-mix TV {d['label_tv']:.3f}")
        if d.get("oov_mass") is not None:
            bits.append(f"unseen word n-grams {_pct(d['oov_mass'])}")
        if d.get("cos_mean") is not None:
            bits.append(f"mean nearest-memory cosine {d['cos_mean']:.3f}")
        if bits:
            s += "Drift (label-free, no thresholds): " + ", ".join(bits) + "."
    return s.strip()


HISTORY_FIELDS = ("row", "kind", "bottle_version", "bottle_memory", "version_out", "memory_out", "N", "N_population",
                  "bottled", "n_cal", "n_usable", "unbiased", "delta", "level", "agree_final", "agree_final_ci",
                  "agree_bottle", "agree_bottle_ci", "agree_vote", "agree_student", "precision_lower", "conf_mean",
                  "label_tv", "oov_mass", "cos_mean", "chunk", "chunks")


def _r6(x):
    return None if x is None else round(float(x), HISTORY_DIGITS)


def _next_row(history: list) -> int:
    """Index of the next learning-curve row: one past the last row's index (stays increasing after the cap drops the
    oldest rows); len(history) for rows without an index."""
    last = history[-1].get("row") if history else None
    return int(last) + 1 if isinstance(last, int) and not isinstance(last, bool) else len(history)


def _history_row(h: dict, art, base_artifact, row: int) -> dict:
    """One learning-curve row (plain types, the HISTORY_FIELDS) for BottleArtifact.history. The intervals are the
    reported Clopper-Pearson ones (`ci`) at h['level']."""
    ag, d = h.get("agree") or {}, h.get("drift") or {}

    def rate(k):
        return _r6((ag.get(k) or {}).get("rate"))

    def ci(k):
        c = (ag.get(k) or {}).get("ci")
        return None if c is None else [_r6(c[0]), _r6(c[1])]
    pop = "population" in str(h.get("scope"))
    return {"row": int(row), "kind": h.get("kind"),
            "bottle_version": None if base_artifact is None else int(base_artifact.version),
            "bottle_memory": 0 if base_artifact is None else len(base_artifact.texts),
            "version_out": int(art.version), "memory_out": len(art.texts),
            "N": None if pop else int(h.get("N")), "N_population": int(h.get("N")) if pop else None,
            "bottled": _r6(h.get("bottled")), "n_cal": h.get("n_cal"), "n_usable": h.get("n_usable"),
            "unbiased": h.get("unbiased"), "delta": _r6((h.get("precision") or {}).get("delta")),
            "level": _r6(h.get("level")), "agree_final": rate("final"), "agree_final_ci": ci("final"), "agree_bottle": rate("bottle"),
            "agree_bottle_ci": ci("bottle"), "agree_vote": rate("vote"), "agree_student": rate("student"),
            "precision_lower": _r6((h.get("precision") or {}).get("precision_lower")),
            "conf_mean": _r6(d.get("conf_mean")), "label_tv": _r6(d.get("label_tv")),
            "oov_mass": _r6(d.get("oov_mass")), "cos_mean": _r6(d.get("cos_mean")), "chunk": None, "chunks": None}


def learning_curve(artifact_or_history) -> list[dict]:
    """The learning-curve rows of a bottle (BottleArtifact.history, or that list itself): one plain dict per
    certified batch of its lineage, oldest first (report-only)."""
    h = getattr(artifact_or_history, "history", artifact_or_history)
    return [dict(r) for r in (h or [])]


def format_learning_curve(rows) -> str:
    """Plain-text table of learning-curve rows (a bottle, its history, or learning_curve() rows)."""
    rows = learning_curve(rows)
    levels = {r.get("level") for r in rows if r.get("level") is not None}
    ci_tag = f"{100 * float(levels.pop()):g}%" if len(levels) == 1 else "CI"
    head = ["row", "kind", "bottle v", "memory", "N", "bottled", "n_cal", f"agree bottle [{ci_tag}]", "agree final",
            "vote", "precision >=", "conf", "oov"]
    lines = [head]
    for r in rows:
        ci = r.get("agree_bottle_ci")
        ab = "-" if r.get("agree_bottle") is None else (
            f"{_pct(r['agree_bottle'])} [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else _pct(r["agree_bottle"]))
        tag = "" if r.get("chunk") is None else f" c{int(r['chunk']) + 1}/{r.get('chunks')}"
        n = r.get("N") if r.get("N") is not None else r.get("N_population")
        lines.append([str(r.get("row")), str(r.get("kind")) + tag,
                      "-" if r.get("bottle_version") is None else str(r["bottle_version"]),
                      str(r.get("bottle_memory")), str(n), _pct(r.get("bottled")), str(r.get("n_cal")), ab,
                      "-" if r.get("agree_final") is None else _pct(r["agree_final"]),
                      "-" if r.get("agree_vote") is None else _pct(r["agree_vote"]),
                      "-" if r.get("precision_lower") is None else _pct(r["precision_lower"]),
                      "-" if r.get("conf_mean") is None else f"{r['conf_mean']:.3f}",
                      "-" if r.get("oov_mass") is None else _pct(r["oov_mass"])])
    w = [max(len(x[j]) for x in lines) for j in range(len(head))]
    return "\n".join("  ".join(x[j].ljust(w[j]) for j in range(len(head))).rstrip() for x in lines)


def _chunked_health(hs: list, sizes: list, N: int, A: int, risk: dict | None, delta: float,
                    level: float = HEALTH_LEVEL) -> dict:
    """Batch-level health of a chunked pour: per-chunk reports plus, for each agreement present in every chunk, the
    stratified estimate sum_j N_j * rate_j / N (unbiased when every chunk's sample is a fixed-size uniform sample of
    its chunk) with a simultaneous interval: each chunk's Clopper-Pearson interval at level 1 - (1 - level)/J
    (Bonferroni), combined with the same weights. precision: from the whole-batch convolution law (risk)."""
    J = len(hs)
    out: dict = {"v": 1, "kind": "chunked", "report_only": True, "level": level, "chunks": J, "N": int(N),
                 "design": "chunked", "unbiased": bool(hs) and all(h.get("unbiased") for h in hs), "agree": {}}
    names = set.intersection(*[set((h.get("agree") or {}).keys()) for h in hs]) if hs else set()
    lv = 1.0 - (1.0 - level) / max(1, J)
    for name in sorted(names):
        ents = [h["agree"][name] for h in hs]
        if any(e.get("rate") is None for e in ents):
            continue
        rate = sum(Nj * e["rate"] for Nj, e in zip(sizes, ents)) / N
        cis = [cp_interval(e["k"], e["n"], lv) for e in ents]
        out["agree"][name] = {"rate": rate, "ci": [sum(Nj * c[0] for Nj, c in zip(sizes, cis)) / N,
                                                   sum(Nj * c[1] for Nj, c in zip(sizes, cis)) / N],
                              "ci_finite": None, "estimator": "stratified (weights N_j / N)",
                              "ci_basis": f"Bonferroni over {J} chunks (each chunk's CP interval at level {lv:.4f})",
                              "k": int(sum(e["k"] for e in ents)), "n": int(sum(e["n"] for e in ents))}
    out["n_usable"] = int(sum(int(h.get("n_usable") or 0) for h in hs))
    out["precision"] = (_precision_entry(risk, A, N, delta) if risk else
                        {"bottled": int(A), "precision_lower": None, "note": "no whole-batch law (plan mismatch)"})
    out["bottled"] = A / max(1, N)
    out["per_chunk"] = [{"N": int(Nj), "summary": h.get("summary"), "unbiased": h.get("unbiased"),
                         "agree": {k: {kk: v.get(kk) for kk in ("k", "n", "rate", "ci", "ci_finite")}
                                   for k, v in (h.get("agree") or {}).items()},
                         "precision_lower": (h.get("precision") or {}).get("precision_lower"),
                         "drift": h.get("drift")} for Nj, h in zip(sizes, hs)]
    out["summary"] = _health_sentence(out) + f" (chunked pour: {J} chunks; per-chunk reports in 'per_chunk')"
    return out


# ------------------------------------------------------------------------------------------------ stream ledger
STREAM_CONDITIONS = (
    "the batch sizes (flush sizes) and the reporting horizon were fixed independently of the outputs: by the "
    "submission sizes, min_batch and flushes scheduled in advance (end of stream, a timer), never by a flush or a stop "
    "chosen after seeing answers, certificates or this report; read after every flush, each report is a fixed-horizon "
    "claim for its own horizon, not a claim that holds uniformly over time",
    "the calibration draws of different batches are independent uniform draws (fresh calibration keys; the lineage "
    "guard refuses any reuse)")


def stream_design(cert: dict, N: int | None = None) -> dict:
    """Label-free design of one certified batch for the stream ledger (certify.StreamLaw), read from its certificate.
    Returns {'ok': True, 'designs': [[[[n, k], ...], N_pop], ...], 'basis'} with one entry per certified population
    (a chunked pour: one per chunk, from its PLAN), or {'ok': False, 'reason'} when the batch's error count has no
    law fixed by its design alone:
      * memo-restricted (n_memo > 0): the certified population R depends on which earlier items the expert answered;
      * budget-capped: the threshold is the budget's, not a k-rule's (only a test inversion bounds it);
      * a chunked pour whose realized chunk designs differ from its plan;
      * no recorded design (risk profile unavailable) or design sizes that do not add up to the batch.
    Single look: (n, k) of the closed-form rule actually applied (any method; mean-risk cap included). Staged or custom
    looks: every pre-specified look's (n, k) (union-bound law). Not certified: the plan's k (k = -1 for a single look
    means nothing can be bottled: U = 0). The design, never the realized outcome, is used, so it is label-free."""
    if not isinstance(cert, dict):
        return {"ok": False, "reason": "no certificate"}
    if cert.get("chunked"):
        if not cert.get("plan_consistent"):
            return {"ok": False, "reason": "chunked pour whose realized chunk designs differ from its plan"}
        cds = cert.get("chunk_designs") or []
        if not cds:
            return {"ok": False, "reason": "chunked pour without recorded chunk designs"}
        designs, basis = [[[[int(n), int(k)]], int(Nj)] for n, Nj, k in cds], "chunked (one law per chunk)"
    else:
        if cert.get("n_memo") or cert.get("N_population") is not None or cert.get("no_calibration_needed"):
            return {"ok": False, "reason": "memo-restricted batch: its certified population depends on which earlier "
                                           "items the expert answered (label-dependent design)"}
        if cert.get("budget_capped"):
            return {"ok": False, "reason": "budget-capped batch: its threshold is the expert budget's, not a k-rule's "
                                           "(only a test inversion bounds it)"}
        looks = (cert.get("risk") or {}).get("looks")
        if not looks or cert.get("N") is None:
            return {"ok": False, "reason": "no pre-specified design recorded (risk profile unavailable)"}
        designs = [[[[int(l["n_cal"]), int(l["k"])] for l in looks], int(cert["N"])]]
        basis = "single look (exact law)" if len(looks) == 1 else "pre-specified looks (union-bound law)"
    tot = sum(d[1] for d in designs)
    if N is not None and tot != int(N):
        return {"ok": False, "reason": f"design populations add up to {tot}, not to the batch size {int(N)}"}
    return {"ok": True, "designs": designs, "basis": basis}


def stream_report(entries: list, eps: float, delta: float, law: StreamLaw | None = None, conditions=STREAM_CONDITIONS,
                  notes=(), deltas=None, withdrawal_possible=()) -> dict:
    """Stream-level tail over certified batches (report-only). entries: one dict per batch, in order, with 'batch',
    'N', 'F' (expert-unusable count) and the stream_design fields ('ok', 'designs' | 'reason'). law: a StreamLaw
    already holding exactly the designs of these entries (else built here).
    status 'exact': P[stream share of final answers != expert > eps' + F/N] <= tail[eps'] for every eps' at once.
    status 'none': at least one batch has no label-free design (memo, budget cap, plan mismatch, ...). Dropping such
    a batch and claiming over the rest would let a label-dependent event pick the covered batches, which the proof
    does not allow, so the whole stream claim is withdrawn (each batch keeps its own guarantee).
    status 'empty': no certified batch yet.
    The probability is over the WHOLE procedure: a run whose claim is withdrawn counts as making no claim, and the
    bound is not conditional on the claim being made (the withdrawal event can correlate with the outcome).
    withdrawal_possible: label-free reasons why this configuration CAN withdraw (expert_budget, memo, chunking);
    reported as 'withdrawal_possible' and named in the sentence."""
    deltas = tuple(sorted({0.5, float(delta), *(STREAM_DELTAS if deltas is None else deltas)}, reverse=True))
    n_b = len(entries)
    base = {"delta": float(delta), "eps": float(eps), "batches": n_b, "conditions": list(conditions),
            "notes": list(notes), "withdrawal_possible": bool(withdrawal_possible),
            "withdrawal_reasons": list(withdrawal_possible),
            "probability_basis": "over the whole procedure; a withdrawn claim counts as no claim; NOT conditional on "
                                 "the claim being made"}
    if not n_b:
        return {"status": "empty", **base, "reason": "no certified batch"}
    bad = [{"batch": e.get("batch"), "reason": e.get("reason")} for e in entries if not e.get("ok")]
    if bad:
        return {"status": "none", **base, "batches_included": 0, "batches_excluded": len(bad), "excluded": bad,
                "reason": "batch(es) without a label-free design; excluding them would make the covered set "
                          "label-dependent, so no stream-level tail is claimed"}
    if law is None:
        law = StreamLaw()
        for e in entries:
            for looks, Nj in e["designs"]:
                law.add_design(looks, Nj)
    N = int(sum(int(e["N"]) for e in entries))
    if law.N != N:
        return {"status": "none", **base, "reason": f"ledger covers {law.N} items but the batches hold {N}"}
    F = int(sum(int(e.get("F") or 0) for e in entries))
    s = law.summary(eps, deltas=deltas, n_unusable=F)
    return {"status": "exact", **base, "batches_included": n_b, "batches_excluded": 0, "excluded": [],
            "N_included": N, **{k: v for k, v in s.items() if k not in ("N", "eps")},
            "scope": "W = #bottled answers != expert over all included batches; stream share of final answers != "
                     "expert <= W/N + F/N (F = expert-routed items without a usable expert answer, observed). Worst "
                     "case over workloads and calibration draws (conditional stochastic dominance, exact convolution "
                     "of the per-batch laws). In ADDITION to each batch's own (eps, delta) guarantee, not instead of "
                     "it; it does not say that no batch fails. The probability is over the whole procedure: a run "
                     "whose claim is withdrawn counts as no claim, and the bound is not conditional on the claim "
                     "being made"}


def stream_sentence(sr: dict) -> str:
    """One sentence for a report's statement (see stream_report)."""
    st = sr.get("status")
    if st == "exact":
        d = sr["delta"]
        ead = sr["eps_at_delta"]
        q = ead.get(f"{d:g}")
        return (f"Stream (fixed horizon: these {sr['batches_included']} batches, {sr['N_included']} items): "
                f"P[share of all final answers != expert > eps' + F/N] <= tail(eps') for every eps' at once (exact "
                f"convolution of the batches' worst-case laws, F/N = {sr['unusable_share']:.2%}), e.g. the stream "
                f"share is <= {q:.2%} with probability >= {1 - d:g} and P[stream share > {sr['eps']:g} + F/N] <= "
                f"{sr['delta_stream']:.2g}. This is in addition to, not instead of, each batch's own (eps, delta) "
                "guarantee, and it does not say that no batch fails. It assumes " + "; and ".join(sr["conditions"])
                + "." + (" This configuration can withdraw the stream claim (" + ", ".join(sr["withdrawal_reasons"])
                         + "); the probability is over the whole procedure, a withdrawn claim counts as no claim, and "
                         "the bound is not conditional on the claim being made." if sr.get("withdrawal_possible")
                         else ""))
    if st == "none":
        return ("No stream-level tail is claimed: " + "; ".join(f"batch #{x['batch']}: {x['reason']}"
                                                               for x in sr.get("excluded", [])[:3])
                + (f" (and {len(sr['excluded']) - 3} more)" if len(sr.get("excluded", [])) > 3 else "")
                + ("" if sr.get("excluded") else str(sr.get("reason"))) + ".")
    return ""


def stream_risk(certificates, eps: float, delta: float, Ns=None) -> dict:
    """Stream-level tail for users who drive run()/pour() themselves: certificates of consecutive batches, in order
    (Ns: their sizes, default certificate['N']). The same claim and conditions as BottleSession.report(); the caller
    is responsible for the conditions the certificates cannot show: batch sizes and the horizon fixed independently
    of the outputs, fresh calibration keys (never reuse a bottle lineage point), and, with cfg.chunking='auto' or
    'cold', no output without a usable answer before a pour (the chunk plan reads the bottle's memory size); with
    chunking='cold' and votes='auto', no pour that passed the cold gate (its split followed the measured forecast)."""
    certs = list(certificates)
    Ns = [c.get("N") if isinstance(c, dict) else None for c in certs] if Ns is None else list(Ns)
    entries = []
    for b, (c, N) in enumerate(zip(certs, Ns)):
        e = stream_design(c, N)
        entries.append({"batch": b, "N": int(N or 0), "F": int((c or {}).get("n_expert_unusable") or 0), **e})
    notes = []
    if any(isinstance(c, dict) and (c.get("chunks") or (c.get("chunking") or {}).get("rule")) for c in certs):
        notes.append("some certificates come from chunked or chunk-gated pours: the caller must ensure that no earlier "
                     "output lacked a usable answer (the chunk plan reads the bottle's memory size) and, under "
                     "chunking='cold' with votes='auto', that no pour passed the cold gate (its split followed the "
                     "measured forecast); BottleSession checks both")
    return stream_report(entries, eps, delta, notes=notes)


def evaluate(result: BottleResult, gold: list | None = None, expert_all: list | None = None) -> dict:
    """Optional evaluation (NOT part of the pipeline cost): gold accuracy and exact realized disagreement with an
    expert labelling of every item (an unusable expert answer counts as disagreement; also reported excluding them)."""
    n = len(result.answers)
    for name, ref in (("gold", gold), ("expert_all", expert_all)):
        if ref is not None and len(ref) != n:
            raise ValueError(f"{name} has {len(ref)} entries, result has {n}")
    out = {"coverage_bottle": result.coverage(), "final_none_rate": float(np.mean([a == NONE for a in result.answers])) if n else 0.0}
    if gold is not None:
        out["final_gold_acc"] = float(np.mean([a == b for a, b in zip(result.answers, gold)]))
        bot = [i for i, p in enumerate(result.provenance) if p == "bottle"]
        out["bottle_gold_acc"] = float(np.mean([result.answers[i] == gold[i] for i in bot])) if bot else None
    if expert_all is not None:
        out["realized_disagreement_with_expert"] = float(np.mean([disagreement(a, e) for a, e in zip(result.answers, expert_all)]))
        ok = [i for i, e in enumerate(expert_all) if e is not None and e != NONE]
        out["realized_disagreement_excl_expert_none"] = (float(np.mean([result.answers[i] != expert_all[i] for i in ok]))
                                                        if ok else None)
        if gold is not None:
            out["expert_all_gold_acc"] = float(np.mean([e == b for e, b in zip(expert_all, gold)]))
    return out
