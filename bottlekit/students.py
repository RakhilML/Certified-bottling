"""Students: cheap classifiers distilled from LLM labels, plus cross-fitting.

Cross-fitting is REQUIRED when the student's training items are part of the workload being certified: a student
scores its own training items over-confidently, which silently breaks held-out certificates (measured: 100% of
splits violated naively; <=4.5% with cross-fitting).

Scales with the label space: the feature budget shrinks as the number of classes grows (coefficients = features x
classes), so 150-class tasks stay within a few hundred MB; tiny or degenerate inputs fall back gracefully
(single class -> constant predictor; empty vocabulary -> majority class) instead of crashing.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .certify import NONE

COEF_BUDGET = 8_000_000      # max (features x classes) of the linear model
TOKEN = r"(?u)\b\w+\b"       # keep 1-character tokens (codes, digits, CJK)
# Student defaults (chosen once on development tasks; not tuned per task). Any value fixed before the batch
# keeps the certificate valid (run() cross-fits whatever student it uses); never choose one on expert calibration labels.
DEFAULT_C = 4.0              # inverse L2 strength of the logistic regression
CROSS_FIT_FOLDS = 5          # out-of-fold folds of cross_fit; any K >= 2 is valid
WORD_NGRAMS = (1, 2)         # word n-gram range
CHAR_NGRAMS = (3, 5)         # char n-gram range
CHAR_ANALYZER = "char_wb"    # sklearn analyzer of the char block ('char' suits unsegmented scripts better)
FEATURE_CLIP = (5_000, 400_000)   # features = clip(COEF_BUDGET // classes, lo, hi)
WORD_DIV = 3                 # the word block gets budget // WORD_DIV features, the char block the rest
MIN_DF_SMALL_N = 50          # min_df='auto': 2 if >= this many training texts, else 1
MAX_ITER = 2000              # lbfgs iterations of the logistic regression
PREDICT_CHUNK = 20_000       # rows per predict_proba call (memory only: results are identical for any value)
STUDENT_KEYS = ("C", "char", "dense_weight", "word_ngrams", "char_ngrams", "char_analyzer", "token_pattern",
                "coef_budget", "feature_clip", "word_div", "min_df", "max_iter")   # BottleConfig.student keys
DRIFT_LOW_PERCENTILE = 10    # the low-tail percentile drift_stats reports (keys conf_p10 / cos_p10): part of the
                             # statistic's definition, report-only and never compared with anything (no thresholds)


class TfidfLR:
    """Word 1-2gram + char 3-5gram TF-IDF (+ optional dense embedding columns) -> multinomial logistic regression.

    Dense columns: `dense` is a row-indexed matrix; fit/predict rows `idx` read `dense[idx]` (cross-fitting). For rows
    that are NOT in that matrix (e.g. a reusable bottle scoring a new batch), pass `dense=` to predict: one row per
    text, same width as at fit time (ValueError otherwise). Dense values are used as given (no rescaling), so the
    cross-fitted run() student and a bottle built from the same embeddings see identical feature scales.
    """

    def __init__(self, C: float = DEFAULT_C, char: bool = True, dense: np.ndarray | None = None, dense_weight: float = 1.0,
                 *, word_ngrams=WORD_NGRAMS, char_ngrams=CHAR_NGRAMS, char_analyzer: str = CHAR_ANALYZER,
                 token_pattern: str = TOKEN, coef_budget: int = COEF_BUDGET, feature_clip=FEATURE_CLIP,
                 word_div: int = WORD_DIV, min_df: int | str = "auto", max_iter: int = MAX_ITER):
        self.C, self.char, self.dense, self.dense_weight = C, char, dense, dense_weight
        self.word_ngrams, self.char_ngrams, self.char_analyzer = tuple(word_ngrams), tuple(char_ngrams), char_analyzer
        self.token_pattern, self.coef_budget, self.feature_clip = token_pattern, int(coef_budget), tuple(feature_clip)
        self.word_div, self.min_df, self.max_iter = int(word_div), min_df, int(max_iter)

    def _vec(self, texts, kind, max_features, min_df):
        if kind == "word":
            v = TfidfVectorizer(ngram_range=self.word_ngrams, sublinear_tf=True, max_features=max_features,
                                min_df=min_df, token_pattern=self.token_pattern, dtype=np.float32)
        else:
            v = TfidfVectorizer(analyzer=self.char_analyzer, ngram_range=self.char_ngrams, sublinear_tf=True,
                                max_features=max_features, min_df=min_df, dtype=np.float32)
        try:
            return v, v.fit_transform(texts)
        except ValueError:          # empty vocabulary (e.g. min_df too high for a tiny input, or no word tokens)
            return None, None

    def _X(self, texts, idx, fit, dense=None):
        if fit:
            n_classes = max(2, len(self.classes_))
            budget = int(np.clip(self.coef_budget // n_classes, *self.feature_clip))
            min_df = (2 if len(texts) >= MIN_DF_SMALL_N else 1) if self.min_df == "auto" else self.min_df
            wb = budget // self.word_div
            self.wv, Xw = self._vec(texts, "word", wb if self.char else budget, min_df)
            self.cv, Xc = self._vec(texts, "char", budget - wb, min_df) if self.char else (None, None)
            parts = [x for x in (Xw, Xc) if x is not None]
        else:
            parts = [v.transform(texts) for v in (self.wv, self.cv) if v is not None]
        if dense is not None:                         # prediction-time rows (override of self.dense[idx])
            parts.append(csr_matrix(np.asarray(dense, np.float32) * self.dense_weight))
        elif self.dense is not None:
            parts.append(csr_matrix(self.dense[np.asarray(idx)].astype(np.float32) * self.dense_weight))
        return hstack(parts).tocsr() if parts else None

    def _check_dense(self, dense, n):
        """Validate a prediction-time dense override (rows == texts, width == fit-time width)."""
        dense = np.asarray(dense)
        if dense.ndim != 2 or dense.shape[0] != n:
            raise ValueError(f"dense has shape {dense.shape}; expected ({n}, d): one embedding row per text")
        if self.dense is None:
            raise ValueError("this student was fitted without dense features; it cannot take dense= at predict time")
        d_fit = np.asarray(self.dense).shape[1]
        if dense.shape[1] != d_fit:
            raise ValueError(f"dense has width {dense.shape[1]}, the student was fitted with width {d_fit}")
        return dense

    def fit(self, texts, labels, idx):
        self.classes_ = sorted(set(labels))
        top = Counter(labels).most_common(1)
        self.const, self.const_share = (top[0][0], top[0][1] / len(labels)) if top else (None, 0.0)
        X = self._X(texts, idx, True) if len(self.classes_) > 1 else None
        if X is None or X.shape[1] == 0:
            self.clf = None          # constant predictor (single class or no features)
            return self
        self.clf = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(X, labels)
        return self

    def predict(self, texts, idx, query: list | None = None, dense: np.ndarray | None = None):
        """Returns (pred labels, confidence of the top label, probability of `query[i]` or None).
        `dense` (optional): embedding rows for THESE texts, used instead of self.dense[idx]."""
        n = len(texts)
        if dense is not None:
            dense = self._check_dense(dense, n)
        if self.clf is None:
            if len(self.classes_) == 1:
                conf = np.ones(n)
            else:  # no features: majority class with its training share as confidence
                conf = np.full(n, self.const_share)
            q = None if query is None else np.array([conf[i] if query[i] == self.const else 0.0 for i in range(n)])
            return [self.const] * n, conf, q
        pred, conf, q = [], np.zeros(n), (None if query is None else np.zeros(n))
        col = {c: j for j, c in enumerate(self.clf.classes_)}
        idx = np.asarray(idx)
        ch = PREDICT_CHUNK
        for a in range(0, n, ch):                     # chunked: memory stays flat for huge workloads
            P = self.clf.predict_proba(self._X(texts[a:a + ch], idx[a:a + ch], False,
                                               None if dense is None else dense[a:a + ch]))
            top = P.argmax(1)
            pred += [self.clf.classes_[j] for j in top]
            conf[a:a + len(P)] = P[np.arange(len(P)), top]
            if query is not None:
                for r in range(len(P)):
                    j = col.get(query[a + r])
                    q[a + r] = P[r, j] if j is not None else 0.0
        return pred, conf, q


def check_student(student: dict | None) -> dict:
    """Validate a BottleConfig.student dict (None -> {}): known keys only (STUDENT_KEYS)."""
    if student is None:
        return {}
    if not isinstance(student, dict):
        raise ValueError("student must be a dict of TfidfLR settings (or None)")
    bad = sorted(set(student) - set(STUDENT_KEYS))
    if bad:
        raise ValueError(f"unknown student keys {bad}; allowed: {list(STUDENT_KEYS)}")
    return dict(student)


def cross_fit(texts: list[str], labels: list[str | None], train_idx: list[int], score_idx: list[int] | None = None,
              folds: int = CROSS_FIT_FOLDS, seed: int = 0, dense: np.ndarray | None = None, factory=None,
              query: list | None = None, return_model: bool = False, groups: list | None = None,
              student: dict | None = None):
    """Score EVERY item with a student that never saw it.

    train_idx items get out-of-fold predictions (K folds); all other items (score_idx, default: the rest) are scored
    by a student trained on all of train_idx. Items whose label is None are never used for training.
    Returns (pred list, conf array) over all len(texts) items (unscored items: pred None, conf 0); with `query`
    (one label per item) also the student probability of that label; with return_model also the full student.
    groups (optional, one hashable key per item, e.g. the expert prompt): group-aware folds, all training copies of
    a key share one fold, so an out-of-fold item is never scored by a student trained on its own twin. Groups are
    permuted with the same random stream and placed by the same array_split boundaries, so all-singleton groups give
    exactly the folds of groups=None. A quality fix only (scores never see expert labels either way).
    student (optional): TfidfLR settings of the default factory (BottleConfig.student; None: the default student).
    """
    skw = check_student(student)
    factory = factory or (lambda: TfidfLR(dense=dense, **skw))
    n = len(texts)
    usable = [i for i in train_idx if labels[i] is not None and labels[i] != "__none__"]
    pred: list[str | None] = [None] * n
    conf = np.zeros(n)
    qprob = np.zeros(n)
    full = None

    def score(model, items):
        p, c, q = model.predict([texts[i] for i in items], items, None if query is None else [query[i] for i in items])
        for r, i in enumerate(items):
            pred[i], conf[i] = p[r], c[r]
            if q is not None:
                qprob[i] = q[r]

    if usable:
        rng = np.random.default_rng(seed)
        if groups is None:
            n_folds = max(2, min(folds, len(usable)))
            fold_list = np.array_split(rng.permutation(usable), n_folds)
        else:
            gmap: dict = {}
            for i in usable:                                   # groups in order of first occurrence
                gmap.setdefault(groups[i], []).append(i)
            glist = list(gmap.values())
            n_folds = max(2, min(folds, len(glist)))
            bounds = np.cumsum([len(x) for x in np.array_split(np.arange(len(usable)), n_folds)])
            fold_items: list = [[] for _ in range(n_folds)]
            pos = 0
            for g in rng.permutation(len(glist)):              # == rng.permutation(usable) when all are singletons
                fold_items[int(np.searchsorted(bounds, pos, side="right"))].extend(glist[g])
                pos += len(glist[g])
            fold_list = [np.asarray(x, dtype=np.int64) for x in fold_items]
            nonempty = [f for f in fold_list if len(f)]
            if len(nonempty) < 2 or any(len(f) >= len(usable) for f in nonempty):
                # one dominant prompt-group would leave a fold with no training complement (its items unscored):
                # fall back to the ungrouped folds
                rng = np.random.default_rng(seed)
                fold_list = np.array_split(rng.permutation(usable), max(2, min(folds, len(usable))))
        for f in fold_list:
            if len(f) == 0:
                continue
            fs = set(f.tolist())
            tr = [i for i in usable if i not in fs]
            if not tr:
                continue
            score(factory().fit([texts[i] for i in tr], [labels[i] for i in tr], tr), [int(i) for i in f])
        usable_set = set(usable)
        rest = [i for i in (score_idx if score_idx is not None else range(n)) if i not in usable_set]
        if rest or return_model:
            full = factory().fit([texts[i] for i in usable], [labels[i] for i in usable], usable)
            if rest:
                score(full, rest)
    out = (pred, conf) if query is None else (pred, conf, qprob)
    return (*out, full) if return_model else out


# ------------------------------------------------------------------------------------------------ drift (report-only)
def drift_stats(bottle, texts, conf, pred, emb=None, seed: int = 0, max_rows: int | None = None,
                low_percentile: float = DRIFT_LOW_PERCENTILE,
                chunk: int = 2048) -> dict:
    """Label-free drift statistics of a NEW batch against a bottle (report-only; no thresholds, no alarms).
    bottle: the BottleArtifact that scored the batch; conf/pred: its max-probability and top label for each new item
    (as pour() computed them); emb: the new items' embeddings (dense bottles only). Reads no expert label, makes no
    LLM call and uses its own random stream (np.random.default_rng(seed)); nothing it returns feeds any decision.
      conf_mean, conf_p10 : mean and 10th percentile of the bottle's top-label probability on the new batch
      label_tv            : total variation between the predicted label mix of the batch and the memory label mix
      oov_mass            : share of the new texts' word 1-2gram tokens (the bottle's own word analyzer) that are not
                            in the bottle's word vocabulary (pruned n-grams count as unseen)
      cos_mean, cos_p10   : (dense bottles with new embeddings) mean and 10th percentile of each new item's largest
                            cosine similarity to the memory embeddings (memory subsampled to <= max_rows rows)
    max_rows: None reads everything; otherwise batches or memories larger than max_rows are read through a seeded,
    label-free subsample (n_used / memory_rows_used say so; Bottler passes BottleConfig.health_max_rows).
    chunk: new-item rows per block of the cosine product (bounds working memory; the result does not depend on it)."""
    n = len(texts)
    out: dict = {"n": int(n)}
    rng = np.random.default_rng(int(seed))
    cap = None if max_rows is None else max(1, int(max_rows))
    sel = np.arange(n) if cap is None or n <= cap else np.sort(rng.choice(n, size=cap, replace=False))
    out["n_used"] = int(len(sel))
    conf = np.asarray(conf, float) if conf is not None else None
    ok = [i for i in sel.tolist() if pred is not None and pred[i] is not None and pred[i] != NONE]
    if conf is not None and ok:
        c = conf[np.asarray(ok, dtype=np.int64)]
        out["conf_mean"], out[f"conf_p{low_percentile:g}"] = float(c.mean()), float(np.percentile(c, low_percentile))
    mem = Counter(getattr(bottle, "labels", []) or [])
    if ok and mem:
        pc = Counter(pred[i] for i in ok)
        keys = set(mem) | set(pc)
        tm, tp = sum(mem.values()), sum(pc.values())
        out["label_tv"] = float(0.5 * sum(abs(pc.get(k, 0) / tp - mem.get(k, 0) / tm) for k in keys))
    wv = getattr(getattr(bottle, "student", None), "wv", None)
    if wv is not None and getattr(wv, "vocabulary_", None):
        an, vocab = wv.build_analyzer(), wv.vocabulary_
        tot = miss = 0
        for i in sel.tolist():
            toks = an(texts[i])
            tot += len(toks)
            miss += sum(1 for t in toks if t not in vocab)
        out["oov_mass"] = float(miss / tot) if tot else None
    M = getattr(bottle, "emb", None)
    if emb is not None and M is not None and len(M) and np.asarray(emb).ndim == 2 and np.asarray(emb).shape[1] == M.shape[1]:
        rows = (np.arange(len(M)) if cap is None or len(M) <= cap
                else np.sort(rng.choice(len(M), size=cap, replace=False)))
        Mm = np.asarray(M[rows], np.float32)
        Mm = Mm / np.maximum(np.linalg.norm(Mm, axis=1, keepdims=True), 1e-12)
        E = np.asarray(emb, np.float32)[sel]
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        step = max(1, int(chunk))
        best = (np.concatenate([(E[a:a + step] @ Mm.T).max(axis=1) for a in range(0, len(E), step)])
                if len(E) else [])
        if len(best):
            out["cos_mean"], out[f"cos_p{low_percentile:g}"] = float(np.mean(best)), float(np.percentile(best, low_percentile))
            out["memory_rows_used"] = int(len(rows))
    return out
