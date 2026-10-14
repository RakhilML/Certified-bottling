"""bottlekit: certified bottling of repetitive LLM classification workloads.

    from bottlekit import LLMClient, load_models, load_workload, Bottler, BottleConfig, evaluate
    res = await Bottler(client, spec, cfg).run(texts)        # cold start: cheap votes + student + certificate
    nxt = await Bottler(client, spec, cfg).pour(res.artifact, new_texts)   # reuse: warm/cold by votes='auto', re-certified
"""

from .certify import (NONE, Certificate, StreamLaw, batch_tail, certified_eps, certify, certify_sequential,
                      default_looks, fixed_sequence, min_batch_for_tax, risk_profile, single_look_size)
from .harness import (BottleArtifact, BottleConfig, BottleResult, Bottler, ExpertFailure, UnknownModelError, cp_interval, evaluate,
                      finite_interval, format_learning_curve, learning_curve, stream_risk)
from .labeling import balanced_orders, label_batched, position_reliability, vote_label, weighted_vote
from .session import BottleSession, Submission, min_certified_size
from .llm import ChatResult, EmbeddingError, LLMClient, ModelSpec, load_models
from .students import TfidfLR, cross_fit, drift_stats
from .tasks import REGISTRY, TaskSpec, Workload, from_csv, from_jsonl, from_records, load_workload, parse_batch, parse_label

__version__ = "0.9.1"

__all__ = [n for n in dir() if not n.startswith("_")]
