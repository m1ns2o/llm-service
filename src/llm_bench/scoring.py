from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateMetrics:
    id: str
    quality: float
    safety: float
    decode_tps: float
    ttft_seconds: float
    size_mb: float
    peak_rss_mb: float
    platform: str


def _dominates(a: CandidateMetrics, b: CandidateMetrics) -> bool:
    maximize = ("quality", "safety", "decode_tps")
    minimize = ("ttft_seconds", "size_mb", "peak_rss_mb")
    no_worse = all(getattr(a, key) >= getattr(b, key) for key in maximize) and all(
        getattr(a, key) <= getattr(b, key) for key in minimize
    )
    strictly_better = any(getattr(a, key) > getattr(b, key) for key in maximize) or any(
        getattr(a, key) < getattr(b, key) for key in minimize
    )
    return no_worse and strictly_better


def pareto_frontier(candidates: list[CandidateMetrics]) -> list[CandidateMetrics]:
    """Return non-dominated candidates in stable input order."""
    frontier = [candidate for candidate in candidates if not any(
        other.id != candidate.id and _dominates(other, candidate) for other in candidates
    )]
    return frontier
