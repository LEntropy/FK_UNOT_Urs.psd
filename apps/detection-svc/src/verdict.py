"""Explainable signal aggregation; this is a similarity verdict, not a legal conclusion."""

from __future__ import annotations


def decide_verdict(
    *, phash_distance: int | None, watermark_match: bool | None, fingerprint_distance: int | None,
    c2pa_status: str | None, match_threshold: int = 20, review_threshold: int = 36
) -> dict:
    reasons: list[str] = []
    score = 0.0
    if watermark_match is True:
        score += 0.55
        reasons.append("embedded watermark matched")
    if phash_distance is not None:
        if phash_distance <= match_threshold:
            score += 0.35
            reasons.append(f"pHash distance {phash_distance} <= {match_threshold}")
        elif phash_distance <= review_threshold:
            score += 0.15
            reasons.append(f"pHash distance {phash_distance} requires review")
    if fingerprint_distance is not None and fingerprint_distance <= match_threshold:
        score += 0.25
        reasons.append("multi-view fingerprint matched")
    if c2pa_status == "VALID":
        score += 0.10
        reasons.append("C2PA manifest validated")
    score = min(score, 1.0)
    if watermark_match is True or score >= 0.55:
        verdict = "MATCH"
    elif score >= 0.15:
        verdict = "REVIEW"
    elif all(value is None for value in (phash_distance, watermark_match, fingerprint_distance)):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO_MATCH"
    return {"verdict": verdict, "confidence": round(score, 4), "reasons": reasons, "ruleVersion": "1.0"}
