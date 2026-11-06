"""RANKING RULES: yours to decide. Edit compute_final_score / apply_diversity / dedupe freely.
Everything else in the pipeline only calls rank()."""


def _s(promo, key):
    v = (promo.get("scores") or {}).get(key)
    if isinstance(v, dict):
        v = v.get("score")
    return float(v) if isinstance(v, (int, float)) else 0.0


def compute_final_score(promo, cfg):
    """DEFAULT: weighted sum of hook/cliffhanger/standalone using config weights.
    TODO (yours): add cold_viewer penalty, hook-type bonus, length preference, etc."""
    w = cfg["ranking"]["weights"]
    return sum(w[k] * _s(promo, k) for k in w)


def overlaps(a, b):
    for sa in a["segments"]:
        for sb in b["segments"]:
            if sa["ep"] == sb["ep"] and sa["start"] < sb["end"] and sb["start"] < sa["end"]:
                return True
    return False


def dedupe(ranked):
    """Keep the best-ranked of any overlapping pair (static)."""
    kept = []
    for p in ranked:
        if not any(overlaps(p, k) for k in kept):
            kept.append(p)
    return kept


def apply_diversity(ranked, cfg):
    """TODO (yours): e.g. cap each hook_type at N in the top-K. Default: pass-through."""
    return ranked


def rank(promos, cfg):
    for p in promos:
        p["final_score"] = round(compute_final_score(p, cfg), 3)
    ranked = sorted(promos, key=lambda p: p["final_score"], reverse=True)
    ranked = apply_diversity(dedupe(ranked), cfg)
    for i, p in enumerate(ranked, 1):
        p["rank"] = i
    return ranked
