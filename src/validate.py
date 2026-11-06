"""Rule-based validation (code). Thresholds come from config.yaml -> validation:."""
from dataclasses import dataclass
from .timeline import normalize_spec, resolve_segment


@dataclass
class Issue:
    severity: str   # "error" rejects the candidate, "warn" is logged only
    code: str
    msg: str

    def __str__(self):
        return f"[{self.severity}] {self.code}: {self.msg}"


def validate_candidate(ep_map: dict, cand: dict, default_ep: int, v: dict):
    """Returns (segments, issues). Candidate is valid iff no issue has severity 'error'."""
    issues, segs = [], []
    specs = cand.get("segments") or []
    if not specs:
        return [], [Issue("error", "no_segments", "candidate has no segments")]
    if len(specs) > v.get("max_segments", 4):
        issues.append(Issue("error", "too_many_segments", f"{len(specs)} > {v.get('max_segments', 4)}"))

    for spec in specs:
        norm = normalize_spec(spec, default_ep)
        if norm is None:
            issues.append(Issue("error", "bad_spec", f"malformed segment {spec!r}"))
            continue
        ep_no, a, b = norm
        ep = ep_map.get(ep_no)
        if ep is None:
            issues.append(Issue("error", "unknown_episode", f"episode {ep_no} not loaded"))
            continue
        seg, err = resolve_segment(ep, a, b, v.get("snap_end", "last_spoken"), v.get("end_pad_s", 0.3))
        if err:
            issues.append(Issue("error", "bad_ids", err))
            continue
        segs.append(seg)
    if not segs:
        return segs, issues

    # order / overlap (only comparable inside the same episode)
    for p, q in zip(segs, segs[1:]):
        if p.ep == q.ep and q.start_id <= p.end_id:
            issues.append(Issue("error", "order_overlap", f"ep{p.ep}: ids {p.start_id}-{p.end_id} then {q.start_id}-{q.end_id}"))
        elif p.ep == q.ep and q.start < p.end:
            issues.append(Issue("error", "time_overlap", f"ep{p.ep}: {p.end:.2f}s overlaps next start {q.start:.2f}s"))

    total = sum(s.duration for s in segs)
    lo, hi = v.get("min_duration_s", 20), v.get("max_duration_s", 60)
    if not lo <= total <= hi:
        issues.append(Issue("error", "duration", f"total {total:.1f}s outside {lo}-{hi}s"))

    # first beat: first non-camera event must start close to clip start
    first = segs[0]
    ep0 = ep_map[first.ep]
    beats = [e for e in ep0.events if first.start_id <= e.id <= first.end_id and e.type != "CAMERA"]
    if beats:
        gap = min(e.start for e in beats) - first.start
        if gap > v.get("first_beat_max_s", 3.0):
            issues.append(Issue("error", "slow_open", f"first action/speech starts {gap:.1f}s into the clip"))

    # cutting through a line of dialogue at either boundary
    for s in segs:
        ep = ep_map[s.ep]
        for e in ep.events:
            if e.spoken and not (s.start_id <= e.id <= s.end_id):
                if e.start < s.end - 0.05 and e.end > s.end + 0.05 and e.start >= s.start:
                    issues.append(Issue("warn", "cuts_line_end", f"ep{s.ep} line {e.id} ('{e.text[:40]}') starts inside clip but is excluded"))
                if e.start < s.start - 0.05 and e.end > s.start + 0.05:
                    issues.append(Issue("warn", "cuts_line_start", f"ep{s.ep} line {e.id} straddles clip start"))
    return segs, issues


def is_valid(issues):
    return not any(i.severity == "error" for i in issues)
