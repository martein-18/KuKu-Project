import csv, json
from pathlib import Path


def _segs_str(p):
    return " | ".join(f"Ep{s['ep']} {s['start_ts']}-{s['end_ts']}" for s in p["segments"])


def write_all(promos, out_dir, run_meta=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "promos.json").write_text(json.dumps({"meta": run_meta or {}, "promos": promos}, indent=2, ensure_ascii=False))
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "episodes", "segments", "total_s", "hook", "cliffhanger", "why", "hook_type",
                    "hook_score", "cliff_score", "standalone_score", "final_score"])
        for p in promos:
            sc = p.get("scores") or {}
            g = lambda k: (sc.get(k) or {}).get("score", "") if isinstance(sc.get(k), dict) else sc.get(k, "")
            w.writerow([p["rank"], ",".join(sorted({str(s['ep']) for s in p['segments']})), _segs_str(p),
                        p["total_duration"], p.get("hook", ""), p.get("cliffhanger", ""), p.get("why", ""),
                        sc.get("hook_type", ""), g("hook"), g("cliffhanger"), g("standalone"), p.get("final_score", "")])
    lines = ["# Promo report", "", f"Model: `{(run_meta or {}).get('model', '?')}` | promos: {len(promos)}", ""]
    for p in promos:
        lines += [f"## #{p['rank']} - {_segs_str(p)} ({p['total_duration']}s)  score {p.get('final_score', '')}",
                  f"- **Hook:** {p.get('hook', '')}", f"- **Cliffhanger:** {p.get('cliffhanger', '')}",
                  f"- **Why it works:** {p.get('why', '')}", f"- **Scores:** `{json.dumps(p.get('scores'), ensure_ascii=False)}`",
                  f"- **Cold viewer:** `{json.dumps(p.get('cold_viewer'), ensure_ascii=False)}`",
                  "", "```", p.get("clip_text", ""), "```", ""]
    (out / "report.md").write_text("\n".join(lines))
