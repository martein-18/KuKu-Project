"""CLI:  python run.py parse            -> outputs/parsed/*.json (no LLM, no key)
        python run.py bible            -> outputs/show_bible.json
        python run.py all [--mock]     -> full pipeline
        python run.py all --episodes 7 9 --out outputs/run_modelB     (compare models via --out)
"""
import argparse, json, sys
from pathlib import Path
from src.config import load_config
from src.parser import parse_dir
from src.llm import make_client
from src.bible import build_bible, load_bible
from src.candidates import generate
from src.validate import validate_candidate, is_valid
from src.score import score_clip, cold_viewer
from src.prompting import clip_text
from src.rank import rank
from src.outputs import write_all
from src.stitch import find_links, pair_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["parse", "bible", "all"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--episodes", nargs="*", type=int)
    ap.add_argument("--out")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--stitch", action="store_true", help="also try candidates across linked episode pairs")
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = Path(a.out or cfg["paths"]["outputs_dir"])
    out.mkdir(parents=True, exist_ok=True)
    eps = parse_dir(cfg["paths"]["data_dir"])
    if a.episodes:
        eps = [e for e in eps if e.ep in a.episodes]
    ep_map = {e.ep: e for e in eps}

    (out / "parsed").mkdir(exist_ok=True)
    bad = 0
    for e in eps:
        (out / "parsed" / f"ep{e.ep:03d}.json").write_text(json.dumps(e.to_dict(), indent=1, ensure_ascii=False))
        if e.warnings:
            bad += 1
            print(f"! Ep {e.ep} ({e.path}): {e.warnings}")
    print(f"parsed {len(eps)} episodes, {sum(len(e.events) for e in eps)} events, {bad} with warnings")
    if a.stage == "parse":
        return

    client = make_client(cfg, out / "raw", force_mock=a.mock)
    bible_path = out / "show_bible.json"
    bible = build_bible(eps, client, bible_path) if (a.stage == "bible" or not bible_path.exists()) else load_bible(bible_path)
    if a.stage == "bible":
        return

    pc, vc = cfg["pipeline"], cfg["validation"]
    promos, rejected = [], []

    def handle(cands, default_ep, source):
        ok = []
        for c in cands:
            segs, issues = validate_candidate(ep_map, c, default_ep, vc)
            if is_valid(issues):
                ok.append((c, segs, issues))
            else:
                rejected.append({"source": source, "candidate": c, "issues": [str(i) for i in issues]})
        return ok

    def run_unit(ep, default_ep, ep_text, source):
        feedback, accepted = "", []
        for attempt in range(pc["max_retries"] + 1):
            cands = generate(ep, bible, client, pc["candidates_per_episode"], feedback,
                             pc["include_camera_in_prompt"], ep_text, default_ep)
            accepted = handle(cands, default_ep, f"{source}#try{attempt}")
            if accepted:
                break
            feedback = "Previous attempt rejected: " + "; ".join(r["issues"][0] for r in rejected[-3:] if r["issues"])
        for c, segs, issues in accepted:
            text = clip_text(ep_map, segs)
            p = {"id": f"{source}-{len(promos)+1}", "source": source,
                 "segments": [s.to_dict() for s in segs],
                 "total_duration": round(sum(s.duration for s in segs), 2),
                 "hook": c.get("hook", ""), "cliffhanger": c.get("cliffhanger", ""), "why": c.get("why", ""),
                 "warnings": [str(i) for i in issues], "clip_text": text,
                 "scores": score_clip(text, c, bible, client)}
            if pc.get("run_cold_viewer", True):
                p["cold_viewer"] = cold_viewer(text, client)
            promos.append(p)

    for e in eps:
        print(f"episode {e.ep} ...")
        run_unit(e, e.ep, None, f"ep{e.ep}")
    if a.stitch:
        for n, m, strength, why in find_links(eps):
            if n in ep_map and m in ep_map:
                print(f"stitch {n}->{m} ({strength}: {why})")
                run_unit(None, n, pair_text(ep_map[n], ep_map[m]), f"stitch{n}-{m}")

    ranked = rank(promos, cfg)
    meta = {"model": cfg["llm"]["model"], "provider": "mock" if a.mock else cfg["llm"]["provider"],
            "episodes": [e.ep for e in eps], "rejected_count": len(rejected)}
    write_all(ranked, out, meta)
    (out / "rejections.json").write_text(json.dumps(rejected, indent=1, ensure_ascii=False))
    print(f"{len(ranked)} promos ranked, {len(rejected)} candidates rejected -> {out}")


if __name__ == "__main__":
    sys.exit(main())
