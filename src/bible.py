import json
from pathlib import Path
from .llm import extract_json
from .prompting import render


def build_bible(episodes, client, out_path):
    eps = "\n".join(f"Ep {e.ep}: {e.logline}\n   continuity: {e.continuity}" for e in episodes)
    raw = client.call(render("bible", episodes=eps), task="bible")
    bible = extract_json(raw)
    Path(out_path).write_text(json.dumps(bible, indent=2, ensure_ascii=False))
    return bible


def load_bible(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def bible_text(bible):
    return json.dumps(bible, ensure_ascii=False, indent=1) if bible else "(none)"
