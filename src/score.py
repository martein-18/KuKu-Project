"""Scoring plumbing. Rubrics live in prompts/score.txt and prompts/cold_viewer.txt (yours)."""
from .llm import extract_json
from .prompting import render
from .bible import bible_text


def _safe(raw):
    try:
        return extract_json(raw)
    except ValueError:
        return {"_parse_error": raw[:200]}


def score_clip(clip, cand, bible, client):
    raw = client.call(render("score", bible=bible_text(bible), clip=clip,
                             hook=cand.get("hook", ""), cliffhanger=cand.get("cliffhanger", "")), task="score")
    return _safe(raw)


def cold_viewer(clip, client):
    """Deliberately gets NO bible: checks the clip stands alone."""
    return _safe(client.call(render("cold_viewer", clip=clip), task="cold_viewer"))
