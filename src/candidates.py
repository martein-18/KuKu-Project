"""Candidate generation plumbing. The *prompt* (prompts/candidates.txt) and model are yours."""
from .llm import extract_json
from .prompting import render, format_episode
from .bible import bible_text


def generate(ep, bible, client, n=3, feedback="", include_camera=True, ep_text=None, default_ep=None):
    prompt = render("candidates", bible=bible_text(bible),
                    episode=ep_text or format_episode(ep, include_camera),
                    n_candidates=n, feedback=feedback)
    raw = client.call(prompt, task=f"candidates:ep{ep.ep if ep else default_ep}")
    try:
        data = extract_json(raw)
    except ValueError:
        return []
    cands = data.get("candidates", data) if isinstance(data, dict) else data
    return [c for c in cands if isinstance(c, dict)]
