"""One function behind which every model lives. Swap models in config.yaml -> llm: only.

Providers:
  openai_compat : any OpenAI-compatible /chat/completions endpoint (OpenAI, Groq, Gemini's
                  OpenAI endpoint, LiteLLM proxy, Ollama...). stdlib only, no extra packages.
  mock          : offline deterministic stub so you can test the pipeline without a key.
Every call is appended untouched to <outputs>/raw/calls.jsonl (evidence for your write-up).
"""
import json, os, time, urllib.request, urllib.error
from pathlib import Path


class LLMError(RuntimeError):
    pass


class BaseClient:
    def __init__(self, cfg, raw_dir):
        self.cfg = cfg
        self.model = cfg.get("model", "mock")
        self.raw = Path(raw_dir)
        self.raw.mkdir(parents=True, exist_ok=True)

    def _log(self, task, prompt, system, response, secs):
        rec = {"ts": time.time(), "task": task, "model": self.model, "system": system,
               "prompt": prompt, "response": response, "seconds": round(secs, 2)}
        with open(self.raw / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def call(self, prompt, system=None, task=""):
        t = time.time()
        out = self._call(prompt, system, task)
        self._log(task, prompt, system, out, time.time() - t)
        return out

    def _call(self, prompt, system, task):
        raise NotImplementedError


class OpenAICompatClient(BaseClient):
    def _call(self, prompt, system, task):
        c = self.cfg
        key = os.environ.get(c.get("api_key_env", "LLM_API_KEY"), "")
        if not key:
            raise LLMError(f"set env var {c.get('api_key_env', 'LLM_API_KEY')}")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "temperature": c.get("temperature", 0.2)}
        if c.get("max_tokens"):
            body["max_tokens"] = c["max_tokens"]
        if c.get("extra_body"):
            body.update(c["extra_body"])
        req = urllib.request.Request(c["base_url"].rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        last = None
        for attempt in range(c.get("retries", 3)):
            try:
                with urllib.request.urlopen(req, timeout=c.get("timeout_s", 120)) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last = f"{e.code} {e.read().decode(errors='replace')[:300]}"
            except Exception as e:  # network, parse
                last = repr(e)
            time.sleep(2 ** attempt)
        raise LLMError(last)


class MockClient(BaseClient):
    """Deterministic fake. Candidate task picks the first events spanning ~35s so that the
    whole pipeline (validators, outputs) can be exercised offline."""
    def _call(self, prompt, system, task):
        if task.startswith("candidates"):
            import re
            rows = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in
                    re.finditer(r"^\[(\d+)\] (\d+\.\d+)-(\d+\.\d+)", prompt, re.M)]
            if not rows:
                return json.dumps({"candidates": []})
            a, t0 = rows[0][0], rows[0][1]
            b = next((i for i, s, e in rows if e - t0 >= 35), rows[-1][0])
            return json.dumps({"candidates": [{"segments": [[a, b]], "hook_event": a, "cliffhanger_event": b,
                                               "hook": "mock hook", "cliffhanger": "mock cliffhanger", "why": "mock"}]})
        if task == "score":
            return json.dumps({"hook": {"score": 3, "reason": "mock"}, "cliffhanger": {"score": 3, "reason": "mock"},
                               "standalone": {"score": 3, "reason": "mock"}, "hook_type": "other"})
        if task == "cold_viewer":
            return json.dumps({"characters": "mock", "conflict": "mock", "next": "mock", "can_answer": True})
        if task == "bible":
            return json.dumps({"premise": "mock", "characters": [], "aliases": {}})
        return "{}"


def make_client(cfg, raw_dir, force_mock=False):
    lc = cfg["llm"]
    if force_mock or lc.get("provider") == "mock":
        return MockClient(lc, raw_dir)
    if lc.get("provider") == "openai_compat":
        return OpenAICompatClient(lc, raw_dir)
    raise LLMError(f"unknown provider {lc.get('provider')!r}")


def extract_json(text):
    """Pull the first JSON object/array out of a model reply (handles ``` fences, preamble)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text or ""):
        if ch in "{[":
            try:
                return dec.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in model reply")
