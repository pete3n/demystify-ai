"""
title: Ollama Logprobs Inspector
description: Streams an Ollama chat completion with per-token logprobs, then renders the response as clickable token chips (colored by confidence) with a top-k alternatives panel — an Open WebUI port of ollama_logprobs.py.
author: you
version: 0.1.0
required_open_webui_version: 0.6.0
license: MIT

Setup
─────
1. Admin Panel → Functions → “+” → paste this file → Save → enable it.
2. Open the Valves (gear icon on the function) and set OLLAMA_BASE_URL:
     • Open WebUI in Docker, Ollama on the host:  http://host.docker.internal:11434
     • Both on bare metal:                        http://localhost:11434
3. Requires Ollama >= 0.12.11 (first version whose API returns logprobs).
4. One model entry appears in the model picker per installed Ollama model,
   prefixed with NAME_PREFIX (default "logprobs/"). Select one and chat.

The reply streams as normal text; when it finishes, an interactive panel is
embedded above the message: every token is a chip colored by the model's
confidence, and clicking a chip shows the top-k alternatives the model
considered at that position.

Note: the embed reports its own height via postMessage, so it works with the
default iframe sandbox settings. No "Same-Origin Access" toggle is needed.
"""

import json
import math

import httpx
from pydantic import BaseModel, Field

# ── Known thinking-block delimiters across models (ported verbatim) ──────
# Each tuple is (open_marker, close_marker).  Add new formats here.
THINK_SPANS = [
    ("<think>",           "</think>"),   # DeepSeek-R1, Qwen3, etc.
    ("<|channel>thought", "<channel|>"), # Gemma 4 (open splits across 2 tokens)
]

# Logprob → chip colour: green (certain) → yellow → red (uncertain).
# logprob range is (-inf, 0]; 0 = 100% confidence.  (Ported verbatim.)
CHIP_STOPS = [
    (-0.05, "#2d5a27", "#a6e3a1", ">95%"),
    (-0.2,  "#3d4a1e", "#d4e88a", ">82%"),
    (-0.5,  "#4a3d1a", "#f9e2af", ">61%"),
    (-1.0,  "#4a2e1a", "#f5b88a", ">37%"),
    (-2.0,  "#4a1e1e", "#f38ba8", ">14%"),
    (-9999, "#2e1e2e", "#cba6f7", "<14%"),
]


class ThinkFilter:
    """
    Think-suppression state machine, ported from stream_completion() in
    ollama_logprobs.py with one improvement: the buffer holds whole token
    entries rather than a flat string, so every surfaced chip keeps the
    exact token boundaries and logprob data the model produced.  (The
    original attributed buffered text to whichever entry flushed it, which
    merged short tokens and skewed their logprobs.)

    Marker detection still operates on the concatenated text, so open/close
    tags split across token boundaries are caught exactly as before; only a
    token that a marker cuts through the middle of gets split, and both
    halves keep that token's logprob data.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.in_think = False
        self.think_close = ""
        self.pending: list[dict] = []
        self.max_hold = max(len(ot) - 1 for ot, _ in THINK_SPANS)

    def _text(self) -> str:
        return "".join(e["token"] for e in self.pending)

    def _take_prefix(self, n: int) -> list:
        """Pop token entries covering the first n chars of buffered text,
        splitting the boundary token if needed."""
        out = []
        while n > 0 and self.pending:
            e = self.pending[0]
            t = e["token"]
            if len(t) <= n:
                out.append(e)
                self.pending.pop(0)
                n -= len(t)
            else:
                out.append({"token": t[:n], "logprob": e["logprob"], "top": e["top"]})
                self.pending[0] = {"token": t[n:], "logprob": e["logprob"], "top": e["top"]}
                n = 0
        return [e for e in out if e["token"]]

    def feed(self, token: str, logprob: float, top: list) -> list:
        td = {"token": token, "logprob": logprob, "top": top}
        if not self.enabled:
            return [td]
        self.pending.append(td)

        out = []
        while True:  # one feed may close a span and open another
            text = self._text()

            if self.in_think:
                idx = text.find(self.think_close)
                if idx == -1:
                    # Discard all but a holdback that could start a split
                    # close marker.
                    hold = len(self.think_close) - 1
                    if len(text) > hold:
                        self._take_prefix(len(text) - hold)  # discard
                    return out
                self._take_prefix(idx + len(self.think_close))  # discard span
                self.in_think = False
                continue

            # Not inside a span: find the earliest open tag, if any.
            best = None
            for open_tag, close_tag in THINK_SPANS:
                i = text.find(open_tag)
                if i != -1 and (best is None or i < best[0]):
                    best = (i, open_tag, close_tag)
            if best is not None:
                i, open_tag, close_tag = best
                out += self._take_prefix(i)          # surface pre-tag text
                self._take_prefix(len(open_tag))     # discard the tag itself
                self.in_think = True
                self.think_close = close_tag
                continue

            # No tag: hold back (longest_open_tag - 1) chars to catch split
            # tags; surface everything before that.
            if len(text) > self.max_hold:
                out += self._take_prefix(len(text) - self.max_hold)
            return out

    def flush(self) -> list:
        """Flush any remaining buffered content after the stream ends."""
        if not self.enabled or self.in_think:
            self.pending = []
            return []
        out = [e for e in self.pending if e["token"]]
        self.pending = []
        return out


def parse_logprob_entries(obj: dict) -> list:
    """
    Ollama logprob structure varies by version (ported verbatim):
      older: obj["logprobs"] is a list of {token, logprob, top_logprobs}
      newer: obj["logprobs"]["content"] is that list
    """
    raw_lp = obj.get("logprobs")
    if isinstance(raw_lp, list):
        return raw_lp
    if isinstance(raw_lp, dict):
        return raw_lp.get("content") or []
    return []


def build_embed_html(tokens: list, model: str) -> str:
    """Self-contained interactive HTML for the Rich UI embed."""
    payload = json.dumps(
        {"model": model, "tokens": tokens},
        ensure_ascii=False,
    ).replace("</", "<\\/")  # keep </script> etc. from closing our tag

    legend_rows = "".join(
        f'<span class="lg" style="background:{bg};color:{fg}">{label}</span>'
        for _, bg, fg, label in CHIP_STOPS
    )
    stops_js = json.dumps([[t, bg, fg] for t, bg, fg, _ in CHIP_STOPS])

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  :root {{
    --bg: #1e1e2e; --panel: #27273d; --border: #3a3a5c;
    --text: #cdd6f4; --dim: #7f849c; --accent: #89b4fa;
    --accent2: #cba6f7; --bar-bg: #313244;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--text);
    font: 13px/1.5 "Segoe UI", system-ui, sans-serif; padding: 12px;
  }}
  .hdr {{
    color: var(--dim); font-size: 11px; font-weight: 600;
    letter-spacing: .04em; margin-bottom: 8px; display: flex;
    justify-content: space-between; gap: 8px; flex-wrap: wrap;
  }}
  .hdr .meta {{ font-weight: 400; letter-spacing: 0; }}
  #chips {{
    display: flex; flex-wrap: wrap; align-items: baseline; row-gap: 3px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px; max-height: 380px; overflow-y: auto;
  }}
  .chip {{
    font: 12px/1.6 ui-monospace, "Cascadia Mono", "Segoe UI Mono", Menlo, monospace;
    white-space: pre; cursor: pointer; padding: 1px 0; border-radius: 2px;
    outline: 1px solid transparent;
  }}
  .chip:hover {{ outline-color: var(--text); }}
  .chip.sel {{ outline: 2px solid var(--accent); background: var(--border) !important;
               color: var(--text) !important; }}
  .nl {{ color: var(--dim); font: 12px ui-monospace, monospace; padding: 0 2px; }}
  .brk {{ flex-basis: 100%; height: 0; }}
  .divider {{ height: 1px; background: var(--border); margin: 10px 0; }}
  #dist-title {{ color: var(--dim); font-size: 12px; margin-bottom: 6px;
                 font-family: ui-monospace, monospace; }}
  #dist-title b {{ color: var(--accent2); }}
  #bars {{ max-height: 320px; overflow-y: auto; }}
  .row {{ display: flex; align-items: center; gap: 6px; padding: 2px 0; }}
  .row .tok {{
    width: 130px; text-align: right; overflow: hidden; text-overflow: ellipsis;
    font: 12px ui-monospace, monospace; white-space: pre; flex: none;
  }}
  .row.hl .tok {{ color: var(--accent2); }}
  .row .track {{ width: 220px; height: 14px; background: var(--bar-bg);
                 flex: none; border-radius: 2px; overflow: hidden; }}
  .row .fill {{ height: 100%; background: var(--accent); }}
  .row.hl .fill {{ background: var(--accent2); }}
  .row .pct {{ width: 54px; color: var(--dim); font-size: 12px; flex: none; }}
  .row.hl .pct {{ color: var(--accent2); }}
  .row .lp {{ color: var(--dim); font: 11px ui-monospace, monospace; }}
  .legend {{ margin-top: 10px; display: flex; gap: 6px; flex-wrap: wrap;
             align-items: center; color: var(--dim); font-size: 11px; }}
  .lg {{ font-size: 11px; font-weight: 600; padding: 1px 6px; border-radius: 3px; }}
  details {{ margin-top: 10px; color: var(--dim); font-size: 11px; }}
  summary {{ cursor: pointer; }}
  textarea {{
    width: 100%; height: 140px; margin-top: 6px; background: var(--bg);
    color: var(--text); border: 1px solid var(--border); border-radius: 4px;
    font: 11px ui-monospace, monospace; padding: 6px;
  }}
</style>
</head>
<body>
  <div class="hdr">
    <span>GENERATED RESPONSE — CLICK ANY TOKEN TO INSPECT</span>
    <span class="meta" id="meta"></span>
  </div>
  <div id="chips"></div>
  <div class="divider"></div>
  <div id="dist-title">Click a token to see its alternatives</div>
  <div id="bars"></div>
  <div class="legend"><span>Token confidence:</span>{legend_rows}</div>
  <details>
    <summary>Debug JSON</summary>
    <textarea id="dbg" readonly onclick="this.select()"></textarea>
  </details>

<script type="application/json" id="lp-data">{payload}</script>
<script>
(function () {{
  var DATA  = JSON.parse(document.getElementById('lp-data').textContent);
  var STOPS = {stops_js};
  var chipsEl = document.getElementById('chips');
  var barsEl  = document.getElementById('bars');
  var titleEl = document.getElementById('dist-title');
  var selected = null;

  document.getElementById('meta').textContent =
    DATA.model + ' \\u00b7 ' + DATA.tokens.length + ' tokens';
  document.getElementById('dbg').value = JSON.stringify(DATA, null, 2);

  function colours(lp) {{
    for (var i = 0; i < STOPS.length; i++)
      if (lp >= STOPS[i][0]) return [STOPS[i][1], STOPS[i][2]];
    var last = STOPS[STOPS.length - 1];
    return [last[1], last[2]];
  }}
  function pct(lp)  {{ return (Math.exp(lp) * 100).toFixed(1) + '%'; }}
  function disp(t)  {{ return (t.trim() === '' && t !== ' ') ? JSON.stringify(t) : t; }}

  // ── Chips ──────────────────────────────────────────────────────────
  DATA.tokens.forEach(function (td, idx) {{
    var parts = td.token.split('\\n');
    parts.forEach(function (part, i) {{
      if (i > 0) {{
        var mark = document.createElement('span');
        mark.className = 'nl';
        mark.textContent = '\\u21b5';
        chipsEl.appendChild(mark);
        var brk = document.createElement('span');
        brk.className = 'brk';
        chipsEl.appendChild(brk);
      }}
      if (!part) return;
      var chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = part;
      var c = colours(td.logprob);
      chip.style.background = c[0];
      chip.style.color = c[1];
      chip.dataset.idx = idx;
      chip.addEventListener('click', function () {{ select(idx, chip); }});
      chipsEl.appendChild(chip);
    }});
  }});

  // ── Chip click → distribution ──────────────────────────────────────
  function select(idx, chip) {{
    if (selected) selected.classList.remove('sel');
    selected = chip;
    chip.classList.add('sel');

    var td = DATA.tokens[idx];
    titleEl.innerHTML = 'Token ' + (idx + 1) + ': <b></b> \\u2014 chosen at ' +
                        pct(td.logprob) + ' probability';
    titleEl.querySelector('b').textContent =
      td.token.trim() ? '"' + td.token + '"' : JSON.stringify(td.token);

    // Merge chosen token in if not already present, sort by logprob desc
    var entries = (td.top || []).slice();
    var present = entries.some(function (e) {{ return e.token === td.token; }});
    if (!present) entries.unshift({{ token: td.token, logprob: td.logprob }});
    entries.sort(function (a, b) {{ return b.logprob - a.logprob; }});

    barsEl.innerHTML = '';
    entries.forEach(function (e) {{
      var row = document.createElement('div');
      row.className = 'row' + (e.token === td.token ? ' hl' : '');
      var w = Math.max(0, Math.min(100, Math.exp(e.logprob) * 100));
      var tok = document.createElement('span');
      tok.className = 'tok';
      tok.textContent = disp(e.token);
      tok.title = JSON.stringify(e.token);
      var track = document.createElement('span');
      track.className = 'track';
      var fill = document.createElement('span');
      fill.className = 'fill';
      fill.style.width = w + '%';
      fill.style.display = 'block';
      track.appendChild(fill);
      var p = document.createElement('span');
      p.className = 'pct';
      p.textContent = pct(e.logprob);
      var lp = document.createElement('span');
      lp.className = 'lp';
      lp.textContent = 'lp=' + e.logprob.toFixed(3);
      row.appendChild(tok); row.appendChild(track);
      row.appendChild(p);   row.appendChild(lp);
      barsEl.appendChild(row);
    }});
  }}

  // ── Iframe height reporting (required with default sandbox) ───────
  function reportHeight() {{
    var h = document.documentElement.scrollHeight;
    parent.postMessage({{ type: 'iframe:height', height: h }}, '*');
  }}
  window.addEventListener('load', reportHeight);
  new ResizeObserver(reportHeight).observe(document.body);
}})();
</script>
</body>
</html>"""


class Pipe:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = Field(
            default="http://localhost:11434",
            description=(
                "Base URL of the Ollama server, reachable FROM THE OPEN WEBUI "
                "BACKEND. If Open WebUI runs in Docker and Ollama on the host, "
                "use http://host.docker.internal:11434"
            ),
        )
        NAME_PREFIX: str = Field(
            default="logprobs/",
            description="Prefix shown before each Ollama model in the model picker.",
        )
        TOP_LOGPROBS: int = Field(
            default=5, ge=1, le=20,
            description="How many alternative tokens to fetch per position (model-dependent max).",
        )
        TEMPERATURE: float = Field(
            default=1.0,
            description="Sampling temperature passed to Ollama.",
        )
        SUPPRESS_THINK: bool = Field(
            default=True,
            description="Strip <think>-style reasoning blocks from the output and the token panel.",
        )
        MAX_PANEL_TOKENS: int = Field(
            default=4000,
            description="Safety cap on tokens stored for the interactive panel (text still streams fully).",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ── One model entry per installed Ollama model (manifold) ────────────
    async def pipes(self):
        base = self.valves.OLLAMA_BASE_URL.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                r = await client.get(f"{base}/api/tags")
                r.raise_for_status()
                models = r.json().get("models", [])
            if not models:
                return [{"id": "error", "name": "No Ollama models found"}]
            return [
                {"id": m["name"], "name": f"{self.valves.NAME_PREFIX}{m['name']}"}
                for m in models
            ]
        except Exception:
            return [{
                "id": "error",
                "name": f"Cannot reach Ollama at {base} — check the OLLAMA_BASE_URL valve",
            }]

    # ── Main entry point ──────────────────────────────────────────────────
    async def pipe(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__=None,
        __task__=None,
    ):
        base = self.valves.OLLAMA_BASE_URL.rstrip("/")
        # body["model"] is "{function_id}.{ollama_model}"; the function id
        # contains no dots, ollama model names may ("llama3.1:8b").
        model = body.get("model", "")
        model = model[model.find(".") + 1:]
        if model == "error":
            return "Ollama is not reachable. Fix the OLLAMA_BASE_URL valve and reload."
        messages = body.get("messages", [])

        # Background tasks (title/tag generation, autocomplete) also route
        # through the selected model — handle them plainly, no logprobs/embed.
        if __task__ is not None:
            return await self._plain_completion(base, model, messages)

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.valves.TEMPERATURE},
            "logprobs": True,
            "top_logprobs": self.valves.TOP_LOGPROBS,
        }

        valves = self.valves

        async def stream():
            tokens: list[dict] = []
            saw_logprobs = False
            think = ThinkFilter(valves.SUPPRESS_THINK)

            def keep(td: dict):
                if len(tokens) < valves.MAX_PANEL_TOKENS:
                    tokens.append(td)

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=5, read=300, write=30, pool=5)
                ) as client:
                    async with client.stream(
                        "POST", f"{base}/api/chat", json=payload
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            obj = json.loads(line)
                            if obj.get("error"):
                                yield f"\n\n> ⚠️ Ollama error: {obj['error']}"
                                return
                            if obj.get("done"):
                                break

                            entries = parse_logprob_entries(obj)
                            if entries:
                                saw_logprobs = True
                                for entry in entries:
                                    if not isinstance(entry, dict):
                                        continue
                                    for td in think.feed(
                                        entry.get("token", ""),
                                        entry.get("logprob", 0.0),
                                        entry.get("top_logprobs") or [],
                                    ):
                                        keep(td)
                                        yield td["token"]
                            else:
                                # No logprobs on this chunk — fall back to the
                                # plain message content so text still streams.
                                content = (obj.get("message") or {}).get("content", "")
                                if content:
                                    for td in think.feed(content, 0.0, []):
                                        yield td["token"]

                for td in think.flush():
                    keep(td)
                    yield td["token"]

            except httpx.ConnectError as e:
                yield (
                    f"\n\n> ⚠️ Cannot reach Ollama at `{base}` from the Open WebUI "
                    f"backend ({e}). If Open WebUI runs in Docker, set the "
                    f"OLLAMA_BASE_URL valve to `http://host.docker.internal:11434`."
                )
                return
            except Exception as e:
                yield f"\n\n> ⚠️ Logprobs inspector error: {type(e).__name__}: {e}"
                return

            # ── Post-stream: emit the interactive panel ───────────────────
            if not saw_logprobs:
                yield (
                    "\n\n> ⚠️ Ollama returned no logprobs. The interactive panel "
                    "was skipped — check that Ollama is v0.12.11 or newer."
                )
                return

            if __event_emitter__ is not None:
                # Normalize top_logprobs entries to a compact shape
                for td in tokens:
                    td["top"] = [
                        {"token": t.get("token", ""), "logprob": t.get("logprob", 0.0)}
                        for t in td["top"]
                        if isinstance(t, dict)
                    ]
                html = build_embed_html(tokens, model)
                await __event_emitter__(
                    {"type": "embeds", "data": {"embeds": [html]}}
                )
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": (
                            f"{len(tokens)} tokens — click chips in the panel "
                            f"above to inspect alternatives"
                        ),
                        "done": True,
                    },
                })

        return stream()

    # ── Plain non-stream completion for background tasks ─────────────────
    async def _plain_completion(self, base: str, model: str, messages: list) -> str:
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f"{base}/api/chat",
                    json={"model": model, "messages": messages, "stream": False},
                )
                r.raise_for_status()
                return (r.json().get("message") or {}).get("content", "")
        except Exception as e:
            return f"Error: {e}"
