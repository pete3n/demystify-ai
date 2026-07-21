"""
Ollama Logprobs Inspector
─────────────────────────
Sends a prompt to a local Ollama model, collects per-token logprobs,
then displays the full response as clickable token chips.  Click any
chip to see the probability distribution of alternatives the model
considered at that position.

Run:    python3 ollama_logprobs.py
Needs:  Python 3.8+, tkinter (standard library), Ollama running locally.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import math
import threading
import urllib.request
import urllib.error

# ── Palette (matches bigram_lm.py) ──────────────────────────────────────
BG       = "#1e1e2e"
PANEL_BG = "#27273d"
BORDER   = "#3a3a5c"
TEXT     = "#cdd6f4"
TEXT_DIM = "#7f849c"
ACCENT   = "#89b4fa"
ACCENT2  = "#cba6f7"
GREEN    = "#a6e3a1"
YELLOW   = "#f9e2af"
RED      = "#f38ba8"
TEAL     = "#94e2d5"
BAR_BG   = "#313244"

# ── Scale factor ──────────────────────────────────────────────────────────
# 1.0 = default, 1.5 = presentation recording, 2.0 = large display
SCALE = 2.0

def s(n):
    """Scale a pixel value by SCALE."""
    return int(n * SCALE)

def fs(n):
    """Scale a font size by SCALE."""
    return max(1, int(n * SCALE))

OLLAMA_BASE = "http://localhost:11434"

# Known thinking-block delimiters across models.
# Each tuple is (open_marker, close_marker).  Add new formats here.
THINK_SPANS = [
    ("<think>",            "</think>"),    # DeepSeek-R1, Qwen3, etc.
    ("<|channel>thought",  "<channel|>"), # Gemma 4 (open splits across 2 tokens; close is <channel|>)
]

# Logprob → background colour: green (certain) → yellow → red (uncertain)
# logprob range is (-inf, 0]; 0 = 100% confidence
CHIP_STOPS = [
    (-0.05, "#2d5a27", "#a6e3a1"),   # >95% — dark green bg, green text
    (-0.2,  "#3d4a1e", "#d4e88a"),   # >82%
    (-0.5,  "#4a3d1a", "#f9e2af"),   # >61%
    (-1.0,  "#4a2e1a", "#f5b88a"),   # >37%
    (-2.0,  "#4a1e1e", "#f38ba8"),   # >14%
    (-9999, "#2e1e2e", "#cba6f7"),   # everything else
]


def logprob_colours(lp: float):
    for threshold, bg, fg in CHIP_STOPS:
        if lp >= threshold:
            return bg, fg
    return CHIP_STOPS[-1][1], CHIP_STOPS[-1][2]


def prob_pct(lp: float) -> str:
    return f"{math.exp(lp)*100:.1f}%"


# ── Ollama API ───────────────────────────────────────────────────────────
def fetch_models() -> list[str]:
    try:
        req = urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=4)
        data = json.loads(req.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def stream_completion(model: str, prompt: str, top_k: int, suppress_think: bool,
                      callback, done_cb, error_cb):
    """
    Stream /api/generate with logprobs.  Calls callback(token_data) per token
    where token_data = {"token": str, "logprob": float, "top_logprobs": [...]}
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 1.0, "think": not suppress_think},
        "logprobs": True,
        "top_logprobs": top_k,
    }).encode()

    in_think = False   # track whether we are inside a thinking span
    think_close = ""  # the close marker we are waiting for
    think_buf = ""    # accumulate tokens to detect split markers
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw_line in resp:
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("done"):
                    break

                # Ollama logprob structure varies by version:
                #   older: obj["logprobs"] is a list of {token, logprob, top_logprobs}
                #   newer: obj["logprobs"]["content"] is that list
                raw_lp = obj.get("logprobs")
                if isinstance(raw_lp, list):
                    lp_content = raw_lp
                elif isinstance(raw_lp, dict):
                    lp_content = raw_lp.get("content") or []
                else:
                    lp_content = []

                for entry in lp_content:
                    if not isinstance(entry, dict):
                        continue
                    token = entry.get("token", "")
                    if suppress_think:
                        think_buf += token

                        if in_think:
                            idx = think_buf.find(think_close)
                            if idx != -1:
                                in_think = False
                                think_buf = think_buf[idx + len(think_close):]
                                # fall through to flush any post-close content below
                            else:
                                continue  # still inside thinking span, discard

                        if not in_think:
                            found_open = False
                            for open_tag, close_tag in THINK_SPANS:
                                idx = think_buf.find(open_tag)
                                if idx != -1:
                                    pre = think_buf[:idx]
                                    think_buf = think_buf[idx + len(open_tag):]
                                    in_think = True
                                    think_close = close_tag
                                    if pre:
                                        callback({
                                            "token":   pre,
                                            "logprob": entry.get("logprob", 0.0),
                                            "top":     entry.get("top_logprobs") or [],
                                        })
                                    found_open = True
                                    break

                            if not found_open:
                                # Hold back (longest_open_tag - 1) chars to catch split tags.
                                # Flush everything before that as safe output.
                                max_hold = max(len(ot) - 1 for ot, _ in THINK_SPANS)
                                if len(think_buf) > max_hold:
                                    safe = think_buf[:-max_hold] if max_hold else think_buf
                                    think_buf = think_buf[-max_hold:] if max_hold else ""
                                    if safe:
                                        callback({
                                            "token":   safe,
                                            "logprob": entry.get("logprob", 0.0),
                                            "top":     entry.get("top_logprobs") or [],
                                        })
                        continue  # suppress_think path always handled above
                    callback({
                        "token":   token,
                        "logprob": entry.get("logprob", 0.0),
                        "top":     entry.get("top_logprobs") or [],
                    })
        # Flush any remaining buffered content after stream ends
        if suppress_think and think_buf and not in_think:
            callback({
                "token":   think_buf,
                "logprob": 0.0,
                "top":     [],
            })
    except urllib.error.URLError as e:
        error_cb(f"Cannot reach Ollama at {OLLAMA_BASE}\n\n{e}")
        return
    except Exception as e:
        import traceback
        error_cb(traceback.format_exc())
        return
    done_cb()


# ── Reusable widgets ─────────────────────────────────────────────────────
def lbl(parent, text="", size=10, color=TEXT, bold=False, **kw):
    kw.setdefault("bg", parent.cget("bg"))
    kw.setdefault("fg", color)
    kw["font"] = ("Segoe UI", fs(size), "bold" if bold else "normal")
    return tk.Label(parent, text=text, **kw)


def section_title(parent, text):
    lbl(parent, text=text, size=9, color=TEXT_DIM, bold=True).pack(anchor="w", pady=(0, s(6)))


def divider(parent):
    tk.Frame(parent, bg=BORDER, height=max(1, s(1))).pack(fill="x", pady=s(8))


def hframe(parent, **kw):
    kw.setdefault("bg", parent.cget("bg"))
    return tk.Frame(parent, **kw)


class BarRow(tk.Frame):
    WIDTH = s(220)

    def __init__(self, parent):
        super().__init__(parent, bg=PANEL_BG)
        self.word_lbl = lbl(self, text="", size=10, color=TEXT, anchor="e", width=14,
                            font=("Segoe UI Mono", fs(10)))
        self.word_lbl.pack(side="left", padx=(0, s(6)))

        self.canvas = tk.Canvas(self, bg=BAR_BG, height=s(14), width=self.WIDTH,
                                highlightthickness=0)
        self.canvas.pack(side="left")
        self.bar = self.canvas.create_rectangle(0, 0, 0, s(14), fill=ACCENT, outline="")

        self.pct_lbl = lbl(self, text="", size=10, color=TEXT_DIM, width=7, anchor="w")
        self.pct_lbl.pack(side="left", padx=(s(6), 0))

        self.lp_lbl = lbl(self, text="", size=9, color=TEXT_DIM, width=8, anchor="w")
        self.lp_lbl.pack(side="left")

    def update(self, token: str, logprob: float, highlight: bool = False):
        display = repr(token) if (token.strip() == "" and token != " ") else token
        self.word_lbl.config(text=display, fg=ACCENT2 if highlight else TEXT)
        prob = math.exp(logprob)
        px = int(prob * self.WIDTH)
        self.canvas.coords(self.bar, 0, 0, px, s(14))
        color = ACCENT2 if highlight else ACCENT
        self.canvas.itemconfig(self.bar, fill=color)
        self.pct_lbl.config(text=f"{prob*100:.1f}%", fg=ACCENT2 if highlight else TEXT_DIM)
        self.lp_lbl.config(text=f"lp={logprob:.3f}", fg=TEXT_DIM)

    def clear(self):
        self.word_lbl.config(text="")
        self.canvas.coords(self.bar, 0, 0, 0, s(14))
        self.pct_lbl.config(text="")
        self.lp_lbl.config(text="")


# ── Main application ─────────────────────────────────────────────────────
class App(tk.Tk):
    MAX_BARS = 16

    def __init__(self):
        super().__init__()
        self.title("Ollama Logprobs Inspector")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(s(960), s(620))

        self._tokens: list[dict] = []          # accumulated token data
        self._selected_idx: int | None = None
        self._chip_widgets: list[tk.Label] = []
        self._streaming = False
        self._suppress_think = tk.BooleanVar(value=True)

        self._build_ui()
        self._load_models()
        self.after(100, self._center)

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── Layout ───────────────────────────────────────────────────────────
    def _build_ui(self):
        paned = tk.PanedWindow(self, orient="horizontal", bg=BG,
                               sashwidth=s(6), sashrelief="flat", bd=0, handlesize=0)
        paned.pack(fill="both", expand=True, padx=s(14), pady=s(14))

        left  = tk.Frame(paned, bg=PANEL_BG, padx=s(12), pady=s(12))
        right = tk.Frame(paned, bg=PANEL_BG, padx=s(12), pady=s(12))
        paned.add(left,  minsize=s(280), width=s(380))
        paned.add(right, minsize=s(300), width=s(580))

        self._build_left(left)
        self._build_right(right)

    # ── LEFT: prompt controls ─────────────────────────────────────────────
    def _build_left(self, p):
        section_title(p, "PROMPT")

        # Model row
        model_row = hframe(p)
        model_row.pack(fill="x", pady=(0, s(8)))
        lbl(model_row, "Model:", size=10, color=TEXT_DIM).pack(side="left", padx=(0, s(6)))
        self.model_var = tk.StringVar(value="Loading…")
        self.model_menu = ttk.Combobox(model_row, textvariable=self.model_var,
                                       state="readonly", width=20, font=("Segoe UI", fs(10)))
        self.model_menu.pack(side="left")

        self._refresh_btn = self._button(model_row, "↻", TEXT_DIM, self._load_models)
        self._refresh_btn.pack(side="left", padx=(s(6), 0))

        # Top-k row
        topk_row = hframe(p)
        topk_row.pack(fill="x", pady=(0, s(10)))
        lbl(topk_row, "Alternatives (top-k):", size=10, color=TEXT_DIM).pack(side="left", padx=(0, s(6)))
        self.topk_var = tk.IntVar(value=5)
        topk_spin = tk.Spinbox(topk_row, from_=1, to=20, textvariable=self.topk_var,
                               width=4, bg=BG, fg=TEXT, insertbackground=TEXT,
                               font=("Segoe UI", fs(10)), relief="flat",
                               highlightthickness=1, highlightbackground=BORDER,
                               buttonbackground=BORDER)
        topk_spin.pack(side="left")
        lbl(topk_row, "  (model-dependent max)", size=9, color=TEXT_DIM).pack(side="left")

        think_row = hframe(p)
        think_row.pack(fill="x", pady=(s(4), 0))
        tk.Checkbutton(
            think_row, text="Suppress thinking output (<think> blocks)",
            variable=self._suppress_think,
            bg=PANEL_BG, fg=TEXT_DIM, selectcolor=BG,
            activebackground=PANEL_BG, activeforeground=TEXT,
            font=("Segoe UI", fs(9)), relief="flat", cursor="hand2",
        ).pack(side="left")

        divider(p)
        lbl(p, "Prompt:", size=9, color=TEXT_DIM).pack(anchor="w", pady=(0, s(4)))

        prompt_wrap = tk.Frame(p, bg=PANEL_BG)
        prompt_wrap.pack(fill="both", expand=True)

        self.prompt_text = tk.Text(
            prompt_wrap, bg="#1e1e2e", fg=TEXT, insertbackground=TEXT,
            font=("Segoe UI", fs(11)), relief="flat", wrap="word",
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, padx=s(8), pady=s(8)
        )
        self.prompt_text.insert("1.0", "Why is the sky blue?")
        self.prompt_text.pack(fill="both", expand=True)

        divider(p)

        btn_row = hframe(p)
        btn_row.pack(fill="x", pady=(0, s(6)))
        self._run_btn = self._button(btn_row, "⚡ Generate", ACCENT, self._on_generate)
        self._run_btn.pack(side="left", fill="x", expand=True, padx=(0, s(6)))
        self._debug_btn = self._button(btn_row, "⎘ Copy debug", TEXT_DIM, self._copy_debug)
        self._debug_btn.pack(side="left")

        self._status = lbl(p, "", size=9, color=TEXT_DIM, anchor="w", wraplength=s(320))
        self._status.pack(fill="x")

        divider(p)

        # Legend
        lbl(p, "Token confidence:", size=9, color=TEXT_DIM, bold=True).pack(anchor="w", pady=(0, s(4)))
        legend_data = [
            (">95%", "#2d5a27", "#a6e3a1"),
            (">82%", "#3d4a1e", "#d4e88a"),
            (">61%", "#4a3d1a", "#f9e2af"),
            (">37%", "#4a2e1a", "#f5b88a"),
            (">14%", "#4a1e1e", "#f38ba8"),
            (" <14%", "#2e1e2e", "#cba6f7"),
        ]
        for pct, bg, fg in legend_data:
            row = hframe(p)
            row.pack(anchor="w", pady=s(1))
            chip = tk.Label(row, text=f" {pct} ", bg=bg, fg=fg,
                            font=("Segoe UI", fs(9), "bold"), padx=s(4), pady=s(1))
            chip.pack(side="left", padx=(0, s(6)))

    # ── RIGHT: response + distribution ───────────────────────────────────
    def _build_right(self, p):
        section_title(p, "GENERATED RESPONSE  — click any token to inspect")

        # Token flow (scrollable canvas of chips)
        token_wrap = tk.Frame(p, bg=PANEL_BG)
        token_wrap.pack(fill="both", expand=True)

        self.token_canvas = tk.Canvas(token_wrap, bg=PANEL_BG, highlightthickness=0)
        tok_vbar = ttk.Scrollbar(token_wrap, orient="vertical",
                                 command=self.token_canvas.yview)
        self.token_canvas.configure(yscrollcommand=tok_vbar.set)
        tok_vbar.pack(side="right", fill="y")
        self.token_canvas.pack(side="left", fill="both", expand=True)

        self.token_frame = tk.Frame(self.token_canvas, bg=PANEL_BG)
        self.token_canvas.create_window((0, 0), window=self.token_frame, anchor="nw")
        self.token_frame.bind("<Configure>", lambda e: self.token_canvas.configure(
            scrollregion=self.token_canvas.bbox("all")))

        self.token_canvas.bind("<Enter>",
            lambda e: self.token_canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.token_canvas.bind("<Leave>",
            lambda e: self.token_canvas.unbind_all("<MouseWheel>"))

        divider(p)

        # Distribution panel
        dist_hdr = hframe(p)
        dist_hdr.pack(fill="x", pady=(0, s(6)))
        self._dist_title = lbl(dist_hdr, "Click a token to see its alternatives",
                               size=9, color=TEXT_DIM)
        self._dist_title.pack(side="left")

        dist_outer = tk.Frame(p, bg=PANEL_BG)
        dist_outer.pack(fill="x")

        dist_canvas = tk.Canvas(dist_outer, bg=PANEL_BG, highlightthickness=0)
        dist_vbar   = ttk.Scrollbar(dist_outer, orient="vertical",
                                    command=dist_canvas.yview)
        dist_canvas.configure(yscrollcommand=dist_vbar.set)
        dist_vbar.pack(side="right", fill="y")
        dist_canvas.pack(side="left", fill="both", expand=True)

        self.dist_frame = tk.Frame(dist_canvas, bg=PANEL_BG)
        dist_canvas.create_window((0, 0), window=self.dist_frame, anchor="nw")
        self.dist_frame.bind("<Configure>", lambda e: dist_canvas.configure(
            scrollregion=dist_canvas.bbox("all")))

        self._bars: list[BarRow] = []
        for _ in range(self.MAX_BARS):
            br = BarRow(self.dist_frame)
            br.pack(anchor="w", pady=s(2))
            self._bars.append(br)

    def _on_mousewheel(self, event):
        self.token_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── Button factory ────────────────────────────────────────────────────
    def _button(self, parent, text, color, cmd):
        btn = tk.Label(parent, text=text, bg=PANEL_BG, fg=color,
                       font=("Segoe UI", fs(10), "bold"),
                       pady=s(6), padx=s(10), cursor="hand2", relief="flat",
                       highlightthickness=1, highlightbackground=color)
        btn.bind("<Button-1>", lambda e: cmd())
        btn.bind("<Enter>",    lambda e: btn.config(bg=color, fg=BG))
        btn.bind("<Leave>",    lambda e: btn.config(bg=PANEL_BG, fg=color))
        return btn

    # ── Debug copy ───────────────────────────────────────────────────────
    def _copy_debug(self):
        if not self._tokens:
            self._status.config(text="Nothing to copy — generate first.", fg=YELLOW)
            return
        data = {
            "token_count": len(self._tokens),
            "plain_text":  "".join(t["token"] for t in self._tokens),
            "tokens": [
                {
                    "i":      i,
                    "token":  t["token"],
                    "repr":   repr(t["token"]),
                    "logprob": t["logprob"],
                    "top":    [{"token": x["token"], "logprob": x["logprob"]}
                               for x in t.get("top", [])],
                }
                for i, t in enumerate(self._tokens)
            ],
        }
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        try:
            import subprocess
            proc = subprocess.run(
                ["wl-copy"], input=payload.encode(), check=True
            )
            self._status.config(
                text=f"Copied {len(self._tokens)} tokens to clipboard (wl-copy).", fg=GREEN)
        except FileNotFoundError:
            # wl-copy not found — fall back to Tk clipboard
            self.clipboard_clear()
            self.clipboard_append(payload)
            self._status.config(
                text=f"Copied {len(self._tokens)} tokens to clipboard (Tk).", fg=GREEN)
        except Exception as e:
            self._status.config(text=f"Copy failed: {e}", fg=RED)

    # ── Model loading ─────────────────────────────────────────────────────
    def _load_models(self):
        self._status.config(text="Fetching models from Ollama…", fg=TEXT_DIM)
        def _fetch():
            models = fetch_models()
            self.after(0, lambda: self._on_models_loaded(models))
        threading.Thread(target=_fetch, daemon=True).start()

    def _on_models_loaded(self, models):
        if models:
            self.model_menu["values"] = models
            self.model_var.set(models[0])
            self._status.config(text=f"{len(models)} model(s) available.", fg=GREEN)
        else:
            self.model_menu["values"] = []
            self.model_var.set("(none found)")
            self._status.config(
                text="Could not reach Ollama. Is it running? (ollama serve)",
                fg=RED)

    # ── Generation ────────────────────────────────────────────────────────
    def _on_generate(self):
        if self._streaming:
            return
        model   = self.model_var.get()
        prompt  = self.prompt_text.get("1.0", "end").strip()
        top_k   = self.topk_var.get()

        if not model or model in ("(none found)", "Loading…"):
            messagebox.showerror("No model", "Select a model first.")
            return
        if not prompt:
            messagebox.showerror("Empty prompt", "Enter a prompt.")
            return

        self._tokens.clear()
        self._selected_idx = None
        self._clear_chips()
        self._clear_bars()
        self._dist_title.config(text="Click a token to see its alternatives")
        self._status.config(text="Generating…", fg=YELLOW)
        self._streaming = True

        suppress_think = self._suppress_think.get()
        def _run():
            stream_completion(
                model, prompt, top_k, suppress_think,
                callback  = lambda td: self.after(0, lambda td=td: self._on_token(td)),
                done_cb   = lambda:    self.after(0, self._on_done),
                error_cb  = lambda msg: self.after(0, lambda msg=msg: self._on_error(msg)),
            )
        threading.Thread(target=_run, daemon=True).start()

    def _on_token(self, td: dict):
        idx = len(self._tokens)
        self._tokens.append(td)
        self._add_chip(idx, td)

    def _on_done(self):
        self._streaming = False
        n = len(self._tokens)
        self._status.config(
            text=f"Done. {n} tokens generated. Click any token to inspect.",
            fg=GREEN)

    def _on_error(self, msg: str):
        self._streaming = False
        self._status.config(text=f"Error: {msg[:120]}", fg=RED)

    # ── Token chips ───────────────────────────────────────────────────────
    def _clear_chips(self):
        for w in self.token_frame.winfo_children():
            w.destroy()
        self._chip_widgets.clear()
        # Reset the wrapping row
        self._current_row = tk.Frame(self.token_frame, bg=PANEL_BG)
        self._current_row.pack(anchor="w", fill="x", pady=s(1))
        self._current_row_width = 0

    def _new_row(self):
        self._current_row = tk.Frame(self.token_frame, bg=PANEL_BG)
        self._current_row.pack(anchor="w", fill="x", pady=s(1))
        self._current_row_width = 0

    def _add_chip(self, idx: int, td: dict):
        token = td["token"]
        lp    = td["logprob"]
        bg, fg = logprob_colours(lp)

        if "\n" in token:
            parts = token.split("\n")
            for i, part in enumerate(parts):
                if i > 0:
                    marker = tk.Label(self._current_row, text="\u21b5",
                                      bg=PANEL_BG, fg=TEXT_DIM,
                                      font=("Segoe UI Mono", fs(10)), padx=s(2), pady=s(2))
                    marker.pack(side="left", padx=0, pady=s(1))
                    self._new_row()
                if part:
                    self._add_chip(idx, dict(td, token=part))
            return

        # Estimate width from font before packing — winfo_reqwidth() returns 1
        # for unattached widgets so we can't measure first then decide.
        # Use a rough character-width estimate: monospace ~8px per char + padding.
        canvas_w = self.token_canvas.winfo_width() or s(500)
        approx_w = len(token) * s(8)
        if self._current_row_width + approx_w > canvas_w - s(10) and self._current_row_width > 0:
            self._new_row()

        chip = tk.Label(
            self._current_row,
            text=token,
            bg=bg, fg=fg,
            font=("Segoe UI Mono", fs(10)),
            padx=0, pady=s(2),
            relief="flat",
            cursor="hand2",
        )
        chip.pack(side="left", padx=0, pady=s(1))
        chip.update_idletasks()
        self._current_row_width += chip.winfo_reqwidth()
        chip.bind("<Button-1>", lambda e, i=idx: self._on_chip_click(i))
        chip.bind("<Enter>", lambda e, c=chip: c.config(
            highlightthickness=1, highlightbackground=TEXT))
        chip.bind("<Leave>", lambda e, c=chip: c.config(highlightthickness=0))
        self._chip_widgets.append(chip)

        self.token_canvas.update_idletasks()
        self.token_canvas.yview_moveto(1.0)

    # ── Chip click → distribution ─────────────────────────────────────────
    def _on_chip_click(self, idx: int):
        # Deselect previous
        if self._selected_idx is not None and self._selected_idx < len(self._chip_widgets):
            prev = self._chip_widgets[self._selected_idx]
            td   = self._tokens[self._selected_idx]
            bg, fg = logprob_colours(td["logprob"])
            prev.config(relief="flat", highlightthickness=0, bg=bg, fg=fg)

        self._selected_idx = idx
        chip = self._chip_widgets[idx]
        chip.config(relief="solid", highlightthickness=max(1, s(2)),
                    highlightbackground=ACCENT, bg=BORDER)

        td = self._tokens[idx]
        self._show_distribution(td)

        token_display = repr(td["token"]) if not td["token"].strip() else f'"{td["token"]}"'
        self._dist_title.config(
            text=f'Token {idx+1}: {token_display}  —  chosen at {prob_pct(td["logprob"])} probability')

    def _show_distribution(self, td: dict):
        chosen_token = td["token"]
        top = td.get("top", [])

        # Merge chosen token in if not already present (can happen with some models)
        tokens_in_top = {t["token"] for t in top}
        entries = list(top)
        if chosen_token not in tokens_in_top:
            entries.insert(0, {"token": chosen_token, "logprob": td["logprob"]})

        # Sort by logprob descending
        entries.sort(key=lambda x: -x["logprob"])

        self._clear_bars()
        for i, bar in enumerate(self._bars):
            if i < len(entries):
                e = entries[i]
                bar.update(e["token"], e["logprob"], highlight=(e["token"] == chosen_token))
                bar.pack(anchor="w", pady=s(2))
            else:
                bar.pack_forget()

    def _clear_bars(self):
        for bar in self._bars:
            bar.clear()
            bar.pack_forget()


# ── ttk style ─────────────────────────────────────────────────────────────
def apply_style():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TScrollbar", background=BORDER, troughcolor=PANEL_BG,
                bordercolor=PANEL_BG, arrowcolor=TEXT_DIM)
    style.configure("TCombobox", fieldbackground=BG, background=BG,
                foreground=TEXT, bordercolor=BORDER, arrowcolor=TEXT_DIM,
                selectbackground=ACCENT, selectforeground=BG)
    style.map("TCombobox",
          fieldbackground=[("readonly", BG)],
          foreground=[("readonly", TEXT)],
          background=[("readonly", PANEL_BG)])


if __name__ == "__main__":
    app = App()
    apply_style()
    app.mainloop()
