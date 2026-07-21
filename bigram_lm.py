"""
Bigram Language Model — interactive educational demo
Run with:  python3 bigram_lm.py
Requires:  Python 3.8+, tkinter (standard library)
"""

import tkinter as tk
from tkinter import ttk
import re
import random

# ── Palette ───────────────────────────────────────────────────────────────
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

# ── Scale factor ──────────────────────────────────────────────────────
# 1.0 = default, 1.5 = presentation recording, 2.0 = large display
SCALE = 2.0

def s(n):
    """Scale a pixel value by SCALE."""
    return int(n * SCALE)

def fs(n):
    """Scale a font size by SCALE."""
    return max(1, int(n * SCALE))

DEFAULT_CORPUS = (
    "the cat sat on the mat the cat ate the rat "
    "the rat ran from the cat a cat is a small animal "
    "the dog sat on the mat the dog chased the cat "
    "the dog is a large animal"
)


# ── Model ─────────────────────────────────────────────────────────────────
class BigramModel:
    def __init__(self):
        self.counts = {}
        self.probs  = {}
        self.vocab  = []
        self.tokens = []

    def train(self, text):
        self.tokens = re.sub(r"[^a-z\s]", " ", text.lower()).split()
        self.counts = {}
        for a, b in zip(self.tokens, self.tokens[1:]):
            self.counts.setdefault(a, {})
            self.counts[a][b] = self.counts[a].get(b, 0) + 1
        self.probs = {}
        for a, nexts in self.counts.items():
            total = sum(nexts.values())
            self.probs[a] = {b: c / total for b, c in nexts.items()}
        self.vocab = sorted(self.probs.keys())

    def distribution(self, word):
        return sorted(self.probs.get(word, {}).items(), key=lambda x: -x[1])

    def sample(self, word):
        d = self.probs.get(word)
        if not d:
            return None
        r, cum = random.random(), 0.0
        for w, p in d.items():
            cum += p
            if r < cum:
                return w
        return list(d.keys())[-1]


# ── Widget helpers ────────────────────────────────────────────────────────
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


# ── Probability bar row ───────────────────────────────────────────────────
class BarRow(tk.Frame):
    WIDTH = s(200)

    def __init__(self, parent):
        super().__init__(parent, bg=PANEL_BG)
        self.word_lbl = lbl(self, text="", size=10, color=TEXT, anchor="e", width=10)
        self.word_lbl.pack(side="left", padx=(0, s(6)))

        self.canvas = tk.Canvas(self, bg=BAR_BG, height=s(14), width=self.WIDTH,
                                highlightthickness=0)
        self.canvas.pack(side="left")
        self.bar = self.canvas.create_rectangle(0, 0, 0, s(14), fill=ACCENT, outline="")

        self.pct_lbl = lbl(self, text="", size=10, color=TEXT_DIM, width=5, anchor="w")
        self.pct_lbl.pack(side="left", padx=(s(6), 0))

    def update(self, word, prob, sampled=False):
        self.word_lbl.config(text=word, fg=ACCENT2 if sampled else TEXT)
        self.canvas.coords(self.bar, 0, 0, int(prob * self.WIDTH), s(14))
        self.canvas.itemconfig(self.bar, fill=ACCENT2 if sampled else ACCENT)
        self.pct_lbl.config(text=f"{prob*100:.0f}%", fg=ACCENT2 if sampled else TEXT_DIM)

    def clear(self):
        self.word_lbl.config(text="")
        self.canvas.coords(self.bar, 0, 0, 0, s(14))
        self.pct_lbl.config(text="")


# ── Main application ──────────────────────────────────────────────────────
class App(tk.Tk):
    MAX_BARS  = 12
    GEN_DELAY = 600

    def __init__(self):
        super().__init__()
        self.title("Bigram Language Model")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(s(980), s(640))

        self.model = BigramModel()
        self._gen_tokens = []
        self._auto_job = None

        self._build_ui()
        self._train(DEFAULT_CORPUS)
        self.after(100, self._center)

    def _center(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── Top-level layout ──────────────────────────────────────────────────
    def _build_ui(self):
        paned = tk.PanedWindow(self, orient="horizontal", bg=BG,
                               sashwidth=s(6), sashrelief="flat", bd=0, handlesize=0)
        paned.pack(fill="both", expand=True, padx=s(14), pady=s(14))

        left   = tk.Frame(paned, bg=PANEL_BG, padx=s(12), pady=s(12))
        middle = tk.Frame(paned, bg=PANEL_BG, padx=s(12), pady=s(12))
        right  = tk.Frame(paned, bg=PANEL_BG, padx=s(12), pady=s(12))

        paned.add(left,   minsize=s(220), width=s(260))
        paned.add(middle, minsize=s(280), width=s(360))
        paned.add(right,  minsize=s(280), width=s(360))

        self._build_corpus_panel(left)
        self._build_matrix_panel(middle)
        self._build_generator_panel(right)

    # ── LEFT: corpus ──────────────────────────────────────────────────────
    def _build_corpus_panel(self, p):
        section_title(p, "TRAINING CORPUS")

        text_wrap = tk.Frame(p, bg=PANEL_BG)
        text_wrap.pack(fill="both", expand=True)

        self.corpus_text = tk.Text(
            text_wrap, bg="#1e1e2e", fg=TEXT, insertbackground=TEXT,
            font=("Segoe UI", fs(10)), relief="flat", wrap="word",
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, padx=s(6), pady=s(6)
        )
        self.corpus_text.insert("1.0", DEFAULT_CORPUS)
        self.corpus_text.pack(fill="both", expand=True)

        divider(p)

        stats = hframe(p)
        stats.pack(fill="x", pady=(0, 8))
        self.stat_tokens = lbl(stats, text="– tokens", size=9, color=TEXT_DIM)
        self.stat_tokens.pack(side="left")
        self.stat_vocab = lbl(stats, text="  – words", size=9, color=TEXT_DIM)
        self.stat_vocab.pack(side="left")

        self._train_btn = self._button(p, "⚡ Train model", ACCENT, self._on_train)
        self._train_btn.pack(fill="x")

    # ── MIDDLE: matrix ────────────────────────────────────────────────────
    def _build_matrix_panel(self, p):
        section_title(p, "MODEL WEIGHTS  (probability matrix)")

        outer = tk.Frame(p, bg=PANEL_BG)
        outer.pack(fill="both", expand=True)

        self.matrix_canvas = tk.Canvas(outer, bg=PANEL_BG, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical",   command=self.matrix_canvas.yview)
        hbar = ttk.Scrollbar(outer, orient="horizontal", command=self.matrix_canvas.xview)
        self.matrix_canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

        vbar.pack(side="right",  fill="y")
        hbar.pack(side="bottom", fill="x")
        self.matrix_canvas.pack(side="left", fill="both", expand=True)

        self.matrix_frame = tk.Frame(self.matrix_canvas, bg=PANEL_BG)
        self.matrix_canvas.create_window((0, 0), window=self.matrix_frame, anchor="nw")
        self.matrix_frame.bind("<Configure>", lambda e: self.matrix_canvas.configure(
            scrollregion=self.matrix_canvas.bbox("all")))

        self.matrix_canvas.bind("<Enter>",
            lambda e: self.matrix_canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.matrix_canvas.bind("<Leave>",
            lambda e: self.matrix_canvas.unbind_all("<MouseWheel>"))

        divider(p)
        lbl(p, "Click a row to inspect that word's distribution →",
            size=9, color=TEXT_DIM).pack(anchor="w")

    def _on_mousewheel(self, event):
        self.matrix_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _draw_matrix(self):
        for w in self.matrix_frame.winfo_children():
            w.destroy()

        vocab     = self.model.vocab
        all_nexts = sorted({w for row in self.model.probs.values() for w in row})
        if not vocab:
            return

        # Column width driven by the longest word, with a minimum to fit "0.00"
        col_w = max(len(w) for w in all_nexts) + 1
        col_w = max(col_w, 5)
        row_w = max(len(w) for w in vocab) + 1
        row_w = max(row_w, 5)

        # The matrix itself uses grid — that's fine because matrix_frame
        # is a dedicated container with no other geometry manager in use.
        tk.Label(self.matrix_frame, text="", bg=PANEL_BG, width=row_w,
                 font=("Segoe UI", fs(8))).grid(row=0, column=0, padx=1, pady=1)
        for j, nxt in enumerate(all_nexts):
            tk.Label(self.matrix_frame, text=nxt, bg=BORDER, fg=TEXT_DIM,
                     font=("Segoe UI", fs(8)), width=col_w, anchor="center",
                     padx=s(2), pady=s(2)).grid(row=0, column=j+1, padx=1, pady=1, sticky="nsew")

        for i, given in enumerate(vocab):
            row_lbl = tk.Label(self.matrix_frame, text=given, bg=BORDER, fg=ACCENT,
                               font=("Segoe UI", fs(8), "bold"), width=row_w,
                               anchor="e", padx=s(4), pady=s(2), cursor="hand2")
            row_lbl.grid(row=i+1, column=0, padx=1, pady=1, sticky="nsew")
            row_lbl.bind("<Button-1>", lambda e, w=given: self._inspect_word(w))

            for j, nxt in enumerate(all_nexts):
                prob = self.model.probs[given].get(nxt, 0.0)
                if prob > 0:
                    rv   = int(40 + prob * 89)
                    gv   = int(40 + prob * 130)
                    bv   = int(80 + prob * 170)
                    bg_c = f"#{rv:02x}{gv:02x}{bv:02x}"
                    fg_c = TEXT if prob > 0.3 else TEXT_DIM
                    txt  = f"{prob:.2f}"
                else:
                    bg_c, fg_c, txt = PANEL_BG, BORDER, "·"

                cell = tk.Label(self.matrix_frame, text=txt, bg=bg_c, fg=fg_c,
                                font=("Segoe UI", fs(8)), width=col_w, anchor="center",
                                padx=s(1), pady=s(2), cursor="hand2")
                cell.grid(row=i+1, column=j+1, padx=1, pady=1, sticky="nsew")
                cell.bind("<Button-1>", lambda e, w=given: self._inspect_word(w))

    # ── RIGHT: generator ──────────────────────────────────────────────────
    def _build_generator_panel(self, p):
        section_title(p, "GENERATOR")

        seed_row = hframe(p)
        seed_row.pack(fill="x", pady=(0, 8))
        lbl(seed_row, "Seed word:", size=10, color=TEXT_DIM).pack(side="left", padx=(0, s(6)))
        self.seed_var = tk.StringVar()
        self.seed_menu = ttk.Combobox(seed_row, textvariable=self.seed_var,
                                      state="readonly", width=12,
                                      font=("Segoe UI", fs(10)))
        self.seed_menu.pack(side="left")
        self.seed_menu.bind("<<ComboboxSelected>>", lambda e: self._reset_gen())

        btn_row = hframe(p)
        btn_row.pack(fill="x", pady=(0, 10))
        self._step_btn  = self._button(btn_row, "Step →",  ACCENT,  self._on_step)
        self._auto_btn  = self._button(btn_row, "▶ Auto",  GREEN,   self._on_auto)
        self._reset_btn = self._button(btn_row, "↺ Reset", YELLOW,  self._on_reset)
        for b in (self._step_btn, self._auto_btn, self._reset_btn):
            b.pack(side="left", padx=(0, s(6)))

        divider(p)

        lbl(p, "Sequence:", size=9, color=TEXT_DIM).pack(anchor="w")
        seq_wrap = hframe(p)
        seq_wrap.pack(fill="x", pady=(4, 0))

        self.seq_canvas = tk.Canvas(seq_wrap, bg=PANEL_BG, height=s(36), highlightthickness=0)
        seq_hbar = ttk.Scrollbar(seq_wrap, orient="horizontal",
                                 command=self.seq_canvas.xview)
        self.seq_canvas.configure(xscrollcommand=seq_hbar.set)
        seq_hbar.pack(side="bottom", fill="x")
        self.seq_canvas.pack(side="top", fill="x")

        self.seq_inner = tk.Frame(self.seq_canvas, bg=PANEL_BG)
        self.seq_canvas.create_window((0, 0), window=self.seq_inner, anchor="nw")
        self.seq_inner.bind("<Configure>", lambda e: self.seq_canvas.configure(
            scrollregion=self.seq_canvas.bbox("all")))

        divider(p)

        lbl(p, "Distribution at current step:", size=9, color=TEXT_DIM).pack(
            anchor="w", pady=(0, s(6)))

        dist_outer = tk.Frame(p, bg=PANEL_BG)
        dist_outer.pack(fill="both", expand=True)

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

        self._bars = []
        for _ in range(self.MAX_BARS):
            br = BarRow(self.dist_frame)
            br.pack(anchor="w", pady=s(2))
            self._bars.append(br)

        self._step_info = lbl(p, "", size=9, color=TEXT_DIM, anchor="w", wraplength=s(320))
        self._step_info.pack(fill="x", pady=(6, 0))

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

    # ── Training ──────────────────────────────────────────────────────────
    def _on_train(self):
        self._train(self.corpus_text.get("1.0", "end").strip())

    def _train(self, text):
        self._stop_auto()
        self.model.train(text)
        self.stat_tokens.config(text=f"{len(self.model.tokens)} tokens")
        self.stat_vocab.config(text=f"  {len(self.model.vocab)} unique")
        self._draw_matrix()
        self.seed_menu["values"] = self.model.vocab
        if self.model.vocab:
            self.seed_var.set(self.model.vocab[0])
        self._reset_gen()

    # ── Matrix inspection ─────────────────────────────────────────────────
    def _inspect_word(self, word):
        self._stop_auto()
        self.seed_var.set(word)
        self._reset_gen()
        self._show_distribution(word, sampled=None)

    # ── Generation ────────────────────────────────────────────────────────
    def _reset_gen(self):
        self._stop_auto()
        seed = self.seed_var.get()
        self._gen_tokens = [seed] if seed else []
        self._render_sequence()
        if seed:
            self._show_distribution(seed, sampled=None)
        self._step_info.config(text="", fg=TEXT_DIM)
        self._set_step_active(True)

    def _on_step(self):
        self._stop_auto()
        self._do_step()

    def _on_auto(self):
        if self._auto_job:
            self._stop_auto()
        else:
            self._auto_btn.config(text="⏸ Pause")
            self._schedule_auto()

    def _schedule_auto(self):
        self._auto_job = self.after(self.GEN_DELAY, self._auto_tick)

    def _auto_tick(self):
        self._auto_job = None
        if self._do_step():
            self._schedule_auto()
        else:
            self._stop_auto()

    def _stop_auto(self):
        if self._auto_job:
            self.after_cancel(self._auto_job)
            self._auto_job = None
        self._auto_btn.config(text="▶ Auto")

    def _on_reset(self):
        self._reset_gen()

    def _do_step(self):
        if not self._gen_tokens:
            return False
        current = self._gen_tokens[-1]
        dist    = self.model.distribution(current)
        if not dist:
            self._step_info.config(
                text=f'"{current}" has no successors in training data — end of chain.',
                fg=RED)
            self._set_step_active(False)
            self._stop_auto()
            return False

        chosen = self.model.sample(current)
        self._gen_tokens.append(chosen)
        self._show_distribution(chosen, sampled=None)
        self._render_sequence()

        top_w, top_p = dist[0]
        self._step_info.config(
            text=(f'"{current}" → sampled "{chosen}"  '
                  f'({len(dist)} options, most likely: "{top_w}" {top_p*100:.0f}%)'),
            fg=TEXT_DIM)
        return True

    # ── Render helpers ────────────────────────────────────────────────────
    def _render_sequence(self):
        for w in self.seq_inner.winfo_children():
            w.destroy()
        for i, tok in enumerate(self._gen_tokens):
            is_last = i == len(self._gen_tokens) - 1
            bg   = ACCENT2 if is_last else (TEAL if i == 0 else BORDER)
            fg   = BG      if is_last else (BG   if i == 0 else TEXT)
            font = ("Segoe UI", fs(10), "bold") if is_last else ("Segoe UI", fs(10))
            tk.Label(self.seq_inner, text=tok, bg=bg, fg=fg, font=font,
                     padx=s(7), pady=s(3)).pack(side="left", padx=s(2), pady=s(4))
        self.seq_canvas.update_idletasks()
        self.seq_canvas.xview_moveto(1.0)

    def _show_distribution(self, word, sampled):
        dist = self.model.distribution(word)
        for i, bar in enumerate(self._bars):
            if i < len(dist):
                w, p = dist[i]
                bar.update(w, p, sampled=(w == sampled))
                bar.pack(anchor="w", pady=s(2))
            else:
                bar.clear()
                bar.pack_forget()

    def _set_step_active(self, active):
        color = ACCENT if active else TEXT_DIM
        self._step_btn.config(fg=color, highlightbackground=color,
                              cursor="hand2" if active else "arrow")
        if active:
            self._step_btn.bind("<Button-1>", lambda e: self._on_step())
        else:
            self._step_btn.unbind("<Button-1>")


# ── ttk style ─────────────────────────────────────────────────────────────
def apply_style():
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("TScrollbar", background=BORDER, troughcolor=PANEL_BG,
                bordercolor=PANEL_BG, arrowcolor=TEXT_DIM)
    s.configure("TCombobox", fieldbackground=BG, background=BG,
                foreground=TEXT, bordercolor=BORDER, arrowcolor=TEXT_DIM,
                selectbackground=ACCENT, selectforeground=BG)
    s.map("TCombobox",
          fieldbackground=[("readonly", BG)],
          foreground=[("readonly", TEXT)],
          background=[("readonly", PANEL_BG)])


if __name__ == "__main__":
    app = App()
    apply_style()
    app.mainloop()
