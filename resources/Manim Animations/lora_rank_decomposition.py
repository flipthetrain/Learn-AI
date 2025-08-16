"""
LoRA: Low-Rank Adaptation — Manim Animation
============================================
Demonstrates how LoRA freezes a large weight matrix W and instead trains
two small matrices A and B whose product approximates the weight update.

Render:
    manim -ql lora_rank_decomposition.py LoRARankDecomposition
"""

from pathlib import Path

from manim import *
from voiceover import generate_voiceovers


NARRATION = {
    "intro": (
        "LoRA, or Low-Rank Adaptation, lets you fine-tune a large language model "
        "by training only a tiny fraction of its parameters — without ever modifying "
        "the original weights."
    ),
    "big_w": (
        "A typical transformer weight matrix W has hundreds of thousands of parameters. "
        "Training all of them is expensive and risks overwriting the knowledge already "
        "baked into the pretrained model."
    ),
    "freeze": (
        "LoRA's first move is to freeze W entirely. It will never receive a gradient update. "
        "Instead, we add a small bypass alongside it."
    ),
    "ab_intro": (
        "The bypass consists of two small matrices: A with shape d by r, "
        "and B with shape r by d, where r is the rank — often just 2 or 4. "
        "These are the only matrices we train."
    ),
    "param_count": (
        "With rank 2, A and B together have just 2,048 trainable parameters — "
        "compared to 262,144 in the original matrix. "
        "That is a 98 percent reduction."
    ),
    "update_rule": (
        "During inference, the effective weight is W plus the product of A and B. "
        "W stays frozen; the learned update lives entirely in the low-rank bypass."
    ),
    "takeaway": (
        "Because W is never modified, you can share one large base model across "
        "many tasks, swapping only the small A and B adapters. "
        "This is the core idea behind parameter-efficient fine-tuning."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def matrix_rect(rows, cols, cell=0.28, color=WHITE, fill=None, fill_opacity=0.15):
    """Return a VGroup of rectangles representing a matrix of shape rows×cols."""
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            rect = Rectangle(
                width=cell, height=cell,
                color=color,
                fill_color=fill or color,
                fill_opacity=fill_opacity,
                stroke_width=0.8,
            )
            rect.move_to(RIGHT * c * cell + DOWN * r * cell)
            cells.add(rect)
    cells.move_to(ORIGIN)
    return cells


def dim_label(text, color=GRAY_B, size=24):
    return Text(text, font_size=size, color=color)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class LoRARankDecomposition(Scene):
    def construct(self):
        vo = generate_voiceovers(NARRATION, Path(__file__).stem)

        # ── Title ──────────────────────────────────────────────────────────
        title = Text("LoRA: Low-Rank Adaptation", font_size=44, color=WHITE)
        subtitle = Text(
            "Train a fraction of the parameters — without touching the original weights",
            font_size=22, color=GRAY_B,
        )
        subtitle.next_to(title, DOWN, buff=0.25)

        if vo.exists("intro"): self.add_sound(vo.path("intro"))
        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP * 0.2))
        self.wait(vo.wait("intro", used=2.0))
        self.play(FadeOut(subtitle), title.animate.to_edge(UP).scale(0.75))
        self.wait(0.3)

        # ── The problem: W is huge ─────────────────────────────────────────
        d, k = 12, 12          # visual grid (represents d_model × d_model)
        W = matrix_rect(d, k, cell=0.28, color=BLUE_C, fill=BLUE_E)
        W.move_to(ORIGIN)

        w_label = Text("W", font_size=52, color=BLUE_C)
        w_label.next_to(W, UP, buff=0.35)
        w_dim   = dim_label("512 × 512 = 262,144 params")
        w_dim.next_to(W, DOWN, buff=0.3)

        if vo.exists("big_w"): self.add_sound(vo.path("big_w"))
        self.play(FadeIn(W, shift=UP * 0.3), Write(w_label))
        self.play(FadeIn(w_dim))
        self.wait(vo.wait("big_w", used=1.5))

        # Freeze W — grey it out and show lock
        if vo.exists("freeze"): self.add_sound(vo.path("freeze"))
        self.play(
            W.animate.set_fill(GRAY_D, opacity=0.3).set_stroke(GRAY, opacity=0.4),
            w_label.animate.set_color(GRAY),
        )
        lock = Text("🔒 frozen", font_size=22, color=GRAY)
        lock.next_to(W, RIGHT, buff=0.4)
        self.play(FadeIn(lock))
        self.wait(vo.wait("freeze", used=1.5))

        # Shift W to the left to make room
        self.play(
            VGroup(W, w_label, w_dim, lock).animate.shift(LEFT * 3.2),
        )

        # ── Introduce A and B ─────────────────────────────────────────────
        r_val = 2              # rank — just 2 columns/rows shown visually
        A = matrix_rect(d, r_val, cell=0.28, color=GREEN_C, fill=GREEN_E)
        B = matrix_rect(r_val, k, cell=0.28, color=YELLOW_C, fill=YELLOW_E)

        # Position A and B to the right of W
        A.move_to(RIGHT * 1.2)
        B.move_to(RIGHT * 1.2)

        A.shift(RIGHT * 0.0)
        B.next_to(A, RIGHT, buff=0.5)

        a_label = Text("A", font_size=52, color=GREEN_C)
        b_label = Text("B", font_size=52, color=YELLOW_C)
        a_label.next_to(A, UP, buff=0.35)
        b_label.next_to(B, UP, buff=0.35)

        a_dim = dim_label("512 × 2")
        b_dim = dim_label("2 × 512")
        a_dim.next_to(A, DOWN, buff=0.3)
        b_dim.next_to(B, DOWN, buff=0.3)

        times = Text("×", font_size=44, color=WHITE)
        times.move_to((A.get_right() + B.get_left()) / 2)

        if vo.exists("ab_intro"): self.add_sound(vo.path("ab_intro"))
        self.play(FadeIn(A, shift=LEFT * 0.3), Write(a_label), FadeIn(a_dim))
        self.wait(0.3)
        self.play(Write(times))
        self.play(FadeIn(B, shift=LEFT * 0.3), Write(b_label), FadeIn(b_dim))
        self.wait(vo.wait("ab_intro", used=2.0))

        # Rank annotation
        r_brace_A = Brace(A, direction=RIGHT, color=GRAY_B)
        r_brace_B = Brace(B, direction=UP, color=GRAY_B)
        r_text = Text("rank r = 2", font_size=22, color=GRAY_B)
        r_text.next_to(r_brace_A, RIGHT, buff=0.2)

        self.play(GrowFromCenter(r_brace_A), GrowFromCenter(r_brace_B), FadeIn(r_text))
        self.wait(1)

        # ── Parameter count comparison ────────────────────────────────────
        orig_count = Text("Original:  512 × 512  = 262,144 trainable params",
                          font_size=22, color=BLUE_C)
        lora_count = Text("LoRA:    (512×2) + (2×512) =   2,048 trainable params",
                          font_size=22, color=GREEN_C)
        saving    = Text("→  98.4 % fewer parameters to train",
                         font_size=24, color=YELLOW_C, weight=BOLD)

        counts = VGroup(orig_count, lora_count, saving).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        counts.to_edge(DOWN, buff=0.55)

        if vo.exists("param_count"): self.add_sound(vo.path("param_count"))
        self.play(FadeIn(orig_count, shift=RIGHT * 0.2))
        self.play(FadeIn(lora_count, shift=RIGHT * 0.2))
        self.play(FadeIn(saving, shift=UP * 0.2))
        self.wait(vo.wait("param_count", used=2.0))

        # ── Show the update rule ──────────────────────────────────────────
        self.play(FadeOut(VGroup(r_brace_A, r_brace_B, r_text, counts)))

        # Collapse A and B into a delta-W symbol
        delta_label = Text("ΔW = A × B", font_size=40, color=GREEN_C)
        delta_label.move_to(VGroup(A, times, B).get_center() + UP * 0.1)

        self.play(
            FadeOut(VGroup(A, B, times, a_label, b_label, a_dim, b_dim)),
            Write(delta_label),
        )
        self.wait(0.5)

        # Reconstruct the full update rule as a VGroup (no LaTeX required)
        rule_w      = Text("W'  =  ", font_size=42, color=WHITE)
        rule_frozen = Text("W", font_size=42, color=GRAY)
        rule_plus   = Text("  +  ", font_size=42, color=WHITE)
        rule_delta  = Text("ΔW", font_size=42, color=GREEN_C)
        rule = VGroup(rule_w, rule_frozen, rule_plus, rule_delta).arrange(RIGHT, buff=0)

        frozen_sub    = Text("(frozen)", font_size=18, color=GRAY)
        trainable_sub = Text("(trainable)", font_size=18, color=GREEN_C)
        frozen_sub.next_to(rule_frozen, DOWN, buff=0.12)
        trainable_sub.next_to(rule_delta, DOWN, buff=0.12)

        rule_group = VGroup(rule, frozen_sub, trainable_sub)
        rule_group.move_to(ORIGIN)

        if vo.exists("update_rule"): self.add_sound(vo.path("update_rule"))
        self.play(
            FadeOut(VGroup(W, w_label, w_dim, lock, delta_label)),
            Write(rule),
        )
        self.play(FadeIn(frozen_sub), FadeIn(trainable_sub))
        self.wait(vo.wait("update_rule", used=2.0))

        # ── Final takeaway ────────────────────────────────────────────────
        takeaway = Text(
            "Only A and B are updated during training.\n"
            "W is never modified — it can be shared across many tasks.",
            font_size=26, color=WHITE,
            line_spacing=1.4,
        )
        takeaway.next_to(rule_group, DOWN, buff=0.6)
        if vo.exists("takeaway"): self.add_sound(vo.path("takeaway"))
        self.play(FadeIn(takeaway, shift=UP * 0.2))
        self.wait(vo.wait("takeaway", used=0.5))

        self.play(FadeOut(Group(*self.mobjects)))
