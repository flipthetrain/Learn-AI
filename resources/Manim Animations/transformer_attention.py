"""
Transformer Self-Attention: The Real Story
==========================================
The actual objective is next-token prediction.
Self-attention is the mechanism that lets every token position
read from every other position so its representation is rich
enough to predict well.  "Attention" is just the name we gave
to that context-mixing step.

Story arc:
  1. The task: predict the next token
  2. The problem: token embeddings are context-free — prediction is bad
  3. Self-attention mixes context across the sequence
  4. Enriched representations → much better prediction distribution
  5. The mixing weights (what we call "attention") are learned by
     backprop to minimise next-token loss — not hand-designed

Render:
    manim -ql transformer_attention.py TransformerAttention
"""

from pathlib import Path
import numpy as np
from manim import *
from voiceover import generate_voiceovers


# ── Narration ─────────────────────────────────────────────────────────────────

NARRATION = {
    "goal": (
        "The transformer has one job: predict the next token. "
        "Everything else — the layers, the heads, the weights — "
        "exists to make that prediction as accurate as possible."
    ),
    "cold_embed": (
        "Each token starts as a fixed embedding vector — "
        "a lookup in a table. "
        "At this point the embedding for 'sat' carries no information "
        "about what came before it. "
        "It's the same vector regardless of context."
    ),
    "bad_pred": (
        "If we try to predict the next token from these cold embeddings, "
        "the model has almost nothing to go on. "
        "The distribution is nearly flat — it's essentially guessing."
    ),
    "mixing_idea": (
        "The key insight: to predict well from position i, "
        "we need information from every other position in the sequence. "
        "Self-attention is the mechanism that does this mixing. "
        "Every position gets to read from every other position."
    ),
    "qkv": (
        "Each embedding is projected into three vectors. "
        "These projections determine how much each position "
        "contributes to every other position's new representation. "
        "The mixing weights are called attention, but that's just a name — "
        "they're learned parameters that minimize prediction loss."
    ),
    "mixing_anim": (
        "The output at each position is a weighted combination "
        "of the value vectors from all positions. "
        "After this step, the embedding at position 'sat' "
        "now encodes information about 'The cat' as well."
    ),
    "good_pred": (
        "Now when we project these enriched embeddings to the vocabulary, "
        "the prediction is sharp. "
        "The model knows a verb follows 'The cat', "
        "and 'sat' rises to the top of the distribution."
    ),
    "learned": (
        "The mixing weights are not designed. "
        "They emerge from gradient descent on the prediction loss. "
        "The model learns which positions to mix "
        "because doing so reduces its error on the next token. "
        "We call it attention because it looks like the model "
        "is paying attention — but mechanistically it's just "
        "a learned weighted average."
    ),
}


# ── Constants ─────────────────────────────────────────────────────────────────

TOKENS     = ["The", "cat", "sat"]
NEXT_TOK   = "on"
N          = len(TOKENS)
FOCUS      = N - 1      # "sat" — the position doing the predicting

T_COLORS   = [BLUE_C, GREEN_C, ORANGE]
FOCUS_CLR  = ORANGE

# Illustrative mixing weights for "sat" reading from ["The", "cat", "sat"]
MIX_W = [0.10, 0.68, 0.22]

# Vocabulary for prediction bar chart
VOCAB       = ["on", "and", "there", "quickly", "meowed", "the"]
COLD_PROBS  = [0.14, 0.13, 0.17, 0.12, 0.11, 0.15]   # nearly flat
WARM_PROBS  = [0.52, 0.14, 0.10, 0.09, 0.06, 0.05]    # sharp after mixing
VOCAB_COLOR = [GREEN_B, BLUE_B, TEAL_B, YELLOW_C, RED_B, GRAY_B]


# ── Helpers ───────────────────────────────────────────────────────────────────

def tok_box(label, color=WHITE, width=1.4, height=0.62, fsize=26):
    rect = RoundedRectangle(
        corner_radius=0.12, width=width, height=height,
        color=color, fill_color=color, fill_opacity=0.2, stroke_width=2,
    )
    lbl = Text(label, font_size=fsize, color=color)
    lbl.move_to(rect.get_center())
    return VGroup(rect, lbl)


def embed_bars(color, seed, n=5):
    import random
    rng = random.Random(seed)
    bars = VGroup()
    for i in range(n):
        h = 0.16 + rng.random() * 0.36
        bar = Rectangle(width=0.1, height=h,
                        fill_color=color, fill_opacity=0.85,
                        stroke_width=0.4, color=color)
        bar.move_to(RIGHT * i * 0.145 + UP * h / 2)
        bars.add(bar)
    bars.move_to(ORIGIN)
    return bars


def prob_bars(probs, colors, labels, bar_width=0.38, scale=2.8):
    """Horizontal bar chart of token probabilities."""
    group = VGroup()
    for i, (p, c, lbl) in enumerate(zip(probs, colors, labels)):
        bar = Rectangle(width=p * scale, height=bar_width,
                        fill_color=c, fill_opacity=0.8,
                        stroke_width=0.5, color=c)
        bar.align_to(LEFT * 0, LEFT)
        bar.move_to(DOWN * i * (bar_width + 0.18) + RIGHT * p * scale / 2)
        tok_lbl = Text(lbl, font_size=14, color=c)
        tok_lbl.next_to(bar, LEFT, buff=0.15)
        pct_lbl = Text(f"{p*100:.0f}%", font_size=13, color=c)
        pct_lbl.next_to(bar, RIGHT, buff=0.1)
        group.add(VGroup(bar, tok_lbl, pct_lbl))
    group.move_to(ORIGIN)
    return group


# ── Scene ─────────────────────────────────────────────────────────────────────

class TransformerAttention(Scene):
    def construct(self):
        vo = generate_voiceovers(NARRATION, Path(__file__).stem)

        # ── Phase 1: The task — predict the next token ─────────────────────
        title = Text("Self-Attention: The Real Story", font_size=38)
        sub   = Text("Next-token prediction, and the mechanism that makes it work",
                     font_size=20, color=GRAY_B)
        sub.next_to(title, DOWN, buff=0.25)

        if vo.exists("goal"): self.add_sound(vo.path("goal"))
        self.play(Write(title))
        self.play(FadeIn(sub))
        self.wait(vo.wait("goal", used=1.5))
        self.play(FadeOut(sub), title.animate.to_edge(UP).scale(0.78))

        # Show the sequence + prediction target
        tokens = VGroup(*[tok_box(TOKENS[i], T_COLORS[i]) for i in range(N)])
        pred_box = tok_box("?", GRAY_B, width=1.4, height=0.62)

        seq = VGroup(*tokens, pred_box)
        seq.arrange(RIGHT, buff=0.5)
        seq.move_to(UP * 1.5)

        self.play(FadeIn(tokens, lag_ratio=0.2))
        self.play(FadeIn(pred_box))

        task_lbl = Text("predict the next token", font_size=18, color=GRAY_B)
        task_lbl.next_to(pred_box, DOWN, buff=0.2)
        arrow_to_pred = Arrow(tokens[-1].get_right(), pred_box.get_left(),
                              buff=0.05, color=GRAY_B, stroke_width=2,
                              max_tip_length_to_length_ratio=0.2)
        self.play(GrowArrow(arrow_to_pred), FadeIn(task_lbl))
        self.wait(0.5)

        # ── Phase 2: Cold embeddings — no context ──────────────────────────
        if vo.exists("cold_embed"): self.add_sound(vo.path("cold_embed"))

        embeds = VGroup(*[embed_bars(T_COLORS[i], seed=i * 17) for i in range(N)])
        for i in range(N):
            embeds[i].scale(0.82)
            embeds[i].next_to(tokens[i], DOWN, buff=0.45)

        embed_lbl = Text("cold embeddings — no context yet", font_size=16, color=GRAY_B)
        embed_lbl.next_to(embeds, DOWN, buff=0.28)

        self.play(FadeOut(arrow_to_pred), FadeOut(task_lbl))
        self.play(FadeIn(embeds, lag_ratio=0.2))
        self.play(FadeIn(embed_lbl))
        self.wait(vo.wait("cold_embed", used=2.5))

        # Highlight that "sat"'s embedding is the same regardless of context
        same_note = Text(
            '"sat" has the same vector here as in "she sat" or "sat down"',
            font_size=15, color=ORANGE,
        )
        same_note.next_to(embed_lbl, DOWN, buff=0.2)
        ring = SurroundingRectangle(embeds[FOCUS], color=ORANGE, buff=0.1, stroke_width=2)
        self.play(Create(ring), FadeIn(same_note))
        self.wait(1.5)
        self.play(FadeOut(same_note), FadeOut(ring))

        # ── Phase 3: Bad prediction from cold embeddings ───────────────────
        if vo.exists("bad_pred"): self.add_sound(vo.path("bad_pred"))

        cold_chart = prob_bars(COLD_PROBS, VOCAB_COLOR, VOCAB)
        cold_chart.scale(0.88)
        cold_chart.to_edge(RIGHT, buff=0.5)
        cold_chart.move_to(cold_chart.get_center() + DOWN * 0.3)

        pred_title = Text("prediction from cold embedding:", font_size=16, color=GRAY_B)
        pred_title.next_to(cold_chart, UP, buff=0.3)

        proj_arrow = Arrow(embeds[FOCUS].get_right(),
                           cold_chart.get_left() + LEFT * 0.2,
                           buff=0.15, color=GRAY_D, stroke_width=2,
                           max_tip_length_to_length_ratio=0.15)
        self.play(GrowArrow(proj_arrow), FadeIn(pred_title))
        self.play(FadeIn(cold_chart, lag_ratio=0.15))
        self.wait(vo.wait("bad_pred", used=3.0))

        self.play(FadeOut(VGroup(cold_chart, pred_title, proj_arrow)))

        # ── Phase 4: The mixing idea ───────────────────────────────────────
        if vo.exists("mixing_idea"): self.add_sound(vo.path("mixing_idea"))

        self.play(FadeOut(embed_lbl))

        mix_note = Text(
            "to predict from position i,  read from every other position",
            font_size=18, color=WHITE,
        )
        mix_note.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(mix_note))

        # All-to-all connection lines
        all_lines = VGroup()
        for i in range(N):
            for j in range(N):
                if i != j:
                    ln = Line(
                        tokens[i].get_bottom(), tokens[j].get_bottom(),
                        color=GRAY_C, stroke_width=1.2,
                    )
                    all_lines.add(ln)

        self.play(Create(all_lines, lag_ratio=0.05))
        self.wait(vo.wait("mixing_idea", used=3.0))
        self.play(FadeOut(all_lines), FadeOut(mix_note))

        # ── Phase 5: Q K V — the learned projection mechanism ─────────────
        if vo.exists("qkv"): self.add_sound(vo.path("qkv"))

        # Show projections as small labelled arrows from each embed
        proj_groups = VGroup()
        for i in range(N):
            center = embeds[i].get_center()
            q_arr = Arrow(center, center + LEFT * 0.5 + UP * 0.6,
                          buff=0.1, color=YELLOW_C, stroke_width=1.8,
                          max_tip_length_to_length_ratio=0.25)
            k_arr = Arrow(center, center + UP * 0.75,
                          buff=0.1, color=TEAL_B, stroke_width=1.8,
                          max_tip_length_to_length_ratio=0.25)
            v_arr = Arrow(center, center + RIGHT * 0.5 + UP * 0.6,
                          buff=0.1, color=PURPLE_B, stroke_width=1.8,
                          max_tip_length_to_length_ratio=0.25)
            proj_groups.add(VGroup(q_arr, k_arr, v_arr))

        # Labels only on the middle token to avoid clutter
        q_lbl = Text("Q", font_size=15, color=YELLOW_C)
        k_lbl = Text("K", font_size=15, color=TEAL_B)
        v_lbl = Text("V", font_size=15, color=PURPLE_B)
        q_lbl.next_to(proj_groups[1][0].get_end(), UP, buff=0.05)
        k_lbl.next_to(proj_groups[1][1].get_end(), UP, buff=0.05)
        v_lbl.next_to(proj_groups[1][2].get_end(), UP, buff=0.05)

        qkv_note = Text(
            "Q · K  =  mixing weight    V  =  what gets mixed in",
            font_size=16, color=GRAY_B,
        )
        qkv_note.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(proj_groups, lag_ratio=0.2))
        self.play(FadeIn(VGroup(q_lbl, k_lbl, v_lbl)))
        self.play(FadeIn(qkv_note))
        self.wait(vo.wait("qkv", used=4.0))
        self.play(FadeOut(VGroup(proj_groups, q_lbl, k_lbl, v_lbl, qkv_note)))

        # ── Phase 6: The actual mixing ─────────────────────────────────────
        if vo.exists("mixing_anim"): self.add_sound(vo.path("mixing_anim"))

        # Show weighted arrows flowing into "sat"'s embed
        mix_arrows = VGroup()
        for i in range(N):
            w = max(0.8, MIX_W[i] * 10)
            arr = Arrow(
                embeds[i].get_center(),
                embeds[FOCUS].get_center() + DOWN * 0.5,
                buff=0.08, color=T_COLORS[i], stroke_width=w,
                max_tip_length_to_length_ratio=0.14,
            )
            weight_lbl = Text(f"{MIX_W[i]:.0%}", font_size=13, color=T_COLORS[i])
            mid = arr.get_center() + UP * 0.18
            weight_lbl.move_to(mid)
            mix_arrows.add(VGroup(arr, weight_lbl))

        new_embed = embed_bars(ORANGE, seed=42)
        new_embed.scale(0.82)
        new_embed.next_to(embeds[FOCUS], DOWN, buff=0.5)

        context_lbl = Text(
            "enriched — now encodes context from the whole sequence",
            font_size=15, color=ORANGE,
        )
        context_lbl.next_to(new_embed, DOWN, buff=0.22)

        ring2 = SurroundingRectangle(new_embed, color=ORANGE, buff=0.1, stroke_width=2)

        self.play(Create(mix_arrows, lag_ratio=0.2))
        self.play(FadeIn(new_embed, shift=DOWN * 0.2))
        self.play(Create(ring2), FadeIn(context_lbl))
        self.wait(vo.wait("mixing_anim", used=3.5))

        # ── Phase 7: Good prediction from enriched embedding ──────────────
        if vo.exists("good_pred"): self.add_sound(vo.path("good_pred"))

        self.play(FadeOut(VGroup(mix_arrows, context_lbl, ring2)))

        warm_chart = prob_bars(WARM_PROBS, VOCAB_COLOR, VOCAB)
        warm_chart.scale(0.88)
        warm_chart.to_edge(RIGHT, buff=0.5)
        warm_chart.move_to(warm_chart.get_center() + DOWN * 0.3)

        warm_title = Text("prediction after context mixing:", font_size=16, color=GRAY_B)
        warm_title.next_to(warm_chart, UP, buff=0.3)

        proj_arrow2 = Arrow(new_embed.get_right(),
                            warm_chart.get_left() + LEFT * 0.2,
                            buff=0.15, color=ORANGE, stroke_width=2,
                            max_tip_length_to_length_ratio=0.15)
        self.play(GrowArrow(proj_arrow2), FadeIn(warm_title))
        self.play(FadeIn(warm_chart, lag_ratio=0.15))
        self.wait(vo.wait("good_pred", used=3.5))

        self.play(FadeOut(Group(
            warm_chart, warm_title, proj_arrow2,
            new_embed, mix_arrows, embeds, tokens, pred_box,
        )))

        # ── Phase 8: What "attention" actually is ─────────────────────────
        if vo.exists("learned"): self.add_sound(vo.path("learned"))

        summary_lines = VGroup(
            Text("The objective:", font_size=22, color=WHITE, weight=BOLD),
            Text("minimise next-token prediction loss", font_size=20, color=GREEN_C),
            Text(" ", font_size=10),
            Text("The mechanism:", font_size=22, color=WHITE, weight=BOLD),
            Text("learned weighted average of value vectors", font_size=20, color=TEAL_B),
            Text(" ", font_size=10),
            Text("The name:", font_size=22, color=WHITE, weight=BOLD),
            Text('"attention" — because it looks like attending', font_size=20, color=GRAY_B),
            Text("but it is just a differentiable context mixer", font_size=18, color=GRAY_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        summary_lines.move_to(ORIGIN)

        for line in summary_lines:
            self.play(FadeIn(line, shift=RIGHT * 0.15), run_time=0.45)

        self.wait(vo.wait("learned", used=len(summary_lines) * 0.45))
        self.play(FadeOut(Group(*self.mobjects)))
