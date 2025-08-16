"""
Mixture of Experts (MoE) — Manim Animation
============================================
Demonstrates sparse top-k expert routing:
  • A router network scores all experts for each token
  • Only the top-k experts (k=2) are activated — the rest are ignored
  • Outputs are weighted and summed
  • Shows the FLOPs advantage over a dense model

Render:
    manim -ql moe_expert_routing.py MoEExpertRouting
"""

from pathlib import Path

from manim import *
from voiceover import generate_voiceovers


NARRATION = {
    "intro": (
        "Mixture of Experts is an architecture that replaces a single large feed-forward "
        "layer with a collection of smaller expert networks. "
        "The key idea is sparsity: only a few experts process each token."
    ),
    "router": (
        "A lightweight router network takes the token embedding as input "
        "and produces a probability score for every expert. "
        "These scores tell the model which experts are most relevant for this token."
    ),
    "experts": (
        "The scores are broadcast to all eight expert networks. "
        "Each expert is a small feed-forward network that specializes "
        "in different types of patterns."
    ),
    "top_k": (
        "Instead of running all eight experts, the model selects only the top two — "
        "the ones with the highest router scores. "
        "The remaining six experts are skipped entirely for this token."
    ),
    "output": (
        "The outputs of the two active experts are combined as a weighted sum, "
        "where the weights come from the router's normalized scores. "
        "This is the gating mechanism."
    ),
    "flops": (
        "Here is why MoE is powerful. A dense model always runs all eight experts — "
        "eight times E floating-point operations per token. "
        "A sparse MoE model runs only two, giving you four times fewer operations "
        "with the same total parameter count."
    ),
}


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
N_EXPERTS   = 8
TOP_K       = 2
ACTIVE_IDX  = [2, 5]   # which experts the router selects (0-indexed)

ROUTER_COLOR = YELLOW_C
EXPERT_COLOR = BLUE_C
ACTIVE_COLOR = GREEN_C
INACTIVE_COLOR = GRAY_D
WEIGHT_COLOR = ORANGE


def expert_box(label, color=EXPERT_COLOR, width=1.3, height=0.65):
    rect = RoundedRectangle(
        corner_radius=0.12, width=width, height=height,
        color=color, fill_color=color, fill_opacity=0.2, stroke_width=2,
    )
    lbl = Text(label, font_size=20, color=color)
    lbl.move_to(rect.get_center())
    return VGroup(rect, lbl)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class MoEExpertRouting(Scene):
    def construct(self):
        vo = generate_voiceovers(NARRATION, Path(__file__).stem)

        # ── Title ──────────────────────────────────────────────────────────
        title = Text("Mixture of Experts (MoE)", font_size=44)
        sub   = Text("Sparse routing: only 2 of 8 experts activate per token",
                     font_size=24, color=GRAY_B)
        sub.next_to(title, DOWN, buff=0.25)

        if vo.exists("intro"): self.add_sound(vo.path("intro"))
        self.play(Write(title))
        self.play(FadeIn(sub, shift=UP * 0.15))
        self.wait(vo.wait("intro", used=1.5))
        self.play(FadeOut(sub), title.animate.to_edge(UP).scale(0.78))

        # ── Input token ────────────────────────────────────────────────────
        input_circle = Circle(radius=0.38, color=WHITE, fill_color=WHITE, fill_opacity=0.12)
        input_text   = Text("token\n x", font_size=22, color=WHITE)
        input_text.move_to(input_circle.get_center())
        input_mob = VGroup(input_circle, input_text)
        input_mob.move_to(UP * 2.8)

        self.play(FadeIn(input_mob, shift=DOWN * 0.3))
        self.wait(0.4)

        # ── Router network ─────────────────────────────────────────────────
        router_rect = RoundedRectangle(
            corner_radius=0.15, width=3.2, height=0.75,
            color=ROUTER_COLOR, fill_color=ROUTER_COLOR, fill_opacity=0.2, stroke_width=2.5,
        )
        router_text = Text("Router  (linear + softmax)", font_size=22, color=ROUTER_COLOR)
        router_text.move_to(router_rect.get_center())
        router_mob = VGroup(router_rect, router_text)
        router_mob.move_to(UP * 1.3)

        input_to_router = Arrow(input_mob.get_bottom(), router_mob.get_top(),
                                buff=0.12, color=GRAY_B, stroke_width=2.5)
        if vo.exists("router"): self.add_sound(vo.path("router"))
        self.play(GrowArrow(input_to_router))
        self.play(FadeIn(router_mob))
        self.wait(0.5)

        # Router produces a score for each expert
        router_note = Text("Produces a score for every expert", font_size=20, color=GRAY_B)
        router_note.next_to(router_mob, RIGHT, buff=0.4)
        self.play(FadeIn(router_note))
        self.wait(vo.wait("router", used=1.5))
        self.play(FadeOut(router_note))

        # ── Expert network boxes ───────────────────────────────────────────
        experts = VGroup(*[
            expert_box(f"Expert {i+1}", color=EXPERT_COLOR)
            for i in range(N_EXPERTS)
        ])
        experts.arrange(RIGHT, buff=0.28)
        experts.move_to(DOWN * 0.5)

        # Arrows from router to each expert
        router_to_experts = VGroup(*[
            Arrow(
                router_mob.get_bottom(),
                experts[i].get_top(),
                buff=0.12, color=GRAY_D, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            )
            for i in range(N_EXPERTS)
        ])

        if vo.exists("experts"): self.add_sound(vo.path("experts"))
        self.play(FadeIn(experts, lag_ratio=0.08))
        self.play(Create(router_to_experts, lag_ratio=0.05))
        self.wait(0.8)

        # ── Show all expert scores ─────────────────────────────────────────
        import random
        random.seed(42)
        raw_scores  = [random.uniform(0.05, 0.35) for _ in range(N_EXPERTS)]
        # Make the two chosen experts clearly highest
        raw_scores[ACTIVE_IDX[0]] = 0.72
        raw_scores[ACTIVE_IDX[1]] = 0.61
        total = sum(raw_scores)
        scores = [s / total for s in raw_scores]

        score_labels = VGroup(*[
            Text(f"{scores[i]:.2f}", font_size=17,
                 color=ACTIVE_COLOR if i in ACTIVE_IDX else GRAY_B)
            for i in range(N_EXPERTS)
        ])
        for i, lbl in enumerate(score_labels):
            lbl.next_to(experts[i], UP, buff=0.12)

        self.play(FadeIn(score_labels, lag_ratio=0.06))
        self.wait(vo.wait("experts", used=2.0))

        # ── Top-k selection: dim inactive experts ──────────────────────────
        top_k_lbl = Text(f"Select top-{TOP_K} experts", font_size=24, color=ACTIVE_COLOR, weight=BOLD)
        top_k_lbl.next_to(router_mob, DOWN, buff=0.15)

        if vo.exists("top_k"): self.add_sound(vo.path("top_k"))
        self.play(FadeIn(top_k_lbl))

        inactive = [i for i in range(N_EXPERTS) if i not in ACTIVE_IDX]
        self.play(
            *[experts[i][0].animate.set_fill(INACTIVE_COLOR, opacity=0.08)
                                   .set_stroke(INACTIVE_COLOR, opacity=0.3)
              for i in inactive],
            *[experts[i][1].animate.set_color(GRAY_D) for i in inactive],
            *[score_labels[i].animate.set_color(GRAY_D) for i in inactive],
            *[router_to_experts[i].animate.set_stroke(color=INACTIVE_COLOR, opacity=0.2)
              for i in inactive],
        )

        # Highlight active experts
        self.play(
            *[experts[i][0].animate.set_fill(ACTIVE_COLOR, opacity=0.3)
                                   .set_stroke(ACTIVE_COLOR)
              for i in ACTIVE_IDX],
            *[experts[i][1].animate.set_color(ACTIVE_COLOR) for i in ACTIVE_IDX],
            *[router_to_experts[i].animate.set_stroke(color=ACTIVE_COLOR, width=3)
              for i in ACTIVE_IDX],
        )
        self.wait(vo.wait("top_k", used=2.0))

        # ── Weighted output combination ────────────────────────────────────
        output_circle = Circle(radius=0.38, color=GREEN_B, fill_color=GREEN_B, fill_opacity=0.15)
        output_text   = Text("output\n  y", font_size=22, color=GREEN_B)
        output_text.move_to(output_circle.get_center())
        output_mob = VGroup(output_circle, output_text)
        output_mob.move_to(DOWN * 2.3)

        sum_lbl = Text(
            "y = Σ  gᵢ · Eᵢ(x)   [i ∈ top-k]",
            font_size=28, color=WEIGHT_COLOR,
        )
        sum_lbl.next_to(output_mob, RIGHT, buff= 0.6)

        active_to_output = VGroup(*[
            Arrow(experts[i].get_bottom(), output_mob.get_top(),
                  buff=0.12, color=ACTIVE_COLOR, stroke_width=2.5,
                  max_tip_length_to_length_ratio=0.18)
            for i in ACTIVE_IDX
        ])

        if vo.exists("output"): self.add_sound(vo.path("output"))
        self.play(FadeIn(output_mob))
        self.play(Create(active_to_output))
        self.play(Write(sum_lbl))
        self.wait(vo.wait("output", used=2.0))

        # ── FLOPs comparison ───────────────────────────────────────────────
        self.play(FadeOut(top_k_lbl))

        flops_title = Text("Why does this matter?", font_size=26, color=WHITE, weight=BOLD)
        flops_title.to_edge(DOWN, buff=1.6)

        dense_flops  = Text("Dense model (8 experts):  8 × E FLOPs per token",
                            font_size=21, color=RED_B)
        sparse_flops = Text(f"MoE model (top-{TOP_K} routing): {TOP_K} × E FLOPs per token",
                            font_size=21, color=GREEN_C)
        saving_text  = Text(f"→  {N_EXPERTS // TOP_K}× fewer FLOPs, same parameter count",
                            font_size=22, color=YELLOW_C, weight=BOLD)

        flops = VGroup(flops_title, dense_flops, sparse_flops, saving_text)
        flops.arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        flops.to_edge(DOWN, buff=0.3)

        if vo.exists("flops"): self.add_sound(vo.path("flops"))
        self.play(FadeIn(flops_title))
        self.play(FadeIn(dense_flops, shift=RIGHT * 0.2))
        self.play(FadeIn(sparse_flops, shift=RIGHT * 0.2))
        self.play(FadeIn(saving_text, shift=UP * 0.1))
        self.wait(vo.wait("flops", used=2.0))

        self.play(FadeOut(Group(*self.mobjects)))
