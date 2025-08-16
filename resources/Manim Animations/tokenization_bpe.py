"""
Tokenization & Byte-Pair Encoding (BPE) — Manim Animation
===========================================================
Shows how raw text is converted to tokens:
  1. Start with individual characters
  2. BPE merge rules fire iteratively, combining common pairs
  3. Final subword tokens are highlighted
  4. Demonstrates why the same word can tokenize differently

Render:
    manim -ql tokenization_bpe.py TokenizationBPE
"""

from pathlib import Path

from manim import *
from voiceover import generate_voiceovers


NARRATION = {
    "intro": (
        "Tokenization is the step that converts raw text into a sequence of numbers "
        "the model can actually process. "
        "Byte-Pair Encoding, or BPE, is the algorithm behind most modern tokenizers."
    ),
    "split_chars": (
        "BPE starts by splitting the input into individual characters. "
        "Every space is represented as a special underscore character "
        "so the model can reconstruct the original spacing."
    ),
    "bpe_merges": (
        "The algorithm then repeatedly finds the most frequent pair of adjacent tokens "
        "and merges them into a single new token. "
        "This fires over and over until the vocabulary reaches its target size."
    ),
    "final_tokens": (
        "The result is a compact set of subword tokens. "
        "Common words become single tokens; rare words are split into meaningful pieces. "
        "Our phrase reduces from fifteen characters to just three tokens."
    ),
    "token_ids": (
        "Each token is then mapped to an integer ID in the model's vocabulary. "
        "These IDs are what get passed as input to the embedding layer."
    ),
    "insight": (
        "The key insight: the model never processes individual letters or full words — "
        "it works with subword token IDs. "
        "Vocabulary size is a trade-off between granularity and sequence length."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOKEN_COLORS = [RED_B, ORANGE, YELLOW_C, GREEN_C, TEAL_B, BLUE_C, PURPLE_B, PINK]


def char_box(char, color=WHITE, size=30):
    """Single character in a coloured box."""
    rect = RoundedRectangle(
        corner_radius=0.1, width=0.55, height=0.6,
        color=color, fill_color=color, fill_opacity=0.2, stroke_width=1.8,
    )
    label = Text(char, font_size=size, color=color)
    label.move_to(rect.get_center())
    return VGroup(rect, label)


def token_box(token, color=WHITE, size=26):
    """Multi-character token in a wider coloured box."""
    width = max(0.55, len(token) * 0.32 + 0.3)
    rect = RoundedRectangle(
        corner_radius=0.12, width=width, height=0.62,
        color=color, fill_color=color, fill_opacity=0.25, stroke_width=2,
    )
    label = Text(token, font_size=size, color=color)
    label.move_to(rect.get_center())
    return VGroup(rect, label)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class TokenizationBPE(Scene):
    def construct(self):
        vo = generate_voiceovers(NARRATION, Path(__file__).stem)

        # ── Title ──────────────────────────────────────────────────────────
        title = Text("Tokenization & Byte-Pair Encoding", font_size=42)
        sub   = Text("How text becomes numbers a model can process", font_size=24, color=GRAY_B)
        sub.next_to(title, DOWN, buff=0.25)

        if vo.exists("intro"): self.add_sound(vo.path("intro"))
        self.play(Write(title))
        self.play(FadeIn(sub, shift=UP * 0.15))
        self.wait(vo.wait("intro", used=1.5))
        self.play(FadeOut(sub), title.animate.to_edge(UP).scale(0.78))

        # ── 1. Raw text ────────────────────────────────────────────────────
        sentence = "the transformer"

        raw_text = Text(f'"{sentence}"', font_size=46, color=WHITE)
        raw_text.move_to(ORIGIN + UP * 1.5)
        self.play(Write(raw_text))
        self.wait(0.8)

        if vo.exists("split_chars"): self.add_sound(vo.path("split_chars"))
        step_lbl = Text("Step 1: split into characters", font_size=24, color=GRAY_B)
        step_lbl.next_to(raw_text, DOWN, buff=0.45)
        self.play(FadeIn(step_lbl))
        self.wait(0.5)

        # ── 2. Character-level split ───────────────────────────────────────
        chars = list(sentence)
        char_boxes = VGroup(*[char_box(c if c != " " else "▁", color=GRAY_A) for c in chars])
        char_boxes.arrange(RIGHT, buff=0.12)
        char_boxes.move_to(ORIGIN + UP * 0.3)

        self.play(
            FadeOut(raw_text),
            FadeOut(step_lbl),
            FadeIn(char_boxes, shift=UP * 0.2),
        )

        char_count = Text(f"{len(chars)} characters", font_size=22, color=GRAY_B)
        char_count.next_to(char_boxes, DOWN, buff=0.3)
        self.play(FadeIn(char_count))
        self.wait(vo.wait("split_chars", used=2.0))

        # ── 3. BPE merge rules ────────────────────────────────────────────
        step2_lbl = Text("Step 2: BPE merge rules fire iteratively", font_size=24, color=GRAY_B)
        step2_lbl.to_edge(UP).shift(DOWN * 0.9)
        if vo.exists("bpe_merges"): self.add_sound(vo.path("bpe_merges"))
        self.play(FadeOut(char_count), FadeIn(step2_lbl))

        # We'll animate a simplified BPE: merge pairs until we reach
        # the tokens:  "the" | "▁" | "trans" | "former"
        # Show merges one at a time with highlight + collapse

        # Map chars to their index positions in char_boxes
        # sentence = "the transformer"
        # indices:    0123456789...
        # Merges to show:
        #   't'+'h' -> 'th'  (indices 0,1)
        #   'th'+'e' -> 'the' (indices 0,1,2)
        #   't'+'r' -> 'tr'  (indices 4,5)  -- inside "transformer"
        #   'tr'+'a' -> 'tra' (4,5,6)
        #   'tra'+'n' -> 'tran' (4,5,6,7)
        #   's'+'f' -> skip (not adjacent in real BPE)
        #   show final split: "the" | "▁" | "trans" | "former"

        # Simpler: just show 3 representative merges then jump to final tokens

        merges = [
            (0, 1, "th", "Merge 't' + 'h' → 'th'"),
            (0, 2, "the", "Merge 'th' + 'e' → 'the'"),
            (4, 5, "tr", "Merge 't' + 'r' → 'tr'  (inside 'transformer')"),
        ]

        # We operate on a mutable list of (token_string, original_indices)
        token_state = [(c, [i]) for i, c in enumerate(chars)]

        current_boxes = char_boxes.copy()
        current_boxes.move_to(ORIGIN + UP * 0.3)

        for merge_tok1, merge_tok2, merged, description in merges:
            # Find the two adjacent boxes to highlight
            # Recalculate positions based on current_boxes
            idx1, idx2 = None, None
            pos = 0
            for ti, (tok, orig) in enumerate(token_state):
                if merge_tok1 in orig or (len(tok) > 0 and chars[merge_tok1] == tok[0] and merge_tok1 in orig):
                    idx1 = ti
                if merge_tok2 in orig or (len(tok) > 0 and chars[merge_tok2] == tok[0] and merge_tok2 in orig):
                    idx2 = ti

            # Just highlight the first two boxes that match our merge step
            # For simplicity, highlight by position in current_boxes
            if idx1 is None or idx2 is None or idx1 >= len(current_boxes) or idx2 >= len(current_boxes):
                continue

            merge_desc = Text(description, font_size=20, color=YELLOW_C)
            merge_desc.next_to(step2_lbl, DOWN, buff=0.3)

            b1 = current_boxes[idx1]
            b2 = current_boxes[idx2]

            # Highlight the two being merged
            self.play(
                b1[0].animate.set_fill(YELLOW_E, opacity=0.6).set_stroke(YELLOW_C),
                b2[0].animate.set_fill(YELLOW_E, opacity=0.6).set_stroke(YELLOW_C),
                FadeIn(merge_desc),
            )
            self.wait(0.6)

            # Collapse them into one new box
            new_color = GREEN_C
            new_b = token_box(merged, color=new_color)
            new_b.move_to((b1.get_center() + b2.get_center()) / 2)

            self.play(
                FadeOut(b1), FadeOut(b2),
                FadeIn(new_b),
            )

            # Rebuild current_boxes with replacement
            new_boxes_list = []
            i = 0
            while i < len(current_boxes):
                if i == idx1:
                    new_boxes_list.append(new_b)
                    i += 2  # skip idx2
                else:
                    new_boxes_list.append(current_boxes[i])
                    i += 1
            current_boxes = VGroup(*new_boxes_list)
            current_boxes.arrange(RIGHT, buff=0.12)
            current_boxes.move_to(ORIGIN + UP * 0.3)

            self.play(current_boxes.animate.arrange(RIGHT, buff=0.12).move_to(ORIGIN + UP * 0.3))
            self.play(FadeOut(merge_desc))
            self.wait(0.4)

        self.play(FadeOut(step2_lbl))
        self.wait(0.3)

        # ── 4. Final tokens ────────────────────────────────────────────────
        step3_lbl = Text("Step 3: final subword tokens", font_size=24, color=GRAY_B)
        step3_lbl.to_edge(UP).shift(DOWN * 0.9)
        if vo.exists("final_tokens"): self.add_sound(vo.path("final_tokens"))
        self.play(FadeIn(step3_lbl))

        # Show the clean final tokenization of "the transformer"
        final_tokens = ["the", "▁trans", "former"]
        final_boxes = VGroup(*[
            token_box(t, color=TOKEN_COLORS[i % len(TOKEN_COLORS)], size=28)
            for i, t in enumerate(final_tokens)
        ])
        final_boxes.arrange(RIGHT, buff=0.2)
        final_boxes.move_to(ORIGIN + DOWN * 0.8)

        final_lbl = Text("Final tokens:", font_size=22, color=GRAY_B)
        final_lbl.next_to(final_boxes, LEFT, buff=0.4)

        self.play(FadeOut(current_boxes))
        self.play(FadeIn(final_lbl), FadeIn(final_boxes, lag_ratio=0.2))

        tok_count = Text(f"{len(final_tokens)} tokens  (was {len(chars)} characters)",
                         font_size=22, color=GREEN_C)
        tok_count.next_to(final_boxes, DOWN, buff=0.4)
        self.play(FadeIn(tok_count))
        self.wait(vo.wait("final_tokens", used=2.0))

        # ── 5. Token IDs ───────────────────────────────────────────────────
        id_lbl = Text("Each token maps to an integer ID in the model's vocabulary:",
                      font_size=21, color=GRAY_B)
        id_lbl.next_to(tok_count, DOWN, buff=0.5)
        if vo.exists("token_ids"): self.add_sound(vo.path("token_ids"))
        self.play(FadeIn(id_lbl))

        token_ids = [482, 5481, 15, 2264]   # illustrative IDs
        id_boxes = VGroup(*[
            VGroup(
                token_box(str(tid), color=GRAY_A, size=22),
            )
            for tid in token_ids[:len(final_tokens)]
        ])
        id_boxes.arrange(RIGHT, buff=0.2)
        id_boxes.next_to(id_lbl, DOWN, buff=0.35)
        id_boxes.shift(RIGHT * (final_boxes.get_center()[0] - id_boxes.get_center()[0]))

        arrows_down = VGroup(*[
            Arrow(final_boxes[i].get_bottom(), id_boxes[i].get_top(),
                  buff=0.1, color=GRAY_B, stroke_width=2, max_tip_length_to_length_ratio=0.2)
            for i in range(len(final_tokens))
        ])

        self.play(Create(arrows_down), FadeIn(id_boxes, lag_ratio=0.2))
        self.wait(vo.wait("token_ids", used=1.5))

        # ── Key insight ────────────────────────────────────────────────────
        if vo.exists("insight"): self.add_sound(vo.path("insight"))
        insight = Text(
            'The model never sees "letters" — only token IDs.\n'
            "Vocabulary size determines granularity vs. sequence length.",
            font_size=22, color=WHITE, line_spacing=1.3,
        )
        insight.to_edge(DOWN, buff=0.35)
        self.play(FadeIn(insight))
        self.wait(vo.wait("insight", used=0.5))

        self.play(FadeOut(Group(*self.mobjects)))
