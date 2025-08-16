"""
RAG Pipeline — Manim Animation
================================
Animates the full Retrieval-Augmented Generation flow:
  Documents → Chunk → Embed → Vector Store → Query → Retrieve → Generate → Answer

Render:
    manim -ql rag_pipeline.py RAGPipeline
"""

from pathlib import Path

from manim import *
from voiceover import generate_voiceovers


NARRATION = {
    "intro": (
        "Retrieval-Augmented Generation, or RAG, is a technique that gives a language model "
        "access to external knowledge without retraining it. "
        "The pipeline has three phases: index, query, and generate."
    ),
    "phase_index": (
        "In the index phase, your documents are split into small chunks, "
        "each chunk is converted to a dense vector embedding, "
        "and all the embeddings are stored in a vector database."
    ),
    "phase_query": (
        "In the query phase, the user's question is embedded using the same model. "
        "The vector store finds the chunks whose embeddings are closest to the question — "
        "these are the most semantically relevant passages."
    ),
    "phase_generate": (
        "In the generate phase, the retrieved chunks are passed to the language model "
        "as context alongside the original question. "
        "The model then generates a grounded, factual answer."
    ),
    "summary": (
        "RAG grounds the model in your documents without any retraining. "
        "Swap the document store and you instantly update what the model knows."
    ),
}


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
DOC_COLOR    = BLUE_D
CHUNK_COLOR  = BLUE_B
VEC_COLOR    = GREEN_C
QUERY_COLOR  = YELLOW_C
MATCH_COLOR  = ORANGE
LLM_COLOR    = PURPLE_B
ANSWER_COLOR = GREEN_B


def box(text, width=2.0, height=0.7, txt_size=22, box_color=WHITE, txt_color=WHITE):
    """Labelled rounded rectangle."""
    rect = RoundedRectangle(corner_radius=0.15, width=width, height=height,
                            color=box_color, fill_color=box_color, fill_opacity=0.15,
                            stroke_width=2)
    label = Text(text, font_size=txt_size, color=txt_color)
    label.move_to(rect.get_center())
    return VGroup(rect, label)


def arrow(start_mob, end_mob, color=GRAY_B):
    return Arrow(
        start_mob.get_right(), end_mob.get_left(),
        buff=0.15, color=color, stroke_width=3,
        max_tip_length_to_length_ratio=0.15,
    )


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class RAGPipeline(Scene):
    def construct(self):
        vo = generate_voiceovers(NARRATION, Path(__file__).stem)

        # ── Title ──────────────────────────────────────────────────────────
        title = Text("Retrieval-Augmented Generation (RAG)", font_size=40)
        if vo.exists("intro"): self.add_sound(vo.path("intro"))
        self.play(Write(title))
        self.wait(vo.wait("intro", used=1.0))
        self.play(title.animate.to_edge(UP).scale(0.8))

        # ── Phase labels ───────────────────────────────────────────────────
        index_label  = Text("① INDEX", font_size=20, color=BLUE_C, weight=BOLD)
        query_label  = Text("② QUERY", font_size=20, color=YELLOW_C, weight=BOLD)
        gen_label    = Text("③ GENERATE", font_size=20, color=PURPLE_B, weight=BOLD)

        # ── PHASE 1: INDEX ────────────────────────────────────────────────
        index_label.to_edge(LEFT).shift(UP * 1.5)
        if vo.exists("phase_index"): self.add_sound(vo.path("phase_index"))
        self.play(FadeIn(index_label))

        # Documents
        docs = VGroup(*[
            box(f"Doc {i+1}", width=1.6, height=0.55, box_color=DOC_COLOR, txt_color=WHITE)
            for i in range(3)
        ]).arrange(DOWN, buff=0.25)
        docs.shift(LEFT * 5 + UP * 0.3)
        self.play(FadeIn(docs, shift=RIGHT * 0.3))

        docs_title = Text("Documents", font_size=20, color=DOC_COLOR)
        docs_title.next_to(docs, UP, buff=0.2)
        self.play(FadeIn(docs_title))
        self.wait(0.5)

        # Chunking arrow + chunks
        chunk_arrow = Arrow(docs.get_right(), docs.get_right() + RIGHT * 1.4,
                            buff=0.1, color=GRAY_B, stroke_width=3)
        chunk_label_txt = Text("chunk", font_size=16, color=GRAY_B)
        chunk_label_txt.next_to(chunk_arrow, UP, buff=0.1)
        self.play(GrowArrow(chunk_arrow), FadeIn(chunk_label_txt))

        chunks = VGroup(*[
            box(f"c{i+1}", width=0.8, height=0.45, box_color=CHUNK_COLOR, txt_color=WHITE)
            for i in range(5)
        ]).arrange(DOWN, buff=0.15)
        chunks.next_to(chunk_arrow, RIGHT, buff=0.15)
        self.play(FadeIn(chunks, shift=RIGHT * 0.2))
        self.wait(0.5)

        # Embedding arrow + vector store
        embed_arrow = Arrow(chunks.get_right(), chunks.get_right() + RIGHT * 1.4,
                            buff=0.1, color=GRAY_B, stroke_width=3)
        embed_lbl = Text("embed", font_size=16, color=GRAY_B)
        embed_lbl.next_to(embed_arrow, UP, buff=0.1)
        self.play(GrowArrow(embed_arrow), FadeIn(embed_lbl))

        # Vector store: scattered dots in a small region
        import random
        random.seed(7)
        dots = VGroup(*[
            Dot(
                point=embed_arrow.get_right() + RIGHT * (0.4 + random.random() * 1.4)
                       + UP * (random.random() * 1.6 - 0.8),
                radius=0.10,
                color=VEC_COLOR,
            )
            for _ in range(5)
        ])
        vs_rect = SurroundingRectangle(dots, color=VEC_COLOR, buff=0.25, corner_radius=0.1)
        vs_title = Text("Vector Store", font_size=18, color=VEC_COLOR)
        vs_title.next_to(vs_rect, UP, buff=0.15)

        self.play(FadeIn(dots, lag_ratio=0.15), Create(vs_rect), FadeIn(vs_title))
        self.wait(vo.wait("phase_index", used=3.5))

        # ── PHASE 2: QUERY ────────────────────────────────────────────────
        query_label.next_to(index_label, DOWN, buff=0.6).to_edge(LEFT)
        if vo.exists("phase_query"): self.add_sound(vo.path("phase_query"))
        self.play(FadeIn(query_label))

        query_box = box("What is LoRA?", width=2.2, height=0.6,
                        box_color=QUERY_COLOR, txt_color=BLACK)
        query_box.next_to(docs, DOWN, buff=1.0).shift(RIGHT * 0.5)
        self.play(FadeIn(query_box, shift=UP * 0.3))
        self.wait(0.3)

        # Query embeds → appears in vector store
        q_dot = Dot(vs_rect.get_center() + LEFT * 0.3 + DOWN * 0.15,
                    radius=0.14, color=QUERY_COLOR)
        q_embed_arrow = Arrow(query_box.get_right(),
                               q_dot.get_center(),
                               buff=0.12, color=QUERY_COLOR, stroke_width=2.5)
        q_embed_lbl = Text("embed query", font_size=15, color=QUERY_COLOR)
        q_embed_lbl.next_to(q_embed_arrow.get_center(), DOWN, buff=0.1)

        self.play(GrowArrow(q_embed_arrow), FadeIn(q_embed_lbl))
        self.play(FadeIn(q_dot))
        self.wait(0.3)

        # Highlight nearest dots
        nearest = [dots[1], dots[3]]
        similarity_circles = VGroup(*[
            Circle(radius=0.22, color=MATCH_COLOR, stroke_width=3).move_to(d.get_center())
            for d in nearest
        ])
        sim_label = Text("top-k similar", font_size=16, color=MATCH_COLOR)
        sim_label.next_to(vs_rect, DOWN, buff=0.2)

        self.play(Create(similarity_circles), FadeIn(sim_label))
        for d in nearest:
            self.play(d.animate.set_color(MATCH_COLOR), run_time=0.3)
        self.wait(vo.wait("phase_query", used=3.0))

        # ── PHASE 3: GENERATE ─────────────────────────────────────────────
        gen_label.next_to(query_label, DOWN, buff=0.6).to_edge(LEFT)
        if vo.exists("phase_generate"): self.add_sound(vo.path("phase_generate"))
        self.play(FadeIn(gen_label))

        # Chunks flow toward LLM box
        llm_box = box("LLM", width=1.8, height=1.0, box_color=LLM_COLOR, txt_color=WHITE)
        llm_box.to_edge(RIGHT).shift(LEFT * 0.6 + DOWN * 0.1)

        self.play(FadeIn(llm_box))

        # Context arrow (retrieved chunks → LLM)
        ctx_arrow = Arrow(vs_rect.get_right(), llm_box.get_left(),
                          buff=0.15, color=MATCH_COLOR, stroke_width=3)
        ctx_lbl = Text("retrieved context", font_size=16, color=MATCH_COLOR)
        ctx_lbl.next_to(ctx_arrow, UP, buff=0.1)
        self.play(GrowArrow(ctx_arrow), FadeIn(ctx_lbl))

        # Query also arrows to LLM
        q_to_llm = Arrow(query_box.get_right() + RIGHT * 0.2, llm_box.get_bottom() + LEFT * 0.1,
                         buff=0.1, color=QUERY_COLOR, stroke_width=2.5)
        self.play(GrowArrow(q_to_llm))
        self.wait(0.5)

        # Answer appears
        answer_box = box("LoRA trains two small\nmatrices A and B...",
                         width=2.6, height=0.9, box_color=ANSWER_COLOR, txt_size=18,
                         txt_color=WHITE)
        answer_box.next_to(llm_box, DOWN, buff=0.5)
        ans_arrow = Arrow(llm_box.get_bottom(), answer_box.get_top(),
                          buff=0.1, color=ANSWER_COLOR, stroke_width=3)

        self.play(GrowArrow(ans_arrow))
        self.play(FadeIn(answer_box, shift=DOWN * 0.2))
        self.wait(vo.wait("phase_generate", used=3.0))

        # ── Summary annotation ────────────────────────────────────────────
        summary = Text(
            "RAG grounds the model in your documents — no retraining required.",
            font_size=22, color=WHITE,
        )
        summary.to_edge(DOWN, buff=0.3)
        if vo.exists("summary"): self.add_sound(vo.path("summary"))
        self.play(FadeIn(summary))
        self.wait(vo.wait("summary", used=0.5))

        self.play(FadeOut(Group(*self.mobjects)))
